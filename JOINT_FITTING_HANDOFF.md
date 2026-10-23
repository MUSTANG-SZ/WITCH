# Joint ACT + MUSTANG2 fitting — handoff log

Branch: `jack/joint-fitting`. Give this file to a new Claude session to pick up the work.
Delete it once the work is merged.

## Goal
1. Merge `perf/likelihood-speedups` into this branch. **Done.** Nothing to merge: it was already
   contained (`1bfe4f3`). `origin` couldn't be fetched from the old machine, so run
   `git fetch && git merge origin/perf/likelihood-speedups` to be sure.
2. Write simulated "unit test" configs for a simple cluster (gNFW + point source + shock), fit jointly
   with ACT maps and MUSTANG2 (M2) TODs, plus an MCMC variant. **Done** (`unit_tests/joint/`).
3. Write `tests/test_joint_act_m2_recovery.py`. It should confirm LM parameter recovery and quantify
   how much adding ACT shrinks the LM errors (expected mostly `alpha`, the large-scale slope).
   **Not written yet. It is blocked on the ACT gradient bug below.**

## Decisions made with the user
- Field: MOOJ1142. Data lives at `$WITCH_DATROOT/MOOJ1142/mustang2/Signal_TOD*.fits` (138 TODs) and
  `$WITCH_DATROOT/MOOJ1142/act/*_{map,ivar}.fits`. The ACT map is 40x40 pixels at 0.5′ in µK_CMB,
  with ivar ≈ 0.0053 (σ ≈ 13.7 µK per pixel).
  The env var is spelled **`WITCH_DATROOT`** (no "A").
- The test compares **LM errors** (σ from the covariance), M2-only vs joint. An MCMC config is
  provided but pytest doesn't run it.
- **Separate point-source models per dataset.** Models listed under `datasets.<name>.point_sources`
  skip both the beam and the prefactor. So `ps_m2` has σ = M2 beam with amp in K_RJ, and `ps_act` has
  σ = ACT beam with amp in µK_CMB. Positions are tied with `parameter_map`; amplitudes are independent.
- No "shock" structure exists. The shock uses the stage-1 `powerlaw_cos` structure (a multiplicative
  pressure enhancement in an ellipsoid), the same approach as `configs/MOOJ1142/MOOJ1142_SHOCK.yaml`.
  Because it is a stage-1 structure, the fast 1D LOS path is off and the model is integrated on the full 3D grid.
- The `gnfw` structure has `r = 1/wu.get_da(z)`. I checked numerically that this makes it identical to
  `a10` to 1e-16, so `r` is held fixed.

## Code changes (committed)
- `witch/fitting.py` `run_lmfit`: **fixed an MPI deadlock / collective mismatch**. Only all-TOD ranks
  called `global_comm.allreduce` to choose the chunked LM path; ACT ranks skipped it. Now every rank
  takes part. When any rank has more than one TOD, map ranks also use the chunked path, with one chunk
  per map. `_metamodel_for_tod` now handles `SolutionSet` (`.solutions`) as well as `TODVec` (`.tods`).
- `witch/fitter.py` `process_maps`: in a sim, pixels with ivar == 0 made the noise scale inf. They now
  get zero noise; they carry no weight anyway.
- `witch/containers/metamodel.py` `_compute_par_map_and_pars`: now raises if tied parameters disagree on
  `to_fit` or priors. Background:
  - `MetaModel.to_fit`, `to_fit_ever` and `priors` take the value from the last model that holds a
    tied index.
  - `model_grad_proj` only adds gradients for each model's own `to_fit` parameters.
  - So a tied parameter that isn't fit in every model gets silently frozen or gets a partial gradient.
    This froze the PS position in the first joint run.

## Configs (committed, `unit_tests/joint/`)
- `joint_sim_m2.yaml`: M2 only, self-contained.
  - Run settings: `sim: True`, `par_offset: 1.1`, `ntods: 8`, `n_rounds: 10`, `maxiter: 30`.
  - Grid: `r_map` 6′, `dr` = `dz` = 3″. `3*fwhm1` of the ACT beam must fit in `2*r_map`.
  - Injected and fitted values:

    | Component | Fixed values | Fitted (truth) |
    |---|---|---|
    | gnfw | — | `dx_1`=3, `dy_1`=−3, `m500`=5e14, `alpha`=1.551, `beta`=5.4905 |
    | shock | geometry, centred at (−30″, 20″) | `amp_sh`=1.5, `kr_sh`=1.0 |
    | ps_m2 | `sigma_m2`=10″/2.3548 | `dx_g`=20, `dy_g`=−15, `amp_m2`=1e-3 K_RJ |

- `joint_sim_act_m2.yaml`: `base: joint_sim_m2.yaml`. Adds:
  - the `act` dataset (`jm.NoiseWhite`, single-Gaussian 120″ beam, prefactor `y2K_CMB(97e9)*1e6`);
  - `ps_act`: `sigma_act` = 120″/2.3548, `amp_act` = 50 µK fitted, `dx_g`/`dy_g` tied to `ps_m2` with
    the same `to_fit` and priors;
  - `parameter_map`, `model_map.act`.
  - Config merging replaces lists, so `metamodel.models` is restated.
- `joint_sim_act_m2_mcmc.yaml`: MH (`method: mh`, 2000 steps, 2 chains, `proposal: cov`).
- Run with at least one rank per TOD, plus one for ACT: `srun -n 9 witcher unit_tests/joint/joint_sim_act_m2.yaml`.
- The old `unit_tests/joint_tests/*.yaml` use the legacy `model:` schema and can't load. Leave them or
  delete them.

## Joint LM blocker — RESOLVED (session 2)
- ACT gradient/curvature were fine: finite differences match `-2*grad` for ACT-only and for the joint
  3-rank metamodel; ACT-only chi2 at truth is ~1608 for 1600 px.
- Real cause: in round 1 `amp_sh` was driven to its lower bound 0, which makes `kr_sh` exactly degenerate
  (zero curvature row). The LM step used plain `jnp.linalg.inv` -> NaN pars -> NaN chi2 -> every step
  "rejected" (delta inf, lambda doubling to 512). Fix: `do_invsafe=True` for the step in both LM loops
  (`witch/fitting.py`). Smoke joint run now converges steadily and recovers truths roughly
  (amp_act 66 +- 4 vs 50 injected on the coarse 2-TOD grid; check on the full config).
- Also fixed: `act_map/funcs.py` `load_maps` failed on big-endian FITS (`>f4`) -> cast to float.
- Open question: why amp_sh hit 0 in round 1 (starting at 1.1x truth). Probably a large Newton step
  from the m500/amp_sh degeneracy; worth checking if the full config still does it.

## Remaining TODO after the fix
1. Check that both full configs converge and recover the truths (8 TODs, 9 ranks).
2. Write `tests/test_joint_act_m2_recovery.py`, following the pattern of `tests/test_rxj1347_tod_recovery.py`:
   - Run `witcher` as a subprocess, using `srun` if `SLURM_JOB_ID` is set and `mpirun` otherwise.
   - Use a tmp wrapper config with `base: <abs path>` and `map: False` to skip mapmaking.
   - Give each run its own `WITCH_OUTROOT`.
   - Read `par_results_final_fit.dill` (`par_names`, `parameters`, `errs`, `to_fit`).
   - Skip if the data, `witcher` or the launcher is missing.
   - Use a module-scoped fixture that runs M2-only and joint once each.
   - Asserts:
     - each fitted parameter is finite with err > 0, and `|fit − truth| ≤ max(3σ, 5%)`;
     - σ_joint(alpha) < 0.8·σ_M2(alpha), with the threshold tuned after a real run;
     - σ_joint ≤ 1.05·σ_M2 for all shared parameters.
   - Print a table of truth, fit ± σ for both runs, and the σ ratio.
   - The tied PS position is reported as `dx_g`/`dy_g` in both runs because both PS models use the
     same parameter names.
3. Add a mock-based test to `tests/test_fitting.py` with one TOD dataset and one map dataset going
   through `_tod_chunked_objective`.
4. Smaller issues found and not fixed:
   - `cfg["fit"]` defaults to `"model" in cfg`, so it is False for metamodel configs. Set `fit: True`.
   - `xfer` is read from `info["info"]` instead of `info["xfer"]` (`fitter.py` `process_tods` and `process_maps`).
   - ACT and M2 white-noise sims all use `jax.random.key(0)`.
   - The beam kernel is truncated at ±1.5·fwhm1 (`utils.py` ~442), so a wide second Gaussian is cut off.
   - The beam FFT convolution is circular with no padding.
   - The `parameter_map` index logic only works if each master comes before its tied parameters.
5. Memory: about 4 GB per rank, dominated by the JIT-compiled model (the full 3D grid here).

## Session 3 (2026-10-23): real MOOJ1142 joint fit, ACT pulls the profile out
Config: `configs/MOOJ1142_Joint/MOOJ1142_joint_M2_ACT_gNFW_rs_peak_subcluster{,_mcmc}.yaml`
(sph_isobeta subcluster + gnfw_rs + `ps_model`, plus ACT with `ps_act`).

- **Fixed (`451ae64`)**: `UnexpectedTracerError` in the chunked LM. `WCSMap.xy` is a jitkasi
  `cached_property`. The per-TOD objectives close over each map, so `xy` was first evaluated inside
  jit and a tracer got cached on the map. `_make_tod_objectives` now touches `tod.xy` eagerly for
  map datasets. The user wants jitkasi left alone; the better fix is for jitkasi to build `xy` with
  numpy or under `jax.ensure_compile_time_eval()`.
- **Open problem**: M2 alone gives gNFW r_s ≈ 24 (the user's number). Joint drives r_s to its upper
  bound of 50, isobeta r to about 47, gNFW beta to about 7.4, and P0 from about 0.15 to 0.036,
  while total chi2 barely moves (about 100 out of 4.3e7). The last LM round on Trillium was
  `.../MOOJ1142/cluster_model_sph_isobeta-gnfw_rs+ps_model_ps_gauss+ps_act_ps_gauss/joint_xray_peak_rs/r-beta-amp-P0-r_s-beta-dx_g-dy_g-sigma-amp_g-amp_g/par_results_fit19.dill`
  (the job was killed before the MCMC; don't trust the joint cluster profile).
- Suspected causes, most likely first (not yet confirmed):
  1. **The ACT noise model ignores the CMB.** The map std is 68 µK, but 1/√ivar is about 14 µK.
     `jm.NoiseWhite` treats the CMB as signal, with about 25× too much weight. Large-scale CMB
     under the cluster can only be absorbed by a more extended SZ profile, and with gamma and
     alpha fixed that means r_s and beta. Fix: a CMB+noise covariance (Fourier space), or
     high-pass both the ACT data and the model.
  2. **The model grid (`r_map` = 3′, so ±180″) covers only the centre of the 20′×20′ ACT map.**
     `bilinear_interp` gives zero outside the grid, so the model is truncated about 1.5 ACT beams
     from the centre.
  3. **`load_maps` zero-means the ACT data but not the model.**
- **Diagnostic to run next (on the personal cluster, not Trillium)**:
  `scripts/act_chisq_diagnostic.py`. It loads only the ACT dataset (one process, no TODs) and,
  for each `par_results_*.dill`, prints the ACT chi2:
  - with the parameters as saved;
  - with the ACT point source amplitude re-solved (exact, since the model is linear in it);
  - with the model zero-meaned as well.

  It also prints the grid coverage of the map and writes model and residual FITS plus a radial
  profile PNG under `<outroot>/.../act_diagnostic/not_fit/`. The script is untested so far.

  ```
  python scripts/act_chisq_diagnostic.py \
      configs/MOOJ1142_Joint/MOOJ1142_joint_M2_ACT_gNFW_rs_peak_subcluster.yaml \
      <m2_only_rs24>/par_results_final_fit.dill <joint>/par_results_fit19.dill \
      --labels m2_only joint
  ```

  Use the M2-only run that gave r_s ≈ 24. The `xray_peak_rs/r-beta-amp-P0-beta-...` run on
  Trillium has r_s fixed at 4.33, so it's the wrong one. Copy the dills over from Trillium.
  Reading the result: if ACT strongly prefers the joint parameters (a large Δchi2 in the
  `ps_solved` column) and the residual or profile shows large-scale structure out to the map
  edge, that confirms cause 1.
