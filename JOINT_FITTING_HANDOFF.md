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

## Current blocker — the joint LM fit doesn't converge
Smoke runs used a coarse grid (`r_map` 4′, `dr` = `dz` = 6″), `ntods: 2`, `n_rounds: 6`, `map: False`,
`model_map: False`.
- **M2 only** (`mpirun -n 2`): converges cleanly every round (Δχ² ≈ 1e-5, λ = 0).
- **Joint** (`mpirun -n 3`): round 1 makes progress. After that every LM step is rejected (Δχ² = inf,
  λ grows to about 5e8), and χ² goes *up* between rounds. Earlier, with m500 = 8e14 and maxiter 3, the
  fit gave `amp_act` = 125 ± 4 against an injected 50, and implausibly small errors.
- **Diagnosis so far:** the ACT dataset's χ² is inconsistent with its gradient/curvature, or the ACT
  noise or data is wrong.

Suspects:
- `BeamConvAndPrefac.apply_grad` / `apply_grad_proj` vs `apply` (`witch/dataset.py` ~175-200).
- Gradient projection for maps in `MetaModel.model_grad_proj` (`witch/containers/metamodel.py` ~340):
  `x, y = data.xy`, `_project_vectorized`.
- The point-source gradient when the model is excluded from metadata.
- ACT `NoiseWhite` normalization.
- Per-round noise re-estimation (`_reestimate_noise` in `fitter.py`) for maps.

**Next step: a finite-difference gradient check on an ACT-only metamodel** (single process, cheap). The
script below hasn't been run yet. Compare `fd dchisq` with `-2*grad`; they should match if grad = Aᵀ N⁻¹ r.
```python
import sys
import numpy as np, jax.numpy as jnp
from witch import fitter
from witch.cfg_loader import load_cfg
from witch.objective import joint_objective
cfg = fitter.load_config({}, sys.argv[1])          # e.g. a small override of joint_sim_act_m2.yaml
del cfg["datasets"]["mustang2"]
del cfg["metamodel"]["model_map"]["mustang2"]
cfg["metamodel"]["models"] = ["cluster_model", "ps_act"]
cfg["metamodel"].pop("parameter_map")
cli = fitter._make_parser().parse_args([sys.argv[1]])
_, _, _, mm = load_cfg(cfg, cli)
pars = jnp.array(mm.parameters); names = np.array(mm.par_names)
chisq, grad, curve = joint_objective(mm, True, True, True)
print("chisq", float(chisq), "npix", mm.datasets[0].datavec[0].data.size)
for i in np.where(np.array(mm.to_fit))[0]:
    h = 1e-4 * max(abs(float(pars[i])), 1.0)
    cs = [float(joint_objective(mm.update(pars.at[i].add(s*h), mm.errs, mm.cov, mm.chisq),
                                True, False, False)[0]) for s in (1, -1)]
    print(names[i], (cs[0]-cs[1])/(2*h), -2*float(grad[i]), float(curve[i, i]))
```
Also check that χ² at the injected truth is about the number of ACT pixels (1600) for the ACT-only
metamodel.

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
