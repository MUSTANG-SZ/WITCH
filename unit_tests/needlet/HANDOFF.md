# Handoff: joint MOOJ1142 M2 + ACT fit and needlet mapmaking (cluster run)

Written 2026-11-19 on `psd` (8 cores, 251 GB, one A6000). Everything here was developed and smoke tested there; the
production run is too big for it and is yours. Read this whole file before starting. Ask the user when something here
does not match what you find (paths, branches, data); do not guess.

## Goal

1. Run the joint MUSTANG-2 (TODs) + ACT (map) fit of MOOJ1142 with WITCH.
2. In the same run, make the joint M2 + ACT needlet maps after the fit: signal, residual, model and the unfiltered model
   ("truth"), with the production needlet settings.
3. Check and report them (HTML report with figures, see "Deliverables"), including how additive the mapmaker is at
   production iteration counts.

## How the joint maps work (read this, it explains the outputs)

- WITCH (`witch/fitter.py: _postfit`) calls `witch/external/joint/funcs.py: joint_post_fit` after the fit when the
  config has a `joint_map` block, on every rank, before the per-dataset postfit (act_map's postfit overwrites its map
  data in place, so the order matters).
- `joint_post_fit` builds one minkasi `JointMapper` (needlet basis, M2 TOD term + ACT map term, CG on the full operator):
  `run()` maps the signal (data), `remap()` sends the residual (data - model) through the same mapper (same noise
  model, response matrices, factorized preconditioner), and model = signal - residual, i.e. the model as it sits in the
  signal map. A finite PCG/CG is not additive (the Krylov coefficients depend on the data), which is why the model map
  is not the model mapped on its own.
- `model_truth.fits` is the fitted model with the M2 beam and prefactor evaluated on the needlet pixels, no filtering.
  It is only made if the needlet pixel equals `coords.dr` (the beam is sampled at `dr`); the config sets both to 4".
  model / truth is the mapmaker's transfer for this model.
- MPI: each WITCH rank holds one dataset (M2 TODs split over the M2 ranks, ACT on one rank). minkasi has no comm
  argument; `witch.external.minkasi.utils.minkasi_comm` points it at the WITCH comm for the joint block. Every rank,
  ACT ones included, holds ALL needlet response matrices: memory per rank is set by the needlet lattice, not by the
  TODs on the rank (see the dry run).

## Code

| repo | branch | needed commits |
|---|---|---|
| WITCH (`git@github.com:MUSTANG-SZ/WITCH.git`) | `needlet-maps` | the commit that adds this file, plus `4f85820` (minkasi comm, HasPrior fix) |
| minkasi (`git@github.com:sievers/minkasi.git`) | `needlet/joint-api` | `f917b85`, `c6e759d` (`JointMapper.remap`), `f6abdc1` (`tests/check_remap.py`) |
| jitkasi | release | >= 3.1.1 (TAN WCS fix) |

If those branches are not on the remote, stop and ask the user to push them. Versions on psd: jax 0.11.2, numpy 2.5.3,
mpi4py 4.0.0, pixell 0.26.1, jitkasi 3.1.1.

Install with uv (project rule), into one venv: `uv venv && source .venv/bin/activate`, then `uv pip install -e` the
three repos (jitkasi, minkasi, WITCH; minkasi builds its C extension, see its README), plus `jax[cuda12]` on GPU nodes.
Check: `python -c "import witch.external.joint.funcs, minkasi.needlet.joint"`.

## Data

`WITCH_DATROOT` must contain `MOOJ1142/mustang2/` (138 `Signal_TOD*.fits`, the MOOJ1142 blacklist in
`minkasi.tools.presets_by_source` removes some) and `MOOJ1142/act/` (`MOOJ1142_act_planck_s08_s22_f090_daynight_{map,ivar}.fits`).
On psd these are under `/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/`. Set `WITCH_OUTROOT` to a scratch directory, and set
`joint_map.options.cache` in the config to an absolute scratch path (needlet response caches, several GB).

## Configs (all in `configs/MOOJ1142_Joint/`)

- `MOOJ1142_joint_needlet.yaml`: the production run. Base: `MOOJ1142_joint_M2_ACT_gNFW_rs_peak_subcluster.yaml`
  (gNFW r_s + peak + subcluster + point sources, ACT with CMB noise and its own point source, `n_rounds: 40`).
  Needlet options: the minkasi baseline `cg: 600, cg_tol: 0, lattice_tol: 1e-3, min_hits: 0` (CG implies `direct`),
  `pixsize: 4` (= `dr`), `freq: 9.7e+10` (write it with a signed exponent: PyYAML reads `97.0e9` as a string).
- `MOOJ1142_joint_needlet_dryrun.yaml`: no fit, no maps, `dry_run`: prints the needlet lattice and memory per rank.
- `MOOJ1142_joint_needlet_smoke.yaml`: no fit, 2 TODs, debug-size needlets; checks the plumbing in minutes.

## Launch rules

- `mpirun --bind-to none -n <N> witcher <config>` (or `srun --cpu-bind=none`). Without `--bind-to none` OpenMPI pins
  ranks to 2 cores and XLA's CPU runtime deadlocks (0% CPU forever). Diagnose hangs with `py-spy dump`, not by waiting.
- N >= 2 (one rank per dataset at least). ACT gets one rank, the M2 TODs are spread over the rest; ranks beyond the number
  of files exit. The base config's header says "one MPI task per TOD plus one for ACT"; `_mpi_fsplit` does allow several
  TODs per rank, so treat that comment as unverified and check what you see.
- Several ranks on one GPU: set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or JAX will grab 75% of the GPU per rank.
  `--cpu True` forces CPU.
- Set `OMP_NUM_THREADS` (minkasi's C code) to cores / ranks per node.

## Steps

0. **Install and smoke test.** On one node, 2 ranks:
   `mpirun --bind-to none -n 2 witcher configs/MOOJ1142_Joint/MOOJ1142_joint_needlet_smoke.yaml --cpu True`.
   Then `python unit_tests/needlet/plot_joint_maps.py <outdir>/needlet --figdir <somewhere>` where `<outdir>` is printed
   as "Outputs can be found in". Expected on psd: `max |model - (signal - residual)|: 0`, corr(model, truth) 0.98,
   no "Not making the truth map" line, remap solves ~2 s against ~30 s for the first solve.
1. **remap check.** `python ../minkasi/tests/check_remap.py --debug --idir <datroot>/MOOJ1142/mustang2 --act-glob
   '<datroot>/MOOJ1142/act/*_ivar.fits'`. Expected: `remap(same data) vs run` ~1e-11, `2 x data` ~1e-11,
   `a + b` ~0.1 (debug: 5 CG iterations). Exit code 0.
2. **Dry run** with the rank count you intend to use:
   `mpirun --bind-to none -n <N> witcher configs/MOOJ1142_Joint/MOOJ1142_joint_needlet_dryrun.yaml`. It prints each
   band's lattice and "per rank: resident X GB, peak while probing ~Y GB, plus the TODs". On psd (2 ranks, all ~106
   TODs after the blacklist) it gave a 360 x 360 map of 4" pixels, 10 bands, 0.15 G matrix entries: resident 1.8 GB,
   peak while probing ~5.2 GB per rank; the result does not depend on N, check that you get the same lattice.
   Pick N and nodes so that every rank has Y + its TODs (~0.1 GB per TOD) + the WITCH fit's own memory, with margin.
   Do not run two probing jobs at once on a node.
3. **Production run:** `mpirun --bind-to none -n <N> witcher configs/MOOJ1142_Joint/MOOJ1142_joint_needlet.yaml`.
   Fit (up to 40 rounds), per-dataset postfit maps, then the joint maps. The M2 response matrices are probed once and
   cached (`<cache>/MOOJ1142_signal_nyq0.001_m2*`); a rerun with the same TODs and noise reuses them (`redo: True` in
   `joint_map.options` to force). Note the run's wall time per stage.
4. **Analysis.** `python unit_tests/needlet/plot_joint_maps.py <outdir>/needlet --figdir <figdir>`: consistency,
   corr(model, truth), peak ratio, radial model / truth (transfer). Also look at the fitted parameters and chi2 (WITCH
   outputs in `<outdir>`), the residual map for structure at the cluster position, and the per-band behaviour if odd.
5. **Additivity at production settings.** The debug numbers (4-10%) come from very unconverged solves. Measure it with
   the production solve on a subset, e.g. `mpirun --bind-to none -n 8 python ../minkasi/tests/check_remap.py --cg 600
   --cg-tol 0 --lattice-tol 1e-3 --ntods-total 40 --idir ... --act-glob ... --outdir <scratch>` (add `--ntods` so
   ranks x ntods >= 40). Report the three numbers. If the a + b number is still large, say so: it decides how much the
   model map depends on the data.

## Deliverables

- An HTML report `reports/needlet_joint_maps_production.html` (self-contained, figures embedded as base64 PNG, like the
  other files in `reports/`; that directory is gitignored, so also tell the user where it is). Content: the takeaway
  first; the fit result (parameters, chi2); the four maps and the transfer figure; the remap and additivity numbers;
  timings and memory per stage; anything that went wrong. Compare with the psd smoke test in
  `reports/needlet_joint_maps.html` if the user has copied it over.
- A short summary back to the user with the numbers and file paths.

## Known gotchas

- act_map's postfit replaces the ACT map data with the residual and then the model. The joint block runs before it
  for that reason. The nonpara branch of `main()` now passes the nonpara metamodel to the per-dataset postfit (it used
  to pass the parametric one); nonpara is off in this config.
- The needlet response caches are keyed on `<name>_signal` plus lattice tags, not on the TOD list or noise. If you change
  the TODs, the noise or the fit between runs, use a new cache directory or `redo: True`.
- `options` in `joint_map` take any `minkasi.needlet.joint.parse_args` option by its dest name (`cg_tol`, `act_cmb`, ...);
  unknown keys raise. `sim_sky` is not usable from WITCH (simulations are done on the WITCH side).
- The truth map is computed on the first rank holding the M2 dataset and broadcast; it is skipped with a warning if the
  needlet pixel differs from `coords.dr`.
- Do not push to `main`, open PRs, or change the minkasi `--sim-sky` path; ask the user first.
