"""
Run WITCH sims over several noise realisations (witcher --sim-seed), a few jobs at a time.
Each (config, seed) pair is one mpirun job writing to its own `...-sim_s<seed>` output directory.
Seed 0 is the default noise realisation so it is usually already run.
"""

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


def run_job(config: str, ranks: int, seed: int, log_dir: str) -> tuple[str, int, int]:
    """
    Run one simulation and fit.

    Parameters
    ----------
    config : str
        WITCH config.
    ranks : int
        Number of MPI tasks, at least one per dataset.
    seed : int
        Noise realisation, passed as --sim-seed.
    log_dir : str
        Directory for the log file.

    Returns
    -------
    config, seed, code : tuple[str, int, int]
        The job and its exit code.
    """
    name = os.path.splitext(os.path.basename(config))[0]
    log = os.path.join(log_dir, f"{name}_s{seed}.log")
    cmd = ["mpirun", "--bind-to", "none", "-n", str(ranks), "witcher", config]
    cmd += ["--sim-seed", str(seed)]
    t0 = time.time()
    with open(log, "w") as f:
        code = subprocess.call(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ | {"JAX_PLATFORMS": "cpu"},
        )
    print(
        f"{name} seed {seed}: exit {code} after {(time.time() - t0) / 60:.1f} min, log {log}",
        flush=True,
    )
    return config, seed, code


def main():
    """Run the jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="WITCH configs")
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        required=True,
        help="MPI tasks per config, in the same order",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--parallel", type=int, default=3, help="Jobs at once")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument(
        "--debug", action="store_true", help="Only the first config and seed"
    )
    args = parser.parse_args()
    if len(args.ranks) != len(args.configs):
        parser.error("Need one --ranks entry per config")
    jobs = [(c, r, s) for s in args.seeds for c, r in zip(args.configs, args.ranks)]
    if args.debug:
        jobs = jobs[:1]
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"{len(jobs)} jobs, {args.parallel} at a time", flush=True)
    with ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(lambda j: run_job(*j, args.log_dir), jobs))
    failed = [(os.path.basename(c), s) for c, s, code in results if code]
    print("all done" if not failed else f"failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
