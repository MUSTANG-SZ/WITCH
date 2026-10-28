"""
Benchmark one MPI layout of a joint ACT+M2 fit on a single node, then estimate the full run.

Run `run` once per layout with the node packed the way one production node would be
(see `bench_joint_layout.sh`), then `summarize` to compare layouts and estimate the run time.
`check` predicts the node memory of a layout from earlier runs so the driver can skip ones
that would not fit.
"""

import argparse
import glob
import json
import os
import resource
import socket
import sys
import time

import numpy as np

N_CG = 26 + 51  # PCG iterations per M2 map with npass=1, see mapmaking.make_maps
N_POSTFIT_MAPS = 3  # signal_res_noise, residual and model maps
FAST_ITERS = 7  # LM iterations of a round that is already converged


def _rss_gb() -> float:
    """
    Peak resident memory of this process.

    Returns
    -------
    rss : float
        Peak RSS in GB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run(args: argparse.Namespace):
    """
    Load the data, make one M2 map and time the objective on this layout.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed `run` arguments.
    """
    import jax

    from bench_objective import timeit
    from witch import fitter
    from witch.cfg_loader import load_cfg
    from witch.objective import joint_objective

    cfg = fitter.load_config({}, args.config)
    cfg["datasets"][args.m2]["ntods"] = args.ntods
    cfg["paths"]["subdir"] = f"bench_layout/{args.label}"
    cfg["map"] = False
    cli = fitter._make_parser().parse_args([args.config, "--cpu", "True"])

    t0 = time.time()
    datasets, _, _, mm = load_cfg(cfg, cli)
    comm = fitter.comm
    t_load = time.time() - t0
    rss_load = _rss_gb()

    # One signal map, the same mapmaking as each postfit map
    cfg["map"] = True
    t0 = time.time()
    for ds in datasets:
        comm.barrier()
        ds.postproc(ds, cfg, mm)
    comm.barrier()
    t_map = time.time() - t0

    res = {}
    for name, flags in {
        "chisq": (True, False, False),
        "lm": (True, True, True),
    }.items():
        ct, rt, _ = timeit(lambda f=flags: joint_objective(mm, *f), args.nrep)
        res[name] = {"compile_s": ct, "run_s": rt}

    t0 = time.time()
    jax.block_until_ready(fitter._reestimate_noise(mm))
    t_reest = time.time() - t0

    rank = {
        "host": socket.gethostname(),
        "dsets": [ds.name for ds in datasets],
        "ntods": sum(len(d.datavec.tods) for d in mm.datasets if d.mode == "tod"),
        "load_s": t_load,
        "reest_s": t_reest,
        "rss_load_gb": rss_load,
        "rss_gb": _rss_gb(),
    }
    ranks = comm.gather(rank, root=0)
    if comm.Get_rank() != 0:
        return
    out = {
        "label": args.label,
        "nranks": comm.Get_size(),
        "threads": int(os.environ.get("SLURM_CPUS_PER_TASK", 0)),
        "ntods": args.ntods,
        "n_rounds": int(cfg.get("n_rounds", 1)),
        "maxiter": int(eval(str(cfg["fitting"].get("maxiter", 10)))),
        "map_s": t_map,
        "timings": res,
        "ranks": ranks,
    }
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, f"{args.label}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("label", "nranks", "threads", "timings")}))


def _load(results: str) -> list[dict]:
    """
    Read every benchmark result in a directory.

    Parameters
    ----------
    results : str
        Directory with the `run` JSON files.

    Returns
    -------
    runs : list[dict]
        The results, most TODs per rank first.
    """
    runs = []
    for path in glob.glob(os.path.join(results, "*.json")):
        with open(path) as f:
            runs += [json.load(f)]
    return sorted(runs, key=lambda r: -max(x["ntods"] for x in r["ranks"]))


def _mem_model(runs: list[dict]) -> tuple[float, float]:
    """
    Fit the peak memory of the M2 ranks as `fixed + per_tod * ntods`.

    Parameters
    ----------
    runs : list[dict]
        Benchmark results.

    Returns
    -------
    fixed : float
        Memory per rank that does not depend on the TODs, in GB.
    per_tod : float
        Memory per TOD, in GB. Zero if all ranks had the same number of TODs.
    """
    pts = np.array(
        [(x["ntods"], x["rss_gb"]) for r in runs for x in r["ranks"] if x["ntods"]]
    )
    if len(np.unique(pts[:, 0])) < 2:
        return float(pts[:, 1].max()), 0.0
    per_tod, fixed = np.polyfit(pts[:, 0], pts[:, 1], 1)
    resid = pts[:, 1] - (fixed + per_tod * pts[:, 0])
    return float(fixed + resid.max()), float(per_tod)  # Shift up to bound every rank


def check(args: argparse.Namespace):
    """
    Exit with status 1 if a layout is predicted not to fit in the node memory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed `check` arguments.
    """
    runs = _load(args.results)
    if not runs:
        return
    fixed, per_tod = _mem_model(runs)
    if per_tod == 0 and args.tods_per_rank > max(x["ntods"] for x in runs[0]["ranks"]):
        return  # One measurement with fewer TODs per rank, can't bound this layout
    need = args.ranks * (fixed + per_tod * args.tods_per_rank)
    print(
        f"Predicted node memory for {args.ranks} ranks x {args.tods_per_rank} TODs: {need:.0f} GB"
    )
    if need > args.mem_gb:
        sys.exit(1)


def summarize(args: argparse.Namespace):
    """
    Print each layout's measurements and the estimated run time on `--nodes` nodes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed `summarize` arguments.
    """
    runs = _load(args.results)
    if not runs:
        sys.exit(f"No results in {args.results}")
    fixed, per_tod = _mem_model(runs)
    print(f"Memory per M2 rank ~ {fixed:.1f} GB + {per_tod:.2f} GB per TOD\n")
    hdr = "layout        ranks thr TOD/rk  rss/rk  node GB  load  map  lm_comp  lm_eval  chisq   est (fast-slow)"
    print(hdr + "\n" + "-" * len(hdr))
    for r in runs:
        m2 = [x for x in r["ranks"] if x["ntods"]]
        k = max(x["ntods"] for x in m2)
        t = r["timings"]
        n_rounds, maxiter = r["n_rounds"], r["maxiter"]
        base = (
            max(x["load_s"] for x in r["ranks"])
            + r["map_s"]  # The signal map made while loading
            + t["chisq"]["compile_s"]
            # run_lmfit builds a new while_loop every round, so LM compiles per round
            + n_rounds * (t["lm"]["compile_s"] + max(x["reest_s"] for x in r["ranks"]))
            + N_POSTFIT_MAPS * r["map_s"]
        )
        fast = base + n_rounds * (1 + FAST_ITERS) * t["lm"]["run_s"]
        slow = base + n_rounds * (1 + maxiter) * t["lm"]["run_s"]
        node = sum(x["rss_gb"] for x in r["ranks"])
        print(
            f"{r['label']:13s} {r['nranks']:5d} {r['threads']:3d} {k:6d} "
            f"{max(x['rss_gb'] for x in m2):6.1f} {node:8.0f} "
            f"{max(x['load_s'] for x in r['ranks']):5.0f} {r['map_s']:4.0f} "
            f"{t['lm']['compile_s']:8.0f} {t['lm']['run_s']:8.2f} {t['chisq']['run_s']:6.2f}   "
            f"{fast / 3600:.1f}-{slow / 3600:.1f} h"
        )
    print(
        f"\nTimes in s. 'est' is for the full fit on {args.nodes} nodes with this many TODs per rank:"
        f" load + signal map + chisq compile + n_rounds x (LM compile + noise re-estimate + (1 + iters) LM evals)"
        f" + {N_POSTFIT_MAPS} postfit maps, with iters = {FAST_ITERS} (converges quickly) to maxiter."
        " The node is packed like a production node, so inter-node allreduces are the only thing missing."
    )


def main():
    """Parse the CLI and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="benchmark this MPI layout (launch with srun)")
    p.add_argument("--config", default="configs/A399_401/A399_401_joint_act_m2.yaml")
    p.add_argument("--m2", default="mustang2", help="name of the TOD dataset")
    p.add_argument("--ntods", type=int, required=True, help="TODs to load on this node")
    p.add_argument("--nrep", type=int, default=3)
    p.add_argument("--label", required=True)
    p.add_argument("--results", required=True, help="directory for the JSON results")
    p.add_argument("--debug", action="store_true", help="1 repetition")
    c = sub.add_parser("check", help="exit 1 if a layout would not fit in memory")
    c.add_argument("--results", required=True)
    c.add_argument("--ranks", type=int, required=True)
    c.add_argument("--tods-per-rank", type=int, required=True)
    c.add_argument("--mem-gb", type=float, default=700.0)
    s = sub.add_parser("summarize", help="compare layouts and estimate the run time")
    s.add_argument("--results", required=True)
    s.add_argument("--nodes", type=int, default=4)
    args = parser.parse_args()
    if args.cmd == "run" and args.debug:
        args.nrep = 1
    {"run": run, "check": check, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
