"""Benchmark the WITCH objective (MH / HMC / LM modes) on a TOD config."""

import argparse
import json
import time

import jax
import numpy as np


def timeit(fn, nrep):
    """
    Time a jitted function.

    Parameters
    ----------
    fn : callable
        Zero-argument callable returning a pytree of jax arrays.
    nrep : int
        Number of timed repetitions after the compile call.

    Returns
    -------
    compile_time : float
        Seconds taken by the first call.
    run_time : float
        Median seconds per call over `nrep` calls.
    out : pytree
        Output of the last call.
    """
    t0 = time.time()
    out = jax.block_until_ready(fn())
    compile_time = time.time() - t0
    times = []
    for _ in range(nrep):
        t0 = time.time()
        out = jax.block_until_ready(fn())
        times.append(time.time() - t0)
    return compile_time, float(np.median(times)), out


def main():
    """Run the benchmark, write timings and reference outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="unit_tests/RXJ1347_a10_single_tod.yaml")
    parser.add_argument("--nrep", type=int, default=20)
    parser.add_argument("--out", default="reports/bench.json")
    parser.add_argument("--ref", default=None, help="npz to save/compare outputs")
    parser.add_argument("--compare", action="store_true", help="compare to --ref")
    parser.add_argument("--debug", action="store_true", help="Few repetitions")
    args = parser.parse_args()
    if args.debug:
        args.nrep = 2

    from witch import fitter
    from witch.cfg_loader import load_cfg
    from witch.objective import joint_objective

    cfg = fitter.load_config({}, args.config)
    cli = fitter._make_parser().parse_args([args.config])
    _, _, _, mm = load_cfg(cfg, cli)

    modes = {"mh": (True, False, False), "hmc": (False, True, False)}
    modes["lm"] = (True, True, True)
    res, outs = {}, {}
    for name, (ll, g, c) in modes.items():
        ct, rt, out = timeit(lambda: joint_objective(mm, ll, g, c), args.nrep)
        res[name] = {"compile_s": ct, "run_s": rt}
        outs.update({f"{name}_{k}": np.asarray(v) for k, v in zip("lgc", out)})
        print(f"{name}: compile {ct:.2f}s run {rt * 1e3:.2f}ms")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.ref and not args.compare:
        np.savez(args.ref, **outs)
    elif args.ref:
        ref = np.load(args.ref)
        for k in ref.files:
            denom = np.abs(ref[k]).max() or 1.0
            print(f"{k}: max rel diff {np.abs(outs[k] - ref[k]).max() / denom:.2e}")


if __name__ == "__main__":
    main()
