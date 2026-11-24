"""Benchmark the WITCH objective (MH / HMC / LM modes) and the MH loop on a TOD config.

Runs on whatever device JAX picks, set `JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda` to choose.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
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


def with_mh_errs(metamodel):
    """
    Give the fitted parameters small errors so MH proposals move.
    Configs loaded without a fit have zero errors, which would make every proposal the current state.

    Parameters
    ----------
    metamodel : MetaModel
        The metamodel to update, modified in place.

    Returns
    -------
    metamodel : MetaModel
        The updated metamodel.
    """
    fit = np.asarray(metamodel.to_fit)
    pars = np.asarray(metamodel.parameters)
    errs = np.where(fit, 1e-3 * np.where(pars != 0, np.abs(pars), 1.0), 0.0)
    return metamodel.update(
        metamodel.parameters, jnp.asarray(errs), metamodel.cov, metamodel.chisq
    )


def time_mh(metamodel, nsteps):
    """
    Time the MH loop as `run_mh` runs it (serial chain, joint likelihood).

    Parameters
    ----------
    metamodel : MetaModel
        The metamodel to sample.
    nsteps : int
        Number of timed MH steps.

    Returns
    -------
    compile_time : float
        Seconds for a 2 step warmup chain, dominated by compilation.
    steps_per_s : float
        MH steps per second over `nsteps` steps.
    acceptance : float
        Acceptance rate of the timed chain.
    """
    from witch.metropolis_hastings import calc_like_joint, metropolis_hastings

    kwargs = dict(calc_like_func=calc_like_joint, progress=False, proposal="diag")
    t0 = time.time()
    metropolis_hastings(metamodel, 2, **kwargs)
    compile_time = time.time() - t0
    t0 = time.time()
    _, acc = metropolis_hastings(metamodel, nsteps, **kwargs)
    return compile_time, nsteps / (time.time() - t0), acc


def main():
    """Run the benchmark, write timings and reference outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="unit_tests/RXJ1347_a10_single_tod.yaml")
    parser.add_argument("--nrep", type=int, default=20)
    parser.add_argument("--out", default="reports/bench.json")
    parser.add_argument("--ref", default=None, help="npz to save/compare outputs")
    parser.add_argument("--compare", action="store_true", help="compare to --ref")
    parser.add_argument("--debug", action="store_true", help="Few repetitions")
    parser.add_argument(
        "--modes", default="mh,hmc,lm", help="Comma separated objective modes"
    )
    parser.add_argument(
        "--mh-steps", type=int, default=0, help="Also time this many MH loop steps"
    )
    parser.add_argument(
        "--profile", default=None, help="Write a jax.profiler trace to this directory"
    )
    args = parser.parse_args()
    if args.debug:
        args.nrep = 2

    from witch import fitter
    from witch.cfg_loader import load_cfg
    from witch.objective import joint_objective

    cfg = fitter.load_config({}, args.config)
    cli = fitter._make_parser().parse_args([args.config])
    _, _, _, mm = load_cfg(cfg, cli)

    datas = [d for ds in mm.datasets for d in ds.datavec]
    meta = {
        "config": args.config,
        "platform": jax.devices()[0].platform,
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "ntod": len(datas),
        "nsamp": int(sum(np.prod(d.data.shape) for d in datas)),
        "nfit": len(mm.to_fit_idx),
    }
    print(meta)

    all_modes = {"mh": (True, False, False), "hmc": (False, True, False)}
    all_modes["lm"] = (True, True, True)
    modes = {m: all_modes[m] for m in args.modes.split(",") if m}
    res, outs = {"meta": meta}, {}
    for name, (ll, g, c) in modes.items():
        ct, rt, out = timeit(lambda: joint_objective(mm, ll, g, c), args.nrep)
        res[name] = {"compile_s": ct, "run_s": rt}
        outs.update({f"{name}_{k}": np.asarray(v) for k, v in zip("lgc", out)})
        print(f"{name}: compile {ct:.2f}s run {rt * 1e3:.2f}ms")
    if args.mh_steps:
        ct, sps, acc = time_mh(with_mh_errs(mm), args.mh_steps)
        res["mh_loop"] = {"compile_s": ct, "steps_per_s": sps, "acceptance": acc}
        print(f"mh loop: warmup {ct:.2f}s {sps:.1f} steps/s (acceptance {acc:.2f})")
    if args.profile:
        with jax.profiler.trace(args.profile, create_perfetto_trace=True):
            for ll, g, c in modes.values():
                for _ in range(5):
                    jax.block_until_ready(joint_objective(mm, ll, g, c))
            if args.mh_steps:
                time_mh(mm, min(args.mh_steps, 50))
        print(f"profile written to {args.profile}")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    if args.ref and not args.compare:
        np.savez(args.ref, **outs)
    elif args.ref:
        ref = np.load(args.ref)
        for k in ref.files:
            if k not in outs:
                continue
            denom = np.abs(ref[k]).max() or 1.0
            print(f"{k}: max rel diff {np.abs(outs[k] - ref[k]).max() / denom:.2e}")


if __name__ == "__main__":
    main()
