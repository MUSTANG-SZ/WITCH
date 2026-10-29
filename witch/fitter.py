"""
Master fitting and map making script.
You typically want to run the `witcher` command instead of this.
"""

import argparse as argp
import glob
import os
import re
import shutil
import sys
import time
from copy import copy, deepcopy
from functools import partial

import corner
import dill as pk
import iteround
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpi4jax
import numpy as np
import yaml
from mpi4py import MPI
from typing_extensions import Any, Unpack

from .containers import Model_xfer
from .containers.metamodel import _compute_metadata_map, _compute_par_map_and_pars
from .fitting import run_lmfit, run_mcmc
from .metropolis_hastings import run_mh
from .nonparametric import para_to_non_para
from .objective import joint_objective

comm = MPI.COMM_WORLD.Clone()
_RUN_START = time.time()

# Fit outputs written by this process, so _archive_old_outputs never moves them.
# Tracked by path because file mtimes are too coarse (and on network filesystems
# too skewed) to tell them apart from the previous run's by time.
_written = set()


def _ours(path):
    """Record `path` as written by this run and return it."""
    _written.add(os.path.abspath(path))
    return path


# Files written by fit_loop and _run_mcmc, relative to the output directory
_FIT_OUTPUTS = (
    "results_*.dill",
    "results_*.yaml",
    "par_results_*.dill",
    "samples_mcmc.npz",
    "corner.png",
    "checkpoints/round_*.pkl",
    "checkpoints/mcmc_step_*.pkl",
)


def print_once(*args: Unpack[tuple[Any, ...]]):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Parameters
    ----------
    *args : Unpack[tuple[Any, ...]]
        Arguments to pass to print.
    """
    if comm.Get_rank() == 0:
        print(*args)
        sys.stdout.flush()


def _make_parser() -> argp.ArgumentParser:
    # Parse arguments
    parser = argp.ArgumentParser(
        description="Fit cluster profiles using mikasi and minkasi_jax"
    )
    parser.add_argument("config", help="Path to config file")
    parser.add_argument(
        "--nofit",
        "-nf",
        action="store_true",
        help="Don't actually fit, just use values from config",
    )
    parser.add_argument(
        "--nosub",
        "-ns",
        action="store_true",
        help="Don't subtract the sim'd or fit model. Useful for mapmaking a sim'd cluster",
    )
    parser.add_argument(
        "--wnoise",
        "-wn",
        action="store_true",
        help="Use whitenoise instead of map noise. Only for use with sim",
    )
    parser.add_argument(
        "--cpu",
        "-cpu",
        default=False,
        type=bool,
        help="If True, then run on CPU. Otherwise and by default run GPU.",
    )
    return parser


def _mpi_fsplit(fnames, comm):
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    # Check that we have the minimum number of procs
    if nproc < len(fnames):
        raise ValueError("More datasets than procs!")

    # Kill extra procs
    flat_fnames = [
        f for fname in [fnames[dset] for dset in fnames.keys()] for f in fname
    ]
    if nproc > len(flat_fnames):
        nproc = len(flat_fnames)
        group = comm.Get_group()
        new_group = group.Incl(list(range(nproc)))
        new_comm = comm.Create(new_group)
        if rank >= len(flat_fnames):
            print(f"More procs than files!, exiting process {rank}")
            MPI.Finalize()
            sys.exit(0)
        comm = new_comm

    # Decide the number of procs for each dataset
    nprocs = {
        dset: 1
        + (nproc - len(fnames))
        * (len(fnames[dset]) - 1)
        / max(1, (len(flat_fnames) - len(fnames)))
        for dset in fnames.keys()
    }
    nprocs = iteround.saferound(nprocs, 0)
    dsets = np.array(
        [
            ds
            for dset in [
                [dset_name] * int(nprocs[dset_name]) for dset_name in fnames.keys()
            ]
            for ds in dset
        ]
    )

    # Die if we messed up
    for dset, f in fnames.items():
        if len(f) < nprocs[dset]:
            raise ValueError(
                f"Too many procs allocated to {dset}! This shouldn't be possible please report it!"
            )

    # Handle my local case
    dset_local = dsets[rank]
    group = comm.Get_group()
    comms_local = {
        dset: comm.Create(group.Incl(list(np.where(dsets == dset)[0])))
        for dset in fnames.keys()
    }
    local_rank = comms_local[dset_local].Get_rank()
    fnames_local = {
        dset: (
            fnames[dset][local_rank :: int(nprocs[dset])] if dset == dset_local else []
        )
        for dset in fnames.keys()
    }

    return fnames_local, comm, comms_local


def process_tods(cfg, dataset, metamodel):
    todvec = dataset.datavec
    info = dataset.info
    rank = todvec.comm.Get_rank()
    nproc = todvec.comm.Get_size()
    sim = cfg.get("sim", False)
    if metamodel is None and sim:
        raise ValueError("metamodel cannot be None when simming!")
    metamodel_cur = metamodel
    xfer = info.get("info", "")
    if xfer:
        metamodel_cur = copy(metamodel)
        metamodel_cur.models = tuple(
            Model_xfer.from_parent(model, xfer) for model in metamodel_cur.models
        )

    dset_ind = metamodel_cur.get_dataset_ind(dataset.name)
    todvec.comm.barrier()
    for i, tod in enumerate(todvec.tods):
        if sim:
            if cfg["wnoise"]:
                temp = jnp.percentile(
                    jnp.diff(tod.data, axis=-1), jnp.array([33.0, 68.0]), axis=-1
                )
                scale = (temp[1] - temp[0]) / jnp.sqrt(8)
                tod.data = (
                    jax.random.normal(
                        jax.random.key(0), shape=tod.data.shape, dtype=tod.data.dtype
                    )
                    * scale[..., None]
                )
            else:
                tod.data *= (-1) ** ((rank + nproc * i) % 2)

            pred = metamodel_cur.model_proj(dset_ind, i)
            tod.data += pred
        tod.compute_noise(
            dataset.noise_class, None, *dataset.noise_args, **dataset.noise_kwargs
        )
    return todvec


def resolve_noise_args(imap, args: tuple, kwargs: dict) -> tuple[list, dict]:
    """
    Evaluate noise arguments that refer to the map, like `compute_noise` does.

    Parameters
    ----------
    imap : WCSMap
        The map that `self` refers to in the arguments.
    args : tuple
        Positional noise arguments, strings starting with `self` are evaluated.
    kwargs : dict
        Keyword noise arguments, string values starting with `self` are evaluated.

    Returns
    -------
    args : list
        The evaluated positional arguments.
    kwargs : dict
        The evaluated keyword arguments.
    """

    def ev(a):
        return eval(a, {"self": imap}) if isinstance(a, str) and a[:4] == "self" else a

    return [ev(a) for a in args], {k: ev(v) for k, v in kwargs.items()}


def process_maps(cfg, dataset, metamodel):
    mapset = dataset.datavec
    info = dataset.info
    sim = cfg.get("sim", False)
    if metamodel is None and sim:
        raise ValueError("metamodel cannot be None when simming!")
    metamodel_cur = metamodel
    xfer = info.get("info", "")
    if xfer:
        metamodel_cur = deepcopy(metamodel)
        metamodel_cur.models = tuple(
            Model_xfer.from_parent(model, xfer) for model in metamodel_cur.models
        )

    dset_ind = metamodel_cur.get_dataset_ind(dataset.name)
    for i, imap in enumerate(mapset):
        if sim:
            if jnp.all(imap.ivar == 0):
                print_once(
                    f"ivar for map {imap.name} is all 0! Filling with a dummy value but you should check your maps!"
                )
                imap.ivar = imap.ivar.at[:].set(cfg.get("default_ivar", 1e8))
            # Pixels with ivar == 0 have no weight in the likelihood, so leave them noiseless
            # rather than letting an inf noise level propagate into the data
            good = imap.ivar > 0
            scale = jnp.where(
                good, 1.0 / jnp.sqrt(jnp.where(good, imap.ivar, 1.0)), 0.0
            )
            print("Map noise: ", float(jnp.mean(scale[good])))
            sample = getattr(dataset.noise_class, "sample", None)
            if sample is not None and not cfg["wnoise"]:
                # The noise model can draw its own correlated noise (e.g. NoiseCMBMap)
                args, kwargs = resolve_noise_args(
                    imap, dataset.noise_args, dataset.noise_kwargs
                )
                # Lets a sim use a different noise model than the fit, to test a mismatch
                kwargs |= cfg["datasets"][dataset.name].get("sim_noise_kwargs", {})
                key = jax.random.key(cfg.get("sim_seed", 0))
                imap.data = sample(key, imap.data, *args, **kwargs)
            else:
                imap.data = (
                    jax.random.normal(
                        jax.random.key(cfg.get("sim_seed", 0)),
                        shape=imap.data.shape,
                        dtype=imap.data.dtype,
                    )
                    * scale
                )
                if not cfg["wnoise"]:
                    print_once(
                        "Noise model cannot be sampled... simming map with white noise"
                    )

            pred = metamodel_cur.model_proj(dset_ind, i)
            imap.data = imap.data + pred
        print("Map scale: ", jnp.mean(jnp.abs(imap.data)))
        imap.compute_noise(
            dataset.noise_class, None, *dataset.noise_args, **dataset.noise_kwargs
        )
    return mapset


def get_outdir(cfg, metamodel, nonpara=False):
    outroot = cfg["paths"]["outroot"]
    if not os.path.isabs(outroot):
        outroot = os.path.join(
            os.environ.get("WITCH_OUTROOT", os.environ["HOME"]), outroot
        )

    name = ""
    if metamodel is not None:
        name = "+".join(
            [
                model.name
                + "_"
                + "-".join([structure.name for structure in model.structures])
                for model in metamodel.models
            ]
        ) + ("_ns" * (not cfg["sub"]))
    # Keep fits to different data (e.g. M2 only vs M2+ACT) apart.
    # Use the config rather than metamodel.datasets, which only has this rank's datasets.
    dsets = "+".join(cfg.get("datasets", {}).keys())
    outdir = os.path.join(outroot, cfg["name"], dsets, name)

    if "subdir" in cfg["paths"]:
        outdir = os.path.join(outdir, cfg["paths"]["subdir"])

    if metamodel is not None:
        if cfg["fit"]:
            outdir = os.path.join(
                outdir,
                "-".join(
                    [
                        name
                        for name, to_fit in zip(
                            metamodel.par_names, metamodel.to_fit_ever
                        )
                        if to_fit
                    ]
                ),
            )
        else:
            outdir = os.path.join(outdir, "not_fit")
    if cfg["sim"] and metamodel is not None:
        outdir += "-sim"
    if nonpara:
        outdir += "-nonpara"

    print_once("Outputs can be found in", outdir)
    if comm.Get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
        # Parameters are positional within a structure, so the key order has to be kept
        with open(os.path.join(outdir, "config.yaml"), "w") as file:
            yaml.dump(cfg, file, sort_keys=False)

    return outdir


def _save_outputs(cfg, metamodel, desc_str):
    """
    Save parameters and cov, etc of model separately from the model itself.
    Useful for loading the parameters without needing WITCH.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary for fitting
    metamodel : MetaModel
        The metamodel containing the fitted parameters and covariance
    desc_str : str
        Description string to append to the output filename

    Returns
    -------
    None
    """
    outdir = cfg["outdir"]
    if comm.Get_rank() != 0:
        return
    res_path = _ours(os.path.join(outdir, f"par_results_{desc_str}.dill"))
    par_dict = {
        "par_names": np.asarray(metamodel.par_names),
        "parameters": np.asarray(metamodel.parameters),
        "errs": np.asarray(metamodel.errs),
        "cov": np.asarray(metamodel.cov),
        "to_fit": np.asarray(metamodel.to_fit),
    }
    with open(res_path, "wb") as f:
        pk.dump(par_dict, f)


def _save_model(cfg, metamodel, desc_str, nonpara=False):
    outdir = cfg["outdir"]
    if comm.Get_rank() != 0:
        return
    res_path = _ours(os.path.join(outdir, f"results_{desc_str}.dill"))
    print_once("Saving results to", res_path)
    metamodel.save(res_path)

    _save_outputs(cfg, metamodel, desc_str)

    if nonpara:
        return

    final = {"metamodel": cfg["metamodel"]}
    for model in metamodel.models:
        final[model.name] = cfg[model.name]
        for model_struct in model.structures:
            structure = cfg[model.name]["structures"][model_struct.name]
            for par, par_name in zip(
                model_struct.parameters, structure["parameters"].keys()
            ):
                final[model.name]["structures"][model_struct.name]["parameters"][
                    par_name
                ]["value"] = [float(cur_par) for cur_par in par.val]
    with open(_ours(os.path.join(outdir, f"results_{desc_str}.yaml")), "w") as file:
        yaml.dump(final, file, sort_keys=False)


def _reestimate_noise(metamodel):
    for i, dataset in enumerate(metamodel.datasets):
        for j, data in enumerate(dataset.datavec):
            pred = metamodel.model_proj(i, j)
            data.compute_noise(
                dataset.noise_class,
                data.data - pred,
                *dataset.noise_args,
                **dataset.noise_kwargs,
            )
    return metamodel


def _save_checkpoint(
    ckpt_path, metamodel, cfg, stage=None, round_num=None, step_num=None
):
    """Save a checkpoint for resuming training"""
    if comm.Get_rank() != 0:
        return

    # Only saves model parameters (which is what we needed to resume)
    state = {
        "cfg": cfg,
        "stage": stage,
        "round": round_num,
        "step": step_num,
    }

    metamodel.save(_ours(ckpt_path), state)


def _read_checkpoint(ckpt_path):
    """
    Load a checkpoint and return model, datasets, start_round, stage, and cfg.
    Note: datasets parameter should be freshly created one from fit_loop (for reestimating noise)
        this function updates the noise estimation based on the loaded model
    """

    with open(ckpt_path, "rb") as f:
        metamodel, state = pk.load(f)

    start_round = state.get("round", -1) + 1
    stage = state.get("stage", None)
    cfg = state.get("cfg", None)

    return metamodel, start_round, stage, cfg


def _archive_old_outputs(outdir):
    """
    Move fit outputs left over from a previous run into `outdir/old_runs/<timestamp>`
    so they can't be mistaken for (or resumed from as) part of this run.
    Files this process wrote (e.g. the parametric fit, when the nonparametric fit shares
    its outdir) are never moved. Maps and other dataset outputs are left alone.
    """
    if comm.Get_rank() != 0:
        return
    stale = [
        path
        for pattern in _FIT_OUTPUTS
        for path in glob.glob(os.path.join(outdir, pattern))
        if os.path.abspath(path) not in _written
    ]
    if len(stale) == 0:
        return
    archive = os.path.join(
        outdir, "old_runs", time.strftime("%Y%m%d-%H%M%S", time.localtime(_RUN_START))
    )
    for path in stale:
        dest = os.path.join(archive, os.path.relpath(path, outdir))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(path, dest)
    print_once(f"[resume] Moved {len(stale)} outputs from a previous run to {archive}")


def _run_fit(
    cfg,
    metamodel,
    r,
    nonpara=False,
):
    metamodel.set_round(r)
    to_fit = np.array(metamodel.to_fit)
    print_once(f"Starting round {r + 1} of fitting with {np.sum(to_fit)} pars free")
    t1 = time.time()
    metamodel, i, delta_chisq, lmd = run_lmfit(
        metamodel,
        eval(str(cfg["fitting"].get("maxiter", "10"))),
        eval(str(cfg["fitting"].get("chitol", "1e-5"))),
    )
    mpi4jax.barrier(comm=comm)
    t2 = time.time()
    print_once(
        f"Took {t2 - t1} s to fit with {i} iterations and final delta chisq of {delta_chisq}, final lambda: {lmd}"
    )
    if float(lmd) > 0:
        print_once(
            "WARNING: final lambda > 0, this round's fit has likely not converged. "
            "The errors below are from the damped curvature and are underestimated."
        )

    print_once(metamodel)
    _save_model(cfg, metamodel, f"fit{r}", nonpara)

    metamodel = _reestimate_noise(metamodel)
    return metamodel, i, float(lmd)


def _mcmc_checkpoint_callback(
    updated_metamodel, callback_state, checkpoint_interval, ckpt_dir, nonpara, cfg
):
    # Callback that _run_mcmc will call every step
    callback_state["step"] += 1

    step = callback_state["step"]

    if step % checkpoint_interval == 0 and comm.Get_rank() == 0:
        ckpt_path = os.path.join(
            ckpt_dir, f"mcmc_step_{step}{'_nonpara' * nonpara}.pkl"
        )
        _save_checkpoint(
            ckpt_path,
            updated_metamodel,
            cfg,
            stage="mcmc",
            step_num=step,
        )
        print_once(f"[checkpoint] Saved MCMC checkpoint at step {step} -> {ckpt_path}")


def _run_mcmc(cfg, metamodel, nonpara=False):
    print_once("Running MCMC")
    # The noise was re-estimated after LM, see the comment in fit_loop
    jax.clear_caches()
    init_pars = np.array(metamodel.parameters.copy())

    # Checkpoint settings
    checkpoint_interval = int(cfg["mcmc"].get("checkpoint_interval", 200))
    ckpt_dir = os.path.join(cfg["outdir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Tracking container for callback to update
    callback_state = {
        "step": 0,
    }

    callback = partial(
        _mcmc_checkpoint_callback,
        callback_state=callback_state,
        checkpoint_interval=checkpoint_interval,
        ckpt_dir=ckpt_dir,
        nonpara=nonpara,
        cfg=cfg,
    )

    mcfg = cfg["mcmc"]
    method = str(mcfg.get("method", "hmc")).lower()
    if method not in ("hmc", "mh"):
        raise ValueError(f"Unknown mcmc method {method}, must be 'hmc' or 'mh'")
    extras = {}
    t1 = time.time()
    if method == "mh":
        metamodel, samples, chains, acc = run_mh(
            metamodel,
            num_samples=int(mcfg.get("num_steps", 10000)),
            num_chains=int(mcfg.get("num_chains", 4)),
            burn_in=float(mcfg.get("burn_in", 0.1)),
            bound=float(mcfg.get("bound", 2)),
            seed=int(mcfg.get("seed", 0)),
            prior_type=str(mcfg.get("prior_type", "uniform")),
            prior_err_scale=float(mcfg.get("prior_err_scale", 1.0)),
            joint=bool(mcfg.get("joint", True)),
            dataset_ind=int(mcfg.get("dataset_index", 0)),
            progress=comm.Get_rank() == 0,
            proposal=str(mcfg.get("proposal", "cov")),
        )
        print_once(f"MH acceptance rates: {acc}")
        extras = dict(
            chains=chains,
            acceptance_rates=acc,
            par_names=np.array(metamodel.par_names),
            to_fit=np.array(metamodel.to_fit),
        )
    else:
        metamodel, samples = run_mcmc(
            metamodel,
            num_steps=int(mcfg.get("num_steps", 5000)),
            num_leaps=int(mcfg.get("num_leaps", 10)),
            step_size=jnp.array(float(mcfg.get("step_size", 0.02))),
            sample_which=int(mcfg.get("sample_which", -1)),
            burn_in=float(mcfg.get("burn_in", 0.1)),
            max_tries=int(mcfg.get("max_tries", 20)),
            # callback=callback, # Disabling for now, impure callbacks in a jax loop are dangerous
        )
    mpi4jax.barrier(comm=comm)
    t2 = time.time()
    print_once(f"Took {t2 - t1} s to run mcmc")
    callback_state["step"] = samples.shape[0]
    callback(metamodel, callback_state=callback_state)

    message = str(metamodel).split("\n")
    message[1] = f"{method.upper()} MCMC estimated pars:"
    print_once("\n".join(message))

    _save_model(cfg, metamodel, "mcmc", nonpara)
    if comm.Get_rank() == 0:
        samples = np.array(samples)
        samps_path = _ours(os.path.join(cfg["outdir"], "samples_mcmc.npz"))
        print_once("Saving samples to", samps_path)
        np.savez_compressed(samps_path, samples=samples, **extras)
        try:
            to_fit = np.array(metamodel.to_fit)
            # ranges = [prior if prior is not None else [0.5 * model.params[i], 2 * model.params[i]] for i, prior in enumerate(model.priors)]
            corner.corner(
                samples,
                labels=np.array(metamodel.par_names)[to_fit],
                truths=init_pars[to_fit],
            )
            plt.savefig(_ours(os.path.join(cfg["outdir"], "corner.png")))
        except Exception as e:
            print_once(f"Failed to make corner plot with error: {str(e)}")
    else:
        # samples are incomplete on non root procs
        del samples

    metamodel = _reestimate_noise(metamodel)
    return metamodel


def fit_loop(metamodel, cfg, comm, nonpara=False):
    if metamodel is None:
        raise ValueError("Can't fit without metamodel defined!")
    # ensure checkpoint directory exists
    ckpt_dir = os.path.join(cfg["outdir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load progress (loading checkpoint if requested)
    load = cfg.get("load_progress", 0)
    start_round = 0
    load_path = None

    if load == -1:
        # Automatically load latest checkpoint round
        ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("round_")]
        if nonpara:
            ckpts = [c for c in ckpts if "nonpara" in c]
        else:
            ckpts = [c for c in ckpts if "nonpara" not in c]

        if len(ckpts) > 0:
            latest = sorted(ckpts, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
            load_path = os.path.join(ckpt_dir, latest)
            print_once(f"[resume] Loading latest checkpoint -> {load_path}")

    elif load > 0:
        # Load specific round number
        load_path = os.path.join(ckpt_dir, f"round_{load}{'_nonpara' * nonpara}.pkl")

        if os.path.exists(load_path):
            print_once(f"[resume] Loading checkpoint round {load} -> {load_path}")
        else:
            print_once(
                f"[resume] Requested round {load} not found, starting at round 0"
            )
            load_path = None
    else:
        print_once("[resume] Not resuming from previous checkpoint")

    # Actually load and validate checkpoint if we found valid path
    if load_path is not None:
        loaded_metamodel, loaded_start_round, *_ = _read_checkpoint(load_path)

        # Only rank 0 saves checkpoints, and the datasets and their maps are per task,
        # so only rank 0 can compare those. Agree across tasks so all fail together.
        compatible = metamodel.check_compatibility(
            loaded_metamodel, check_local=comm.Get_rank() == 0
        )
        if not comm.allreduce(int(compatible), op=MPI.MIN):
            raise ValueError("Checkpoint is incompatible!")

        # Take the fit state from the checkpoint but keep this task's datasets
        loaded_metamodel.global_comm = metamodel.global_comm
        loaded_metamodel.datasets = metamodel.datasets
        loaded_metamodel.model_map = metamodel.model_map
        loaded_metamodel.metadata_map = metamodel.metadata_map
        start_round = loaded_start_round
        metamodel = _reestimate_noise(loaded_metamodel)

        print_once(f"[resume] Checkpoint OK. Resuming from round {start_round}")
    elif cfg.get("archive_old_outputs", True):
        # Starting fresh, so stale rounds and checkpoints from an earlier run would
        # mix with ours (and a later load_progress: -1 could resume from them)
        _archive_old_outputs(cfg["outdir"])
    comm.barrier()

    if cfg["sim"]:
        # Remove structs we deliberately want to leave out of model
        metamodel = metamodel.remove_structs(cfg)
        params = jnp.array(metamodel.parameters)
        par_offset = cfg.get("par_offset", 1.1)
        params = params.at[metamodel.to_fit_ever].multiply(
            par_offset
        )  # Don't start at exactly the right value
        metamodel = metamodel.update(
            pars=params, errs=metamodel.errs, cov=metamodel.cov, chisq=metamodel.chisq
        )

    # Compile objective function
    # Noise models live in pytree aux data and can't be compared once re-estimated,
    # so drop any jit cache entries keyed on old noise objects
    jax.clear_caches()
    print_once("Compiling objective function")
    t0 = time.time()
    chisq, _, _ = joint_objective(metamodel, do_grad=False, do_curve=False)
    jax.block_until_ready(chisq)
    metamodel = metamodel.update(
        pars=metamodel.parameters, errs=metamodel.errs, cov=metamodel.cov, chisq=chisq
    )
    print_once(f"Took {time.time() - t0} s to compile")

    print_once(
        re.sub(r"^Round 1.*\n?", "Starting pars:\n", str(metamodel), flags=re.MULTILINE)
    )
    # A round counts as converged if it took a single iteration, or if it ended
    # undamped (lambda = 0) with every free parameter moving by less than partol
    # times its error. Stop after converged_rounds converged rounds in a row.
    partol = float(eval(str(cfg["fitting"].get("partol", "0.1"))))
    converged_rounds = int(cfg["fitting"].get("converged_rounds", 2))
    consecutive_converged_rounds = 0
    # mcmc.lm: False skips LM and samples from the starting parameters
    run_lm = cfg.get("mcmc", {}).get("lm", True)
    for r in range(start_round, metamodel.n_rounds if run_lm else 0):
        pars_before = np.array(metamodel.parameters)
        metamodel, iterations, lmd = _run_fit(
            cfg,
            metamodel,
            r,
            nonpara,
        )
        pars_after = np.array(metamodel.parameters)
        errs = np.array(metamodel.errs)
        free = np.array(metamodel.to_fit) & (errs > 0)
        shift = np.max(np.abs(pars_after - pars_before)[free] / errs[free], initial=0.0)
        converged = iterations == 1 or (lmd == 0 and shift < partol)
        converged = comm.allreduce(int(converged), op=MPI.MIN) == 1
        if converged:
            consecutive_converged_rounds += 1
        else:
            consecutive_converged_rounds = 0
        print_once(
            f"Largest parameter shift this round: {shift:.3g} sigma "
            f"({consecutive_converged_rounds}/{converged_rounds} converged rounds)"
        )
        mpi4jax.barrier(comm=comm)

        # Checkpoint after each round
        ckpt_path = os.path.join(ckpt_dir, f"round_{r}{'_nonpara' * nonpara}.pkl")
        _save_checkpoint(
            ckpt_path,
            metamodel,
            cfg,
            stage="fit_round",
            round_num=r,
        )
        print_once(f"[checkpoint] Saved LM state after round {r} -> {ckpt_path}")
        if consecutive_converged_rounds >= converged_rounds:
            print_once(
                f"Stopping fitting after {converged_rounds} consecutive converged rounds"
            )
            break

    if "mcmc" in cfg and cfg["mcmc"].get("run", True):
        metamodel = _run_mcmc(
            cfg,
            metamodel,
            nonpara,
        )
        mpi4jax.barrier(comm=comm)

    # Save final pars
    _save_model(cfg, metamodel, "final_fit", nonpara)

    return metamodel


def deep_merge(a: dict, b: dict) -> dict:
    """
    Based on https://gist.github.com/angstwad/bf22d1822c38a92ec0a9?permalink_comment_id=3517209
    """
    result = deepcopy(a)
    for bk, bv in b.items():
        av = result.get(bk)
        if isinstance(av, dict) and isinstance(bv, dict):
            result[bk] = deep_merge(av, bv)
        else:
            result[bk] = deepcopy(bv)
    return result


def load_config(start_cfg, cfg_path):
    """
    We want to load a config and if it has the key "base",
    load that as well and merge them.
    We only want to take things from base that are not in the original config
    so we merge the original into the newly loaded one.
    """
    with open(cfg_path) as file:
        new_cfg = yaml.safe_load(file)
    cfg = deep_merge(new_cfg, start_cfg)
    if "base" in new_cfg:
        base_path = new_cfg["base"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(cfg_path), base_path)
        return load_config(cfg, base_path)
    return cfg


def main():
    parser = _make_parser()
    args = parser.parse_args()

    if args.cpu == True:
        jax.config.update("jax_platform_name", "cpu")

    from .cfg_loader import load_cfg

    cfg = load_config({}, args.config)
    datasets, outdir, _, metamodel = load_cfg(cfg, args)
    comm.barrier()

    # Now we fit
    to_fit = cfg.get("fit", True)
    resample = cfg.get("resample", False)
    if to_fit and outdir is not None:
        metamodel = fit_loop(metamodel, cfg, comm)
        for dataset, mdata in zip(datasets, metamodel.datasets):
            dataset.data = mdata
            dataset.postfit(dataset, cfg, metamodel)
        metamodel.datasets = tuple(dataset.data for dataset in datasets)
        if "nonpara" in cfg and cfg["nonpara"]["convert"]:
            print_once("Setting up nonparametric model")
            n_rounds = cfg["nonpara"].get("n_rounds", None)
            nonpara_models = []
            mm_cfg = cfg.get("metamodel", {})
            for model in metamodel.models:
                mnonpara = cfg[model.name].get("nonpara", {})
                to_copy = mnonpara.get("to_copy", [])
                if len(to_copy) == 0:
                    nonpara_models += [copy(model)]
                    continue
                rmax = mnonpara.get("rmax", 180.0)
                struct_num = mnonpara.get("struct_num", 0)
                sig_params = mnonpara.get("sig_params", ["amp", "P0"])
                default = mnonpara.get("default", (0, 10, 20, 30, 50, 80, 120, 180))
                nonpara_model = para_to_non_para(
                    model,
                    n_rounds,
                    to_copy,
                    rmax,
                    struct_num,
                    sig_params,
                    default,
                    mm_cfg,
                )
                nonpara_models += [nonpara_model]
            nonpara_models = tuple(nonpara_models)
            par_map, pars, errs, cov = _compute_par_map_and_pars(mm_cfg, nonpara_models)
            metadata_map = _compute_metadata_map(nonpara_models, metamodel.datasets)
            nonparametamodel = copy(metamodel)
            nonparametamodel.models = nonpara_models
            nonparametamodel.parameter_map = par_map
            nonparametamodel.parameters = pars
            nonparametamodel.errs = errs
            nonparametamodel.cov = cov
            nonparametamodel.metadata_map = metadata_map
            nonparametamodel = nonparametamodel.update(
                pars=pars, errs=errs, cov=nonparametamodel.cov, chisq=metamodel.chisq
            )

            outdir = get_outdir(cfg, nonparametamodel)
            cfg["outdir"] = outdir
            for dataset in datasets:
                os.makedirs(os.path.join(outdir, dataset.name), exist_ok=True)
                dataset.info["outdir"] = outdir

            nonparametamodel = fit_loop(nonparametamodel, cfg, comm, True)
            for dataset, mdata in zip(datasets, nonparametamodel.datasets):
                dataset.data = mdata
                dataset.postfit(dataset, cfg, metamodel)
            nonparametamodel.datasets = tuple(dataset.data for dataset in datasets)

    else:
        for dataset, mdata in zip(datasets, metamodel.datasets):
            dataset.data = mdata
            dataset.postfit(dataset, cfg, metamodel)
        metamodel.datasets = tuple(dataset.data for dataset in datasets)

    print_once("Outputs can be found in", outdir)
