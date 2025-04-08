"""
Master fitting and map making script.
You typically want to run the `witcher` command instead of this.
"""

import argparse as argp
import os
import sys
import time
from copy import deepcopy
from importlib import import_module

import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mpi4jax
import numpy as np
import yaml
from mpi4py import MPI
from typing_extensions import Any, Unpack

from . import utils as wu
from .containers import Model, Model_xfer
from .dataset import DataSet
from .fitting import fit_dataset, run_mcmc

comm = MPI.COMM_WORLD.Clone()


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
    if nproc > len(fnames):
        nproc = len(fnames)
        group = comm.Get_group()
        new_group = group.Incl(list(range(nproc)))
        new_comm = comm.Create(new_group)
        if rank >= len(fnames):
            print(f"More procs than files!, exiting process {rank}")
            MPI.Finalize()
            sys.exit(0)
        comm = new_comm
    return fnames[rank::nproc], comm


def process_tods(cfg, dataset, model):
    todvec = dataset.datavec
    rank = todvec.comm.Get_rank()
    nproc = todvec.comm.Get_size()
    sim = cfg.get("sim", False)
    if model is None and sim:
        raise ValueError("model cannot be None when simming!")
    model_cur = model
    xfer = dataset.info.get("info", "")
    if xfer:
        model_cur = Model_xfer.from_parent(model, xfer)

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
                tod.data *= tod.data * (-1) ** ((rank + nproc * i) % 2)

            pred = model_cur.to_tod(
                tod.x * wu.rad_to_arcsec, tod.y * wu.rad_to_arcsec
            ).block_until_ready()
            tod.data = tod.data + pred
        tod.data = tod.data - jnp.mean(tod.data, axis=-1)[..., None]
        tod.compute_noise(
            dataset.noise_class, None, *dataset.noise_args, **dataset.noise_kwargs
        )
    return todvec


def process_maps(cfg, dataset, model):
    sim = cfg.get("sim", False)
    mapset = dataset.datavec
    if model is None and sim:
        raise ValueError("model cannot be None when simming!")
    model_cur = model
    xfer = dataset.info.get("info", "")
    if xfer:
        model_cur = Model_xfer.from_parent(model, xfer)

    for imap in mapset:
        if sim:
            if jnp.all(imap.ivar == 0):
                print_once(
                    f"ivar for map {imap.name} is all 0! Filling with a dummy value but you should check your maps!"
                )
                imap.ivar = imap.ivar.at[:].set(cfg.get("default_ivar", 1e8))
            scale = 1.0 / jnp.sqrt(imap.ivar)
            avg_scale = np.nanmean(scale)
            scale = jnp.nan_to_num(
                scale, nan=avg_scale, posinf=avg_scale, neginf=avg_scale
            )
            imap.data = (
                jax.random.normal(
                    jax.random.key(0), shape=imap.data.shape, dtype=imap.data.dtype
                )
                * scale
            )
            if not cfg["wnoise"]:
                print_once(
                    "Only white noise currently supported for map sims... simming map with white noise"
                )

            x, y = imap.xy
            pred = model_cur.to_map(
                x * wu.rad_to_arcsec, y * wu.rad_to_arcsec
            ).block_until_ready()
            imap.data = imap.data + pred
        imap.data = imap.data - jnp.mean(imap.data)
        imap.compute_noise(
            dataset.noise_class, None, *dataset.noise_args, **dataset.noise_kwargs
        )
    return mapset


def get_outdir(cfg, model):
    outroot = cfg["paths"]["outroot"]
    if not os.path.isabs(outroot):
        outroot = os.path.join(
            os.environ.get("WITCH_OUTROOT", os.environ["HOME"]), outroot
        )

    name = ""
    if model is not None:
        name = model.name + ("_ns" * (not cfg["sub"]))
    outdir = os.path.join(outroot, cfg["name"], name)
    if "subdir" in cfg["paths"]:
        outdir = os.path.join(outdir, cfg["paths"]["subdir"])
    if model is not None:
        if cfg["fit"]:
            outdir = os.path.join(
                outdir,
                "-".join(
                    [
                        name
                        for name, to_fit in zip(model.par_names, model.to_fit_ever)
                        if to_fit
                    ]
                ),
            )
        else:
            outdir = os.path.join(outdir, "not_fit")
    if cfg["sim"] and model is not None:
        outdir += "-sim"

    print_once("Outputs can be found in", outdir)
    if comm.Get_rank() == 0:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config.yaml"), "w") as file:
            yaml.dump(cfg, file)

    return outdir


def _save_model(cfg, model, outdir, desc_str):
    if comm.Get_rank() != 0:
        return
    res_path = os.path.join(outdir, f"results_{desc_str}.dill")
    print_once("Saving results to", res_path)
    model.save(res_path)

    final = {"model": cfg["model"]}
    for i, (struct_name, structure) in zip(
        model.original_order, cfg["model"]["structures"].items()
    ):
        model_struct = model.structures[i]
        for par, par_name in zip(
            model_struct.parameters, structure["parameters"].keys()
        ):
            final["model"]["structures"][struct_name]["parameters"][par_name][
                "value"
            ] = [float(cur_par) for cur_par in par.val]
    with open(os.path.join(outdir, f"results_{desc_str}.yaml"), "w") as file:
        yaml.dump(final, file)


def _reestimate_noise(model, dataset):
    for data in dataset.datavec:
        if dataset.mode == "tod":
            pred = model.to_tod(
                data.x * wu.rad_to_arcsec, data.y * wu.rad_to_arcsec
            ).block_until_ready()
        else:
            x, y = data.xy
            pred = model.to_map(x * wu.rad_to_arcsec, y * wu.rad_to_arcsec)
        data.compute_noise(
            dataset.noise_class,
            data.data - pred,
            *dataset.noise_args,
            **dataset.noise_kwargs,
        )
    return dataset


def _run_fit(cfg, model, dataset, outdir, r):
    model.cur_round = r
    to_fit = np.array(model.to_fit)
    print_once(f"Starting round {r+1} of fitting with {np.sum(to_fit)} pars free")
    t1 = time.time()
    model, i, delta_chisq = fit_dataset(
        model,
        dataset,
        eval(str(cfg["fitting"].get("maxiter", "10"))),
        eval(str(cfg["fitting"].get("chitol", "1e-5"))),
    )
    _ = mpi4jax.barrier(comm=comm)
    t2 = time.time()
    print_once(
        f"Took {t2 - t1} s to fit with {i} iterations and final delta chisq of {delta_chisq}"
    )

    print_once(model)
    _save_model(cfg, model, outdir, f"fit{r}")

    dataset = _reestimate_noise(model, dataset)
    return model, dataset


def _run_mcmc(cfg, model, dataset, outdir):
    print_once("Running MCMC")
    init_pars = np.array(model.pars.copy())
    t1 = time.time()
    model, samples = run_mcmc(
        model,
        dataset,
        num_steps=int(cfg["mcmc"].get("num_steps", 5000)),
        num_leaps=int(cfg["mcmc"].get("num_leaps", 10)),
        step_size=float(cfg["mcmc"].get("step_size", 0.02)),
        sample_which=int(cfg["mcmc"].get("sample_which", -1)),
    )
    _ = mpi4jax.barrier(comm=comm)
    t2 = time.time()
    print_once(f"Took {t2 - t1} s to run mcmc")

    message = str(model).split("\n")
    message[1] = "MCMC estimated pars:"
    print_once("\n".join(message))

    _save_model(cfg, model, outdir, "mcmc")
    if comm.Get_rank() == 0:
        samples = np.array(samples)
        samps_path = os.path.join(outdir, f"samples_mcmc.npz")
        print_once("Saving samples to", samps_path)
        np.savez_compressed(samps_path, samples=samples)
        try:
            to_fit = np.array(model.to_fit)
            corner.corner(
                samples,
                labels=np.array(model.par_names)[to_fit],
                truths=init_pars[to_fit],
            )
            plt.savefig(os.path.join(outdir, "corner.png"))
        except Exception as e:
            print_once(f"Failed to make corner plot with error: {str(e)}")
    else:
        # samples are incomplete on non root procs
        del samples

    dataset = _reestimate_noise(model, dataset)
    return model, dataset


def fit_loop(
    model, cfg, dataset, outdir, noise_class, noise_args, noise_kwargs, info, comm
):
    if model is None:
        raise ValueError("Can't fit without a model defined!")
    if cfg["sim"]:
        # Remove structs we deliberately want to leave out of model
        for struct_name in cfg["model"]["structures"]:
            if cfg["model"]["structures"][struct_name].get("to_remove", False):
                model.remove_struct(struct_name)
        params = jnp.array(model.pars)
        par_offset = cfg.get("par_offset", 1.1)
        params = params.at[model.to_fit_ever].multiply(
            par_offset
        )  # Don't start at exactly the right value
        model.update(params, model.errs, model.chisq)

    print_once("Compiling objective function")
    t0 = time.time()
    chisq, *_ = dataset.objective(model, dataset.datavec, dataset.mode)
    model = model.update(model.pars, model.errs, chisq)
    print_once(f"Took {time.time() - t0} s to compile")

    message = str(model).split("\n")
    message[1] = "Starting pars:"
    print_once("\n".join(message))
    for r in range(model.n_rounds):
        model, dataset = _run_fit(
            cfg,
            model,
            dataset,
            outdir,
            r,
        )
        _ = mpi4jax.barrier(comm=comm)

    if "mcmc" in cfg and cfg["mcmc"].get("run", True):
        model, dataset = _run_mcmc(
            cfg,
            model,
            dataset,
            outdir,
        )
        _ = mpi4jax.barrier(comm=comm)
    # Save final pars
    final = {"model": cfg["model"]}
    for i, (struct_name, structure) in zip(
        model.original_order, cfg["model"]["structures"].items()
    ):
        model_struct = model.structures[i]
        for par, par_name in zip(
            model_struct.parameters, structure["parameters"].keys()
        ):
            final["model"]["structures"][struct_name]["parameters"][par_name][
                "value"
            ] = par.val
    with open(os.path.join(outdir, "fit_params.yaml"), "w") as file:
        yaml.dump(final, file)

    return model


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
    if "MJ_TODROOT" in os.environ:
        if "WITCH_DATROOT" not in os.environ:
            print_once(
                "You are using MJ_TODROOT, this is depreciated in favor of WITCH_DATROOT. Setting WITCH_DATROOT to the value of MJ_TODROOT..."
            )
            os.environ["WITCH_DATROOT"] = os.environ["MJ_TODROOT"]
        else:
            print_once(
                "Both WITCH_DATROOT and MJ_TODROOT provided! MJ_TODROOT is depreciated, using WITCH_DATROOT..."
            )

    if "MJ_OUTROOT" in os.environ:
        if "WITCH_OUTROOT" not in os.environ:
            print_once(
                "You are using MJ_OUTROOT, this is depreciated in favor of WITCH_OUTROOT. Setting WITCH_OUTROOT to the value of MJ_OUTROOT..."
            )
            os.environ["WITCH_OUTROOT"] = os.environ["MJ_OUTROOT"]
        else:
            print_once(
                "Both WITCH_OUTROOT and MJ_OUTROOT provided! MJ_OUTROOT is depreciated, using WITCH_OUTROOT..."
            )

    parser = _make_parser()
    args = parser.parse_args()

    if args.cpu == True:
        jax.config.update("jax_platform_name", "cpu")

    # TODO: Serialize cfg to a data class (pydantic?)
    cfg = load_config({}, args.config)
    cfg["fit"] = cfg.get("fit", "model" in cfg)
    cfg["sim"] = cfg.get("sim", False)
    cfg["wnoise"] = cfg.get("wnoise", False)
    cfg["map"] = cfg.get("map", True)
    cfg["sub"] = cfg.get("sub", True)
    cfg["mem_starved"] = cfg.get("mem_starved", False)
    if args.wnoise:
        cfg["wnoise"] = True
    if args.nosub:
        cfg["sub"] = False
    if args.nofit:
        cfg["fit"] = False

    # Do imports
    for module, name in cfg.get("imports", {}).items():
        mod = import_module(module)
        if isinstance(name, str):
            locals()[name] = mod
        elif isinstance(name, list):
            for n in name:
                locals()[n] = getattr(mod, n)
        else:
            raise TypeError("Expect import name to be a string or a list")

    # Get the functions needed to work with out dataset
    dset_name = list(cfg["datasets"].keys())[0]
    if "load_tods" in cfg["datasets"][dset_name]["funcs"]:
        cfg["datasets"][dset_name]["funcs"]["load"] = cfg["datasets"][dset_name][
            "funcs"
        ]["load_tods"]
    elif "load_maps" in cfg["datasets"][dset_name]["funcs"]:
        cfg["datasets"][dset_name]["funcs"]["load"] = cfg["datasets"][dset_name][
            "funcs"
        ]["load_maps"]
    get_files = eval(cfg["datasets"][dset_name]["funcs"]["get_files"])
    load = eval(cfg["datasets"][dset_name]["funcs"]["load"])
    get_info = eval(cfg["datasets"][dset_name]["funcs"]["get_info"])
    make_beam = eval(cfg["datasets"][dset_name]["funcs"]["make_beam"])
    preproc = eval(cfg["datasets"][dset_name]["funcs"]["preproc"])
    postproc = eval(cfg["datasets"][dset_name]["funcs"]["postproc"])
    postfit = eval(cfg["datasets"][dset_name]["funcs"]["postfit"])
    dataset = DataSet(
        dset_name, get_files, load, get_info, make_beam, preproc, postproc, postfit
    )

    # Get data
    fnames = dataset.get_files(dset_name, cfg)
    global comm
    fnames, comm = _mpi_fsplit(fnames, comm)
    dataset.datavec = dataset.load(dset_name, cfg, fnames, comm)

    # Get any info we need specific to an expiriment
    dataset.info = dataset.get_info(dset_name, cfg, dataset.datavec)

    # Get the beam
    beam = dataset.make_beam(dset_name, cfg, dataset.info)

    # Define the model and get stuff setup fitting
    if "model" in cfg:
        model = Model.from_cfg(cfg, beam)
    else:
        model = None
        print_once("No model defined, setting fit, sim, and sub to False")
        cfg["fit"] = False
        cfg["sim"] = False
        cfg["sub"] = False

    # Setup noise
    dataset.info["noise_class"] = eval(
        str(cfg["datasets"][dset_name]["noise"]["class"])
    )
    dataset.info["noise_args"] = tuple(
        eval(str(cfg["datasets"][dset_name]["noise"]["args"]))
    )
    dataset.info["noise_kwargs"] = eval(
        str(cfg["datasets"][dset_name]["noise"]["kwargs"])
    )

    # Make sure we have the dataset set up properly
    dataset.check_completeness()

    # Get output
    if "base" in cfg.keys():
        del cfg["base"]  # We've collated to the cfg files so no need to keep the base
    outdir = get_outdir(cfg, model)
    os.makedirs(os.path.join(outdir, dset_name), exist_ok=True)
    dataset.info["outdir"] = outdir

    # Process the data
    preproc(dset_name, cfg, dataset.datavec, model, dataset.info)
    if dataset.mode == "tod":
        dataset.datavec = process_tods(
            cfg,
            dataset,
            model,
        )
    elif dataset.mode == "map":
        dataset.datavec = process_maps(
            cfg,
            dataset,
            model,
        )
    dataset.datavec = jax.block_until_ready(dataset.datavec)
    postproc(dset_name, cfg, dataset.datavec, model, dataset.info)

    # Now we fit
    if cfg["fit"]:
        model = fit_loop(
            model,
            cfg,
            dataset,
            outdir,
            noise_class,
            noise_args,
            noise_kwargs,
            info,
            comm,
        )

    postfit(dset_name, cfg, dataset.datavec, model, dataset.info)

    if "nonpara" in cfg:
        nonpara_model = model.para_to_non_para()
        to_copy = cfg["nonpara"].get("to_copy", "")
        n_rounds = cfg["nonpara"].get("n_rounds", None)
        if to_copy == "":
            raise ValueError("To copy must be specified")
        nonpara_model = model.para_to_non_para(n_rounds=n_rounds, to_copy=to_copy)
        oudir = get_outdir(cfg, model)
        fit_loop(
            nonpara_model,
            cfg,
            dataset,
            outdir,
            noise_class,
            noise_args,
            noise_kwargs,
            info,
            comm,
        )
        postfit(dset_name, cfg, dataset, nonpara_model, info)

    print_once("Outputs can be found in", outdir)
