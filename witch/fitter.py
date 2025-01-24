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

import jax
import jax.numpy as jnp
import minkasi
import mpi4jax
import numpy as np
import yaml
from jitkasi.tod import TOD
from mpi4py import MPI
from typing_extensions import Any, Unpack

from . import utils as wu
from .containers import Model, Model_xfer
from .fitting import fit_tods, objective

comm = MPI.COMM_WORLD.Clone()

import pdb


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


def process_tods(cfg, todvec, noise_class, noise_args, noise_kwargs, model, xfer=""):
    rank = todvec.comm.Get_rank()
    nproc = todvec.comm.Get_size()
    sim = cfg.get("sim", False)
    if model is None and sim:
        raise ValueError("model cannot be None when simming!")
    model_cur = model
    if xfer:
        model_cur = Model_xfer.from_parent(model, xfer)

    for i, tod in enumerate(todvec.tods):
        if sim:
            if cfg["wnoise"]:
                temp = jnp.percentile(jnp.diff(tod.data), jnp.array([33.0, 68.0]))
                scale = (temp[1] - temp[0]) / jnp.sqrt(8)
                tod.data = scale * jax.random.normal(
                    jax.random.key(0), shape=tod.data.shape, dtype=tod.data.dtype
                )
            else:
                tod.data *= tod.data * (-1) ** ((rank + nproc * i) % 2)

            pred = model_cur.to_tod(
                tod.x * wu.rad_to_arcsec, tod.y * wu.rad_to_arcsec
            ).block_until_ready()
            tod.data = tod.data + pred
        tod.compute_noise(noise_class, None, *noise_args, **noise_kwargs)
    return todvec


def get_outdir(cfg, model):
    outroot = cfg["paths"]["outroot"]
    if not os.path.isabs(outroot):
        outroot = os.path.join(
            os.environ.get("MJ_OUTROOT", os.environ["HOME"]), outroot
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
    if minkasi.myrank == 0:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config.yaml"), "w") as file:
            yaml.dump(cfg, file)

    return outdir


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
    # Check if we have MPI
    if minkasi.comm is None:
        raise RuntimeError("Running without MPI is not currently supported!")

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
    # TODO: make protocols for these and check them
    dset_name = list(cfg["datasets"].keys())[0]
    load_tods = eval(cfg["datasets"][dset_name]["funcs"]["load_tods"])
    get_info = eval(cfg["datasets"][dset_name]["funcs"]["get_info"])
    make_beam = eval(cfg["datasets"][dset_name]["funcs"]["make_beam"])
    preproc = eval(cfg["datasets"][dset_name]["funcs"]["preproc"])
    postproc = eval(cfg["datasets"][dset_name]["funcs"]["postproc"])
    postfit = eval(cfg["datasets"][dset_name]["funcs"]["postfit"])

    # Get TODs
    todvec = load_tods(dset_name, cfg, comm)

    # Get any info we need specific to an expiriment
    info = get_info(dset_name, cfg, todvec)

    # Get the beam
    beam = make_beam(dset_name, cfg, info)

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
    noise_class = eval(str(cfg["datasets"][dset_name]["noise"]["class"]))
    noise_args = tuple(eval(str(cfg["datasets"][dset_name]["noise"]["args"])))
    noise_kwargs = eval(str(cfg["datasets"][dset_name]["noise"]["kwargs"]))
    info["noise_class"] = noise_class
    info["noise_args"] = noise_args
    info["noise_kwargs"] = noise_kwargs

    # Get output
    if "base" in cfg.keys():
        del cfg["base"]  # We've collated to the cfg files so no need to keep the base
    outdir = get_outdir(cfg, model)
    info["outdir"] = outdir

    # Process the TODs
    preproc(dset_name, cfg, todvec, model, info)
    todvec = process_tods(
        cfg, todvec, noise_class, noise_args, noise_kwargs, model, info["xfer"]
    )
    todvec = jax.block_until_ready(todvec)
    postproc(dset_name, cfg, todvec, model, info)

    # Now we fit
    if cfg["fit"]:
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
        model, *_ = objective(model.pars, model, todvec, model.errs)
        print_once(f"Took {time.time() - t0} s to compile")

        message = str(model).split("\n")
        message[1] = "Starting pars:"
        print_once("\n".join(message))
        for r in range(model.n_rounds):
            model.cur_round = r
            to_fit = np.array(model.to_fit)
            print_once(
                f"Starting round {r+1} of fitting with {np.sum(to_fit)} pars free"
            )
            t1 = time.time()
            model, i, delta_chisq = fit_tods(
                model,
                todvec,
                eval(str(cfg["fitting"].get("maxiter", "10"))),
                eval(str(cfg["fitting"].get("chitol", "1e-5"))),
            )
            _ = mpi4jax.barrier(comm=comm)
            t2 = time.time()
            print_once(
                f"Took {t2 - t1} s to fit with {i} iterations and final delta chisq of {delta_chisq}"
            )

            print_once(model)

            if minkasi.myrank == 0:
                res_path = os.path.join(outdir, f"results_{r}.dill")
                print_once("Saving results to", res_path)
                # TODO: switch to h5?
                model.save(res_path)

            # Reestimate noise
            for tod in todvec:
                pred = model.to_tod(tod.x * wu.rad_to_arcsec, tod.y * wu.rad_to_arcsec)
                tod.compute_noise(
                    noise_class, tod.data - pred, *noise_args, **noise_kwargs
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

    postfit(dset_name, cfg, todvec, model, info)

    print_once("Outputs can be found in", outdir)
