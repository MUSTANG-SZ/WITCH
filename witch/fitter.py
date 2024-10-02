"""
Master fitting and map making script.
You typically want to run the `witcher` command instead of this.
"""

import argparse as argp
import glob
import os
import sys
import time
from copy import deepcopy

import jax as jax
import minkasi
import numpy as np
import yaml
from minkasi.tools import presets_by_source as pbs
from typing_extensions import Any, Unpack

from . import mapmaking as mm
from . import utils as wu
from .containers import Model


def print_once(*args: Unpack[tuple[Any, ...]]):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Parameters
    ----------
    *args : Unpack[tuple[Any, ...]]
        Arguments to pass to print.
    """
    if minkasi.myrank == 0:
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


def load_tods(cfg: dict) -> minkasi.tods.TodVec:
    todroot = cfg["paths"]["tods"]
    if not os.path.isabs(todroot):
        todroot = os.path.join(
            os.environ.get("MJ_TODROOT", os.environ["HOME"]), todroot
        )
    tod_names = glob.glob(os.path.join(todroot, cfg["paths"]["glob"]))
    bad_tod, _ = pbs.get_bad_tods(
        cfg["name"], ndo=cfg["paths"]["ndo"], odo=cfg["paths"]["odo"]
    )
    if "cut" in cfg["paths"]:
        bad_tod += cfg["paths"]["cut"]
    tod_names = minkasi.tods.io.cut_blacklist(tod_names, bad_tod)
    tod_names.sort()
    ntods = cfg["minkasi"].get("ntods", None)
    tod_names = tod_names[:ntods]
    print(minkasi.nproc, len(tod_names))
    if minkasi.nproc > len(tod_names):
        minkasi.nproc = len(tod_names)
    if minkasi.myrank >= len(tod_names):
        print(f"More procs than TODs!, exiting process {minkasi.myrank}")
        sys.exit(0)
    tod_names = tod_names[minkasi.myrank :: minkasi.nproc]
    print(minkasi.myrank, tod_names)
    minkasi.barrier()  # Is this needed?

    todvec = minkasi.tods.TodVec()
    for fname in tod_names:
        dat = minkasi.tods.io.read_tod_from_fits(fname)
        minkasi.tods.processing.truncate_tod(dat)
        minkasi.tods.processing.downsample_tod(dat)
        minkasi.tods.processing.truncate_tod(dat)
        # figure out a guess at common mode and (assumed) linear detector drifts/offset
        # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
        dd, pred2, cm = minkasi.tods.processing.fit_cm_plus_poly(
            dat["dat_calib"], cm_ord=3, full_out=True
        )
        dat["dat_calib"] = dd
        dat["pred2"] = pred2
        dat["cm"] = cm

        tod = minkasi.tods.Tod(dat)
        todvec.add_tod(tod)

    return todvec


def process_tods(
    cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
) -> str:
    bowling = cfg.get("bowling", {})
    sub_poly = bowling.get("sub_poly", False)
    method = bowling.get("method", "pred2")
    degree = bowling.get("degree", 2)
    sim = cfg.get("sim", False)
    if model is None and sim:
        raise ValueError("model cannot be None when simming!")
    for i, tod in enumerate(todvec.tods):
        ipix = skymap.get_pix(tod)
        tod.info["ipix"] = ipix

        if sub_poly:
            tod.set_apix()
            for j in range(tod.info["dat_calib"].shape[0]):
                x, y = (
                    tod.info["apix"][j],
                    tod.info["dat_calib"][j] - tod.info[method][j],
                )
                res = np.polynomial.polynomial.polyfit(x, y, degree)
                tod.info["dat_calib"][j] -= np.polynomial.polynomial.polyval(x, res)

        if sim:
            if cfg["wnoise"]:
                temp = np.percentile(np.diff(tod.info["dat_calib"]), [33, 68])
                scale = (temp[1] - temp[0]) / np.sqrt(8)
                tod.info["dat_calib"] = np.random.normal(
                    0, scale, size=tod.info["dat_calib"].shape
                )
            else:
                tod.info["dat_calib"] *= (-1) ** (
                    (minkasi.myrank + minkasi.nproc * i) % 2
                )

            pred = model.to_tod(
                tod.info["dx"] * wu.rad_to_arcsec, tod.info["dy"] * wu.rad_to_arcsec
            )
            tod.info["dat_calib"] += np.array(pred)

        tod.set_noise(noise_class, *noise_args, **noise_kwargs)
    if sub_poly:
        return f"-{method}_{degree}"
    return ""


def get_outdir(cfg, bowl_str, model):
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
    outdir += bowl_str
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
    if args.wnoise:
        cfg["wnoise"] = True
    if args.nosub:
        cfg["sub"] = False
    if args.nofit:
        cfg["fit"] = False

    # Get TODs
    todvec = load_tods(cfg)

    # make a template map with desired pixel size an limits that cover the data
    # todvec.lims() is MPI-aware and will return global limits, not just
    # the ones from private TODs
    lims = todvec.lims()
    pixsize = cfg.get("pix_size", 2.0 / wu.rad_to_arcsec)
    skymap = minkasi.maps.SkyMap(lims, pixsize)

    # Define the model and get stuff setup for minkasi
    if "model" in cfg:
        model = Model.from_cfg(cfg, pix_size=pixsize, lims=tuple(lims))
    else:
        model = None
        print_once("No model defined, setting fit, sim, and sub to False")
        cfg["fit"] = False
        cfg["sim"] = False
        cfg["sub"] = False

    # Deal with bowling and simming in TODs and setup noise
    noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
    noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
    noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
    bowl_str = process_tods(
        cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
    )

    # Get output
    if "base" in cfg.keys():
        del cfg["base"]  # We've collated to the cfg files so no need to keep the base
    outdir = get_outdir(cfg, bowl_str, model)

    # Make signal maps
    if cfg.get("sig_map", cfg.get("map", True)):
        print_once("Making signal map")
        mm.make_maps(
            todvec,
            skymap,
            noise_class,
            noise_args,
            noise_kwargs,
            os.path.join(outdir, "signal"),
            cfg["minkasi"]["npass"],
            cfg["minkasi"]["dograd"],
        )
    else:
        print_once(
            "Not making signal map, this means that your starting noise may be more off"
        )

    # Now we fit
    if cfg["fit"]:
        if model is None:
            raise ValueError("Can't fit without a model defined!")
        funs = [model.minkasi_helper]
        params = np.array(model.pars)
        npars = np.array([len(params)])
        prior_vals = model.priors
        priors = [None if prior is None else "flat" for prior in prior_vals]

        if cfg["sim"]:
            # Remove structs we deliberately want to leave out of model
            for struct_name in cfg["model"]["structures"]:
                if cfg["model"]["structures"][struct_name].get("to_remove", False):
                    model.remove_struct(struct_name)
            params = np.array(model.pars)
            npars = np.array([len(params)])
            prior_vals = model.priors
            priors = [None if prior is None else "flat" for prior in prior_vals]
            params[model.to_fit_ever] *= 1.1  # Don't start at exactly the right value
            model.update(list(params), model.errs, model.chisq)

        message = str(model).split("\n")
        message[1] = "Starting pars:"
        print_once("\n".join(message))
        for i in range(model.n_rounds):
            model.cur_round = i
            to_fit = np.array(model.to_fit)
            print_once(
                f"Starting round {i+1} of fitting with {np.sum(to_fit)} pars free"
            )
            t1 = time.time()
            (
                pars_fit,
                chisq,
                _,
                errs,
            ) = minkasi.fitting.fit_timestreams_with_derivs_manyfun(
                funs,
                model.pars,
                npars,
                todvec,
                to_fit,
                maxiter=cfg["minkasi"]["maxiter"],
                priors=priors,
                prior_vals=prior_vals,
            )
            minkasi.comm.barrier()
            t2 = time.time()
            print_once("Took", t2 - t1, "seconds to fit")

            model.update(pars_fit, errs, chisq)
            print_once(model)

            if minkasi.myrank == 0:
                res_path = os.path.join(outdir, f"results_{i}.dill")
                print_once("Saving results to", res_path)
                # TODO: switch to h5?
                model.save(res_path)

            # Reestimate noise
            for i, tod in enumerate(todvec.tods):
                pred = model.to_tod(
                    tod.info["dx"] * wu.rad_to_arcsec,
                    tod.info["dy"] * wu.rad_to_arcsec,
                )

                tod.set_noise(
                    noise_class,
                    tod.info["dat_calib"] - pred,
                    *noise_args,
                    **noise_kwargs,
                )
            minkasi.barrier()
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
                ] = float(par.val)
        with open(os.path.join(outdir, "fit_params.yaml"), "w") as file:
            yaml.dump(final, file)

    # If we arenn't mapmaking then we can stop here
    if cfg["sub"] is False or not cfg.get("res_map", cfg.get("map", True)):
        return

    # Compute residual and either set it to the data or use it for noise
    if model is None:
        raise ValueError("Somehow trying to make a residual map with no model defined!")
    for i, tod in enumerate(todvec.tods):
        pred = model.to_tod(
            tod.info["dx"] * wu.rad_to_arcsec,
            tod.info["dy"] * wu.rad_to_arcsec,
        )
        if cfg["sub"]:
            tod.info["dat_calib"] -= np.array(pred)
            tod.set_noise(noise_class, *noise_args, **noise_kwargs)
        else:
            tod.set_noise(
                noise_class, tod.info["dat_calib"] - pred, *noise_args, **noise_kwargs
            )

    # Make residual maps
    print_once("Making residual map")
    mm.make_maps(
        todvec,
        skymap,
        noise_class,
        noise_args,
        noise_kwargs,
        os.path.join(outdir, "residual"),
        cfg["minkasi"]["npass"],
        cfg["minkasi"]["dograd"],
    )
