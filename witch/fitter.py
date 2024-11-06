"""
Master fitting and map making script.
You typically want to run the `witcher` command instead of this.
"""

import argparse as argp
import glob
import os
import pdb
import sys
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import minkasi
import mpi4jax
import numpy as np
import yaml
from astropy.convolution import Gaussian2DKernel, convolve
from jitkasi import noise as jn
from jitkasi.tod import TOD, TODVec
from minkasi.tools import presets_by_source as pbs
from mpi4py import MPI
from typing_extensions import Any, Unpack

from . import grid
from . import mapmaking as mm
from . import utils as wu
from .containers import Model
from .fitting import fit_tods, objective

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


def load_tods(cfg: dict, comm: MPI.Intracomm) -> TODVec:
    rank = comm.Get_rank()
    nproc = comm.Get_size()
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
    if nproc > len(tod_names):
        minkasi.nproc = len(tod_names)
        nproc = len(tod_names)
    if rank >= len(tod_names):
        print(f"More procs than TODs!, exiting process {rank}")
        sys.exit(0)
    tod_names = tod_names[rank::nproc]

    tods = []
    for fname in tod_names:
        dat = minkasi.tods.io.read_tod_from_fits(fname)
        minkasi.tods.processing.truncate_tod(dat)
        minkasi.tods.processing.downsample_tod(dat)
        tod = minkasi.tods.Tod(dat)

        tods += [from_minkasi_tod(deepcopy(tod))]
    todvec = TODVec(tods, comm)

    return todvec


def to_minkasi(todvec: TODVec, delete=False) -> minkasi.tods.TodVec:
    todvec_minkasi = minkasi.tods.TodVec()
    for tod in todvec:
        dat = {
            "dat_calib": np.ascontiguousarray(np.array(tod.data)),
            "dx": np.ascontiguousarray(np.array(tod.x)),
            "dy": np.ascontiguousarray(np.array(tod.y)),
        }
        dat.update(tod.meta)
        noise = None
        if isinstance(tod.noise, jn.NoiseWrapper):
            noise = tod.noise.ext_inst
        if delete:
            del tod
        tod_minkasi = minkasi.tods.Tod(dat)
        tod_minkasi.noise = noise
        todvec_minkasi.add_tod(tod_minkasi)
    if delete:
        del todvec
    return todvec_minkasi


def from_minkasi(
    todvec_minkasi: minkasi.tods.TodVec, comm: MPI.Intracomm, delete=False
) -> TODVec:
    tods = []
    for tod_minkasi in todvec_minkasi.tods:
        tod = from_minkasi_tod(tod_minkasi)
        if delete:
            del tod_minkasi
        tods += [tod]
    todvec = TODVec(tods, comm)
    return todvec


def from_minkasi_tod(tod_minkasi: minkasi.tods.Tod) -> TOD:
    meta = deepcopy(tod_minkasi.info)
    data = jnp.array(meta["dat_calib"])
    del meta["dat_calib"]
    x = jnp.array(meta["dx"]).block_until_ready()
    del meta["dx"]
    y = jnp.array(meta["dy"]).block_until_ready()
    del meta["dy"]
    noise = from_minkasi_noise(tod_minkasi)
    tod = TOD(data, x, y, meta=meta, noise=noise)

    return tod


def from_minkasi_noise(tod_minkasi):
    if tod_minkasi.noise is None:
        return jn.NoiseI()
    return jn.NoiseWrapper(
        deepcopy(tod_minkasi.noise),
        "apply_noise",
        False,
        jax.ShapeDtypeStruct(
            tod_minkasi.info["dat_calib"].shape, tod_minkasi.info["dat_calib"].dtype
        ),
    )


def process_tods(cfg, todvec, noise_args, noise_kwargs, model):
    rank = todvec.comm.Get_rank()
    nproc = todvec.comm.Get_size()
    sim = cfg.get("sim", False)
    if model is None and sim:
        raise ValueError("model cannot be None when simming!")
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

            pred = model.to_tod(
                tod.x * wu.rad_to_arcsec, tod.y * wu.rad_to_arcsec
            ).block_until_ready()
            tod.data = tod.data + pred
        tod.compute_noise(jn.NoiseWrapper, None, *noise_args, **noise_kwargs)
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

    # Get TODs
    todvec = load_tods(cfg, comm)

    # make a template map with desired pixel size an limits that cover the data
    # todvec.lims is MPI-aware and will return global limits, not just
    # the ones from private TODs
    lims = todvec.lims.block_until_ready()
    lims = [np.float64(lim) for lim in np.array(lims)]
    pixsize = cfg.get("pix_size", 2.0 / wu.rad_to_arcsec)
    skymap = minkasi.maps.SkyMap(lims, pixsize)

    # Define the model and get stuff setup for minkasi
    if "model" in cfg:
        model = Model.from_cfg(cfg)
    else:
        model = None
        print_once("No model defined, setting fit, sim, and sub to False")
        cfg["fit"] = False
        cfg["sim"] = False
        cfg["sub"] = False

    # Setup noise
    noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
    noise_args = tuple(eval(str(cfg["minkasi"]["noise"]["args"])))
    noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
    noise_args_full = (noise_class, "__call__", "apply_noise", False) + noise_args

    # Get output
    if "base" in cfg.keys():
        del cfg["base"]  # We've collated to the cfg files so no need to keep the base
    outdir = get_outdir(cfg, model)

    # Make Noisemaps
    if cfg.get("noise_map", False):
        print_once("Making noise map")
        noise_vec = to_minkasi(todvec, False)  # We always take a mem hit here...
        noise_skymap = minkasi.maps.SkyMap(lims, pixsize)
        for i, tod in enumerate(noise_vec.tods):
            tod.info["dat_calib"] *= (
                (-1) ** i ** ((minkasi.myrank + minkasi.nproc * i) % 2)
            )
        minkasi.barrier()
        noise_mapset = mm.make_maps(
            noise_vec,
            noise_skymap,
            noise_class,
            noise_args,
            noise_kwargs,
            os.path.join(outdir, "noise"),
            0,
            cfg["minkasi"]["dograd"],
            return_maps=True,
        )

        if noise_mapset is None:
            raise ValueError("Noise mapset is none?")
        nmap = noise_mapset.maps[0].map
        kernel = Gaussian2DKernel(int(10 / (pixsize * wu.rad_to_arcsec)))
        nmap = convolve(nmap, kernel)

        flags = wu.get_radial_mask(nmap, pixsize * wu.rad_to_arcsec, 60.0)
        print_once(
            "Noise in central 1 arcmin is {:.2f}uK".format(np.std(nmap[flags]) * 1e6)
        )

    todvec = process_tods(cfg, todvec, noise_args_full, noise_kwargs, model)
    todvec = jax.block_until_ready(todvec)

    # Make signal maps
    if cfg.get("sig_map", cfg.get("map", True)):
        todvec_minkasi = to_minkasi(todvec, cfg["mem_starved"])
        print_once("Making signal map")
        mm.make_maps(
            todvec_minkasi,
            skymap,
            noise_class,
            noise_args,
            noise_kwargs,
            os.path.join(outdir, "signal"),
            cfg["minkasi"]["npass"],
            cfg["minkasi"]["dograd"],
        )
        if cfg["mem_starved"]:
            todvec = from_minkasi(todvec, comm, True)
        else:
            for tod, tod_minkasi in zip(todvec, todvec_minkasi.tods):
                tod.noise = from_minkasi_noise(tod_minkasi)
            del todvec_minkasi
    else:
        print_once(
            "Not making signal map, this means that your starting noise may be more off"
        )

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
            params = params.at[model.to_fit_ever].multiply(
                1.1
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
                cfg["minkasi"].get("maxiter", 10),
                cfg["minkasi"].get("chitol", 1e-5),
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
                    jn.NoiseWrapper, tod.data - pred, *noise_args_full, **noise_kwargs
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
                ] = float(par.val)
        with open(os.path.join(outdir, "fit_params.yaml"), "w") as file:
            yaml.dump(final, file)

    # Residual map (or with noise from residual)
    if cfg.get("res_map", cfg.get("map", True)):
        # Compute residual and either set it to the data or use it for noise
        if model is None:
            raise ValueError("Somehow trying to make a residual map with no model defined!")
        todvec = to_minkasi(todvec, False)
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
                    noise_class,
                    tod.info["dat_calib"] - pred,
                    *noise_args,
                    **noise_kwargs,
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
  
    #Make Model maps
    if cfg.get("model_map", False):
        print_once("Making model map")
        if model is None:
            raise ValueError("Somehow trying to make a model map with no model defined!")
        model_todvec = deepcopy(todvec)
        model_skymap = minkasi.maps.SkyMap(lims, pixsize)
        model_cfg = deepcopy(cfg)
        model_cfg["sim"] = True
        model_todvec = process_tods(cfg, model_todvec, noise_args_full, noise_kwargs, model)
        model_todvec = to_minkasi(model_todvec, False)
        mm.make_maps(
            model_todvec,
            model_skymap,
            noise_class,
            noise_args,
            noise_kwargs,
            os.path.join(outdir, "model"),
            cfg["minkasi"]["npass"],
            cfg["minkasi"]["dograd"],
        )
        model.xyz = grid.make_grid_from_wcs(
            model_skymap.wcs,
            model_skymap.map.shape[0],
            model_skymap.map.shape[1],
            0.00116355,
            0.00000969,
        )
        model_skymap.map = model.model
        if minkasi.myrank == 0:
            model_skymap.write(os.path.join(outdir, "model/truth.fits"))

    print_once("Outputs can be found in", outdir)
