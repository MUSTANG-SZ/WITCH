import glob
import os
import sys
from copy import deepcopy

import jax.numpy as jnp
import minkasi
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from jitkasi.tod import TODVec
from minkasi.tools import presets_by_source as pbs
from mpi4py import MPI

import witch.utils as wu
from witch import grid
from witch.containers import Model
from witch.fitter import print_once, process_tods

from . import mapmaking as mm
from .utils import from_minkasi, from_minkasi_noise, from_minkasi_tod, to_minkasi


def get_files(dset_name: str, cfg: dict) -> list:
    todroot = cfg.get("data", cfg["paths"]["tods"])
    if not os.path.isabs(todroot):
        todroot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), todroot
        )
    todroot_dset = os.path.join(todroot, dset_name)
    if os.path.isdir(todroot_dset):
        todroot = todroot_dset
    tod_names = glob.glob(
        os.path.join(
            todroot,
            cfg["datasets"][dset_name].get("glob", cfg["paths"].get("glob", "*.fits")),
        )
    )
    bad_tod, _ = pbs.get_bad_tods(
        cfg["name"], ndo=cfg["paths"]["ndo"], odo=cfg["paths"]["odo"]
    )
    if "cut" in cfg["paths"]:
        bad_tod += cfg["paths"]["cut"]
    tod_names = minkasi.tods.io.cut_blacklist(tod_names, bad_tod)
    tod_names.sort()
    ntods = cfg["datasets"][dset_name].get("ntods", None)
    tod_names = tod_names[:ntods]

    return tod_names


def load_tods(dset_name: str, cfg: dict, fnames: list, comm: MPI.Intracomm) -> TODVec:
    tods = []
    for fname in fnames:
        dat = minkasi.tods.io.read_tod_from_fits(fname)
        minkasi.tods.processing.truncate_tod(dat)
        minkasi.tods.processing.downsample_tod(dat)
        tod = minkasi.tods.Tod(dat)

        tods += [from_minkasi_tod(deepcopy(tod))]
    todvec = TODVec(tods, comm)

    return todvec


def get_info(dset_name: str, cfg: dict, todvec: TODVec) -> dict:
    # Setup minkasi noise
    noise_class = eval(str(cfg["datasets"][dset_name]["minkasi_noise"]["class"]))
    noise_args = tuple(eval(str(cfg["datasets"][dset_name]["minkasi_noise"]["args"])))
    noise_kwargs = eval(str(cfg["datasets"][dset_name]["minkasi_noise"]["kwargs"]))

    # make a template map with desired pixel size an limits that cover the data
    # todvec.lims is MPI-aware and will return global limits, not just
    # the ones from private TODs
    lims = todvec.lims.block_until_ready()
    lims = [np.float64(lim) for lim in np.array(lims)]
    pixsize = cfg.get("pix_size", 2.0 / wu.rad_to_arcsec)
    skymap = minkasi.maps.SkyMap(lims, pixsize)

    prefactor = eval(str(cfg["datasets"][dset_name]["prefactor"]))

    return {
        "mode": "tod",
        "lims": lims,
        "pixsize": pixsize,
        "skymap": skymap,
        "minkasi_noise_class": noise_class,
        "minkasi_noise_args": noise_args,
        "minkasi_noise_kwargs": noise_kwargs,
        "copy_noise": cfg["datasets"][dset_name]["copy_noise"],
        "xfer": cfg["datasets"][dset_name].get("xfer", ""),
        "prefactor": prefactor,
    }


def make_beam(dset_name: str, cfg: dict, info: dict):
    _ = info
    dr = eval(str(cfg["coords"]["dr"]))
    beam = wu.beam_double_gauss(
        dr,
        eval(str(cfg["datasets"][dset_name]["beam"]["fwhm1"])),
        eval(str(cfg["datasets"][dset_name]["beam"]["amp1"])),
        eval(str(cfg["datasets"][dset_name]["beam"]["fwhm2"])),
        eval(str(cfg["datasets"][dset_name]["beam"]["amp2"])),
    )

    return beam


def preproc(dset_name: str, cfg: dict, todvec: TODVec, model: Model, info: dict):
    _ = model
    lims = info["lims"]
    pixsize = info["pixsize"]
    noise_class = info["minkasi_noise_class"]
    noise_args = info["minkasi_noise_args"]
    noise_kwargs = info["minkasi_noise_kwargs"]
    outdir = info["outdir"]
    copy_noise = info["copy_noise"]
    if not cfg.get("noise_map", False):
        return

    print_once("Making noise map")
    noise_vec = to_minkasi(
        todvec, copy_noise, False
    )  # We always take a mem hit here...
    noise_skymap = minkasi.maps.SkyMap(lims, pixsize)
    for i, tod in enumerate(noise_vec.tods):
        tod.info["dat_calib"] *= (-1) ** i ** ((minkasi.myrank + minkasi.nproc * i) % 2)
    minkasi.barrier()
    noise_mapset = mm.make_maps(
        noise_vec,
        noise_skymap,
        noise_class,
        noise_args,
        noise_kwargs,
        os.path.join(outdir, dset_name, "noise"),
        0,
        cfg["datasets"][dset_name]["dograd"],
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


def postproc(dset_name: str, cfg: dict, todvec: TODVec, model: Model, info: dict):
    _ = model
    skymap = info["skymap"]
    noise_class = info["minkasi_noise_class"]
    noise_args = info["minkasi_noise_args"]
    noise_kwargs = info["minkasi_noise_kwargs"]
    outdir = info["outdir"]
    copy_noise = info["copy_noise"]
    # Make signal maps
    if cfg.get("sig_map", cfg.get("map", True)):
        todvec_minkasi = to_minkasi(todvec, copy_noise, cfg["mem_starved"])
        print_once("Making signal map")
        mapset = mm.make_maps(
            todvec_minkasi,
            skymap,
            noise_class,
            noise_args,
            noise_kwargs,
            os.path.join(outdir, dset_name, "signal"),
            cfg["datasets"][dset_name]["npass"],
            cfg["datasets"][dset_name]["dograd"],
            return_maps=True,
        )
        if mapset is None:
            raise ValueError("No mapset returned!")
        if cfg["mem_starved"]:
            todvec = from_minkasi(todvec_minkasi, todvec.comm, True)
        elif copy_noise:
            for tod, tod_minkasi in zip(todvec, todvec_minkasi.tods):
                tod.noise = from_minkasi_noise(tod_minkasi)
        else:
            for tod, tod_minkasi in zip(todvec, todvec_minkasi.tods):
                mat = 0 * tod_minkasi.info["dat_calib"].copy()
                for m in mapset.maps:
                    m.map2tod(tod_minkasi, mat)
                mat -= np.mean(mat, axis=-1)[..., None]  # Avoid overflow
                tod.recompute_noise(
                    data=tod.data - jnp.array(mat),
                    *info["noise_args"],
                    **info["noise_kwargs"],
                )
        del todvec_minkasi
    else:
        print_once(
            "Not making signal map, this means that your starting noise may be more off"
        )


def postfit(dset_name: str, cfg: dict, todvec: TODVec, model: Model, info: dict):
    lims = info["lims"]
    pixsize = info["pixsize"]
    skymap = info["skymap"]
    noise_class = info["noise_class"]
    noise_args = info["noise_args"]
    noise_kwargs = info["noise_kwargs"]
    minkasi_noise_class = info["minkasi_noise_class"]
    minkasi_noise_args = info["minkasi_noise_args"]
    minkasi_noise_kwargs = info["minkasi_noise_kwargs"]
    outdir = info["outdir"]
    # Residual map (or with noise from residual)
    if cfg.get("res_map", cfg.get("map", True)):
        # Compute residual and either set it to the data or use it for noise
        if model is None:
            raise ValueError(
                "Somehow trying to make a residual map with no model defined!"
            )
        todvec_minkasi = to_minkasi(todvec, False)
        for tod in todvec_minkasi.tods:
            pred = model.to_tod(
                tod.info["dx"] * wu.rad_to_arcsec,
                tod.info["dy"] * wu.rad_to_arcsec,
            )
            if cfg["sub"]:
                tod.info["dat_calib"] -= np.array(pred)
                tod.set_noise(
                    minkasi_noise_class, *minkasi_noise_args, **minkasi_noise_kwargs
                )
            else:
                tod.set_noise(
                    minkasi_noise_class,
                    tod.info["dat_calib"] - pred,
                    *minkasi_noise_args,
                    **minkasi_noise_kwargs,
                )

        # Make residual maps
        if cfg["sub"]:
            print_once("Making residual map")
            name = "residual"
        else:
            print_once("Making signal map with residual noise")
            name = "signal_res_noise"
        mm.make_maps(
            todvec_minkasi,
            skymap,
            minkasi_noise_class,
            minkasi_noise_args,
            minkasi_noise_kwargs,
            os.path.join(outdir, dset_name, name),
            cfg["datasets"][dset_name]["npass"],
            cfg["datasets"][dset_name]["dograd"],
        )

    # Make Model maps
    if cfg.get("model_map", False):
        print_once("Making model map")
        if model is None:
            raise ValueError(
                "Somehow trying to make a model map with no model defined!"
            )
        model_todvec = todvec.copy(deep=True)
        model_skymap = minkasi.maps.SkyMap(lims, pixsize)
        model_cfg = deepcopy(cfg)
        model_cfg["sim"] = True
        model_todvec = process_tods(
            cfg, model_todvec, noise_class, noise_args, noise_kwargs, model
        )
        model_todvec = to_minkasi(model_todvec, False)
        mm.make_maps(
            model_todvec,
            model_skymap,
            minkasi_noise_class,
            minkasi_noise_args,
            minkasi_noise_kwargs,
            os.path.join(outdir, dset_name, "model"),
            cfg["datasets"][dset_name]["npass"],
            cfg["datasets"][dset_name]["dograd"],
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
