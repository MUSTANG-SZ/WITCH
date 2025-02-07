import glob
import os
import sys

import jax.numpy as jnp
from astropy.io import fits
from astropy.wcs import WCS
from jitkasi.solutions import SolutionSet, maps
from mpi4py import MPI

import witch.utils as wu
from witch import grid
from witch.containers import Model
from witch.fitter import print_once


def load_maps(dset_name: str, cfg: dict, comm: MPI.Intracomm) -> SolutionSet:
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    maproot = cfg.get("data", cfg["paths"]["maps"])
    if not os.path.isabs(maproot):
        maproot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), maproot
        )
    maproot_dset = os.path.join(maproot, dset_name)
    if os.path.isdir(maproot_dset):
        maproot = maproot_dset
    map_glob = cfg["datasets"][dset_name].get("glob", "*_map.fits")
    map_names = glob.glob(os.path.join(maproot, map_glob))
    ivar_names = glob.glob(
        os.path.join(
            maproot, cfg["datasets"][dset_name].get("ivar_glob", "*_ivar.fits")
        )
    )
    map_names.sort()
    ivar_names.sort()
    nmaps = cfg["datasets"][dset_name].get("nmaps", None)
    map_names = map_names[:nmaps]
    ivar_names = ivar_names[:nmaps]
    if nproc > len(map_names):
        nproc = len(map_names)
    if rank >= len(map_names):
        print(f"More procs than maps!, exiting process {rank}")
        sys.exit(0)
    map_names = map_names[rank::nproc]
    ivar_names = ivar_names[rank::nproc]

    imaps = []
    for fname in map_names:
        name = os.path.basename(fname)[: (1 - len(map_glob))]
        # The actual map
        f: fits.HDUList = fits.open(fname)
        wcs = WCS(f[0].header)  # type: ignore
        dat = jnp.array(f[0].data.copy().T)  # type: ignore
        f.close()
        # The ivar map
        f: fits.HDUList = fits.open(fname)
        ivar = jnp.array(f[0].data.copy().T)  # type: ignore
        f.close()
        imaps += [maps.WCSMap(name, dat, comm, wcs, "nn", ivar=ivar)]
    mapset = SolutionSet(imaps, comm)

    return mapset


def get_info(dset_name: str, cfg: dict, mapset: SolutionSet) -> dict:
    _ = (dset_name, cfg, mapset)
    return {
        "mode": "map",
    }


def make_beam(dset_name: str, cfg: dict, info: dict):
    # TODO: Maybe just load from a file?
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


def preproc(dset_name: str, cfg: dict, mapset: SolutionSet, model: Model, info: dict):
    _ = (dset_name, cfg, mapset, model, info)


def postproc(dset_name: str, cfg: dict, mapset: SolutionSet, model: Model, info: dict):
    _ = (dset_name, cfg, mapset, model, info)


def postfit(dset_name: str, cfg: dict, mapset: SolutionSet, model: Model, info: dict):
    outdir = info["outdir"]
    # Residual map (or with noise from residual)
    if cfg.get("res_map", cfg.get("map", True)):
        # Compute residual and either set it to the data or use it for noise
        if model is None:
            raise ValueError(
                "Somehow trying to make a residual map with no model defined!"
            )
        for imap in mapset:
            pred = model.to_map(*imap.xy)
            imap.data = imap.data - pred

            hdu = fits.PrimaryHDU(data=imap.data, header=imap.wcs.to_header())
            hdul = fits.HDUList([hdu])
            hdul.writeto(
                os.path.join(outdir, dset_name, f"{imap.name}_residual.fits"),
                overwrite=True,
            )

    # Make Model maps
    if cfg.get("model_map", cfg.get("map", True)):
        print_once("Making model map")
        if model is None:
            raise ValueError(
                "Somehow trying to make a model map with no model defined!"
            )
        for imap in mapset:
            pred = model.to_map(*imap.xy)
            imap.data = pred

            hdu = fits.PrimaryHDU(data=imap.data, header=imap.wcs.to_header())
            hdul = fits.HDUList([hdu])
            hdul.writeto(
                os.path.join(outdir, dset_name, f"{imap.name}_truth.fits"),
                overwrite=True,
            )
