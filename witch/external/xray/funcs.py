import glob
import os

import jax.numpy as jnp
from astropy.io import fits
from astropy.wcs import WCS
from jax import Array
from jitkasi.solutions import SolutionSet, maps
from mpi4py import MPI

import witch.utils as wu
from witch.containers import Model
from witch.fitter import print_once

from ...objective import poisson_objective


def get_files(dset_name: str, cfg: dict) -> list:
    maproot = cfg["paths"].get("data", cfg["paths"]["xmaps"])
    if not os.path.isabs(maproot):
        maproot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), maproot
        )
    maproot_dset = os.path.join(maproot, dset_name)
    if os.path.isdir(maproot_dset):
        maproot = maproot_dset
    map_glob = cfg["datasets"][dset_name].get("glob", "*data*.fits")
    map_names = glob.glob(os.path.join(maproot, map_glob))
    map_names.sort()
    nmaps = cfg["datasets"][dset_name].get("nmaps", None)
    fnames = map_names[:nmaps]
    return fnames


def load_maps(
    dset_name: str, cfg: dict, fnames: list, comm: MPI.Intracomm
) -> SolutionSet:
    map_names = fnames
    map_glob = cfg["datasets"][dset_name].get("glob", "*data*.fits")

    imaps = []
    for fname in map_names:
        name = os.path.basename(fname)[: (1 - len(map_glob))]
        # The actual map
        f: fits.HDUList = fits.open(fname)
        wcs = WCS(f[0].header)  # type: ignore
        dat = jnp.array(f[0].data.copy().T)  # type: ignore
        f.close()
        imaps += [maps.WCSMap(name, dat, comm, wcs, "nn")]
    mapset = SolutionSet(imaps, comm)
    return mapset


# def get_metadata():
#
#    return exp_maps, psf_maps, back_maps


def get_info(dset_name: str, cfg: dict, mapset: SolutionSet) -> dict:
    _ = (dset_name, cfg, mapset)
    prefactor = eval(str(cfg["datasets"][dset_name]["prefactor"]))
    return {
        "mode": "map",
        "prefactor": prefactor,
        "objective": poisson_objective,
    }


def make_beam(dset_name: str, cfg: dict, info: dict) -> Array:
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
            x, y = imap.xy
            pred = model.to_map(x * wu.rad_to_arcsec, y * wu.rad_to_arcsec)
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
            x, y = imap.xy
            pred = model.to_map(x * wu.rad_to_arcsec, y * wu.rad_to_arcsec)
            imap.data = pred

            hdu = fits.PrimaryHDU(data=imap.data, header=imap.wcs.to_header())
            hdul = fits.HDUList([hdu])
            hdul.writeto(
                os.path.join(outdir, dset_name, f"{imap.name}_truth.fits"),
                overwrite=True,
            )
