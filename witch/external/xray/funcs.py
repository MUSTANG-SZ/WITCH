import glob
import os

import jax.numpy as jnp
from astropy.io import fits
from astropy.wcs import WCS
from jax import Array
from jitkasi.solutions import SolutionSet, maps
from mpi4py import MPI

import witch.utils as wu
from witch.containers import MetaModel
from witch.dataset import DataSet
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


def get_info(dset_name: str, cfg: dict, mapset: SolutionSet) -> dict:
    _ = (dset_name, cfg, mapset)
    prefactor = eval(str(cfg["datasets"][dset_name]["prefactor"]))
    return {
        "mode": "map",
        "prefactor": prefactor,
        "objective": poisson_objective,
    }


def make_metadata(dset_name: str, cfg: dict, comm: MPI.Intracomm) -> tuple:
    maproot = cfg["paths"].get("data", cfg["paths"]["xmaps"])
    if not os.path.isabs(maproot):
        maproot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), maproot
        )
    maproot_dset = os.path.join(maproot, dset_name)
    if os.path.isdir(maproot_dset):
        maproot = maproot_dset

    # import beams
    beam_glob = cfg["datasets"][dset_name].get("glob", "*psf*.fits")
    beam_fnames = glob.glob(os.path.join(maproot, beam_glob))
    beam_fnames.sort()
    beams = []
    for fname in beam_fnames:
        name = os.path.splitext(os.path.basename(fname))[0]
        # The actual map
        f: fits.HDUList = fits.open(fname)
        wcs = WCS(f[0].header)  # type: ignore
        dat = jnp.array(f[0].data.copy().T)  # type: ignore
        f.close()
        beams += [maps.WCSMap(name, dat, comm, wcs, "nn")]

    return beams


def make_exp_maps(dset_name: str, cfg: dict, comm: MPI.Intracomm) -> Array:
    maproot = cfg["paths"].get("data", cfg["paths"]["xmaps"])
    if not os.path.isabs(maproot):
        maproot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), maproot
        )
    maproot_dset = os.path.join(maproot, dset_name)
    if os.path.isdir(maproot_dset):
        maproot = maproot_dset
    map_glob = cfg["datasets"][dset_name].get("glob", "*exp*.fits")
    fnames = glob.glob(os.path.join(maproot, map_glob))
    fnames.sort()

    exp_maps = []
    for fname in exp_fnames:
        name = os.path.splitext(os.path.basename(fname))[0]
        # The actual map
        f: fits.HDUList = fits.open(fname)
        wcs = WCS(f[0].header)  # type: ignore
        dat = jnp.array(f[0].data.copy().T)  # type: ignore
        f.close()
        exp_maps += [maps.WCSMap(name, dat, comm, wcs, "nn")]

    return exp_maps


def make_back_map(dset_name: str, cfg: dict, comm: MPI.Intracomm) -> Array:
    maproot = cfg["paths"].get("data", cfg["paths"]["xmaps"])
    if not os.path.isabs(maproot):
        maproot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), maproot
        )
    maproot_dset = os.path.join(maproot, dset_name)
    if os.path.isdir(maproot_dset):
        maproot = maproot_dset
    map_glob = cfg["datasets"][dset_name].get("glob", "*back*.fits")
    fname = glob.glob(os.path.join(maproot, map_glob))[0]

    name = os.path.splitext(os.path.basename(fname))[0]
    # The actual map
    f: fits.HDUList = fits.open(fname)
    wcs = WCS(f[0].header)  # type: ignore
    dat = jnp.array(f[0].data.copy().T)  # type: ignore
    f.close()
    back_map = [maps.WCSMap(name, dat, comm, wcs, "nn")]

    meta_list = [beams, exp_maps, back_map]
    metadata = tuple(meta_list)
    print("type metadata: ", type(metadata[0][0]))
    return metadata


def preproc(dset: DataSet, cfg: dict, metamodel: MetaModel):
    _ = (dset, cfg, metamodel)


def postproc(dset: DataSet, cfg: dict, metamodel: MetaModel):
    _ = (dset, cfg, metamodel)


def postfit(dset: DataSet, cfg: dict, metamodel: MetaModel):
    outdir = dset.info["outdir"]
    # Residual map (or with noise from residual)
    res_map = cfg.get("res_map", cfg.get("map", True))
    mod_map = cfg.get("model_map", cfg.get("map", True))
    if res_map or mod_map:
        # Compute residual and either set it to the data or use it for noise
        if metamodel is None:
            raise ValueError(
                "Somehow trying to make a residual or model map with no model defined!"
            )
        dset_ind = metamodel.get_dataset_ind(dset.name)
        for i, imap in enumerate(dset.datavec):
            pred = metamodel.model_proj(dset_ind, i)

            if res_map:
                imap.data = imap.data - pred
                hdu = fits.PrimaryHDU(data=imap.data, header=imap.wcs.to_header())
                hdul = fits.HDUList([hdu])
                hdul.writeto(
                    os.path.join(outdir, dset.name, f"{imap.name}_residual.fits"),
                    overwrite=True,
                )

            if mod_map:
                imap.data = pred
                hdu = fits.PrimaryHDU(data=imap.data, header=imap.wcs.to_header())
                hdul = fits.HDUList([hdu])
                hdul.writeto(
                    os.path.join(outdir, dset.name, f"{imap.name}_truth.fits"),
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
