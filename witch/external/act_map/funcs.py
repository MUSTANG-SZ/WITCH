import glob
import os

import jax.numpy as jnp
from astropy.io import fits
from astropy.wcs import WCS
from jitkasi.solutions import SolutionSet, maps
from mpi4py import MPI

import witch.utils as wu
from witch.containers import MetaModel
from witch.dataset import BeamConvAndPrefac, DataSet, MetaData

from ...objective import chisq_objective


def get_files(dset_name: str, cfg: dict) -> list:
    maproot = cfg["paths"].get("data", cfg["paths"]["maps"])
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
    fnames = [(m, i) for m, i in zip(map_names, ivar_names)]
    return fnames


def load_maps(
    dset_name: str, cfg: dict, fnames: list, comm: MPI.Intracomm
) -> SolutionSet:
    """
    Load the ACT maps and ivar maps.

    Parameters
    ----------
    dset_name : str
        Name of the dataset in the config.
    cfg : dict
        The config. If `datasets.<dset_name>.zero_mean` is True (default) the
        ivar weighted mean is subtracted from each map.
    fnames : list
        `(map, ivar)` file name pairs from `get_files`.
    comm : MPI.Intracomm
        Communicator for the maps.

    Returns
    -------
    SolutionSet
        The loaded maps.
    """
    map_names = [f[0] for f in fnames]
    ivar_names = [f[1] for f in fnames]
    map_glob = cfg["datasets"][dset_name].get("glob", "*_map.fits")

    imaps = []
    for fname, iname in zip(map_names, ivar_names):
        name = os.path.basename(fname)[: (1 - len(map_glob))]
        # The actual map
        f: fits.HDUList = fits.open(fname)
        wcs = WCS(f[0].header)  # type: ignore
        # Keep the FITS (y, x) layout: `WCSMap.xy` flattens its meshgrid in (y, x) order, so a
        # transposed map would be fit against a transposed model.
        dat = jnp.array(f[0].data.astype(float))  # type: ignore
        f.close()
        if dat.shape[0] != dat.shape[1]:
            raise ValueError(f"WCSMap.xy only supports square maps, got {dat.shape}")
        # The ivar map
        f: fits.HDUList = fits.open(iname)
        ivar = jnp.array(f[0].data.astype(float))  # type: ignore
        f.close()
        if cfg["datasets"][dset_name].get("zero_mean", True):
            # Match the zero-mean convention of the other datasets, this is a cutout
            # so its mean is not zero. The mean is ivar weighted and ignores empty pixels.
            dat = dat - jnp.sum(dat * ivar) / jnp.sum(ivar)
        imaps += [maps.WCSMap(name, dat, comm, wcs, "nn", ivar=ivar)]
    mapset = SolutionSet(imaps, comm)

    return mapset


def get_info(dset_name: str, cfg: dict, mapset: SolutionSet) -> dict:
    _ = (dset_name, cfg, mapset)
    prefactor = eval(str(cfg["datasets"][dset_name]["prefactor"]))
    return {
        "mode": "map",
        "prefactor": prefactor,
        "objective": chisq_objective,
        "point_sources": cfg["datasets"][dset_name].get("point_sources", []),
    }


def make_metadata(dset_name: str, cfg: dict, info: dict) -> tuple[MetaData, ...]:
    dr = eval(str(cfg["coords"]["dr"]))
    beam = wu.beam_double_gauss(
        dr,
        eval(str(cfg["datasets"][dset_name]["beam"]["fwhm1"])),
        eval(str(cfg["datasets"][dset_name]["beam"]["amp1"])),
        eval(str(cfg["datasets"][dset_name]["beam"]["fwhm2"])),
        eval(str(cfg["datasets"][dset_name]["beam"]["amp2"])),
    )

    return (
        BeamConvAndPrefac(
            beam, info["prefactor"], exclude=tuple(info["point_sources"])
        ),
    )


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
