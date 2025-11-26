import glob
import os
from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.wcs import WCS
from jax import Array, vmap
from jax.tree_util import register_pytree_node_class
from jitkasi.solutions import SolutionSet, maps
from mpi4py import MPI

import witch.utils as wu
from witch.containers import MetaModel
from witch.dataset import DataSet, MetaData
from witch.fitter import print_once

from ...objective import poisson_objective


def convolve(img, PSF):
    """
    Parameters
    ----------
    img : np.ndarray
        The map to be smoothed.

    res : float
        FWHM (resolution) of the smoothing (arcmin).

    dim_pixel : float
        Pixel size in the map (arcmin).

    Returns
    -------
    img_convolved: np.ndarray
        Same map but smoothed.
    """

    img_convolved = convolve_fft(img, PSF)

    return img_convolved


convolve_vec = vmap(convolve, in_axes=(0, None))


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


def load_exps(dset_name: str, cfg: dict, struct_names: list):
    # def __init__(self, dset_name: str, cfg: dict):
    maproot = cfg["paths"].get("data", cfg["paths"]["xmaps"])
    if not os.path.isabs(maproot):
        maproot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), maproot
        )
    maproot_dset = os.path.join(maproot, dset_name)
    if os.path.isdir(maproot_dset):
        maproot = maproot_dset

    # import exposure maps
    exp_glob = cfg["datasets"][dset_name].get("glob", "*exp*.fits")
    exp_fnames = glob.glob(os.path.join(maproot, exp_glob))
    order_map = {item: index for index, item in enumerate(struct_names)}
    exp_fnames = sorted(exp_fnames, key=lambda x: order_map.get(x, float("inf")))
    exp_maps = []
    for strc_name in struct_names:
        for fname in exp_fnames:
            if strc_name in fname:
                name = os.path.splitext(os.path.basename(fname))[0]
                # The actual map
                f: fits.HDUList = fits.open(fname)
                wcs = WCS(f[0].header)  # type: ignore
                dat = jnp.array(f[0].data.copy().T)  # type: ignore
                f.close()
                exp_maps += [dat]
    return exp_maps


def load_beams(dset_name: str, cfg: dict, struct_names: list):
    # def __init__(self, dset_name: str, cfg: dict):
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
    order_map = {item: index for index, item in enumerate(struct_names)}
    beam_fnames = sorted(beam_fnames, key=lambda x: order_map.get(x, float("inf")))
    beams = []
    for strc_name in struct_names:
        for fname in beam_fnames:
            if strc_name in fname:
                name = os.path.splitext(os.path.basename(fname))[0]
                # The actual map
                f: fits.HDUList = fits.open(fname)
                wcs = WCS(f[0].header)  # type: ignore
                dat = jnp.array(f[0].data.copy().T)  # type: ignore
                f.close()
                beams += [dat]
    return beams


def load_back(dset_name: str, cfg: dict):
    maproot = cfg["paths"].get("data", cfg["paths"]["xmaps"])
    if not os.path.isabs(maproot):
        maproot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), maproot
        )
    maproot_dset = os.path.join(maproot, dset_name)
    if os.path.isdir(maproot_dset):
        maproot = maproot_dset

    back_glob = cfg["datasets"][dset_name].get("glob", "*back*.fits")
    back_fname = glob.glob(os.path.join(maproot, back_glob))[0]
    back_name = os.path.splitext(os.path.basename(back_fname))[0]
    # The actual map
    f: fits.HDUList = fits.open(back_fname)
    wcs = WCS(f[0].header)  # type: ignore
    dat = jnp.array(f[0].data.copy().T)  # type: ignore
    f.close()
    return dat


@register_pytree_node_class
@dataclass
class ExpConvProj(MetaData):
    exp_map: Array
    beam_map: Array

    def apply(self, model: Array) -> Array:
        return self.exp_map * convolve(model, self.beam_map)

    def apply_grad(self, model_grad: Array) -> Array:
        # Using *c* as convolution below
        # d/dx_i ( exp_map*(beam_map *c* model))
        # = d/dx_i(exp_map*beam_map *c* model))
        # = exp_map*beam_map *c* d/dx_i(model)
        # = exp_map*beam_map *c* model_grad
        return convolve_vec(model_grad, self.exp_map * self.beam_map)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.exp_map, self.beam_map)
        aux_data = tuple()

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        _ = aux_data

        return cls(*children)


@register_pytree_node_class
@dataclass
class BackgroundProj(MetaData):
    back_map: Array

    def apply(self, model: Array) -> Array:
        return model + self.back_map

    def apply_grad(self, model_grad: Array) -> Array:
        return model_grad  # Functions for making this a pytree

    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:

        children = (self.back_map,)
        aux_data = tuple()

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        _ = aux_data

        return cls(children[0])


def make_metadata(dset_name: str, cfg: dict, info: dict) -> tuple[MetaData, ...]:
    _ = info
    dr = eval(str(cfg["coords"]["dr"]))
    struct_names = cfg["model"]["structures"].keys()

    exp_maps = load_exps(dset_name, cfg, struct_names)
    beam_maps = load_beams(dset_name, cfg, struct_names)
    back_map = load_back(dset_name, cfg)

    metadata = []
    for idx in range(len(struct_names)):
        metadata += [ExpConvProj(exp_maps[idx], beam_maps[idx])]
    metadata += [BackgroundProj(back_map)]

    return tuple(metadata)


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


"""
def make_metadata(dset_name: str, cfg: dict, comm: MPI.Intracomm) -> tuple:
    maproot = cfg["paths"].get("data", cfg["paths"]["xmaps"])
    if not os.path.isabs(maproot):
        maproot = os.path.join(
            os.environ.get("WITCH_DATROOT", os.environ["HOME"]), maproot
        )
    maproot_dset = os.path.join(maproot, dset_name)
    if os.path.isdir(maproot_dset):
        maproot = maproot_dset

    back_glob = cfg["datasets"][dset_name].get("glob", "*back*.fits")
    back_fname = glob.glob(os.path.join(maproot, back_glob))[0]
    back_name = os.path.splitext(os.path.basename(back_fname))[0]
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
"""
