import glob
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
import minkasi
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from jax import Array
from jax.tree_util import register_pytree_node_class
from jitkasi.tod import TODVec
from minkasi.tools import presets_by_source as pbs
from mpi4py import MPI

import witch.utils as wu
from witch import grid
from witch.containers import MetaModel
from witch.dataset import DataSet, MetaData
from witch.fitter import print_once, process_tods

from ...objective import chisq_objective
from ...utils import beam_conv, beam_conv_vec
from . import mapmaking as mm
from .utils import from_minkasi, from_minkasi_noise, from_minkasi_tod, to_minkasi


def get_files(dset_name: str, cfg: dict) -> list:
    todroot = cfg["paths"].get("data", cfg["paths"]["tods"])
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


def update_minkasi(inst, comm):
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    for attr in dir(inst):
        if attr == "comm":
            inst.comm = comm
        elif attr == "nproc":
            inst.nproc = nproc
        elif attr == "myrank":
            inst.myrank = rank
        elif attr[0] == "_":
            continue
        elif hasattr(inst, "__package__") and "minkasi" in inst.__package__:
            update_minkasi(getattr(inst, attr), comm)
    return


def load_tods(dset_name: str, cfg: dict, fnames: list, comm: MPI.Intracomm) -> TODVec:
    # Update the minkasi comm here
    if minkasi.have_mpi:
        update_minkasi(minkasi, comm)
        update_minkasi(mm.minkasi, comm)
    # Now load
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
        "objective": chisq_objective,
        "lims": lims,
        "pixsize": pixsize,
        "skymap": skymap,
        "minkasi_noise_class": noise_class,
        "minkasi_noise_args": noise_args,
        "minkasi_noise_kwargs": noise_kwargs,
        "copy_noise": cfg["datasets"][dset_name]["copy_noise"],
        "xfer": cfg["datasets"][dset_name].get("xfer", ""),
        "prefactor": prefactor,
        "point_sources": cfg["datasets"][dset_name].get("point_sources", []),
    }


@register_pytree_node_class
@dataclass
class BeamConvAndPrefac(MetaData):
    beam: Array
    prefactor: Array

    def apply(self, model: Array) -> Array:
        return self.prefactor * beam_conv(model, self.beam)

    def apply_grad(self, model_grad: Array) -> Array:
        return self.prefactor * beam_conv_vec(model_grad, self.beam)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:

        children = (self.beam, self.prefactor)
        aux_data = (self.include, self.exclude)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        include, exclude = aux_data

        return cls(*children, include=include, exclude=exclude)


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
    _ = metamodel
    lims = dset.info["lims"]
    pixsize = dset.info["pixsize"]
    noise_class = dset.info["minkasi_noise_class"]
    noise_args = dset.info["minkasi_noise_args"]
    noise_kwargs = dset.info["minkasi_noise_kwargs"]
    outdir = dset.info["outdir"]
    copy_noise = dset.info["copy_noise"]
    if not cfg.get("noise_map", False):
        return

    print_once("Making noise map")
    noise_vec = to_minkasi(
        dset.datavec, copy_noise, False
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
        os.path.join(outdir, dset.name, "noise"),
        0,
        cfg["datasets"][dset.name]["dograd"],
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


def postproc(dset: DataSet, cfg: dict, metamodel: MetaModel):
    _ = metamodel
    skymap = dset.info["skymap"]
    noise_class = dset.info["minkasi_noise_class"]
    noise_args = dset.info["minkasi_noise_args"]
    noise_kwargs = dset.info["minkasi_noise_kwargs"]
    outdir = dset.info["outdir"]
    copy_noise = dset.info["copy_noise"]
    # Make signal maps
    if cfg.get("sig_map", cfg.get("map", True)):
        todvec_minkasi = to_minkasi(dset.datavec, copy_noise, cfg["mem_starved"])
        print_once("Making signal map")
        mapset = mm.make_maps(
            todvec_minkasi,
            skymap,
            noise_class,
            noise_args,
            noise_kwargs,
            os.path.join(outdir, dset.name, "signal"),
            cfg["datasets"][dset.name]["npass"],
            cfg["datasets"][dset.name]["dograd"],
            return_maps=True,
        )
        if mapset is None:
            raise ValueError("No mapset returned!")
        todvec = dset.datavec
        if cfg["mem_starved"]:
            todvec = from_minkasi(todvec_minkasi, dset.datavec.comm, True)
        elif copy_noise:
            for tod, tod_minkasi in zip(dset.datavec, todvec_minkasi.tods):
                tod.noise = from_minkasi_noise(tod_minkasi)
        else:
            for tod, tod_minkasi in zip(todvec, todvec_minkasi.tods):
                mat = 0 * tod_minkasi.info["dat_calib"].copy()
                for m in mapset.maps:
                    m.map2tod(tod_minkasi, mat)
                mat -= np.mean(mat, axis=-1)[..., None]  # Avoid overflow
                tod.recompute_noise(
                    data=tod.data - jnp.array(mat),
                    *dset.noise_args,
                    **dset.noise_kwargs,
                )

        if cfg.get("ntods_map", False):

            if minkasi.nproc > 1:
                print(
                    "Warning: running bunched TOD mapmaking with mpi is not reccomeneded. The bundles will be NPROC x NTODS_MAP and may lead to hanging code."
                )
            step = int(cfg.get("ntods_map", 1))
            todvec_copy = dset.datavec.copy(deep=True)
            for i in range(int(np.ceil(len(todvec_minkasi.tods) / step))):
                cur_tods = todvec_copy.tods[i * step : (i + 1) * step]
                cur_vec = TODVec(cur_tods, todvec_copy.comm)
                cur_vec = to_minkasi(cur_vec, copy_noise, cfg["mem_starved"])
                mapset = mm.make_maps(
                    cur_vec,
                    skymap,
                    noise_class,
                    noise_args,
                    noise_kwargs,
                    os.path.join(outdir, dset.name, "signal_{}".format(i)),
                    cfg["datasets"][dset.name]["npass"],
                    cfg["datasets"][dset.name]["dograd"],
                    return_maps=False,
                )
                del cur_tods
                del cur_vec
            print("DONE INDV TODS")
            del todvec_copy

        del todvec_minkasi

    else:
        print_once(
            "Not making signal map, this means that your starting noise may be more off"
        )


def postfit(dset: DataSet, cfg: dict, metamodel: MetaModel):
    lims = dset.info["lims"]
    pixsize = dset.info["pixsize"]
    skymap = dset.info["skymap"]
    minkasi_noise_class = dset.info["minkasi_noise_class"]
    minkasi_noise_args = dset.info["minkasi_noise_args"]
    minkasi_noise_kwargs = dset.info["minkasi_noise_kwargs"]
    outdir = dset.info["outdir"]
    # Residual map (or with noise from residual)
    if cfg.get("res_map", cfg.get("map", True)):
        # Compute residual and either set it to the data or use it for noise
        if metamodel is None:
            raise ValueError(
                "Somehow trying to make a residual map with no model defined!"
            )
        dataset_ind = metamodel.get_dataset_ind(dset.name)
        todvec_minkasi = to_minkasi(dset.datavec, False)
        for i, tod in enumerate(todvec_minkasi.tods):
            pred = metamodel.model_proj(dataset_ind, i)
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
            os.path.join(outdir, dset.name, name),
            cfg["datasets"][dset.name]["npass"],
            cfg["datasets"][dset.name]["dograd"],
        )

    # Make Model maps
    if cfg.get("model_map", False):
        print_once("Making model map")
        if metamodel is None:
            raise ValueError(
                "Somehow trying to make a model map with no model defined!"
            )
        model_todvec = dset.datavec.copy(deep=True)
        model_skymap = minkasi.maps.SkyMap(lims, pixsize)
        model_cfg = deepcopy(cfg)
        model_cfg["sim"] = True
        model_todvec = process_tods(cfg, dset, metamodel)
        model_todvec = to_minkasi(model_todvec, False)
        mm.make_maps(
            model_todvec,
            model_skymap,
            minkasi_noise_class,
            minkasi_noise_args,
            minkasi_noise_kwargs,
            os.path.join(outdir, dset.name, "model"),
            cfg["datasets"][dset.name]["npass"],
            cfg["datasets"][dset.name]["dograd"],
        )
        xyz = grid.make_grid_from_wcs(
            model_skymap.wcs,
            model_skymap.map.shape[0],
            model_skymap.map.shape[1],
            0.00116355,
            0.00000969,
        )
        tmpmodel = deepcopy(metamodel)
        for tm in tmpmodel.models:
            tm.xyz = xyz
        dataset_ind = tmpmodel.get_dataset_ind(dset.name)
        model_skymap.map = tmpmodel.model_grid(dataset_ind)
        if minkasi.myrank == 0:
            model_skymap.write(os.path.join(outdir, "model/truth.fits"))
