"""
Dataset functions for NIKA2 TODs.

Only file discovery and loading are NIKA2 specific.
Everything downstream (noise, mapmaking, metadata) reuses the minkasi dataset functions.
"""

import glob
import os
from copy import deepcopy

import minkasi
import numpy as np
from jitkasi.tod import TODVec
from mpi4py import MPI

import witch.utils as wu
from witch.dataset import BeamConvAndPrefac, MetaData

from ..minkasi import mapmaking as mm
from ..minkasi.funcs import get_info as _minkasi_get_info
from ..minkasi.funcs import postfit, postproc, preproc, update_minkasi  # noqa: F401
from ..minkasi.utils import from_minkasi_tod
from .io import read_xdr


def _beam_pars(dset_name: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Get the multi-gaussian beam parameters from the config.

    Returns
    -------
    fwhms : np.ndarray
        FWHM of each gaussian in arcseconds.
    amps : np.ndarray
        Relative peak amplitude of each gaussian.
    extent : float
        The beam is evaluated out to this many times the largest FWHM.
    """
    beam_cfg = cfg["datasets"][dset_name]["beam"]
    fwhms = np.atleast_1d(np.array(eval(str(beam_cfg["fwhms"])), dtype=float))
    amps = np.atleast_1d(np.array(eval(str(beam_cfg["amps"])), dtype=float))
    extent = float(eval(str(beam_cfg.get("extent", 1.5))))
    return fwhms, amps, extent


def get_info(dset_name: str, cfg: dict, todvec: TODVec) -> dict:
    """
    Same as the minkasi `get_info` but the prefactor can reference
    `beam_omega` (solid angle in arcsec^2 of the configured beam, normalized to a peak of 1)
    and `colour_corr` (the dataset's `colour_corr`, default 1).
    """
    dset_cfg = cfg["datasets"][dset_name]
    fwhms, amps, _ = _beam_pars(dset_name, cfg)
    beam_omega = wu.gauss_solid_angle(fwhms, amps)
    colour_corr = float(eval(str(dset_cfg.get("colour_corr", 1.0))))
    prefactor = float(
        eval(
            str(dset_cfg["prefactor"]),
            {"wu": wu, "np": np},
            {"cfg": cfg, "beam_omega": beam_omega, "colour_corr": colour_corr},
        )
    )
    cfg_local = deepcopy(cfg)
    cfg_local["datasets"][dset_name]["prefactor"] = repr(prefactor)
    info = _minkasi_get_info(dset_name, cfg_local, todvec)
    info["beam_omega"] = beam_omega
    info["colour_corr"] = colour_corr
    return info


def make_metadata(dset_name: str, cfg: dict, info: dict) -> tuple[MetaData, ...]:
    dr = eval(str(cfg["coords"]["dr"]))
    r_map = eval(str(cfg["coords"]["r_map"]))
    fwhms, amps, extent = _beam_pars(dset_name, cfg)
    beam = wu.beam_multi_gauss(dr, fwhms, amps, extent, max_npix=2 * int(r_map / dr))

    return (
        BeamConvAndPrefac(
            beam, info["prefactor"], exclude=tuple(info["point_sources"])
        ),
    )


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
            cfg["datasets"][dset_name].get(
                "glob", cfg["paths"].get("glob", "stru_scan*_2mm_tot.xdr")
            ),
        )
    )
    bad_tod = list(cfg["paths"].get("cut", []))
    tod_names = minkasi.tods.io.cut_blacklist(tod_names, bad_tod)
    tod_names.sort()
    ntods = cfg["datasets"][dset_name].get("ntods", None)
    tod_names = tod_names[:ntods]

    return tod_names


def load_tods(dset_name: str, cfg: dict, fnames: list, comm: MPI.Intracomm) -> TODVec:
    if minkasi.have_mpi:
        update_minkasi(minkasi, comm)
        update_minkasi(mm.minkasi, comm)

    dset_cfg = cfg["datasets"][dset_name]
    max_nan_frac = float(dset_cfg.get("max_nan_frac", 0.1))
    tods = []
    for fname in fnames:
        if fname.endswith(".xdr"):
            dat = read_xdr(fname, max_nan_frac=max_nan_frac)
        elif fname.endswith("-imb.fits"):
            raise NotImplementedError(
                "Raw NIKA2 IMBFITS calibration is not yet implemented, use calibrated xdr files"
            )
        else:
            raise ValueError(f"Unrecognized NIKA2 file type: {fname}")
        if dat is None:
            continue
        minkasi.tods.processing.truncate_tod(dat)
        if dset_cfg.get("downsample", False):
            minkasi.tods.processing.downsample_tod(dat)
        tod = minkasi.tods.Tod(dat)

        tods += [from_minkasi_tod(deepcopy(tod))]
    todvec = TODVec(tods, comm)

    return todvec
