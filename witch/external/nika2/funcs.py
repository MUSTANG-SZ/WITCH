"""
Dataset functions for NIKA2 TODs.

Only file discovery and loading are NIKA2 specific.
Everything downstream (noise, mapmaking, metadata) reuses the minkasi dataset functions.
"""

import glob
import os
from copy import deepcopy

import minkasi
from jitkasi.tod import TODVec
from mpi4py import MPI

from ..minkasi import mapmaking as mm
from ..minkasi.funcs import (  # noqa: F401
    get_info,
    make_metadata,
    postfit,
    postproc,
    preproc,
    update_minkasi,
)
from ..minkasi.utils import from_minkasi_tod
from .io import read_xdr


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
