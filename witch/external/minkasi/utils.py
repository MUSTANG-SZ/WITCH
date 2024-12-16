from copy import deepcopy

import jax
import jax.numpy as jnp
import minkasi
import numpy as np
from jitkasi import noise as jn
from jitkasi.tod import TOD, TODVec
from mpi4py import MPI


def to_minkasi(
    todvec: TODVec, copy_noise: bool = True, delete: bool = False
) -> minkasi.tods.TodVec:
    todvec_minkasi = minkasi.tods.TodVec()
    for tod in todvec:
        dat = {
            "dat_calib": np.ascontiguousarray(np.array(tod.data)),
            "dx": np.ascontiguousarray(np.array(tod.x)),
            "dy": np.ascontiguousarray(np.array(tod.y)),
        }
        dat.update(tod.meta)
        noise = None
        if (
            copy_noise
            and isinstance(tod.noise, jn.NoiseWrapper)
            and tod.noise.ext_inst.__module__ == "minkasi.mapmaking.noise"
        ):
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
    todvec_minkasi: minkasi.tods.TodVec,
    comm: MPI.Intracomm,
    copy_noise: bool = True,
    delete: bool = False,
) -> TODVec:
    tods = []
    for tod_minkasi in todvec_minkasi.tods:
        tod = from_minkasi_tod(tod_minkasi, copy_noise)
        if delete:
            del tod_minkasi
        tods += [tod]
    todvec = TODVec(tods, comm)
    return todvec


def from_minkasi_tod(tod_minkasi: minkasi.tods.Tod, copy_noise: bool = True) -> TOD:
    meta = deepcopy(tod_minkasi.info)
    data = jnp.array(meta["dat_calib"])
    del meta["dat_calib"]
    x = jnp.array(meta["dx"]).block_until_ready()
    del meta["dx"]
    y = jnp.array(meta["dy"]).block_until_ready()
    del meta["dy"]
    noise = jn.NoiseI()
    if copy_noise:
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
