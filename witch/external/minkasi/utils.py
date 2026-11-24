from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import minkasi
import numpy as np
from jitkasi import noise as jn
from jitkasi.tod import TOD, TODVec
from mpi4py import MPI


def update_minkasi(inst: Any, comm: MPI.Intracomm):
    """
    Point minkasi at a communicator. minkasi has no comm argument: each of its modules holds its own copy of
    `comm`, `myrank` and `nproc` (imported from minkasi.parallel), so this walks the minkasi modules reachable
    from `inst` and overwrites those attributes. Modules imported later pick up the patched minkasi.parallel.
    `have_mpi` is not changed.

    Parameters
    ----------
    inst : Any
        Module (or object) to patch, usually the `minkasi` package itself.
    comm : MPI.Intracomm
        Communicator minkasi should reduce over.
    """
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
        elif "minkasi" in (getattr(inst, "__package__", None) or ""):
            update_minkasi(getattr(inst, attr), comm)


@contextmanager
def minkasi_comm(comm: MPI.Intracomm) -> Iterator[None]:
    """
    Temporarily point minkasi at `comm` (see update_minkasi), restoring the previous communicator on exit, even
    on an exception. WITCH points minkasi at each dataset's sub-communicator when loading TODs, so anything that
    needs minkasi collectives across datasets (e.g. the joint needlet mapmaker) must run inside this, on every
    rank of `comm`.

    Parameters
    ----------
    comm : MPI.Intracomm
        Communicator for the duration of the block, e.g. `witch.fitter.comm`.

    Yields
    ------
    None
    """
    if not minkasi.have_mpi:
        yield
        return
    old = minkasi.parallel.comm
    update_minkasi(minkasi, comm)
    try:
        yield
    finally:
        update_minkasi(minkasi, old)


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
