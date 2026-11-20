import os

import minkasi
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from minkasi.needlet.act import ACT_OSAMP
from minkasi.needlet.joint import JointMapper, options
from minkasi.tods.processing import fit_cm_plus_poly
from mpi4py import MPI
from numpy.typing import NDArray
from pixell import enmap

from witch.containers import MetaModel
from witch.dataset import DataSet
from witch.external.minkasi.funcs import model_on_skymap
from witch.external.minkasi.utils import minkasi_comm, to_minkasi
from witch.fitter import print_once

VALID_METHODS = ["needlet"]


def reduce_ds(datasets: tuple[DataSet, ...], targets: list[str]) -> tuple[DataSet, ...]:
    """
    Reduce datasets to only those that are relevant for the joint mapmaking process. Currently only supports the needlet method.

    Parameters
    ----------
    datasets : tuple[DataSet]
        A tuple of DataSet objects, each representing a different dataset to be combined in the joint mapmaking.
    targets : list[str]
        A list of dataset names that are relevant for the joint mapmaking process.

    Returns
    -------
    tuple[DataSet]
        A tuple of DataSet objects that are relevant for the joint mapmaking process.
    """
    return tuple(ds for ds in datasets if ds.name in targets)


def _project(
    ds: DataSet, i: int, data: NDArray | None, metamodel: MetaModel, kind: str
) -> NDArray:
    """
    What a map kind maps from item i (TOD or map) of a dataset: the data ("signal"), the data minus the model
    ("residual", the data itself if there is no model) or the model ("model").

    Parameters
    ----------
    ds : DataSet
        The dataset.
    i : int
        Index of the TOD or map in ds.datavec.
    data : NDArray | None
        Its data (unused for "model").
    metamodel : MetaModel
        The fitted metamodel.
    kind : str
        "signal", "residual" or "model".

    Returns
    -------
    NDArray
        The array to map, in the layout of the data.
    """
    if kind == "signal" or (kind == "residual" and len(metamodel.models) == 0):
        return data
    pred = np.asarray(metamodel.model_proj(metamodel.get_dataset_ind(ds.name), i))
    return pred if kind == "model" else data - pred


def _tod_input(
    ds: DataSet, i: int, data: NDArray, metamodel: MetaModel, kind: str, detrend: bool
) -> NDArray:
    """
    What a map kind maps from TOD i of a dataset (see _project), with the per-detector polynomial drifts removed by
    minkasi's fit_cm_plus_poly if `detrend`, as minkasi.tods.io.read_tods does for the standalone mapper. WITCH's
    load_tods leaves the drifts in (the WITCH noise model handles them), but the minkasi noise model the joint mapper
    uses expects them gone: left in, they are 3-10x the white noise per sample and land in the large scales and at
    the edge of the scan. The fit is done on what is mapped, so the residual is P(data - model), filtered like the
    signal P(data).

    Parameters
    ----------
    ds : DataSet
        The dataset.
    i : int
        Index of the TOD in ds.datavec.
    data : NDArray
        Its data, (ndet, nsamp).
    metamodel : MetaModel
        The fitted metamodel.
    kind : str
        "signal", "residual" or "model".
    detrend : bool
        Remove the polynomial drifts.

    Returns
    -------
    NDArray
        C-contiguous float64 array to map, (ndet, nsamp).
    """
    dat = np.array(_project(ds, i, data, metamodel, kind), dtype=float, order="C")
    return fit_cm_plus_poly(dat) if detrend else dat


def m2_todvec(
    datasets: tuple[DataSet, ...], metamodel: MetaModel, kind: str, detrend: bool
) -> minkasi.tods.TodVec:
    """
    The TOD datasets of this rank as one minkasi TodVec holding what `kind` maps (see _tod_input). Ranks without TOD
    datasets get an empty TodVec, which still takes part in the collectives.

    Parameters
    ----------
    datasets : tuple[DataSet, ...]
        The datasets of this rank.
    metamodel : MetaModel
        The fitted metamodel.
    kind : str
        "signal", "residual" or "model".
    detrend : bool
        Remove the per-detector polynomial drifts (fit_cm_plus_poly).

    Returns
    -------
    minkasi.tods.TodVec
        TODs of this rank. Noise is not copied: JointMapper sets its own noise model.
    """
    todvec = minkasi.tods.TodVec()
    for ds in datasets:
        if ds.mode != "tod":
            continue
        for i, tod in enumerate(to_minkasi(ds.datavec, copy_noise=False).tods):
            tod.info["dat_calib"] = _tod_input(
                ds, i, tod.info["dat_calib"], metamodel, kind, detrend
            )
            todvec.add_tod(tod)
    return todvec


def m2_data(
    datasets: tuple[DataSet, ...], metamodel: MetaModel, kind: str, detrend: bool
) -> list[NDArray]:
    """
    What `kind` maps (see _tod_input) for every TOD of this rank, in the order of m2_todvec (for JointMapper.remap).

    Parameters
    ----------
    datasets : tuple[DataSet, ...]
        The datasets of this rank.
    metamodel : MetaModel
        The fitted metamodel.
    kind : str
        "signal", "residual" or "model".
    detrend : bool
        Remove the per-detector polynomial drifts (fit_cm_plus_poly).

    Returns
    -------
    list[NDArray]
        One array per TOD of this rank (empty on ranks without TODs).
    """
    return [
        _tod_input(ds, i, np.asarray(tod.data), metamodel, kind, detrend)
        for ds in datasets
        if ds.mode == "tod"
        for i, tod in enumerate(ds.datavec)
    ]


def truth_map(
    datasets: tuple[DataSet, ...],
    cfg: dict,
    metamodel: MetaModel,
    comm: MPI.Intracomm,
    targets: list[str],
    skymap: minkasi.maps.SkyMap,
) -> NDArray | None:
    """
    The unfiltered model on the joint map's pixels, as the first TOD dataset of `targets` sees it (its beam and
    prefactor), computed on the first rank holding that dataset and broadcast. Collective.

    Parameters
    ----------
    datasets : tuple[DataSet, ...]
        The datasets of this rank.
    cfg : dict
        WITCH config.
    metamodel : MetaModel
        The fitted metamodel.
    comm : MPI.Intracomm
        Communicator spanning all datasets.
    targets : list[str]
        The joint map's datasets.
    skymap : minkasi.maps.SkyMap
        The joint map's pixelization.

    Returns
    -------
    NDArray | None
        The model on the skymap pixels on every rank (evaluated on the fit grid's pixel size coords.dr, at which the
        beam is sampled, and interpolated if the joint map's pixels differ); None (with a warning) if no target is a
        TOD dataset.
    """
    tod_names = [
        name for name in targets if "load_tods" in cfg["datasets"][name]["funcs"]
    ]
    if not tod_names:
        print_once(f"Not making the truth map: no TOD dataset in {targets}")
        return None
    dr = float(eval(str(cfg["coords"]["dr"])))
    held = [ds for ds in datasets if ds.name == tod_names[0]]
    root = comm.allgather(bool(held)).index(True)
    truth = None
    if held:
        truth = model_on_skymap(
            metamodel, metamodel.get_dataset_ind(tod_names[0]), skymap, dr
        )
    return comm.bcast(truth, root=root)


def _write(skymap: minkasi.maps.SkyMap, data: NDArray, path: str, comm: MPI.Intracomm):
    """
    Write a map on the joint map's pixelization from rank 0.

    Parameters
    ----------
    skymap : minkasi.maps.SkyMap
        The joint map's pixelization; its map is set to `data`.
    data : NDArray
        The map.
    path : str
        FITS file to write.
    comm : MPI.Intracomm
        Communicator spanning all datasets.
    """
    skymap.map = data
    if comm.Get_rank() == 0:
        skymap.write(path)


def act_pairs(
    datasets: tuple[DataSet, ...], metamodel: MetaModel, comm: MPI.Intracomm, kind: str
) -> list[tuple[enmap.ndmap, enmap.ndmap]]:
    """
    (map, ivar) of every map dataset on every rank, holding what `kind` maps (see _project), gathered to all ranks
    (the ACT term is computed on each).

    Parameters
    ----------
    datasets : tuple[DataSet, ...]
        The datasets of this rank.
    metamodel : MetaModel
        The fitted metamodel.
    comm : MPI.Intracomm
        Communicator spanning all datasets.
    kind : str
        "signal", "residual" or "model".

    Returns
    -------
    list[tuple[enmap.ndmap, enmap.ndmap]]
        (map, ivar) per map, in pixell (dec, ra) layout and the dataset units (uK_CMB for act_map).
    """
    local = []
    for ds in datasets:
        if ds.mode != "map":
            continue
        for i, imap in enumerate(ds.datavec):
            dat = _project(ds, i, np.asarray(imap.data), metamodel, kind)
            # WITCH maps are (x, y) = FITS transposed; send plain arrays and the header string
            local.append(
                (dat.T, np.asarray(imap.ivar).T, imap.wcs.to_header().tostring())
            )
    pairs = []
    for dat, ivar, hdr in (p for part in comm.allgather(local) for p in part):
        wcs = WCS(fits.Header.fromstring(hdr)).celestial
        pairs.append((enmap.ndmap(dat, wcs), enmap.ndmap(ivar, wcs)))
    return pairs


def needlet_pixsize(pairs: list[tuple[enmap.ndmap, enmap.ndmap]]) -> float:
    """
    The needlet map pixel the ACT maps force. minkasi's ACT term models every ACT pixel as ACT_OSAMP x ACT_OSAMP
    needlet pixels (and refuses anything else), so the needlet pixel is the ACT pixel / ACT_OSAMP: 3" for 30" ACT
    maps. It only sets the joint map's resolution, not the fit.

    Parameters
    ----------
    pairs : list[tuple[enmap.ndmap, enmap.ndmap]]
        (map, ivar) of the ACT maps (act_pairs).

    Returns
    -------
    float
        Needlet pixel size in arcsec.

    Raises
    ------
    ValueError
        If the ACT maps do not all have the same square pixels.
    """
    sizes = (
        np.array(
            [
                np.abs(np.diag(ivar.wcs.celestial.pixel_scale_matrix))
                for _, ivar in pairs
            ]
        )
        * 3600
    )
    if not np.allclose(sizes, sizes[0, 0], rtol=1e-3):
        raise ValueError(
            f"The joint map needs ACT maps with the same square pixels, got {sizes.round(4).tolist()} arcsec"
        )
    return float(sizes[0, 0]) / ACT_OSAMP


def joint_post_fit(
    datasets: tuple[DataSet, ...],
    cfg: dict,
    metamodel: MetaModel,
    comm: MPI.Comm,
    outdir: str,
) -> dict[str, NDArray]:
    """
    Post-fit function for multi-scale mapmaking, configured by cfg["joint_map"]. Currently only supports the needlet
    method. Called by witch.fitter when the config has a joint_map block.

    Collective: must be called on every rank of `comm`, whatever datasets the rank holds, and before the per-dataset
    postfit (act_map's postfit overwrites its map data with the residual and the model). The config checks below
    only read cfg, so all ranks raise together rather than some hanging in a collective.

    Parameters
    ----------
    datasets : tuple[DataSet, ...]
        This rank's DataSet objects; those named in joint_map.datasets are combined in the joint mapmaking.
    cfg : dict
        WITCH config dictionary.
    metamodel : MetaModel
        WITCH metamodel object.
    comm : MPI.Comm
        Communicator spanning all datasets (witch.fitter.comm).
    outdir : str
        Output directory; maps go to <outdir>/<method>/<kind>_*.fits.

    joint_map.maps lists the kinds to make: "signal" (the data), "residual" (data minus model, the default) and
    "model". All kinds share one mapper (the residual goes through the signal's with JointMapper.remap), and the
    model map is signal - residual: the model as it sits in the signal map. With "model", the unfiltered model on
    the same pixels is also written (model_truth.fits, returned as "truth"; evaluated at coords.dr and interpolated
    onto the joint map's pixels if they differ). joint_map.detrend (default True) removes the per-detector polynomial drifts from what is mapped from
    TODs, as the standalone mapper does (see _tod_input). With ACT maps the needlet pixel is set from them
    (needlet_pixsize, overriding joint_map.options.pixsize): it only sets the joint map's resolution.

    Returns
    -------
    dict[str, NDArray]
        Final real space map per map kind, on every rank.

    Raises
    ------
    KeyError
        If joint_map.method or joint_map.datasets is not set, or datasets names an unknown dataset.
    ValueError
        If joint_map.maps has an unknown kind, or asks for "model" without a model.
    NotImplementedError
        If the specified method is not "needlet".
    """
    jcfg = cfg["joint_map"]
    if "method" not in jcfg:
        raise KeyError(f"joint_map.method must be set, one of {VALID_METHODS}")
    method = jcfg["method"]
    if method not in VALID_METHODS:
        raise NotImplementedError(
            f"Method '{method}' is not implemented, only {VALID_METHODS} are supported."
        )
    targets = jcfg.get("datasets")
    if not targets:
        raise KeyError("joint_map.datasets must be a non-empty list of dataset names")
    if unknown := set(targets) - set(cfg["datasets"]):
        raise KeyError(
            f"joint_map.datasets has unknown datasets {sorted(unknown)}, known: {list(cfg['datasets'])}"
        )
    kinds = jcfg.get("maps", ["residual"])
    if bad := set(kinds) - {"residual", "signal", "model"}:
        raise ValueError(
            f"joint_map.maps has unknown kinds {sorted(bad)}, use residual, signal and/or model"
        )
    if "model" in kinds and len(metamodel.models) == 0:
        raise ValueError("joint_map.maps has model but there is no model")

    detrend = jcfg.get("detrend", True)  # as minkasi.tods.io.read_tods, see _tod_input

    datasets = reduce_ds(datasets, targets=targets)
    out = {}
    odir = os.path.join(outdir, method)
    if comm.Get_rank() == 0:
        os.makedirs(odir, exist_ok=True)
    if method == "needlet":
        # One mapper for all kinds: it maps the signal (or the residual if that is all that is asked for), and the
        # residual goes through the same mapper. model = signal - residual is then the model as it sits in the
        # signal map, whatever the (data dependent) filtering of the finite PCG/CG.
        first = "residual" if set(kinds) == {"residual"} else "signal"
        pairs = act_pairs(datasets, metamodel, comm, first)
        over = dict(jcfg.get("options", {}))
        if pairs:  # the ACT pixels set the needlet pixel (see needlet_pixsize)
            pixsize = needlet_pixsize(pairs)
            if "pixsize" in over and not np.isclose(over["pixsize"], pixsize):
                print_once(
                    f'joint_map.options.pixsize {over["pixsize"]:g}" replaced by {pixsize:g}" (ACT pixel / {ACT_OSAMP})'
                )
            over["pixsize"] = pixsize
        with minkasi_comm(comm):
            opts = options(
                {
                    "name": f"{cfg['name']}_{first}",  # the response caches depend on the noise, so on the data
                    "outroot": os.path.join(odir, first),
                    **over,
                }
            )
            mapper = JointMapper(
                opts,
                todvec=m2_todvec(datasets, metamodel, first, detrend),
                act_maps=pairs,
            )
            out[first] = mapper.run()
            if not mapper.solved:  # dry_run / symcheck
                return {}
            if first == "signal" and {"residual", "model"} & set(kinds):
                out["residual"] = mapper.remap(
                    m2_data(datasets, metamodel, "residual", detrend),
                    act_pairs(datasets, metamodel, comm, "residual"),
                    os.path.join(odir, "residual"),
                )
            if "model" in kinds:
                skymap = mapper.wmap.real_map
                out["model"] = out["signal"] - out["residual"]
                _write(
                    skymap, out["model"], os.path.join(odir, "model_final.fits"), comm
                )
                truth = truth_map(datasets, cfg, metamodel, comm, targets, skymap)
                if truth is not None:
                    out["truth"] = truth
                    _write(skymap, truth, os.path.join(odir, "model_truth.fits"), comm)
        out = {kind: m for kind, m in out.items() if kind in kinds or kind == "truth"}

    else:
        raise NotImplementedError(
            f"Method '{method}' is not implemented. Not sure how you got here TBH."
        )

    return out
