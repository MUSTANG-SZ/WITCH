"""
I/O for NIKA2 time ordered data.

Currently supports the calibrated (Jy/beam) IDL ``.xdr`` save files produced by
the NIKA2 pipeline (``stru_scan<date>s<scan>_<band>_<kind>.xdr``).
"""

import os
import re
from typing import Any, Optional

import numpy as np
from minkasi.tods.processing import gapfill_lin
from scipy.io import readsav

XDR_RE = re.compile(r"stru_scan(\d+)s(\d+)_(\w+?)_(\w+)\.xdr$")


def _unwrap_deg(ang: np.ndarray) -> np.ndarray:
    """
    Wrap angles (in degrees) into a continuous range centered on their median.
    The result lies within [med - 180, med + 180) with med in [0, 360).
    """
    med = np.nanmedian(ang) % 360.0
    return med + ((ang - med + 180.0) % 360.0) - 180.0


def read_xdr(fname: str, max_nan_frac: float = 0.1) -> Optional[dict[str, Any]]:
    """
    Read a calibrated NIKA2 TOD from an IDL ``.xdr`` save file.

    Parameters
    ----------
    fname : str
        Path to the ``stru_scan*.xdr`` file.
    max_nan_frac : float, default: 0.1
        KIDs with a larger fraction of NaN samples than this are cut.
        Remaining NaNs are linearly gapfilled.

    Returns
    -------
    dat : dict | None
        Dict that can be passed to ``minkasi.tods.Tod``.
        ``dat_calib`` is in Jy/beam, ``dx`` and ``dy`` are RA/Dec in radians.
        Returns ``None`` if no KIDs survive the cuts.
    """
    m = XDR_RE.search(os.path.basename(fname))
    if m is None:
        raise ValueError(f"{fname} does not look like a NIKA2 stru_scan xdr file")
    date, scan_num, band, kind = m.groups()

    raw = readsav(fname)
    if "scan" not in raw:
        raise ValueError(f"{fname} has no 'scan' structure")
    data = raw["scan"]
    names = data.dtype.names

    units = [u.decode() if isinstance(u, bytes) else str(u) for u in data["UNITS"][0]]
    if not any("Jy/beam" in u for u in units):
        raise ValueError(f"{fname} is not in Jy/beam, units are {units}")

    signal = np.asarray(data["SIGNAL_PLW"][0], dtype="float64")
    nan_frac = np.mean(~np.isfinite(signal), axis=-1)
    good_dets = np.where(nan_frac <= max_nan_frac)[0]
    if len(good_dets) == 0:
        print(f"No good detectors in tod {fname}")
        return None
    ndet = len(good_dets)
    ndata = signal.shape[1]

    dat: dict[str, Any] = {}
    dat["pixid"] = good_dets
    for key in ("ARRAY", "ACQBOX", "ACQSUBBAND"):
        if key in names:
            dat[key.lower()] = np.asarray(data[key][0])[good_dets]

    signal = signal[good_dets]
    signal[~np.isfinite(signal)] = np.nan
    signal, cuts = gapfill_lin(signal)
    dat["dat_calib"] = np.ascontiguousarray(signal, dtype="float64")
    dat["cuts"] = cuts

    ra = np.asarray(data["XRA_BOLOS_PLW"][0], dtype="float64")[good_dets]
    dec = np.asarray(data["YDEC_BOLOS_PLW"][0], dtype="float64")[good_dets]
    dat["dx"] = np.ascontiguousarray(np.deg2rad(_unwrap_deg(ra)))
    dat["dy"] = np.ascontiguousarray(np.deg2rad(dec))

    subscans = np.asarray(data["IND_SUBSCANS"][0])
    dat["subscans"] = subscans
    dat["turnarounds"] = np.where(np.diff(subscans))[0]

    dat["dt"] = float(np.median(np.diff(data["TIME_S"][0])))
    if "ELEV" in names:
        dat["elev"] = float(data["ELEV"][0])
    if "TAU_EFF" in names:
        dat["tau_eff"] = float(data["TAU_EFF"][0])

    dat["fname"] = fname
    dat["calinfo"] = {
        "calinfo": True,
        "date": date,
        "scan": scan_num,
        "band": band,
        "kind": kind,
        "units": "Jy/beam",
    }

    ff = 180 / np.pi
    print(
        f"ndet {ndet} ndata {ndata} on {fname} with lims "
        f"{dat['dx'].min() * ff:.4f} {dat['dx'].max() * ff:.4f} "
        f"{dat['dy'].min() * ff:.4f} {dat['dy'].max() * ff:.4f}"
    )

    return dat
