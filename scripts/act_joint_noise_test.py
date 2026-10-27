"""
Offline test of a joint multiband noise model for the A399-A401 ACT maps.

Fixes the per-band cluster and bridge templates (from the 3 band fits, see below) and solves the
linear generalised least squares problem for a cluster scale and a bridge amplitude with two
covariances: the current one (CMB independent in each band) and the joint one where the CMB is the
same sky in every band. Reports the bridge amplitude and its error for each.

Templates (uK per band, as written by witcher into `<outroot>/A399_401/act090+act150+act220`):
  clusters: `act_3band_nobridge` fit, bridge: `act_3band_bridgeonly` run with --nofit (P0 = 1e-3).

Usage
-----
    python scripts/act_joint_noise_test.py [--debug]
"""

import argparse
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.linalg import cho_factor, cho_solve

from act_split_noise import coadd_and_noise, load_band_splits
from witch.noise import (
    cmb_covariance_multiband,
    noise_covariance,
    ring_cross_spectra,
)

BANDS = {"f090": ("act090", 2.1), "f150": ("act150", 1.4), "f220": ("act220", 1.0)}
P0_TEMPLATE = 1e-3


def load_band(datroot: Path, root: Path, band: str, dset: str, debug: bool) -> tuple:
    """
    Load the data, ivar and the two templates for one band.

    Parameters
    ----------
    datroot : Path
        Directory with the cutouts.
    root : Path
        Directory with the 3 band fit outputs.
    band, dset : str
        Band name (e.g. `f090`) and the dataset name in the config.
    debug : bool
        If True crop to the central 16x16 pixels.

    Returns
    -------
    dat, ivar, clus, bridge : np.ndarray
        Flattened maps with the ivar weighted mean removed from `dat`.
    """
    dat = fits.getdata(datroot / f"A399_401_all_{band}_map.fits").astype(float)
    ivar = fits.getdata(datroot / f"A399_401_all_{band}_ivar.fits").astype(float)
    dat = dat - np.sum(dat * ivar) / np.sum(ivar)

    def tmpl(sub: str) -> np.ndarray:
        f = next(root.glob(f"*/{sub}/**/{dset}/*_truth.fits"))
        return fits.getdata(f).astype(float)

    maps = [dat, ivar, tmpl("act_3band_nobridge"), tmpl("act_3band_bridgeonly")]
    if debug:
        maps = [m[32:48, 32:48] for m in maps]
    return tuple(m.ravel() for m in maps)


def gls(
    ninv_apply, dat: np.ndarray, temps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalised least squares amplitudes and covariance.

    Parameters
    ----------
    ninv_apply : callable
        Applies the inverse noise covariance to a vector or a matrix of column vectors.
    dat : np.ndarray
        Stacked data.
    temps : np.ndarray
        Templates with shape `(ntemp, ndata)`.

    Returns
    -------
    amp, cov : np.ndarray
        Best fit amplitudes and their covariance.
    """
    nt = ninv_apply(temps.T)
    fisher = temps @ nt
    cov = np.linalg.inv(fisher)
    return cov @ (nt.T @ dat), cov


def main() -> None:
    """Build both covariances and compare the bridge amplitude."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outroot", type=Path, default=Path(os.environ["WITCH_OUTROOT"]))
    ap.add_argument(
        "--datroot",
        type=Path,
        default=Path(os.environ["WITCH_DATROOT"]) / "A399_401/act",
    )
    ap.add_argument(
        "--white-scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="Multipliers on the ivar white noise variance, one fit each",
    )
    ap.add_argument(
        "--noise",
        choices=["ivar", "splits"],
        default="ivar",
        help="White ivar noise on the all_* coadds, or noise estimated from the splits",
    )
    ap.add_argument("--mean-sigma", type=float, default=1e3)
    ap.add_argument("--debug", action="store_true", help="Central 16x16 pixels only")
    args = ap.parse_args()
    root = args.outroot / "A399_401/act090+act150+act220"
    bands = [
        load_band(args.datroot, root, b, d, args.debug) for b, (d, _) in BANDS.items()
    ]
    nz = None
    if args.noise == "splits":  # data and noise both from the split maps
        sp = [
            coadd_and_noise(*load_band_splits(b, 44.612, 13.301, 80.0, 2))
            for b in BANDS
        ]
        nz = [x[1] for x in sp]
        crop = np.s_[32:48, 32:48] if args.debug else np.s_[:, :]
        cos = [x[0][crop] for x in sp]
        bands = [
            (c.ravel() - c.mean(), *b[1:]) for c, b in zip(cos, bands, strict=True)
        ]
        if args.debug:
            nz = [z[..., 32:48, 32:48] for z in nz]
    npix = bands[0][0].size
    side = int(np.sqrt(npix))
    nb = len(bands)
    dat = np.concatenate([b[0] for b in bands])
    white = np.concatenate([1.0 / b[1] for b in bands])
    clus = np.concatenate([b[2] for b in bands])
    brid = np.concatenate([b[3] for b in bands]) / P0_TEMPLATE
    temps = np.stack([clus, brid])
    fwhms = [f for _, f in BANDS.values()]

    cmb = cmb_covariance_multiband((side, side), 1.0, fwhms)
    prior = np.zeros_like(cmb)
    for i in range(nb):  # per band offset priors
        sl = slice(i * npix, (i + 1) * npix)
        prior[sl, sl] = args.mean_sigma**2

    def diag_blocks(c: np.ndarray) -> np.ndarray:
        """Keep only the band auto blocks of a stacked covariance."""
        out = np.zeros_like(c)
        for i in range(nb):
            sl = slice(i * npix, (i + 1) * npix)
            out[sl, sl] = c[sl, sl]
        return out

    cases = []
    if args.noise == "ivar":
        for k in args.white_scale:
            cov = cmb + np.diag(k * white) + prior
            cases += [
                (f"ivar x{k:g}, independent", diag_blocks(cov)),
                (f"ivar x{k:g}, joint CMB", cov),
            ]
    else:
        n = nz[0].shape[-1]
        ix, iy = np.indices((side, side))
        off = (n - side) // 2
        power, _ = ring_cross_spectra(np.stack(nz))
        ncov = noise_covariance(power, ix.ravel() + off, iy.ravel() + off)
        cases = [
            ("split noise, independent", diag_blocks(cmb + ncov + prior)),
            ("split noise + joint CMB", cmb + ncov + prior),
            ("split noise, no cross noise, joint CMB", cmb + diag_blocks(ncov) + prior),
            ("split noise only (no CMB)", ncov + prior),
        ]
    for label, c in cases:
        fac = cho_factor(c)
        amp, acov = gls(lambda x, fac=fac: cho_solve(fac, x), dat, temps)
        err = np.sqrt(np.diag(acov))
        chi2 = float(dat @ cho_solve(fac, dat) - amp @ np.linalg.inv(acov) @ amp)
        print(
            f"{label:40s} cluster scale {amp[0]:.3f} +- {err[0]:.3f}, "
            f"bridge P0 {amp[1] * P0_TEMPLATE:.2e} +- {err[1] * P0_TEMPLATE:.2e} "
            f"({amp[1] / err[1]:.2f} sigma), chi2/N {chi2 / dat.size:.3f}"
        )


if __name__ == "__main__":
    main()
