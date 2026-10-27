"""
Empirical multiband ACT noise covariance from the 4-way split maps.

Each band is rebuilt from the per-array split maps (ivar weighted over arrays). Sign-flipped
combinations of the splits that sum to zero have no sky signal and give independent noise
realisations of the coadd. Their ring averaged cross power spectra give a stationary
`(nband*npix, nband*npix)` noise covariance that keeps cross band (e.g. atmospheric) correlations.

Usage
-----
    python scripts/act_split_noise.py [--debug]   # prints the noise level per band and ring
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from witch.noise import ring_cross_spectra

BRIDGE = Path("/mnt/welch/ACT/bridge")
ARRAYS = {"f090": ["pa5", "pa6"], "f150": ["pa4", "pa5", "pa6"], "f220": ["pa4"]}
SIGNS = np.array([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]], dtype=float)


def _bin(arr: np.ndarray, nbin: int) -> np.ndarray:
    """Sum `nbin` x `nbin` blocks."""
    n = arr.shape[0] // nbin
    return arr.reshape(n, nbin, n, nbin).sum(axis=(1, 3))


def load_band_splits(
    band: str, ra: float, dec: float, width: float, nbin: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one band's split maps on the same patch `make_act_cutout.py` uses.

    Parameters
    ----------
    band : str
        Band name, e.g. `f090`.
    ra, dec : float
        Patch centre in degrees.
    width : float
        Patch side length in arcmin.
    nbin : int
        Rebinning factor.

    Returns
    -------
    maps, ivar : np.ndarray
        Arrays of shape `(4, n, n)`: the array combined map and ivar of each split.
    """
    maps = np.zeros((4, 0, 0))
    for k in range(4):
        num = den = 0.0
        for arr in ARRAYS[band]:
            stem = BRIDGE / f"dr6v2_{arr}_{band}_4way_set{k}"
            hdr = fits.getheader(f"{stem}_map_srcfree.fits")
            wcs = WCS(hdr, naxis=2)
            n = int(round(width / (abs(hdr["CDELT1"]) * 60) / nbin)) * nbin
            cx, cy = (float(c) for c in wcs.wcs_world2pix([[ra, dec]], 0)[0])
            x0, y0 = int(round(cx - n / 2)), int(round(cy - n / 2))
            sl = np.s_[y0 : y0 + n, x0 : x0 + n]
            m = np.nan_to_num(
                fits.getdata(f"{stem}_map_srcfree.fits")[0][sl].astype(float)
            )
            iv = np.nan_to_num(fits.getdata(f"{stem}_ivar.fits")[sl].astype(float))
            num = num + _bin(m * iv, nbin)
            den = den + _bin(iv, nbin)
        if k == 0:
            maps, ivar = np.zeros((4, *den.shape)), np.zeros((4, *den.shape))
        maps[k] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        ivar[k] = den
    return maps, ivar


def coadd_and_noise(
    maps: np.ndarray, ivar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coadd the splits and make signal free noise realisations of the coadd.

    Parameters
    ----------
    maps, ivar : np.ndarray
        Split maps and ivar with shape `(4, n, n)`.

    Returns
    -------
    coadd : np.ndarray
        The ivar weighted coadd, shape `(n, n)`.
    noise : np.ndarray
        Three noise realisations with the noise level of the coadd, shape `(3, n, n)`.
    """
    tot = ivar.sum(axis=0)
    w = ivar / np.where(tot > 0, tot, 1.0)
    coadd = (w * maps).sum(axis=0)
    resid = w * (maps - coadd)  # no common signal, sums to zero over the splits
    return coadd, np.einsum("jk,kxy->jxy", SIGNS, resid)


def main() -> None:
    """Print the empirical noise level per band next to the ivar one."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--debug", action="store_true", help="20' patch")
    args = ap.parse_args()
    width = 20.0 if args.debug else 80.0
    noise, ivar_rms = [], []
    for band in ARRAYS:
        maps, ivar = load_band_splits(band, 44.612, 13.301, width, 2)
        _, nz = coadd_and_noise(maps, ivar)
        noise.append(nz)
        ivar_rms.append(np.median(1 / np.sqrt(ivar.sum(axis=0))))
        print(band, "split noise std", nz.std(), "ivar rms", ivar_rms[-1])
    power, ring = ring_cross_spectra(np.stack(noise))
    print(
        "ring 1, 5, 20 auto power:",
        [power[:, :, ring == r][0, 0, 0] for r in (1, 3, min(20, ring.max()))],
    )


if __name__ == "__main__":
    main()
