"""
Cut a (map, ivar) pair of ACT temperature maps down to a small, optionally rebinned, patch.

The ACT bridge maps hold (T, Q, U), only T is kept. Rebinning uses the ivar weighted mean for
the map and the sum for the ivar. The dense `NoiseCMBMap` needs a small number of pixels,
which is why the 0.5' maps are binned.

Usage
-----
    python scripts/make_act_cutout.py --srcfree --ra 44.612 --dec 13.301 --width 80 --bin 2 --out <dir>
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

BRIDGE = Path("/mnt/welch/ACT/bridge")


def cutout(
    mapf: Path, ivarf: Path, ra: float, dec: float, width: float, nbin: int
) -> tuple[fits.PrimaryHDU, fits.PrimaryHDU]:
    """
    Cut out and rebin the temperature map and its ivar.

    Parameters
    ----------
    mapf : Path
        Map file with shape `(3, ny, nx)`.
    ivarf : Path
        Ivar file with shape `(3, 3, ny, nx)`.
    ra, dec : float
        Centre of the patch in degrees.
    width : float
        Side length of the patch in arcmin.
    nbin : int
        Rebinning factor.

    Returns
    -------
    map_hdu, ivar_hdu : fits.PrimaryHDU
        The cut out map and ivar, with a WCS describing the rebinned pixels.
    """
    hdr = fits.getheader(mapf)
    wcs = WCS(hdr, naxis=2)
    pix = abs(hdr["CDELT1"]) * 60.0
    n = int(round(width / pix / nbin)) * nbin
    cx, cy = (float(c) for c in wcs.wcs_world2pix([[ra, dec]], 0)[0])
    x0, y0 = int(round(cx - n / 2)), int(round(cy - n / 2))
    dat = fits.getdata(mapf)[0, y0 : y0 + n, x0 : x0 + n].astype(float)
    ivar = fits.getdata(ivarf)[0, 0, y0 : y0 + n, x0 : x0 + n].astype(float)
    ivar = np.nan_to_num(ivar)
    dat = np.nan_to_num(dat)
    m = n // nbin
    ivar_b = ivar.reshape(m, nbin, m, nbin).sum(axis=(1, 3))
    num = (dat * ivar).reshape(m, nbin, m, nbin).sum(axis=(1, 3))
    dat_b = np.where(ivar_b > 0, num / np.where(ivar_b > 0, ivar_b, 1.0), 0.0)
    new = WCS(naxis=2)
    new.wcs.ctype = wcs.wcs.ctype
    new.wcs.cunit = wcs.wcs.cunit
    new.wcs.crval = wcs.wcs.crval
    new.wcs.cdelt = wcs.wcs.cdelt * nbin
    new.wcs.crpix = (np.asarray(wcs.wcs.crpix) - 1 - [x0, y0] + 0.5) / nbin + 0.5
    h = new.to_header()
    return fits.PrimaryHDU(dat_b, h), fits.PrimaryHDU(ivar_b, h)


def cutout_y(
    mapf: Path, ra: float, dec: float, width: float, nbin: int
) -> tuple[np.ndarray, fits.Header]:
    """
    Cut out and rebin (by averaging) a full sky 2D map such as an ACT ILC Compton-y map.

    Parameters
    ----------
    mapf : Path
        2D map file.
    ra, dec : float
        Centre of the patch in degrees.
    width : float
        Side length of the patch in arcmin.
    nbin : int
        Rebinning factor.

    Returns
    -------
    dat : np.ndarray
        The `(n, n)` rebinned patch, NaNs set to 0.
    header : fits.Header
        Header with a WCS describing the rebinned pixels.
    """
    with fits.open(mapf, memmap=True) as hdul:
        wcs = WCS(hdul[0].header, naxis=2)
        pix = abs(hdul[0].header["CDELT1"]) * 60.0
        n = int(round(width / pix / nbin)) * nbin
        cx, cy = (float(c) for c in wcs.wcs_world2pix([[ra, dec]], 0)[0])
        x0, y0 = int(round(cx - n / 2)), int(round(cy - n / 2))
        dat = np.nan_to_num(
            np.array(hdul[0].data[y0 : y0 + n, x0 : x0 + n], dtype=float)
        )
    m = n // nbin
    new = WCS(naxis=2)
    new.wcs.ctype = wcs.wcs.ctype
    new.wcs.cunit = wcs.wcs.cunit
    new.wcs.crval = wcs.wcs.crval
    new.wcs.cdelt = wcs.wcs.cdelt * nbin
    new.wcs.crpix = (np.asarray(wcs.wcs.crpix) - 1 - [x0, y0] + 0.5) / nbin + 0.5
    return dat.reshape(m, nbin, m, nbin).mean(axis=(1, 3)), new.to_header()


def main() -> None:
    """Parse arguments and write the cutouts."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--freq", default="f090")
    ap.add_argument(
        "--ymap",
        type=Path,
        help="Cut a 2D Compton-y map instead, written with a unit ivar as A399_401_y_{map,ivar}.fits",
    )
    ap.add_argument(
        "--srcfree", action="store_true", help="Use the point source subtracted map"
    )
    ap.add_argument("--ra", type=float, default=44.612)
    ap.add_argument("--dec", type=float, default=13.301)
    ap.add_argument("--width", type=float, default=80.0, help="arcmin")
    ap.add_argument("--bin", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--debug", action="store_true", help="Small 20' patch")
    args = ap.parse_args()
    if args.ymap is not None:
        dat, hdr = cutout_y(
            args.ymap, args.ra, args.dec, 20.0 if args.debug else args.width, args.bin
        )
        args.out.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(dat, hdr).writeto(
            args.out / "A399_401_y_map.fits", overwrite=True
        )
        fits.PrimaryHDU(np.ones_like(dat), hdr).writeto(
            args.out / "A399_401_y_ivar.fits", overwrite=True
        )
        print(dat.shape, "std", float(dat.std()), "min", float(dat.min()))
        return
    sfx = "_srcfree" if args.srcfree else ""
    width = 20.0 if args.debug else args.width
    mh, ih = cutout(
        BRIDGE / f"all_{args.freq}_map{sfx}.fits",
        BRIDGE / f"all_{args.freq}_ivar.fits",
        args.ra,
        args.dec,
        width,
        args.bin,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    mh.writeto(args.out / f"A399_401_all_{args.freq}_map.fits", overwrite=True)
    ih.writeto(args.out / f"A399_401_all_{args.freq}_ivar.fits", overwrite=True)
    print(mh.data.shape, "ivar>0 frac:", float((ih.data > 0).mean()))


if __name__ == "__main__":
    main()
