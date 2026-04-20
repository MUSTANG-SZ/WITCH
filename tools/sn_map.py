##########################################
# Small script for generating a SN map   #
# from a WITCH output signal/noise map.  #
# Caution that the map-space noise may   #
# not be well constrained for e.g. M2.   #
##########################################

import argparse as argp

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from pixell import enmap


def _make_parser() -> argp.ArgumentParser:
    parser = argp.ArgumentParser(
        description="Make a signal to noise map given an input signal and noise map"
    )
    parser.add_argument(
        "ipath",
        help="Path to input files. Should be the directory containing both the noise and signal directory.",
    )
    parser.add_argument(
        "--opath",
        "-op",
        default=None,
        help="Path to output map. If none, uses input path.",
    )
    parser.add_argument(
        "--smooth", "-s", type=int, default=3.0, help="Smoothing to apply in pixels"
    )

    return parser


def main():
    parser = _make_parser()
    args = parser.parse_args()

    if args.opath is None:
        opath = args.ipath
    else:
        opath = args.opath
    ipath = args.ipath

    kernel = Gaussian2DKernel(args.smooth)
    nmap = enmap.read_map(ipath + "noise/initial.fits")
    smap = enmap.read_map(ipath + "signal/niter_1.fits")
    wcs = smap.wcs
    nmap = np.abs(nmap)  # Gotta take the absolute value before smoothing
    nmap = convolve(nmap, kernel)
    smap = convolve(smap, kernel)
    snmap = (
        -1 * smap / nmap
    )  # Compton y is negative so this makes the S/n map easier to interperet
    snmap[np.isnan(snmap) | np.isinf(snmap)] = 0

    snmap = enmap.ndmap(snmap, wcs=wcs)

    enmap.write_map(opath + "snmap.fits", snmap)


if __name__ == "__main__":
    main()
