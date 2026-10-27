"""
Empirical noise power spectrum of an ACT Compton-y map from control fields.

Following Hincks et al. 2022 the covariance is measured on nearby fields rather than modelled.
Fields on a grid around the target (same declination band, so similar depth) are cut out, the
ones that are empty or have much more variance than the median (bright clusters) are dropped,
and the ring averaged power of the rest is written for `witch.noise.NoiseSpectrum`.

Usage
-----
    python scripts/act_y_noise.py --ymap <y map> --out <power.npy> [--debug]
"""

import argparse
from pathlib import Path

import numpy as np
from make_act_cutout import cutout_y

from witch.noise import ring_cross_spectra


def control_centres(ra: float, dec: float, nra: int = 5, step: float = 1.6) -> list:
    """
    Centres of control fields around the target, skipping the target itself.

    Parameters
    ----------
    ra, dec : float
        Target centre in degrees.
    nra : int, default: 5
        Number of RA steps either side.
    step : float, default: 1.6
        Spacing in degrees, larger than the 80' field.

    Returns
    -------
    centres : list[tuple[float, float]]
        `(ra, dec)` of each field in degrees.
    """
    cs = [(ra + i * step, dec) for i in range(-nra, nra + 1) if i]
    cs += [(ra + i * step, dec + d) for d in (-1.5, 1.5) for i in range(-3, 4)]
    return cs


def local_centres(ra: float, dec: float, width: float = 80.0) -> list:
    """
    Centres of the non-overlapping fields tiling a small map, excluding the central ones.

    Parameters
    ----------
    ra, dec : float
        Target centre in degrees (CAR pixels are square in `(ra, dec)`).
    width : float, default: 80.0
        Field side in arcmin.

    Returns
    -------
    centres : list[tuple[float, float]]
        `(ra, dec)` of the 12 outer fields of a 4x4 grid.
    """
    off = np.array([-1.5, -0.5, 0.5, 1.5]) * width / 60.0
    return [
        (ra + a, dec + b)
        for a in off
        for b in off
        if max(abs(a), abs(b)) > 0.8 * width / 60.0
    ]


def main() -> None:
    """Measure and save the noise power."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ymap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ra", type=float, default=44.612)
    ap.add_argument("--dec", type=float, default=13.301)
    ap.add_argument("--width", type=float, default=80.0, help="arcmin")
    ap.add_argument("--bin", type=int, default=2)
    ap.add_argument("--max-std-ratio", type=float, default=1.5)
    ap.add_argument(
        "--local",
        action="store_true",
        help="Tile the outer part of a small (6.4 deg) map instead of using a wide area",
    )
    ap.add_argument("--debug", action="store_true", help="Few small fields")
    args = ap.parse_args()
    width = 20.0 if args.debug else args.width
    cs = (
        local_centres(args.ra, args.dec, width)
        if args.local
        else control_centres(args.ra, args.dec, 1 if args.debug else 5)
    )
    fields = [cutout_y(args.ymap, r, d, width, args.bin)[0] for r, d in cs]
    keep = [f for f in fields if np.mean(f == 0) < 0.01]
    stds = np.array([f.std() for f in keep])
    good = [
        f - f.mean()
        for f, s in zip(keep, stds, strict=True)
        if s < args.max_std_ratio * np.median(stds)
    ]
    target = cutout_y(args.ymap, args.ra, args.dec, width, args.bin)[0]
    print(f"{len(cs)} fields, {len(keep)} non-empty, {len(good)} used")
    print(
        f"control std median {np.median(stds):.3e} range {stds.min():.3e}-{stds.max():.3e}, target std {target.std():.3e}"
    )
    power, _ = ring_cross_spectra(np.stack(good)[None])
    np.save(args.out, power[0, 0])
    print("wrote", args.out, power[0, 0].shape)


if __name__ == "__main__":
    main()
