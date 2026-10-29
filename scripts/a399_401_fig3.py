"""
Replicate Fig. 3 of Hincks et al. (2022, arXiv:2107.04611): the best-fit 'Ellip-beta+mesa' model.

Shows the model for both clusters and the bridge, convolved with the ACT beam, in Compton y,
over the 80' y-map cutout. As in the paper, the green contours are of the data, not the model,
at 3, 5, ..., 15 sigma (Fig. 1). The 1' cutout is noisier per pixel than the paper's quoted
3.0e-6, so by default the data are smoothed by 2' FWHM, the median residual is removed (in place of
the paper's fitted background plane) and sigma is the robust rms of the smoothed residual.
This reproduces the look of the paper's contours. The model is `<dir>/acty/A399_401_truth.fits`, written by the
postfit of an ACT (or joint ACT+M2) fit, so after an MCMC run it is the posterior estimate.

Usage
-----
    python scripts/a399_401_fig3.py --dir <fit output dir> --out reports/a399_401_fig3.png [--debug]
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

LEVELS = np.arange(3, 17, 2)  # Contour levels of Fig. 1, in sigma
SMOOTH = 2.0  # FWHM in arcmin of the smoothing applied to the data before contouring
VMAX = 10.0  # Colour scale of Figs. 1 and 3, in 1e-5 y


def load(path: Path) -> tuple[np.ndarray, WCS]:
    """
    Read a map and its celestial WCS.

    Parameters
    ----------
    path : Path
        FITS file with the map in the primary HDU.

    Returns
    -------
    data : np.ndarray
        The map.
    wcs : WCS
        Its WCS.
    """
    hdu = fits.open(path)[0]
    return np.asarray(hdu.data, dtype=float), WCS(hdu.header)


def main() -> None:
    """Plot the model with the data contours."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="Fit output directory")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("WITCH_DATROOT", "."))
        / "A399_401/act_y/A399_401_y_map.fits",
        help="y-map the contours are drawn from",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--sigma",
        type=float,
        help="Contour sigma, default the rms of the smoothed residual",
    )
    ap.add_argument(
        "--smooth",
        type=float,
        default=SMOOTH,
        help="Data smoothing FWHM in arcmin, 0 for none",
    )
    ap.add_argument("--debug", action="store_true", help="Low resolution output")
    args = ap.parse_args()

    model, wcs = load(args.dir / "acty/A399_401_truth.fits")
    data, _ = load(args.data)
    data = data - np.median(data - model)
    pix = abs(wcs.wcs.cdelt[1]) * 60
    if args.smooth:
        data = gaussian_filter(data, args.smooth / 2.355 / pix)
    resid = data - model
    rms = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    sigma = args.sigma or rms
    print(f"contour sigma {sigma:.2e}, robust rms of the smoothed residual {rms:.2e}")

    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(projection=wcs)
    im = ax.imshow(model * 1e5, origin="lower", cmap="magma_r", vmin=0.0, vmax=VMAX)
    ax.contour(data / sigma, levels=LEVELS, colors="green", linewidths=0.8)
    ax.coords[0].set_axislabel("RA (ICRS)")
    ax.coords[1].set_axislabel("Dec (ICRS)")
    ax.coords[0].set_major_formatter("hh:mm")
    ax.coords[1].set_major_formatter("dd:mm")
    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"Compton-$y$ ($\times 10^5$)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=60 if args.debug else 200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
