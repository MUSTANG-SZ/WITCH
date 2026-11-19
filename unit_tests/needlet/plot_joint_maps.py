"""
Check and plot the joint_map (needlet) outputs of a WITCH run: signal, residual, model (= signal - residual) and the
unfiltered model (truth). Prints the consistency checks and the model/truth agreement, and saves a map panel and the
azimuthally averaged model and truth with their ratio (the mapmaker's radial transfer for this model).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

KINDS = ["signal_final", "residual_final", "model_final", "model_truth"]


def radial_profile(
    m: np.ndarray, center: tuple[int, int], nbin: int, rmax: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Azimuthal average of a map around a pixel.

    Parameters
    ----------
    m : np.ndarray
        Map.
    center : tuple[int, int]
        Pixel to average around.
    nbin : int
        Number of radial bins.
    rmax : int
        Outer radius in pixels.

    Returns
    -------
    r : np.ndarray
        Bin centres in pixels.
    prof : np.ndarray
        Mean of each bin.
    """
    ii, jj = np.indices(m.shape)
    rr = np.hypot(ii - center[0], jj - center[1])
    edges = np.linspace(0, rmax, nbin + 1)
    idx = np.digitize(rr.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbin)
    prof = np.bincount(idx[ok], m.ravel()[ok], nbin) / np.maximum(
        np.bincount(idx[ok], minlength=nbin), 1
    )
    return 0.5 * (edges[1:] + edges[:-1]), prof


def main() -> None:
    """Load the maps, print the checks, save the figures."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outdir", help="The run's <outdir>/needlet directory")
    p.add_argument("--figdir", default=".", help="Where to save the figures")
    p.add_argument(
        "--half", type=int, default=60, help="Half size in pixels of the map panels"
    )
    p.add_argument(
        "--debug", action="store_true", help="Smaller panels and fewer radial bins"
    )
    args = p.parse_args()
    half, nbin = (30, 10) if args.debug else (args.half, 30)
    m = {k: fits.getdata(os.path.join(args.outdir, f"{k}.fits")) for k in KINDS}
    hdr = fits.getheader(os.path.join(args.outdir, "model_truth.fits"))
    pix = abs(hdr["CDELT1"]) * 3600
    hits = fits.getdata(os.path.join(args.outdir, "signal_hits.fits"))
    diff = np.abs(m["model_final"] - (m["signal_final"] - m["residual_final"])).max()
    print(f"max |model - (signal - residual)|: {diff:.2e} (0 by construction)")
    c = np.unravel_index(np.argmax(np.abs(m["model_truth"])), m["model_truth"].shape)
    for k, v in m.items():
        print(
            f"{k:15s} min {v.min():+.3e} max {v.max():+.3e} at truth peak {v[c]:+.3e}"
        )
    inside = hits > np.percentile(hits[hits > 0], 50)
    r = np.corrcoef(m["model_final"][inside], m["model_truth"][inside])[0, 1]
    print(
        f"corr(model, truth) where hits > median: {r:.3f}; peak ratio model/truth {m['model_final'][c] / m['model_truth'][c]:.3f}"
    )
    sl = tuple(slice(max(ci - half, 0), ci + half) for ci in c)
    lim = np.abs(m["model_truth"][sl]).max() * 1e6
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (k, v) in zip(axs, m.items()):
        im = ax.imshow(
            v[sl].T * 1e6, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim
        )
        ax.set_title(k)
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=axs, label="uK_RJ", shrink=0.8)
    fig.savefig(
        os.path.join(args.figdir, "joint_maps.png"), dpi=110, bbox_inches="tight"
    )
    rpix, pm = radial_profile(m["model_final"], c, nbin, half)
    _, pt = radial_profile(m["model_truth"], c, nbin, half)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    a0.plot(rpix * pix, pt * 1e6, label="truth (unfiltered)")
    a0.plot(rpix * pix, pm * 1e6, label="model map (signal - residual)")
    a0.set_ylabel("uK_RJ"), a0.legend()
    a1.plot(rpix * pix, pm / pt)
    a1.axhline(1, color="k", lw=0.5)
    a1.set_ylabel("model / truth"), a1.set_xlabel("radius [arcsec]")
    fig.savefig(
        os.path.join(args.figdir, "joint_model_transfer.png"),
        dpi=110,
        bbox_inches="tight",
    )
    print(
        "radial model/truth:",
        " ".join(f'{x * pix:.0f}":{y:.2f}' for x, y in zip(rpix[::3], (pm / pt)[::3])),
    )


if __name__ == "__main__":
    main()
