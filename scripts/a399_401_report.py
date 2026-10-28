"""
Write reports/a399_401_act.html: figures and chi-squared comparison for the ACT-only A399-A401 fits.

Reads the model and residual maps of the free-bridge run plus the chi-squared of the three runs.

Usage
-----
    python scripts/a399_401_report.py [--outroot $WITCH_OUTROOT]
"""

import argparse
import base64
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter

RUNS = {  # subdir: (label, chi2, n extra pars, note) from the logs in the fit outputs
    "act_only_nobridge": ("Two gNFW, no bridge", 5050.23, 0, "m500 399/401 free"),
    "act_only": (
        "Two gNFW + free bridge",
        5048.08,
        5,
        "r_c at lower bound, not converged",
    ),
    "act_only_ampfit": (
        "Two gNFW + bridge, amp only",
        5049.58,
        1,
        "P0 = 0.00085 +- 0.00106",
    ),
}


RUNS3 = [  # f090+f150+f220 joint fit, independent noise model per band
    ("Two gNFW, no bridge", 16323.67, 0, "m500 3.8e14 / 7.9e14"),
    ("Two gNFW + free bridge", 16320.35, 5, "r_c at lower bound, not converged"),
    ("Two gNFW + bridge, amp only", 16322.37, 1, "P0 = 0.00096 +- 0.00085"),
]


SPLITS = [  # scripts/act_joint_noise_test.py --noise splits, linear GLS with fixed templates
    ("Independent bands, split noise", "1.006 +- 0.064", "0.9 +- 0.9", "1.1", 0.786),
    (
        "Joint CMB + split noise (with cross-band noise)",
        "0.865 +- 0.055",
        "1.5 +- 0.8",
        "1.9",
        1.093,
    ),
    (
        "Joint CMB + split noise, no cross-band noise",
        "0.882 +- 0.057",
        "1.4 +- 0.8",
        "1.7",
        1.051,
    ),
    ("Split noise only, no CMB", "1.033 +- 0.012", "1.0 +- 0.2", "4.7", 6.616),
    (
        "Joint CMB + ivar white noise (previous)",
        "-0.318 +- 0.016",
        "9.2 +- 0.3",
        "32",
        6.099,
    ),
]


YFIT = [  # clusters only fit to ilc_SZ_mask80_yy, 6400 pixels, configs/A399_401/A399_401_ymap_*.yaml
    ("alpha fixed (A10, 1.051)", "6.04 +- 0.21", "7.64 +- 0.20", "-", 6663.86),
    (
        "alpha free",
        "6.30 +- 1.24",
        "9.12 +- 1.47",
        "1.03 +- 0.10 / 0.96 +- 0.07",
        6662.37,
    ),
]
YBEAM = {78: 6666.36, 84: 6665.15, 96: 6663.86, 108: 6663.94, 120: 6665.34}


def table(runs) -> str:
    """HTML rows of (label, chi2, d chi2 vs the first run, extra pars, note)."""
    return "".join(
        f"<tr><td>{lab}</td><td>{c:.2f}</td><td>{c - runs[0][1]:+.2f}</td><td>{n}</td><td>{note}</td></tr>"
        for lab, c, n, note in runs
    )


def png(fig) -> str:
    """Return a figure as an inline base64 `<img>` tag."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return (
        f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
    )


def main() -> None:
    """Make the figures and the report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outroot", type=Path, default=Path(os.environ["WITCH_OUTROOT"]))
    ap.add_argument("--out", type=Path, default=Path("reports/a399_401_act.html"))
    ap.add_argument("--debug", action="store_true", help="Skip the figures")
    args = ap.parse_args()
    base = args.outroot / "A399_401/act/a399_gnfw+a401_gnfw+bridge_cylindrical_beta"
    figs = ""
    if not args.debug:
        act = next(base.glob("act_only/*/act"))
        res = fits.getdata(next(act.glob("*_residual.fits")))
        mod = fits.getdata(next(act.glob("*_truth.fits")))
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
        for a, m, t in zip(
            ax, (res + mod, mod, res), ("data", "model", "residual"), strict=True
        ):
            im = a.imshow(
                gaussian_filter(m, 1.0),
                origin="lower",
                cmap="RdBu_r",
                vmin=-150,
                vmax=150,
            )
            a.set_title(t + " (smoothed 1 px)")
        fig.colorbar(im, ax=ax, label="uK_CMB")
        figs = png(fig)
    rows = table(
        [RUNS[k] for k in ("act_only_nobridge", "act_only", "act_only_ampfit")]
    )
    rows3 = table(RUNS3)
    yrows = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e:.2f}</td></tr>"
        for a, b, c, d, e in YFIT
    )
    ybeam = ", ".join(f"{k}: {v:.1f}" for k, v in YBEAM.items())
    yfig = ""
    if not args.debug:
        yact = next(args.outroot.glob("A399_401/acty/*/ymap/*/acty"))
        yres = fits.getdata(next(yact.glob("*_residual.fits")))
        ymod = fits.getdata(next(yact.glob("*_truth.fits")))
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
        for a, m, t, v in zip(
            ax,
            (yres + ymod, ymod, yres),
            ("data", "model", "residual"),
            ((-2e-5, 1.2e-4), (-2e-5, 1.2e-4), (-5e-5, 5e-5)),
            strict=True,
        ):
            im = a.imshow(m, origin="lower", vmin=v[0], vmax=v[1], cmap="RdBu_r")
            a.set_title(t)
            fig.colorbar(im, ax=a)
        yfig = png(fig)
    split_rows = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e:.3f}</td></tr>"
        for a, b, c, d, e in SPLITS
    )
    html = f"""<!doctype html><meta charset=utf-8><title>A399-A401 ACT-only</title>
<h1>ACT-only fit of A399, A401 and the bridge</h1>
<p>98 GHz ACT coadd (point source subtracted), 80'x80' cutout binned to 1', 6400 pixels, noise = CMB covariance
+ ivar white noise. Model: two gNFW (A10 shape, free m500 and centre) + cylindrical beta bridge. Positions, redshifts
and masses from Hincks et al. 2022 (arXiv:2107.04611).</p>
<h2>f090 only (6400 pixels)</h2><table border=1 cellpadding=4><tr><th>Run</th><th>chi2</th><th>d chi2</th><th>extra pars</th><th>note</th></tr>{rows}</table>
<h2>Joint f090+f150+f220 (19200 pixels)</h2>
<table border=1 cellpadding=4><tr><th>Run</th><th>chi2</th><th>d chi2</th><th>extra pars</th><th>note</th></tr>{rows3}</table>
<p>Each band has its own noise model, so the common primary CMB is treated as independent noise per band
and the combined errors are underestimated. No bridge detection either way.</p>
<h2>Joint noise model from the splits (linear GLS, fixed templates)</h2>
<p>Noise covariance estimated from sign-flipped combinations of the 4 splits per array
(scripts/act_split_noise.py), ring averaged cross-band spectra, stationary. Data are the split coadds.</p>
<table border=1 cellpadding=4><tr><th>Covariance</th><th>cluster scale</th><th>bridge P0 (1e-6)</th><th>bridge sigma</th><th>chi2/N</th></tr>{split_rows}</table>
<h2>Clusters only, ACT DR6 ILC Compton-y map</h2>
<p>ilc_SZ_mask80_yy (the standard ilc_SZ_yy is empty at dec +13), 80' cutout binned to 1', beam 1.6' assumed,
noise = stationary spectrum from 24 control fields. chi2/N = 1.04. Paper m500: 8.7 and 10.2 (1e14).</p>
<table border=1 cellpadding=4><tr><th>Model</th><th>A399 m500 (1e14)</th><th>A401 m500 (1e14)</th><th>alpha</th><th>chi2</th></tr>{yrows}</table>
<p>Beam FWHM sweep (arcsec: chi2): {ybeam}. {yfig}</p>
<p>Fitted m500: A399 3.9e14, A401 7.8e14 (paper, from beta models + Suzaku: 8.7e14, 10.2e14).</p>{figs}"""
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(html)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
