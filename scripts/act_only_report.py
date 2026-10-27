"""
Write reports/act_only.html: figures for the ACT-only MOOJ1142 fit.

Reads the model and residual maps written by the real-data config and plots data, model, residual
and the chi-squared profile over alpha (values from `scripts/act_alpha_profile.py`).

Usage
-----
    python scripts/act_only_report.py <fit output dir>/act
"""

import argparse
import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

ALPHA = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
DCHI2 = [4.97, 2.03, 1.12, 0.68, 0.42, 0.26, 0.14, 0.06, 0.0]


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
    ap.add_argument("actdir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/act_only.html"))
    args = ap.parse_args()
    res = fits.getdata(next(args.actdir.glob("*_residual.fits")))
    mod = fits.getdata(next(args.actdir.glob("*_truth.fits")))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, m, t in zip(
        ax, (res + mod, mod, res), ("data", "model", "residual"), strict=True
    ):
        im = a.imshow(m, origin="lower", cmap="RdBu_r", vmin=-100, vmax=100)
        a.set_title(t)
    fig.colorbar(im, ax=ax, label="uK_CMB")
    maps = png(fig)
    fig, a = plt.subplots(figsize=(5, 3.5))
    a.plot(ALPHA, DCHI2, "o-")
    a.set(
        xlabel="alpha (fixed)",
        ylabel="chi2 - min",
        title="Real ACT map, m500 and amp refit",
    )
    prof = png(fig)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(
        f"""<!doctype html><meta charset="utf-8"><title>ACT-only fitting</title>
<h1>ACT-only fitting of MOOJ1142</h1>
<h2>Takeaway</h2>
<p>For a cluster of MOOJ1142's size, ACT adds essentially nothing on the profile shape. The map is 40x40 pixels
at 0.5' with a 2' beam, so the cluster is nearly unresolved and alpha is unconstrained (see the profile below).
The map does detect the integrated signal, but only as one combination of m500 and the point source amplitude.
That may still serve as an overall flux anchor in a joint fit; it cannot help with shape.
Untested guess: a free ACT point source amplitude is degenerate with m500 because the beam blends them,
so in a joint fit it is tied to M2 by position only and could absorb cluster signal.
Suggested joint-fit setup: keep ACT only as a flux constraint with the shape parameters fixed or set by M2, or drop it.</p>
<p>The CMB + foreground covariance is crude (foreground is an l^2 power law with D_l(3000)=100 uK^2) and should be improved.
It is not expected to change the conclusion above, but this has not been tested; the mismatch scan only varied the CMB scale.</p>
<p>Simulation (<code>unit_tests/act/</code>): white noise recovers m500, alpha and the point source amplitude
within 1 sigma. With CMB noise the LM fit stalls at chitol 1e-5 (float32 chi2) and the quoted alpha error is
not trustworthy.</p>
<h2>Real map: data, model, residual</h2>{maps}
<h2>Chi2 profile over alpha</h2>{prof}
<p>Over alpha 0.5-4.5 chi2 changes by 5, and by 1.1 between 1.5 and 4.5. The minimum is at the upper prior edge, so
the data give roughly alpha &gt; 1.4 and no upper bound. The LM error of about 0.4 is misleading.</p>
"""
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
