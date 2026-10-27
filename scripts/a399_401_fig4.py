"""
Replicate Fig. 4 of Hincks et al. (2022, arXiv:2107.04611): the posterior of the 'Ellip-beta+mesa' fit.

Reads the samples written by `witcher configs/A399_401/A399_401_ymap_isobeta_mesa_mcmc.yaml` and converts
the isobeta amplitudes to the paper's central Compton y, A = prefactor * amp * r_3 * sqrt(pi) Gamma(1.5 beta - 0.5) / Gamma(1.5 beta),
and the mesa amplitude to A_fil = prefactor * amp. The core radius is the larger of r_1 and r_2.
The blue lines mark the LM best fit.

Usage
-----
    python scripts/a399_401_fig4.py --dir <fit output dir> --out reports/a399_401_fig4.png [--debug]
"""

import argparse
from pathlib import Path

import corner
import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

from witch import utils as wu

Z = 0.072735  # `constants.z` of the config, sets the y prefactor
# Column layout of the full parameter vector: two isobeta clusters (9 pars) then the mesa (6 pars)
CLUSTERS = {"A399": 0, "A401": 9}
PAPER = {  # Table 2, 'Ellip-beta+mesa'
    "A399": (8.1, 0.81, 3.0),
    "A401": (12.6, 0.82, 2.6),
    "A_fil": 1.10,
}
LABELS = [
    r"$A_{401}(10^{-5}y)$",
    r"$A_{399}(10^{-5}y)$",
    r"$A_{fil}(10^{-5}y)$",
    r"$\beta_{401}$",
    r"$\beta_{399}$",
    r"$rc_{401}(')$",
    r"$rc_{399}(')$",
]


# Axis ranges read off the paper's Fig. 4, in the order of LABELS
RANGES = [
    (9.5, 15.5),
    (5.5, 11.0),
    (0.3, 2.0),
    (0.62, 1.35),
    (0.6, 1.7),
    (1.3, 5.5),
    (1.0, 8.0),
]


def physical(pars: np.ndarray, prefactor: float) -> np.ndarray:
    """
    Convert the model parameters to the quantities plotted in Fig. 4.

    Parameters
    ----------
    pars : np.ndarray
        Full parameter vectors with shape `(..., 24)`.
    prefactor : float
        Dataset prefactor that turns the integrated pressure into Compton y.

    Returns
    -------
    out : np.ndarray
        `(..., 7)` array of A_401, A_399, A_fil (1e-5 y), beta_401, beta_399, rc_401, rc_399 (arcmin).
    """
    amps, betas, rcs = {}, {}, {}
    for name, i in CLUSTERS.items():
        r1, r2, r3, beta, amp = (pars[..., i + k] for k in (3, 4, 5, 7, 8))
        norm = r3 * np.sqrt(np.pi) * gamma(1.5 * beta - 0.5) / gamma(1.5 * beta)
        amps[name] = prefactor * amp * norm * 1e5
        betas[name] = beta
        rcs[name] = np.maximum(r1, r2) / 60.0
    a_fil = prefactor * pars[..., 23] * 1e5
    return np.stack(
        [
            amps["A401"],
            amps["A399"],
            a_fil,
            betas["A401"],
            betas["A399"],
            rcs["A401"],
            rcs["A399"],
        ],
        axis=-1,
    )


def main() -> None:
    """Make the corner plot and print the posterior summary next to the paper's values."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="Fit output directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--burn", type=float, default=0.1, help="Burn-in fraction")
    ap.add_argument("--debug", action="store_true", help="Use every 50th sample")
    args = ap.parse_args()

    chains = np.load(args.dir / "samples_mcmc.npz", allow_pickle=True)["chains"]
    chains = chains[:, int(args.burn * chains.shape[1]) :]
    best = np.asarray(
        dill.load((args.dir / "par_results_fit3.dill").open("rb"))["parameters"]
    )
    prefactor = float(wu.get_da(Z) * wu.XMpc / wu.me)
    samples = physical(chains.reshape(-1, chains.shape[-1]), prefactor)
    truths = physical(best, prefactor)
    if args.debug:
        samples = samples[::50]

    fig = corner.corner(
        samples,
        labels=LABELS,
        truths=truths,
        truth_color="C0",
        range=RANGES,
        levels=(0.68, 0.95, 0.997),
        fill_contours=True,
        plot_datapoints=False,
        smooth=1.0,
        color="k",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)

    q = np.percentile(samples, [16, 50, 84], axis=0)
    paper = [
        PAPER["A401"][0],
        PAPER["A399"][0],
        PAPER["A_fil"],
        PAPER["A401"][1],
        PAPER["A399"][1],
        PAPER["A401"][2],
        PAPER["A399"][2],
    ]
    print(f"{'param':14s} {'median':>8s}  -    +   {'LM':>7s} {'paper':>7s}")
    for lab, lo, med, hi, t, p in zip(
        [s.replace("$", "") for s in LABELS],
        q[0],
        q[1],
        q[2],
        truths,
        paper,
        strict=True,
    ):
        print(f"{lab:14s} {med:8.3f} {med - lo:5.3f} {hi - med:5.3f} {t:7.3f} {p:7.2f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
