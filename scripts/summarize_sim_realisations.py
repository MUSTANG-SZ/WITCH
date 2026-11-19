"""
Summarise WITCH sim fits over noise realisations (see run_sim_realisations.py): for each run
and fitted parameter, the mean, the scatter between realisations, the mean reported error and
the pulls (fit - truth) / error, as an HTML report with a figure.
"""

import argparse
import base64
import glob
import io
import os
import re

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2, validated for CVD
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"


def seed_of(outdir: str) -> int:
    """
    The noise realisation of an output directory.

    Parameters
    ----------
    outdir : str
        Output directory ending in `-sim` (seed 0) or `-sim_s<seed>`.

    Returns
    -------
    seed : int
        The seed.
    """
    m = re.search(r"-sim(?:_s(\d+))?$", outdir.rstrip("/"))
    if m is None:
        raise ValueError(f"{outdir} is not a sim output directory")
    return int(m.group(1) or 0)


def collect(pattern: str, debug: bool) -> dict[str, dict[int, tuple[float, float]]]:
    """
    Read the final fit of every realisation matching a glob.

    Parameters
    ----------
    pattern : str
        Glob of the output directories.
    debug : bool
        Only read the first two.

    Returns
    -------
    fits : dict[str, dict[int, tuple[float, float]]]
        For each fitted parameter (`structure.parameter`) the value and error by seed.
    """
    dirs = sorted(
        d
        for d in glob.glob(pattern)
        if os.path.exists(os.path.join(d, "results_final_fit.dill"))
    )
    fits: dict[str, dict[int, tuple[float, float]]] = {}
    for d in dirs[: 2 if debug else None]:
        with open(os.path.join(d, "results_final_fit.dill"), "rb") as f:
            metamodel = dill.load(f)[0]
        for model in metamodel.models:
            names = [
                f"{s}.{p}" for s, p in zip(model.par_struct_names, model.par_names)
            ]
            for name, val, err, fit in zip(names, model.pars, model.errs, model.to_fit):
                if fit:
                    fits.setdefault(name, {})[seed_of(d)] = (float(val), float(err))
    return fits


def stats(fits: dict[int, tuple[float, float]], truth: float) -> dict[str, float]:
    """
    Summary statistics of one parameter over realisations.

    Parameters
    ----------
    fits : dict[int, tuple[float, float]]
        Value and error by seed.
    truth : float
        The input value.

    Returns
    -------
    stats : dict[str, float]
        n, mean, scatter (std between realisations), mean error, bias in units of the
        error on the mean, and mean and std of the pulls.
    """
    val, err = np.array(list(fits.values())).T
    pull = (val - truth) / err
    scatter = val.std(ddof=1) if len(val) > 1 else np.nan
    return {
        "n": len(val),
        "mean": val.mean(),
        "scatter": scatter,
        "mean_err": err.mean(),
        "bias_sig": (val.mean() - truth) / (scatter / np.sqrt(len(val))),
        "pull_mean": pull.mean(),
        "pull_std": pull.std(ddof=1) if len(val) > 1 else np.nan,
    }


def figure(runs: dict, params: list[str], truth: float) -> str:
    """
    Fitted value per realisation with its error, one panel per parameter.

    Parameters
    ----------
    runs : dict
        Label -> output of `collect`.
    params : list[str]
        Parameters to plot.
    truth : float
        The input value.

    Returns
    -------
    png : str
        The figure as a base64 PNG.
    """
    fig, axs = plt.subplots(1, len(params), figsize=(6 * len(params), 4), sharey=True)
    axs = np.atleast_1d(axs)
    offsets = np.linspace(-0.15, 0.15, len(runs))
    for ax, par in zip(axs, params):
        ax.axhline(truth, color=MUTED, lw=1, ls="--", zorder=0)
        for (label, fits), color, off in zip(runs.items(), COLORS, offsets):
            if par not in fits:
                continue
            seeds = np.array(sorted(fits[par]))
            val, err = np.array([fits[par][s] for s in seeds]).T
            ax.errorbar(
                seeds + off, val, err, fmt="o", ms=5, lw=1.5, color=color, label=label
            )
        ax.set_title(par, color=INK, loc="left")
        ax.set_xlabel("noise realisation (seed)", color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    axs[0].set_ylabel("fitted amplitude", color=MUTED)
    axs[0].text(
        axs[0].get_xlim()[0], truth, " truth", va="bottom", color=MUTED, fontsize=9
    )
    axs[-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    return base64.b64encode(buf.getvalue()).decode()


def html(title: str, runs: dict, truth: float, notes: str) -> str:
    """
    The report.

    Parameters
    ----------
    title : str
        Report title.
    runs : dict
        Label -> output of `collect`.
    truth : float
        The input value.
    notes : str
        Free text (HTML) shown above the results.

    Returns
    -------
    page : str
        The HTML page.
    """
    params = sorted({p for fits in runs.values() for p in fits})
    rows = []
    for label, fits in runs.items():
        for par in params:
            if par not in fits:
                continue
            s = stats(fits[par], truth)
            rows.append(
                f"<tr><td>{label}</td><td>{par}</td><td>{s['n']}</td><td>{s['mean']:.3f}</td>"
                f"<td>{s['scatter']:.3f}</td><td>{s['mean_err']:.3f}</td><td>{s['bias_sig']:+.2f}</td>"
                f"<td>{s['pull_mean']:+.2f}</td><td>{s['pull_std']:.2f}</td></tr>"
            )
    png = figure(runs, params, truth)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fcfcfb;--ink:#1f1f1e;--muted:#6b6a63;--rule:#e4e3de}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 16px}}
h1{{font-size:1.4rem}} p,li{{color:var(--ink)}} .muted{{color:var(--muted)}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:1rem 0}}
th,td{{padding:.35rem .7rem;border-bottom:1px solid var(--rule);text-align:right}}
th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}} th{{color:var(--muted);font-weight:500}}
img{{max-width:100%;height:auto}}
</style></head><body>
<h1>{title}</h1>
{notes}
<p class="muted">Truth = {truth}. Scatter is the standard deviation of the fitted values between realisations;
bias is (mean &minus; truth) in units of scatter/&radic;n; pulls are (fit &minus; truth)/reported error,
so a calibrated, unbiased fit has pull mean &asymp; 0 and pull std &asymp; 1.</p>
<table><thead><tr><th>run</th><th>parameter</th><th>n</th><th>mean</th><th>scatter</th><th>mean error</th>
<th>bias (&sigma; of mean)</th><th>pull mean</th><th>pull std</th></tr></thead><tbody>
{"".join(rows)}</tbody></table>
<img alt="Fitted amplitude per noise realisation for each run, truth dashed" src="data:image/png;base64,{png}">
</body></html>"""


def main():
    """Make the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label=glob of the output directories, can be repeated",
    )
    parser.add_argument(
        "--params", nargs="+", help="Only these parameters (structure.parameter)"
    )
    parser.add_argument("--truth", type=float, default=-1.0)
    parser.add_argument("--title", default="Sim recovery over noise realisations")
    parser.add_argument("--notes", default="", help="HTML shown above the results")
    parser.add_argument("--out", required=True, help="Output HTML file")
    parser.add_argument("--debug", action="store_true", help="Two realisations per run")
    args = parser.parse_args()
    runs = {}
    for spec in args.run:
        label, pattern = spec.split("=", 1)
        runs[label] = collect(pattern, args.debug)
        if args.params:
            runs[label] = {p: v for p, v in runs[label].items() if p in args.params}
        print(label, {p: len(v) for p, v in runs[label].items()})
    with open(args.out, "w") as f:
        f.write(html(args.title, runs, args.truth, args.notes))
    print("wrote", args.out)


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
