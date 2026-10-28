"""
Compare the SZ and X-ray pressure profiles of MOO 1142 and represent the error with 90% confidence intervals.

This is a copy of compare_xray_SZ_pressure_profiles_errors.py that additionally computes and plots the
full (n_r, n_r) covariance of each pressure-profile ensemble, to check whether the 90% CL bands (which
only probe the diagonal of this covariance) are hiding strong off-diagonal correlations between radial bins.

Created in 2026-05, covariance diagnostics added 2026-09.
"""

from __future__ import annotations

# Standard-library imports.
import argparse
import builtins
import sys
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import numpy as np
import numpy.random._pickle as numpy_random_pickle
import scipy.special
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter

try:
    from .compare_xray_SZ_pressure_profiles_errors import (
        format_radius_axes,
        format_shared_pressure_axes,
        kpc_per_arcsec,
    )
    from .profile_models import gNFW, iso_beta
except ImportError:
    from compare_xray_SZ_pressure_profiles_errors import (
        format_radius_axes,
        format_shared_pressure_axes,
        kpc_per_arcsec,
    )
    from profile_models import gNFW, iso_beta

# -------------------------------------------------------------------------------------------- #
location = Path(__file__).resolve().parent

# Globally relevant constants and parameters.
R_values = np.arange(0.1, 100.0, 0.1)
PROFILE_SAMPLE_COUNT = 1000

# XMM profile information
xmm_state_path = Path("/mnt/welch/XRAY/pmc_final.state")
ALPHA_MAIN = 2.26
GAMMA_MAIN = 0.465
F_S_MAIN = 1.632e4
F_S_SUB = 2.597e5
KT_MAIN_KEV = 6.76
KT_SUB_KEV = 7.13
XMM_PIXEL_SCALE_ARCSEC = 2.5

# Plot styling.
PLOT_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
}
PLOT_STYLE = {
    "M2 rs_fixed": ("blue", "-"),
    "M2 rs_free": ("darkorange", "--"),
    "XMM": ("red", "-"),
}

# M2 profile information: MCMC runs with r_s fixed / free.
m2_reduction_dir = Path("/mnt/welch/USERS/jorlo/Reductions/MOOJ1142")
M2_RUNS = {
    "rs_fixed": (m2_reduction_dir / "r-beta-amp-P0-beta-dx_g-dy_g-sigma-amp_g", False),
    "rs_free": (
        m2_reduction_dir / "r-beta-amp-P0-r_s-beta-dx_g-dy_g-sigma-amp_g",
        True,
    ),
}
M2_ALPHA = 2.26
M2_GAMMA = 0.465
M2_R_S_FIXED = 4.33


# -------------------------------------------------------------------------------------------- #
def load_xmm_state(state_path: Path) -> dict:
    # Provenance: standalone 90% CL script.
    # Intent: load the saved pocoMC state outside the original notebook session by patching the
    # legacy objects expected by the serialized RNG payload.
    """Load the saved pocoMC state with compatibility shims for this environment."""
    sys.modules["__builtin__"] = builtins
    original_randomstate_ctor = numpy_random_pickle.__randomstate_ctor
    original_bitgen_ctor = numpy_random_pickle.__bit_generator_ctor

    def patched_randomstate_ctor(bit_generator_name: str = "MT19937", *args):
        # Ignore extra pickle args so older numpy random-state payloads still load.
        return original_randomstate_ctor(bit_generator_name)

    def patched_bitgen_ctor(bit_generator_name: str = "MT19937", *args):
        # Ignore extra pickle args for the bit-generator payload as well.
        return original_bitgen_ctor(bit_generator_name)

    numpy_random_pickle.__randomstate_ctor = patched_randomstate_ctor
    numpy_random_pickle.__bit_generator_ctor = patched_bitgen_ctor
    try:
        with state_path.open("rb") as file_handle:
            return dill.load(file_handle)
    finally:
        numpy_random_pickle.__randomstate_ctor = original_randomstate_ctor
        numpy_random_pickle.__bit_generator_ctor = original_bitgen_ctor


def get_xmm_posterior_samples(state_path: Path, sample_count: int) -> np.ndarray:
    # Provenance: adapted from radial_profile_and_uncert_n.ipynb posterior access.
    # Intent: reconstruct the same posterior tail used for the notebook pressure-profile summaries,
    # but from the serialized state instead of a live sampler object.
    """Return the last sample_count samples, i.e. the final posterior stage (earlier rows are intermediate pocoMC tempering stages)."""
    state = load_xmm_state(state_path)
    samples = state["particles"].get("x", flat=True)
    return samples[-sample_count:]


def compute_profile_covariance(p_profiles: list[np.ndarray]) -> np.ndarray:
    # Provenance: colleague's suggested sanity check on 2026-09.
    # Intent: expose the full (n_r, n_r) covariance across radial bins, since the percentile bands
    # only summarize the diagonal and can hide strong bin-to-bin correlations.
    """Return the (n_r, n_r) covariance matrix of a pressure-profile ensemble."""
    return np.cov(np.asarray(p_profiles).T)


def plot_profile_covariance(
    cov: np.ndarray, r_vals: np.ndarray, title: str, output_path: Path
) -> None:
    """Save an imshow diagnostic plot of a radial-bin covariance matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = (r_vals[0], r_vals[-1], r_vals[-1], r_vals[0])
    log_cov = np.where(np.abs(cov) > 0.0, np.log10(np.abs(cov)), np.nan)
    im = ax.imshow(log_cov, extent=extent, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel('r (")')
    ax.set_ylabel('r (")')
    fig.colorbar(im, ax=ax, label="log10(abs(covariance))")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def xmm_profile_params(samples: np.ndarray) -> dict[str, np.ndarray]:
    # Provenance: adapted from radial_profile_and_uncert_n.ipynb, posterior-profile construction.
    # Intent: convert each posterior sample into the physical profile parameters, with radii in arcsec.
    """
    Convert XMM posterior samples into profile parameters.

    Args:
        samples: XMM posterior samples, columns ordered as the saved notebook fit
            (x_main, y_main, x_sub, y_sub, rs_main, rc_sub, beta_main, beta_sub, a_main, a_sub, a_bkg).

    Returns:
        Map "main" -> (n, 3) array of gNFW (P0, r_s, beta) and "sub" -> (n, 3) array of isobeta (amp, r_c, beta).
    """
    rs_main, rc_sub, beta_main, beta_sub, a_main, a_sub = samples[:, 4:10].T
    p0_main = np.sqrt(a_main / F_S_MAIN) * KT_MAIN_KEV
    amp_sub = (
        np.sqrt(
            a_sub
            * scipy.special.gamma(3.0 * beta_sub)
            / (F_S_SUB * scipy.special.gamma(3.0 * beta_sub - 0.5))
        )
        * KT_SUB_KEV
    )
    return {
        "main": np.column_stack(
            [p0_main, rs_main * XMM_PIXEL_SCALE_ARCSEC, beta_main / 2.0]
        ),
        "sub": np.column_stack([amp_sub, rc_sub * XMM_PIXEL_SCALE_ARCSEC, beta_sub]),
    }


def compute_xmm_pressure_profiles(
    samples: np.ndarray, r_vals: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build the XMM main-cluster and west-subcluster pressure-profile ensembles."""
    params = xmm_profile_params(samples)
    return (
        [
            gNFW(r_vals, p_0, r_s, ALPHA_MAIN, beta, GAMMA_MAIN)
            for p_0, r_s, beta in params["main"]
        ],
        [iso_beta(r_vals, p_0, r_c, beta) for p_0, r_c, beta in params["sub"]],
    )


def load_M2_samples(run_dir: Path, sample_count: int, seed: int = 0) -> np.ndarray:
    """
    Load a random subset of the flattened M2 MCMC samples of a run.

    Args:
            run_dir: Reduction directory containing samples_mcmc.npz.
            sample_count: Number of samples to draw (without replacement).
            seed: Seed of the random subsampling.

    Returns:
            Array of shape (sample_count, n_fit_par), columns ordered as the fitted parameters in the run directory name.
    """
    samples = np.load(run_dir / "samples_mcmc.npz")["samples"]
    rng = np.random.default_rng(seed)
    return samples[
        rng.choice(len(samples), size=min(sample_count, len(samples)), replace=False)
    ]


def m2_profile_params(samples: np.ndarray, *, r_s_fit: bool) -> dict[str, np.ndarray]:
    """
    Split M2 MCMC samples into profile parameters.

    Args:
        samples: MCMC samples with columns (r, beta, amp, P0, [r_s,] beta, dx_g, dy_g, sigma, amp_g), the
            bracketed r_s column being present only if r_s was fit.
        r_s_fit: Whether r_s is a fitted parameter; otherwise it is fixed to M2_R_S_FIXED.

    Returns:
        Map "main" -> (n, 3) array of gNFW (P0, r_s, beta) and "sub" -> (n, 3) array of isobeta (amp, r_c, beta).
    """
    r_c, beta_sub, amp, p_0 = samples[:, :4].T
    r_s, beta_main = (
        (samples[:, 4], samples[:, 5])
        if r_s_fit
        else (np.full(len(samples), M2_R_S_FIXED), samples[:, 4])
    )
    return {
        "main": np.column_stack([p_0, r_s, beta_main]),
        "sub": np.column_stack([amp, r_c, beta_sub]),
    }


def compute_M2_pressure_profiles(
    samples: np.ndarray,
    r_vals: np.ndarray,
    *,
    r_s_fit: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build the M2 main-cluster (gNFW) and west-subcluster (isobeta) pressure-profile ensembles from MCMC samples.

    Args:
        samples: MCMC samples, see m2_profile_params.
        r_vals: Radii at which to evaluate the profiles.
        r_s_fit: Whether r_s is a fitted parameter; otherwise it is fixed to M2_R_S_FIXED.

    Returns:
        Lists of main and sub pressure profiles, one entry per sample.
    """
    params = m2_profile_params(samples, r_s_fit=r_s_fit)
    return (
        [
            gNFW(r_vals, p_0, r_s, M2_ALPHA, beta, M2_GAMMA)
            for p_0, r_s, beta in params["main"]
        ],
        [iso_beta(r_vals, p_0, r_c, beta) for p_0, r_c, beta in params["sub"]],
    )


# -------------------------------------------------------------------------------------------- #
def plot_profile_comparison(
    ensembles: dict[str, tuple[list[np.ndarray], list[np.ndarray]]],
    r_vals: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot the median and 90% confidence band of the main and west-subcluster pressure profiles side by side,
    with the axis formatting of compare_xray_SZ_pressure_profiles_errors.py.

    Args:
            ensembles: Map from label to the (main, sub) lists of sampled profiles.
            r_vals: Radii in arcsec at which the profiles are evaluated.
            output_path: Where to save the figure.
    """
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5), sharey=True, gridspec_kw={"wspace": 0.0}
        )
        _draw_profile_panels(axes, ensembles, r_vals)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


PANEL_TITLES = ("Main Cluster Pressure Profile", "West Subcluster Pressure Profile")
PARAM_LABELS = {
    "main": (r"$P_0$ (keV cm$^{-3}$)", r'$r_s$ (")', r"$\beta$"),
    "sub": (r"amp (keV cm$^{-3}$)", r'$r_c$ (")', r"$\beta$"),
}


def _draw_profile_panel(
    ax: plt.Axes,
    ensembles: dict[str, tuple[list[np.ndarray], list[np.ndarray]]],
    r_vals: np.ndarray,
    i: int,
) -> None:
    """Draw the median + 90% band of component i (0 main, 1 sub) for every source onto ax, with radius axes."""
    ax.set_title(PANEL_TITLES[i])
    for label, pair in ensembles.items():
        color, ls = PLOT_STYLE[label]
        low, med, high = np.percentile(np.asarray(pair[i]), [5, 50, 95], axis=0)
        ax.loglog(r_vals, med, c=color, ls=ls, label=label)
        ax.fill_between(r_vals, low, high, color=color, alpha=0.3)
    ax.legend()
    format_radius_axes(ax, kpc_per_arcsec, hide_rightmost_label=not i)


def _draw_profile_panels(
    axes: np.ndarray,
    ensembles: dict[str, tuple[list[np.ndarray], list[np.ndarray]]],
    r_vals: np.ndarray,
) -> None:
    """Draw the main (left) and west-subcluster (right) median + 90% band panels onto axes."""
    for i, ax in enumerate(axes):
        _draw_profile_panel(ax, ensembles, r_vals, i)
    format_shared_pressure_axes(*axes)
    axes[0].figure.subplots_adjust(wspace=0.0)


def _credible_levels(hist: np.ndarray, fracs: tuple[float, ...]) -> list[float]:
    """Return the histogram heights enclosing each fraction of the total mass (highest fraction first -> lowest level)."""
    flat = np.sort(hist.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return [flat[np.searchsorted(cum, f)] for f in fracs]


def _own_range(x: np.ndarray, pad: float = 0.2) -> tuple[float, float]:
    """Return the 0.5-99.5 percentile range of x, padded by the given fraction of its width."""
    lo, hi = np.percentile(x, [0.5, 99.5])
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def plot_corner(
    fig: plt.Figure,
    samples: dict[str, np.ndarray],
    labels: tuple[str, ...],
    n_bins: int = 30,
) -> None:
    """
    Draw an overlaid corner plot of several sample sets onto fig.

    Parameters whose samples are constant for a source (i.e. fixed in that fit) are skipped for that source.
    Each source is binned over its own range so that narrow posteriors are not over-smoothed by the
    shared (wider) axis limits.

    Args:
        fig: Figure or SubFigure to draw on.
        samples: Map from source label (key of PLOT_STYLE) to an (n, k) array of samples.
        labels: The k axis labels.
        n_bins: Histogram bins per axis.
    """
    k = len(labels)
    axes = fig.subplots(
        k,
        k,
        squeeze=False,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05, "left": 0.12, "right": 0.98},
    )
    varies = {name: np.ptp(s, axis=0) > 0 for name, s in samples.items()}
    ranges = []
    for j in range(k):
        cols = np.concatenate(
            [s[:, j] for name, s in samples.items() if varies[name][j]]
        )
        lo, hi = np.percentile(cols, [0.5, 99.5])
        pad = 0.1 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            for name, s in samples.items():
                if not (varies[name][i] and varies[name][j]):
                    continue
                color = PLOT_STYLE[name][0]
                if i == j:
                    ax.hist(
                        s[:, i],
                        bins=n_bins,
                        range=_own_range(s[:, i]),
                        density=True,
                        histtype="step",
                        color=color,
                        lw=1.5,
                    )
                    continue
                hist, xe, ye = np.histogram2d(
                    s[:, j],
                    s[:, i],
                    bins=n_bins,
                    range=[_own_range(s[:, j]), _own_range(s[:, i])],
                )
                hist = gaussian_filter(hist, 1.0)
                l95, l68 = _credible_levels(hist, (0.95, 0.68))
                xc, yc = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
                ax.contourf(
                    xc,
                    yc,
                    hist.T,
                    levels=[l95, l68, hist.max() * 1.001],
                    colors=[to_rgba(color, 0.25), to_rgba(color, 0.5)],
                )
                ax.contour(
                    xc, yc, hist.T, levels=[l95, l68], colors=color, linewidths=1.0
                )
            ax.set_xlim(ranges[j])
            if i != j:
                ax.set_ylim(ranges[i])
            else:
                ax.set_yticks([])
            ax.tick_params(labelbottom=i == k - 1, labelleft=j == 0 and i > 0)
            if i == k - 1:
                ax.set_xlabel(labels[j])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])


def plot_profile_and_corner(
    ensembles: dict[str, tuple[list[np.ndarray], list[np.ndarray]]],
    params: dict[str, dict[str, np.ndarray]],
    r_vals: np.ndarray,
    component: str,
    output_path: Path,
) -> None:
    """
    Save a two-panel figure of one component: its pressure profile (left) and a corner plot of the
    profile parameters of the samples used to build it (right), for every source.

    Args:
        ensembles: Map from label to the (main, sub) lists of sampled profiles.
        params: Map from label to the {"main", "sub"} parameter arrays of the same samples.
        r_vals: Radii in arcsec at which the profiles are evaluated.
        component: "main" or "sub".
        output_path: Where to save the figure.
    """
    i = ("main", "sub").index(component)
    with plt.rc_context(PLOT_RC):
        fig = plt.figure(figsize=(17, 6.5))
        left, right = fig.subfigures(1, 2, width_ratios=[1, 1.1], wspace=0.0)
        left.subplots_adjust(left=0.12, right=0.97)
        ax = left.subplots()
        _draw_profile_panel(ax, ensembles, r_vals, i)
        ax.set(ylim=(1e-7, 1.0), ylabel=r"P (keV cm$^{-3}$)")
        plot_corner(
            right,
            {label: p[component] for label, p in params.items()},
            PARAM_LABELS[component],
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Compute the radial-bin covariance, co-plotted 90% bands and parameter corner plots of the M2 and XMM pressure profiles, all from MCMC chains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        choices=[*M2_RUNS, "both"],
        default="both",
        help="Which M2 MCMC run to use.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Quick test: fewer samples and a coarser radial grid.",
    )
    args = parser.parse_args()

    n_samples = 50 if args.debug else PROFILE_SAMPLE_COUNT
    r_vals = R_values[::10] if args.debug else R_values
    cov_plot_dir = location / "plots/pressure/covariance"
    cov_plot_dir.mkdir(parents=True, exist_ok=True)

    # Pressure-profile ensembles per source: XMM first, then each requested M2 run.
    ensembles, params = {}, {}
    for name in M2_RUNS if args.run == "both" else [args.run]:
        run_dir, r_s_fit = M2_RUNS[name]
        m2_samples = load_M2_samples(run_dir, n_samples)
        ensembles[f"M2 {name}"] = compute_M2_pressure_profiles(
            m2_samples, r_vals, r_s_fit=r_s_fit
        )
        params[f"M2 {name}"] = m2_profile_params(m2_samples, r_s_fit=r_s_fit)
    xmm_samples = get_xmm_posterior_samples(xmm_state_path, n_samples)
    ensembles["XMM"] = compute_xmm_pressure_profiles(xmm_samples, r_vals)
    params["XMM"] = xmm_profile_params(xmm_samples)

    for label, pair in ensembles.items():
        for component, profiles in zip(("main", "sub"), pair):
            plot_profile_covariance(
                compute_profile_covariance(profiles),
                r_vals,
                f"{label} {component} profile covariance",
                cov_plot_dir / f"{label.replace(' ', '_')}_{component}_cov.png",
            )
    plot_profile_comparison(
        ensembles,
        r_vals,
        cov_plot_dir.parent / "MOO_1142_main+west_pressure_profiles_w_errors.png",
    )
    for component in ("main", "sub"):
        plot_profile_and_corner(
            ensembles,
            params,
            r_vals,
            component,
            cov_plot_dir.parent
            / f"MOO_1142_{component}_pressure_profile_and_corner.png",
        )


if __name__ == "__main__":
    main()
