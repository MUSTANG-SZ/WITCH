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


def compute_xmm_pressure_profiles(
    samples: np.ndarray, r_vals: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # Provenance: adapted from radial_profile_and_uncert_n.ipynb, posterior-profile construction.
    # Intent: turn each posterior sample into a main-cluster and west-subcluster pressure profile.
    """Build the XMM main-cluster and west-subcluster pressure-profile ensembles."""
    p_profiles_main = []
    p_profiles_sub = []

    for sample in samples:
        # Match the posterior parameter ordering used by the saved notebook fit.
        (
            _x_main_mcmc,
            _y_main_mcmc,
            _x_sub_mcmc,
            _y_sub_mcmc,
            rs_main_mcmc,
            rc_sub_mcmc,
            beta_main_mcmc,
            beta_sub_mcmc,
            a_main_mcmc,
            a_sub_mcmc,
            _a_bkg_mcmc,
        ) = sample

        # Keep the notebook normalization and shape conventions, with radii converted to arcsec.
        main_profile = gNFW(
            r=r_vals,
            p_0=np.sqrt(a_main_mcmc / F_S_MAIN) * KT_MAIN_KEV,
            r_s=rs_main_mcmc * XMM_PIXEL_SCALE_ARCSEC,
            alpha=ALPHA_MAIN,
            beta=beta_main_mcmc / 2.0,
            gamma=GAMMA_MAIN,
        )
        sub_profile = iso_beta(
            r=r_vals,
            p_0=np.sqrt(
                a_sub_mcmc
                * scipy.special.gamma(3.0 * beta_sub_mcmc)
                / (F_S_SUB * scipy.special.gamma(3.0 * beta_sub_mcmc - 0.5))
            )
            * KT_SUB_KEV,
            r_c=rc_sub_mcmc * XMM_PIXEL_SCALE_ARCSEC,
            beta=beta_sub_mcmc,
        )

        p_profiles_main.append(main_profile)
        p_profiles_sub.append(sub_profile)

    return p_profiles_main, p_profiles_sub


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


def compute_M2_pressure_profiles(
    samples: np.ndarray,
    r_vals: np.ndarray,
    *,
    r_s_fit: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build the M2 main-cluster (gNFW) and west-subcluster (isobeta) pressure-profile ensembles from MCMC samples.

    Args:
            samples: MCMC samples with columns (r, beta, amp, P0, [r_s,] beta, dx_g, dy_g, sigma, amp_g), the
                    bracketed r_s column being present only if r_s was fit.
            r_vals: Radii at which to evaluate the profiles.
            r_s_fit: Whether r_s is a fitted parameter; otherwise it is fixed to M2_R_S_FIXED.

    Returns:
            Lists of main and sub pressure profiles, one entry per sample.
    """
    p_profiles_main = []
    p_profiles_sub = []

    for sample in samples:
        r_c_sub, beta_sub, amp_sub, p_0 = sample[:4]
        r_s, beta_main = (
            (sample[4], sample[5]) if r_s_fit else (M2_R_S_FIXED, sample[4])
        )

        p_profiles_main.append(gNFW(r_vals, p_0, r_s, M2_ALPHA, beta_main, M2_GAMMA))
        p_profiles_sub.append(iso_beta(r_vals, amp_sub, r_c_sub, beta_sub))

    return p_profiles_main, p_profiles_sub


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


def _draw_profile_panels(
    axes: np.ndarray,
    ensembles: dict[str, tuple[list[np.ndarray], list[np.ndarray]]],
    r_vals: np.ndarray,
) -> None:
    """Draw the main (left) and west-subcluster (right) median + 90% band panels onto axes."""
    for i, (ax, title) in enumerate(
        zip(axes, ("Main Cluster Pressure Profile", "West Subcluster Pressure Profile"))
    ):
        ax.set_title(title)
        for label, pair in ensembles.items():
            color, ls = PLOT_STYLE[label]
            low, med, high = np.percentile(np.asarray(pair[i]), [5, 50, 95], axis=0)
            ax.loglog(r_vals, med, c=color, ls=ls, label=label)
            ax.fill_between(r_vals, low, high, color=color, alpha=0.3)
        ax.legend()
        format_radius_axes(ax, kpc_per_arcsec, hide_rightmost_label=not i)
    format_shared_pressure_axes(*axes)
    axes[0].figure.subplots_adjust(wspace=0.0)


def main() -> None:
    """Compute the radial-bin covariance and co-plotted 90% bands of the M2 and XMM pressure profiles, all from MCMC chains."""
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
    ensembles = {}
    for name in M2_RUNS if args.run == "both" else [args.run]:
        run_dir, r_s_fit = M2_RUNS[name]
        ensembles[f"M2 {name}"] = compute_M2_pressure_profiles(
            load_M2_samples(run_dir, n_samples), r_vals, r_s_fit=r_s_fit
        )
    xmm_samples = get_xmm_posterior_samples(xmm_state_path, n_samples)
    ensembles["XMM"] = compute_xmm_pressure_profiles(xmm_samples, r_vals)

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


if __name__ == "__main__":
    main()
