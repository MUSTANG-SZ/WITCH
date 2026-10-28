"""
Compare the SZ and X-ray pressure profiles of MOO 1142 and represent the error with 90% confidence intervals.

This combines the base two-panel comparison plot from compare_xray_SZ_profiles.py
with the posterior-profile construction used in radial_profile_and_uncert_n.ipynb.

Created in 2026-05.
"""
from __future__ import annotations

# Standard-library imports.
import builtins
from datetime import date
import sys
import types
from pathlib import Path

from astropy import units as u
from astropy.cosmology import Planck18

import dill
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpy.random._pickle as numpy_random_pickle
import pandas as pd
import scipy.special

try:
	from .profile_models import gNFW, iso_beta
except ImportError:
	from profile_models import gNFW, iso_beta

# -------------------------------------------------------------------------------------------- #
location = Path(__file__).resolve().parent

# Globally relevant constants and parameters.
z = 1.189
kpc_per_arcsec = Planck18.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value
R_values = np.arange(0.1, 100.0, 0.1)
PROFILE_SAMPLE_COUNT = 1000

# XMM profile information
xmm_fit_dir = location.parent / "XMM/Barbavara_fit_2026-04"
xmm_state_path = xmm_fit_dir / "gnfw_circ+beta_circ_acfixed_NESTED/pmc_final.state"
ALPHA_MAIN = 2.26
GAMMA_MAIN = 0.465
F_S_MAIN = 1.632e4
F_S_SUB = 2.597e5
KT_MAIN_KEV = 6.76
KT_SUB_KEV = 7.13
XMM_PIXEL_SCALE_ARCSEC = 2.5

# M2 profile information
m2_fit_dir = location.parent / "M2/MOO1142_Xray_priors_subcluster"
m2_fit_cov_path = m2_fit_dir / "par_results_final_fit.dill"
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
	"""Return the tail of the saved posterior samples used for the notebook profiles."""
	state = load_xmm_state(state_path)
	samples = state["particles"].get("x", flat=True)
	return samples[-sample_count:]

def compute_profile_percentiles(
	p_profiles_main: list[np.ndarray],
	p_profiles_sub: list[np.ndarray],
) -> dict[str, np.ndarray]:
	"""Summarize main and subcluster pressure-profile ensembles with 90% intervals."""
	p_p5_main, p_p50_main, p_p95_main = np.percentile(p_profiles_main, [5, 50, 95], axis=0)
	p_p5_sub, p_p50_sub, p_p95_sub = np.percentile(p_profiles_sub, [5, 50, 95], axis=0)

	return {
		"P_p5_main": p_p5_main,
		"P_p50_main": p_p50_main,
		"P_p95_main": p_p95_main,
		"P_p5_sub": p_p5_sub,
		"P_p50_sub": p_p50_sub,
		"P_p95_sub": p_p95_sub,
	}

def compute_xmm_pressure_profiles(samples: np.ndarray, r_vals: np.ndarray) -> dict[str, np.ndarray]:
	# Provenance: adapted from radial_profile_and_uncert_n.ipynb, posterior-profile construction.
	# Intent: turn each posterior sample into a main-cluster and west-subcluster pressure profile,
	# then summarize the ensemble with the same 5/50/95 percentile curves used in the notebook.
	"""Build the XMM median and 90% interval pressure profiles for both subclusters."""
	p_profiles_main = []
	p_profiles_sub = []

	for sample in samples:
		# Match the posterior parameter ordering used by the saved notebook fit.
		(
			x_main_mcmc,
			y_main_mcmc,
			x_sub_mcmc,
			y_sub_mcmc,
			rs_main_mcmc,
			rc_sub_mcmc,
			beta_main_mcmc,
			beta_sub_mcmc,
			a_main_mcmc,
			a_sub_mcmc,
			a_bkg_mcmc,
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
				a_sub_mcmc * scipy.special.gamma(3.0 * beta_sub_mcmc)
				/ (F_S_SUB * scipy.special.gamma(3.0 * beta_sub_mcmc - 0.5))
			)
			* KT_SUB_KEV,
			r_c=rc_sub_mcmc * XMM_PIXEL_SCALE_ARCSEC,
			beta=beta_sub_mcmc,
		)

		p_profiles_main.append(main_profile)
		p_profiles_sub.append(sub_profile)

	return compute_profile_percentiles(p_profiles_main, p_profiles_sub)

def load_M2_covariance(cov_path: Path) -> dict[str, pd.DataFrame]:
	"""Load covariance of M2 fit."""
	with open(cov_path, "rb") as f:
		M2_fit_par_dict = dill.load(f)

	return M2_fit_par_dict

def report_M2_fit_parameters(
	M2_fit_par_dict: dict[str, pd.DataFrame],
	expected_values: np.ndarray,
	nonzero: tuple[np.ndarray, ...],
	cov: np.ndarray,
) -> None:
	"""Print constrained M2 fit parameters and their 1-sigma uncertainties."""
	mydiag = np.diag(cov)
	print("M2 fit parameters and 1-sigma uncertainties:")
	for pn, expected, unc in zip(M2_fit_par_dict["par_names"][nonzero], expected_values[nonzero], np.sqrt(mydiag)):
		print(pn, ':', np.round(expected, 4), '+/-', np.round(unc, 4))

def compute_M2_pressure_profiles(
	M2_fit_par_dict: dict[str, pd.DataFrame],
	r_vals: np.ndarray,
	*,
	report_fit_parameters: bool = True,
) -> dict[str, np.ndarray]:
	expected_values = M2_fit_par_dict["parameters"]   # mean if PURELY Gaussian.

	# Check for zero rows in the covariance matrix, which would indicate unconstrained parameters/parameters that were not fit.
	rowsum = np.sum(M2_fit_par_dict["cov"], axis=0)
	#print(rowsum.shape)
	nonzero = np.where(rowsum != 0)

	# Extract the submatrix of the covariance corresponding to the nonzero rows/columns.
	cov_intermediate = M2_fit_par_dict["cov"][nonzero]
	cov = cov_intermediate[:,nonzero[0]]
	#print(cov.shape)

	# Optional: print out parameter values and errors
	if report_fit_parameters:
		report_M2_fit_parameters(M2_fit_par_dict, expected_values, nonzero, cov)

	# Generate samples from the multivariate normal distribution defined by the expected values and covariance.
	L = np.linalg.cholesky(cov)
	rng = np.random.default_rng()
	ngoodpar = cov.shape[0]
	rand_gen = rng.normal(size=(PROFILE_SAMPLE_COUNT, ngoodpar))  # Unscaled random values. (Or just scaled to unity)
	rotated = np.dot(L,rand_gen.T)  # Both scales the random values and rotates according to covariance (correlation). 
	# rotated is "Rotating about zero" so lets add the expected values back in to get the full distribution of parameter values
	generated = rotated.T + expected_values[nonzero]   # Add means back in, full distribution.

	# Optional: print out the generated parameter values for inspection.
	#print(M2_fit_par_dict['par_names'][nonzero])

	p_profiles_main = []
	p_profiles_sub = []

	for sample in generated:
		#print(sample.shape)
		pressure_main = gNFW(r_vals, sample[3], sample[4], sample[6], sample[7], sample[5])  # alpha, beta, gamma in gNFW call.
		pressure_sub = iso_beta(r_vals, sample[2], sample[0], sample[1])
		
		p_profiles_main.append(pressure_main)
		p_profiles_sub.append(pressure_sub)

	return compute_profile_percentiles(p_profiles_main, p_profiles_sub)

def format_radius_axes(
	axis: plt.Axes,
	kpc_per_arcsec: float,
	*,
	hide_leftmost_label: bool = False,
	hide_rightmost_label: bool = False,
) -> None:
	# Provenance: adapted from compare_xray_SZ_pressure_profiles.py, shared radius-axis formatting.
	# Intent: keep both panels on the same arcsec and kpc axis scheme while allowing the seam labels
	# to be suppressed where the panels touch.
	"""Apply the shared arcsec and kpc radius axis formatting."""
	bottom_tick_labels = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
	if hide_leftmost_label:
		bottom_tick_labels[0] = ""
	if hide_rightmost_label:
		bottom_tick_labels[-1] = ""

	axis.set_xlim(10, 90)
	axis.set_xlabel('r (")')
	axis.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
	axis.set_xticklabels(bottom_tick_labels)

	axis_top = axis.secondary_xaxis("top", functions=(lambda x: x * kpc_per_arcsec, lambda x: x / kpc_per_arcsec))
	axis_top.set_xlabel("r (kpc)")
	axis_top.set_xticks([100, 200, 300, 400, 500, 600, 700])
	axis_top.set_xticklabels(["100", "200", "300", "400", "500", "600", "700"])

def format_shared_pressure_axes(left_axis: plt.Axes, right_axis: plt.Axes) -> None:
	# Provenance: standalone 90% CL script, shared two-panel layout.
	# Intent: place one effective pressure scale between the panels while preserving readable ticks on
	# the outer left and right edges of the combined figure.
	"""Apply the shared log-pressure axis styling for the two-panel figure."""
	def configure_pressure_ticks(axis: plt.Axes, *, show_left: bool, show_right: bool) -> None:
		axis.set_yscale("log")
		axis.set_ylim(1e-7, 1e0)
		axis.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
		axis.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
		axis.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
		axis.tick_params(
			axis="y",
			which="both",
			left=show_left,
			labelleft=show_left,
			right=show_right,
			labelright=show_right,
		)

	left_axis.set_ylim(1e-7, 1e0)
	left_axis.set_ylabel("P (keV cm$^{-3}$)")
	left_axis.set_zorder(2)
	left_axis.patch.set_visible(False)
	configure_pressure_ticks(left_axis, show_left=True, show_right=False)

	configure_pressure_ticks(right_axis, show_left=False, show_right=False)
	right_axis.spines["left"].set_visible(False)

	center_axis = left_axis.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
	configure_pressure_ticks(center_axis, show_left=False, show_right=True)
	center_axis.tick_params(axis="y", which="both", labelright=False)

	outer_right_axis = right_axis.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
	configure_pressure_ticks(outer_right_axis, show_left=False, show_right=True)
# -------------------------------------------------------------------------------------------- #
def main() -> None:
	output_date = date.today().isoformat()

	# Read in the covariance of the M2 fit and summarize the pressure-profile uncertainty.
	m2_fit_cov = load_M2_covariance(m2_fit_cov_path)
	m2_profiles = compute_M2_pressure_profiles(m2_fit_cov, R_values)

	# Load the XMM posterior samples and compute the pressure profiles for each sample, then summarize with the median and 90% interval curves.
	xmm_samples = get_xmm_posterior_samples(xmm_state_path, PROFILE_SAMPLE_COUNT)
	xmm_profiles = compute_xmm_pressure_profiles(xmm_samples, R_values)

	# Make figure
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, gridspec_kw={"wspace": 0.0})

	ax1.set_title("Main Cluster Pressure Profile")
	ax1.loglog(R_values, m2_profiles["P_p50_main"], c="blue", label="M2")
	ax1.fill_between(R_values, m2_profiles["P_p5_main"], m2_profiles["P_p95_main"], color="blue", alpha=0.3)
	ax1.loglog(R_values, xmm_profiles["P_p50_main"], c="red", label="XMM")
	ax1.fill_between(R_values, xmm_profiles["P_p5_main"], xmm_profiles["P_p95_main"], color="red", alpha=0.3)
	ax1.legend()
	format_radius_axes(ax1, kpc_per_arcsec, hide_rightmost_label=True)

	ax2.set_title("West Subcluster Pressure Profile")
	ax2.loglog(R_values, m2_profiles["P_p50_sub"], c="blue", label="M2")
	ax2.fill_between(R_values, m2_profiles["P_p5_sub"], m2_profiles["P_p95_sub"], color="blue", alpha=0.3)
	ax2.loglog(R_values, xmm_profiles["P_p50_sub"], c="red", label="XMM")
	ax2.fill_between(R_values, xmm_profiles["P_p5_sub"], xmm_profiles["P_p95_sub"], color="red", alpha=0.3)
	ax2.legend()
	format_radius_axes(ax2, kpc_per_arcsec)
	format_shared_pressure_axes(ax1, ax2)

	fig.subplots_adjust(wspace=0.0)
	plt.savefig(location / f"plots/pressure/{output_date}_MOO_1142_main+west_pressure_profiles_w_errors.png", dpi=300)
	#plt.show()

if __name__ == "__main__":
	main()