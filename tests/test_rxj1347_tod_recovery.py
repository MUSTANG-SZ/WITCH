import os
import shutil
import subprocess
from pathlib import Path

import dill
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RXJ1347_CONFIG = REPO_ROOT / "unit_tests" / "RXJ1347_tod_recovery.yaml"
RXJ1347_MH_CONFIG = REPO_ROOT / "unit_tests" / "RXJ1347_mh.yaml"
TOD_ROOT = Path(os.environ.get("WITCH_DATROOT", Path.home())) / "RXJ1347" / "mustang2"
RXJ1347_TODS = sorted(TOD_ROOT.glob("Signal_TOD*.fits"))


INJECTED = {"m500": 1.5e15, "alpha": 1.551, "amp_g": 0.002}


def _run_witcher(config, outdir, timeout=1800):
    """
    Run witcher on a config with 4 MPI ranks, skipping if requirements are missing.

    Parameters
    ----------
    config : Path
        Config to run.
    outdir : Path
        Output root, passed through WITCH_OUTROOT.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    None
    """
    if len(RXJ1347_TODS) < 4:
        pytest.skip("Requires at least four RXJ1347 TOD FITS files")
    pytest.importorskip("minkasi")
    pytest.importorskip("jitkasi")
    mpirun = shutil.which("mpirun")
    witcher = shutil.which("witcher")
    if mpirun is None or witcher is None:
        pytest.skip("Requires mpirun and witcher for the four-rank fitting regression")

    env = os.environ.copy()
    env["WITCH_OUTROOT"] = str(outdir)
    print(f"Launching {config.name} with 4 MPI ranks", flush=True)
    result = subprocess.run(
        [mpirun, "-n", "4", witcher, str(config)],
        check=False,
        capture_output=True,
        env=env,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr.decode()[-2000:]


def _check_recovery(recovered, label, tol=0.05):
    """
    Assert that recovered values are within a relative tolerance of the injected ones.

    Parameters
    ----------
    recovered : dict
        Parameter name to recovered value.
    label : str
        Label for the printed report.
    tol : float
        Allowed relative error.

    Returns
    -------
    None
    """
    failures = []
    for parameter, true_value in INJECTED.items():
        relative_error = abs(recovered[parameter] - true_value) / abs(true_value)
        passed = relative_error <= tol
        print(
            f"{'PASS' if passed else 'FAIL'} [{label}] {parameter}: "
            f"injected={true_value:.8g}, recovered={recovered[parameter]:.8g}, "
            f"relative_error={relative_error:.2%} (< {tol:.0%}: {passed})",
            flush=True,
        )
        if not passed:
            failures.append(parameter)
    assert not failures, f"[{label}] parameters outside {tol:.0%} tolerance: {failures}"


def test_rxj1347_simulated_a10_gaussian_recovers_input_parameters(tmp_path):
    _run_witcher(RXJ1347_CONFIG, tmp_path / "output")

    result_files = list((tmp_path / "output").rglob("par_results_final_fit.dill"))
    assert len(result_files) == 1
    with result_files[0].open("rb") as result_file:
        fit_result = dill.load(result_file)

    fitted = dict(zip(fit_result["par_names"], fit_result["parameters"], strict=True))
    _check_recovery(fitted, "LM")


def test_rxj1347_mcmc_recovers_input_parameters(tmp_path):
    """Runs the LM fit followed by MH MCMC and checks the posterior against the inputs."""
    _run_witcher(RXJ1347_MH_CONFIG, tmp_path / "output", timeout=3600)

    sample_files = list((tmp_path / "output").rglob("samples_mcmc.npz"))
    assert len(sample_files) == 1
    with np.load(sample_files[0]) as npz:
        samples = npz["samples"]
        fit_names = npz["par_names"][npz["to_fit"]]
    assert samples.shape[1] == len(fit_names)
    assert np.all(np.isfinite(samples))

    medians = dict(zip(fit_names, np.median(samples, axis=0), strict=True))
    stds = dict(zip(fit_names, np.std(samples, axis=0), strict=True))
    for parameter in INJECTED:
        print(f"{parameter}: median={medians[parameter]:.6g} +/- {stds[parameter]:.3g}")
    _check_recovery(medians, "MCMC")
