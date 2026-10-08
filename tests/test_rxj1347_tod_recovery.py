import os
import shutil
import subprocess
from pathlib import Path

import dill
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RXJ1347_CONFIG = REPO_ROOT / "unit_tests" / "RXJ1347_tod_recovery.yaml"
TOD_ROOT = Path(os.environ.get("WITCH_DATROOT", Path.home())) / "RXJ1347" / "mustang2"
RXJ1347_TODS = sorted(TOD_ROOT.glob("Signal_TOD*.fits"))


def test_rxj1347_simulated_a10_gaussian_recovers_input_parameters(tmp_path):
    if len(RXJ1347_TODS) < 4:
        pytest.skip("Requires at least four RXJ1347 TOD FITS files")
    pytest.importorskip("minkasi")
    pytest.importorskip("jitkasi")
    mpirun = shutil.which("mpirun")
    witcher = shutil.which("witcher")
    if mpirun is None or witcher is None:
        pytest.skip("Requires mpirun and witcher for the four-rank fitting regression")

    env = os.environ.copy()
    env["WITCH_OUTROOT"] = str(tmp_path / "output")
    print("Launching RXJ1347 fit with 4 MPI ranks", flush=True)
    result = subprocess.run(
        [
            mpirun,
            "-n",
            "4",
            witcher,
            str(RXJ1347_CONFIG),
        ],
        check=False,
        capture_output=True,
        env=env,
        timeout=1800,
    )
    assert result.returncode == 0

    result_files = list((tmp_path / "output").rglob("par_results_final_fit.dill"))
    assert len(result_files) == 1
    with result_files[0].open("rb") as result_file:
        fit_result = dill.load(result_file)

    fitted = dict(zip(fit_result["par_names"], fit_result["parameters"], strict=True))
    injected = {"m500": 1.5e15, "alpha": 1.551, "amp_g": 0.002}
    failures = []
    for parameter, true_value in injected.items():
        recovered_value = fitted[parameter]
        relative_error = abs(recovered_value - true_value) / abs(true_value)
        passed = relative_error <= 0.05
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} {parameter}: injected={true_value:.8g}, "
            f"recovered={recovered_value:.8g}, "
            f"relative_error={relative_error:.2%} (< 5%: {passed})",
            flush=True,
        )
        if not passed:
            failures.append(parameter)

    assert not failures, f"Parameters outside 5% recovery tolerance: {failures}"
