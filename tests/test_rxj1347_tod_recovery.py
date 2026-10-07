import os
import shutil
import subprocess
import sys
from pathlib import Path

import dill
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RXJ1347_CONFIG = REPO_ROOT / "unit_tests" / "RXJ1347_a10.yaml"
TOD_ROOT = Path(os.environ.get("WITCH_DATROOT", Path.home())) / "RXJ1347" / "mustang2"
RXJ1347_TODS = sorted(TOD_ROOT.glob("Signal_TOD*.fits"))


def test_rxj1347_simulated_a10_gaussian_recovers_input_parameters(tmp_path):
    if len(RXJ1347_TODS) < 4:
        pytest.skip("Requires at least four RXJ1347 TOD FITS files")
    num_tods = int(os.environ.get("RXJ1347_TEST_NTODS", "4"))
    if num_tods < 4 or num_tods > len(RXJ1347_TODS):
        pytest.skip(f"Requires between 4 and {len(RXJ1347_TODS)} RXJ1347 TODs")
    pytest.importorskip("minkasi")
    pytest.importorskip("jitkasi")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("Requires mpirun for the two-rank fitting regression")

    test_config = tmp_path / "rxj1347_simulated_fit.yaml"
    test_config.write_text(
        yaml.safe_dump(
            {
                "base": str(RXJ1347_CONFIG),
                "sim": True,
                "par_offset": 1.1,
                "paths": {
                    "outroot": str(tmp_path / "output"),
                    "subdir": "rxj1347_tod_recovery",
                },
                "datasets": {"mustang2": {"ntods": num_tods}},
                "fitting": {"maxiter": 30, "chitol": 1e-5},
            }
        )
    )

    result = subprocess.run(
        [
            mpirun,
            "-n",
            "2",
            sys.executable,
            "-c",
            "from witch.fitter import main; main()",
            str(test_config),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    result_files = list((tmp_path / "output").rglob("par_results_final_fit.dill"))
    assert len(result_files) == 1
    with result_files[0].open("rb") as result_file:
        fit_result = dill.load(result_file)

    fitted = dict(zip(fit_result["par_names"], fit_result["parameters"], strict=True))
    injected = {"m500": 1.5e15, "alpha": 1.551, "amp_g": 0.002}
    for parameter, true_value in injected.items():
        relative_error = abs(fitted[parameter] - true_value) / abs(true_value)
        assert relative_error <= 0.05, (
            f"{parameter} recovered as {fitted[parameter]:.8g}; "
            f"injected value is {true_value:.8g} "
            f"(relative error {relative_error:.2%})"
        )
