"""Simulated MOOJ1142 fits of M2 alone and M2+ACT, checking recovery and the gain from ACT."""

import os
import shutil
import subprocess
from pathlib import Path

import dill
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
JOINT_DIR = REPO_ROOT / "unit_tests" / "joint"
DATA_ROOT = Path(os.environ.get("WITCH_DATROOT", Path.home())) / "MOOJ1142"
NTODS = 8
INJECTED = {
    "dx_1": 3.0,
    "dy_1": -3.0,
    "m500": 5e14,
    "alpha": 1.551,
    "beta": 5.4905,
    "amp_sh": 1.5,
    "kr_sh": 1.0,
    "dx_g": 20.0,
    "dy_g": -15.0,
    "amp_m2": 1e-3,
    "amp_act": 50.0,
}
# Mean ACT-to-M2 error ratio on alpha was ~0.6 in the reference run
ALPHA_SIGMA_RATIO = 0.8
SHARED_SIGMA_RATIO = 1.05


def _launcher(nranks: int) -> list[str]:
    """
    Command prefix that starts `nranks` MPI tasks, using srun inside a SLURM job.

    Parameters
    ----------
    nranks : int
        Number of MPI tasks.

    Returns
    -------
    list[str]
        The launcher and its arguments, or an empty list if none is available.
    """
    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        return ["srun", "-n", str(nranks)]
    if shutil.which("mpirun"):
        return ["mpirun", "--oversubscribe", "-n", str(nranks)]
    return []


def _run_fit(config: str, nranks: int, workdir: Path) -> dict[str, tuple[float, float]]:
    """
    Run `witcher` on one of the joint configs and read back the fitted parameters.

    Parameters
    ----------
    config : str
        Name of the yaml file in `unit_tests/joint`.
    nranks : int
        Number of MPI tasks to launch.
    workdir : Path
        Directory for the wrapper config and the outputs.

    Returns
    -------
    dict[str, tuple[float, float]]
        Map from the name of each fitted parameter to its value and error.
    """
    wrapper = workdir / "wrapper.yaml"
    wrapper.write_text(f'base: "{JOINT_DIR / config}"\nmap: False\nmodel_map: False\n')
    env = os.environ.copy()
    env["WITCH_OUTROOT"] = str(workdir / "output")
    result = subprocess.run(
        [*_launcher(nranks), shutil.which("witcher"), str(wrapper)],
        check=False,
        capture_output=True,
        env=env,
        timeout=3600,
    )
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    files = list((workdir / "output").rglob("par_results_final_fit.dill"))
    assert len(files) == 1
    with files[0].open("rb") as f:
        res = dill.load(f)
    return {
        name: (float(par), float(err))
        for name, par, err, fit in zip(
            res["par_names"], res["parameters"], res["errs"], res["to_fit"], strict=True
        )
        if fit
    }


@pytest.fixture(scope="module")
def fits(tmp_path_factory) -> dict[str, dict[str, tuple[float, float]]]:
    """Fit the M2-only and the joint ACT+M2 simulations once each."""
    if len(list((DATA_ROOT / "mustang2").glob("Signal_TOD*.fits"))) < NTODS:
        pytest.skip(f"Requires {NTODS} MOOJ1142 TOD FITS files")
    if len(list((DATA_ROOT / "act").glob("*_map.fits"))) < 1:
        pytest.skip("Requires the MOOJ1142 ACT map")
    pytest.importorskip("minkasi")
    pytest.importorskip("jitkasi")
    if shutil.which("witcher") is None or not _launcher(1):
        pytest.skip("Requires witcher and mpirun or srun")
    out = {}
    for key, config, nranks in (
        ("m2", "joint_sim_m2.yaml", NTODS),
        ("joint", "joint_sim_act_m2.yaml", NTODS + 1),
    ):
        workdir = tmp_path_factory.mktemp(key)
        out[key] = _run_fit(config, nranks, workdir)
    _print_table(out)
    return out


def _print_table(fits: dict[str, dict[str, tuple[float, float]]]) -> None:
    """Print the injected values, both fits and the error ratios."""
    print(f"\n{'par':8}{'truth':>10}{'M2':>22}{'joint':>22}{'sig ratio':>10}")
    for name, (par, err) in fits["joint"].items():
        m2 = fits["m2"].get(name)
        m2_str = f"{m2[0]:12.4g} ±{m2[1]:.2g}" if m2 else ""
        ratio = f"{err / m2[1]:10.2f}" if m2 else ""
        print(
            f"{name:8}{INJECTED[name]:10.4g}{m2_str:>22}{par:12.4g} ±{err:.2g}{ratio}"
        )


@pytest.mark.parametrize("key", ["m2", "joint"])
def test_parameters_recovered(fits, key):
    """Each fitted parameter is finite, has an error, and is within max(3 sigma, 5%) of the input."""
    failures = []
    for name, (par, err) in fits[key].items():
        tol = max(3 * err, 0.05 * abs(INJECTED[name]))
        if not (np.isfinite(par) and np.isfinite(err) and err > 0):
            failures.append(f"{name}: non-finite fit or error ({par}, {err})")
        elif abs(par - INJECTED[name]) > tol:
            failures.append(f"{name}: {par:.5g} ± {err:.2g} vs {INJECTED[name]}")
    assert not failures, failures


def test_act_tightens_errors(fits):
    """Adding ACT shrinks the error on alpha and does not inflate any shared error."""
    m2, joint = fits["m2"], fits["joint"]
    assert joint["alpha"][1] < ALPHA_SIGMA_RATIO * m2["alpha"][1]
    worse = [n for n in m2 if joint[n][1] > SHARED_SIGMA_RATIO * m2[n][1]]
    assert not worse, worse
