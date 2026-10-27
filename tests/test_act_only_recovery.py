"""ACT-only simulated MOOJ1142 fit: injected parameters are recovered and the CMB noise sampler is sane."""

import os
import shutil
import subprocess
from pathlib import Path

import dill
import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "unit_tests" / "act" / "act_sim.yaml"
DATA_ROOT = Path(os.environ.get("WITCH_DATROOT", Path.home())) / "MOOJ1142"
INJECTED = {"m500": 5e14, "alpha": 1.551, "amp_act": 50.0}


@pytest.fixture(scope="module")
def fit(tmp_path_factory) -> dict[str, tuple[float, float]]:
    """Run the white noise ACT-only sim once and read back the fitted parameters and errors."""
    if not list((DATA_ROOT / "act").glob("*_map.fits")):
        pytest.skip("Requires the MOOJ1142 ACT map")
    pytest.importorskip("jitkasi")
    if shutil.which("witcher") is None:
        pytest.skip("Requires witcher")
    workdir = tmp_path_factory.mktemp("act_only")
    wrapper = workdir / "wrapper.yaml"
    wrapper.write_text(f'base: "{CONFIG}"\nmap: False\nmodel_map: False\n')
    env = os.environ | {"WITCH_OUTROOT": str(workdir / "output")}
    result = subprocess.run(
        [shutil.which("witcher"), str(wrapper)],
        check=False,
        capture_output=True,
        env=env,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    files = list((workdir / "output").rglob("par_results_final_fit.dill"))
    assert len(files) == 1
    with files[0].open("rb") as f:
        res = dill.load(f)
    return {
        name: (float(par), float(err))
        for name, par, err, fitted in zip(
            res["par_names"], res["parameters"], res["errs"], res["to_fit"], strict=True
        )
        if fitted
    }


def test_parameters_recovered(fit) -> None:
    """Each parameter has a finite, non-tiny error and lies within 3 sigma of the input."""
    assert set(fit) == set(INJECTED)
    for name, (par, err) in fit.items():
        assert np.isfinite(par) and np.isfinite(err)
        # Errors a thousand times below the input would be the damped-curvature symptom
        assert err > 1e-3 * abs(
            INJECTED[name]
        ), f"{name} error {err} is implausibly small"
        assert abs(par - INJECTED[name]) < 3 * err, f"{name}: {par} +- {err}"


def test_cmb_sample_properties() -> None:
    """The CMB sample has more variance than white noise (on a small patch, after mean removal), zero weighted mean and no noise in empty pixels."""
    pytest.importorskip("camb")
    from witch.noise import NoiseCMBMap

    shape = (12, 12)
    ivar = jnp.full(shape, 0.0053).at[0, :].set(0.0)
    noise = np.asarray(
        NoiseCMBMap.sample(jax.random.key(1), jnp.zeros(shape), ivar, 0.5 / 60)
    )
    assert noise.shape == shape
    assert np.all(noise[0] == 0)
    assert abs(np.sum(noise * np.asarray(ivar)) / np.sum(ivar)) < 1e-3
    assert noise[1:].std() > 1.2 * np.sqrt(1 / 0.0053)
