"""Simulated A399-A401 isobeta + mesa fit on a white noise Compton-y map: injected parameters are recovered and the bridge is detected."""

import os
import shutil
import subprocess
from pathlib import Path

import dill
import numpy as np
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "unit_tests" / "a399_401" / "a399_401_sim.yaml"
SIGMA_Y = 5.2e-6  # Pixel rms of the ACT+Planck y map in source free corners
MIN_BRIDGE_SNR = 5.0
# Header of the ACT+Planck CAR cutout (80x80 pixels of 1')
HEADER = {
    "CRPIX1": 2717.25,
    "CRPIX2": -757.25,
    "CDELT1": -1 / 60,
    "CDELT2": 1 / 60,
    "CUNIT1": "deg",
    "CUNIT2": "deg",
    "CTYPE1": "RA---CAR",
    "CTYPE2": "DEC--CAR",
    "CRVAL1": 0.0,
    "CRVAL2": 0.0,
    "RADESYS": "ICRS",
}


def _write_maps(root: Path, sigma: float = SIGMA_Y, npix: int = 80) -> None:
    """
    Write a blank map and a constant ivar map with the cutout's WCS to `root`.

    Parameters
    ----------
    root : Path
        Directory to write `A399_401_y_map.fits` and `A399_401_y_ivar.fits` to.
    sigma : float, default: SIGMA_Y
        White noise rms per pixel in Compton y, sets the ivar.
    npix : int, default: 80
        Number of pixels on each side.
    """
    root.mkdir(parents=True)
    for name, value in (("map", 0.0), ("ivar", sigma**-2)):
        hdu = fits.PrimaryHDU(data=np.full((npix, npix), value))
        hdu.header.update(HEADER)
        hdu.writeto(root / f"A399_401_y_{name}.fits")


def _injected(cfg: dict) -> np.ndarray:
    """
    Injected parameter vector, in the metamodel's order (a399, a401, bridge; 9 + 9 + 6 parameters).

    Parameters
    ----------
    cfg : dict
        Merged config.

    Returns
    -------
    values : np.ndarray
        The `value` of every parameter in the config.
    """
    return np.array(
        [
            float(par["value"])
            for model in ("a399", "a401", "bridge")
            for struct in cfg[model]["structures"].values()
            for par in struct["parameters"].values()
        ]
    )


@pytest.fixture(scope="module")
def fit(tmp_path_factory) -> dict[str, np.ndarray]:
    """Run the sim once and return the fitted parameters, errors, flags and names with the injected values."""
    pytest.importorskip("jitkasi")
    if shutil.which("witcher") is None:
        pytest.skip("Requires witcher")
    from witch.fitter import load_config

    workdir = tmp_path_factory.mktemp("a399_401_sim")
    _write_maps(workdir / "data" / "A399_401_sim")
    env = os.environ | {
        "WITCH_DATROOT": str(workdir / "data"),
        "WITCH_OUTROOT": str(workdir / "output"),
    }
    result = subprocess.run(
        [shutil.which("witcher"), str(CONFIG), "-cpu", "True"],
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
        "par": np.asarray(res["parameters"]),
        "err": np.asarray(res["errs"]),
        "fitted": np.asarray(res["to_fit"], dtype=bool),
        "names": np.asarray(res["par_names"]),
        "truth": _injected(load_config({}, CONFIG)),
    }


def test_parameters_recovered(fit) -> None:
    """Every free parameter has a finite, non-tiny error and lies within 3 sigma of the input."""
    assert fit["truth"].shape == fit["par"].shape
    for i in np.flatnonzero(fit["fitted"]):
        par, err, truth = fit["par"][i], fit["err"][i], fit["truth"][i]
        label = f"{i} ({fit['names'][i]})"
        assert np.isfinite(par) and np.isfinite(err), label
        # Errors a thousand times below the input would be the damped-curvature symptom
        assert err > 1e-3 * abs(truth), f"{label} error {err} is implausibly small"
        assert abs(par - truth) < 3 * err, f"{label}: {par} +- {err}, input {truth}"


def test_bridge_detected(fit) -> None:
    """The mesa amplitude is recovered with a significance above `MIN_BRIDGE_SNR`."""
    amp, err = fit["par"][23], fit["err"][23]
    assert amp / err > MIN_BRIDGE_SNR, f"bridge amp {amp} +- {err}"
