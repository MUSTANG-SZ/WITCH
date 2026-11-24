"""
Pytest configuration and shared fixtures for WITCH tests.
Reference outputs for structures without closed forms are stored in pytest cache and tracked in the repository so anyone can pull and run tests against a known-good state.
"""

import jax.numpy as jnp
import jitkasi.noise as jn
import numpy as np
import pytest
from jitkasi.tod import TOD, TODVec
from mpi4py import MPI

from witch import grid, structure
from witch.containers import MetaModel
from witch.dataset import DataSetData
from witch.objective import chisq_objective
from witch.utils import NullComm, rad_to_arcsec

# ---------------------------------------------------------------------------
# Shared grid fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_grid():
    """
    Standard grid for most tests.
    """
    r_map = 60.0
    dr = 2.0
    dz = 2.0
    xyz = grid.make_grid(r_map, dr, dr, dz, 0.0, 0.0)
    return xyz, dz


@pytest.fixture
def wide_grid():
    """
    Wide grid needed for slowly-decaying profiles (power-law tails).
    """
    r_map = 180.0
    dr = 2.0
    dz = 2.0
    xyz = grid.make_grid(r_map, dr, dr, dz, 0.0, 0.0)
    return xyz, dz


# ---------------------------------------------------------------------------
# Reference output cache fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gnfw_reference(request):
    """
    Reference output for gnfw model.]
    """
    cache_key = "structure/gnfw_reference"
    cached = request.config.cache.get(cache_key, None)

    if cached is not None:
        return np.array(cached)

    # Compute reference output
    r_map = 60.0
    dr = 2.0
    dz = 2.0
    xyz = grid.make_grid(r_map, dr, dr, dz, 0.0, 0.0)

    model = structure.gnfw(
        dx=0.0,
        dy=0.0,
        dz=0.0,
        r=1.0,
        P0=8.403,
        c500=1.177,
        m500=1e15,
        gamma=0.3081,
        alpha=1.0510,
        beta=5.4905,
        z=0.5,
        xyz=xyz,
    )

    # Integrate along z to get 2D map
    model_2d = np.array(np.sum(model, axis=2) * dz)

    request.config.cache.set(cache_key, model_2d.tolist())
    return model_2d


@pytest.fixture(scope="session")
def a10_reference(request):
    """
    Reference output for a10 model.
    """
    cache_key = "structure/a10_reference"
    cached = request.config.cache.get(cache_key, None)

    if cached is not None:
        return np.array(cached)

    # Compute reference output using Arnaud 2010 best-fit parameters
    r_map = 60.0
    dr = 2.0
    dz = 2.0
    xyz = grid.make_grid(r_map, dr, dr, dz, 0.0, 0.0)

    model = structure.a10(
        dx=0.0,
        dy=0.0,
        dz=0.0,
        theta=1.0,
        P0=8.403,
        c500=1.177,
        m500=1e15,
        gamma=0.3081,
        alpha=1.0510,
        beta=5.4905,
        z=0.5,
        xyz=xyz,
    )

    model_2d = np.array(np.sum(model, axis=2) * dz)

    request.config.cache.set(cache_key, model_2d.tolist())
    return model_2d


# ---------------------------------------------------------------------------
# Synthetic TOD metamodel
# ---------------------------------------------------------------------------


def correlated_noise(rng, ndet, nsamp):
    """Common mode and per detector random walks plus white noise."""
    common = np.cumsum(rng.standard_normal(nsamp))
    drift = np.cumsum(rng.standard_normal((ndet, nsamp)), axis=-1) * 0.1
    gains = 1 + 0.1 * rng.standard_normal((ndet, 1))
    return gains * common + drift + 0.05 * rng.standard_normal((ndet, nsamp))


def _tod_metamodel(ntod=2, comm=None):
    """
    Small gaussian model and TODs with correlated noise, of odd and even lengths,
    with `jitkasi.noise.NoiseSmoothedSVD` noise and `chisq_objective`.
    """
    rng = np.random.default_rng(3)
    cfg = {
        "coords": {"r_map": 60.0, "dr": 2.0, "dz": 2.0, "x0": 0.0, "y0": 0.0},
        "model": {
            "structures": {
                "ps": {
                    "structure": "gaussian",
                    "parameters": {
                        "dx": {"value": 3.0, "to_fit": True},
                        "dy": {"value": -2.0},
                        "sigma": {"value": 8.0, "to_fit": True},
                        "amp": {"value": 1.0, "to_fit": True},
                    },
                }
            }
        },
    }
    tods = []
    for i in range(ntod):
        ndet, nsamp = 10, 301 - i
        t = np.linspace(0, 6 * np.pi, nsamp)
        offs = rng.uniform(-20, 20, (ndet, 2))
        x = (40 * np.sin(t)[None, :] + offs[:, :1]) / rad_to_arcsec
        y = (40 * np.sin(1.3 * t)[None, :] + offs[:, 1:]) / rad_to_arcsec
        dat = correlated_noise(rng, ndet, nsamp)
        tod = TOD(jnp.asarray(dat), jnp.asarray(x), jnp.asarray(y))
        tod.compute_noise(jn.NoiseSmoothedSVD, None, fwhm=10.0)
        tods.append(tod)
    dataset = DataSetData(
        "test",
        "tod",
        chisq_objective,
        TODVec(tods, MPI.COMM_WORLD),
        (),
        jn.NoiseSmoothedSVD,
        (),
        {"fwhm": 10.0},
    )
    return MetaModel.from_config(NullComm() if comm is None else comm, cfg, (dataset,))


@pytest.fixture
def tod_metamodel():
    """
    Factory for a small real `MetaModel` on synthetic TODs with correlated noise,
    call as `tod_metamodel(ntod=2, comm=None)`.
    """
    return _tod_metamodel
