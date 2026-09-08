"""
Pytest configuration and shared fixtures for WITCH tests.
Reference outputs for structures without closed forms are stored in pytest cache and tracked in the repository so anyone can pull and run tests against a known-good state.
"""

import numpy as np
import pytest

from witch import grid, structure

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
