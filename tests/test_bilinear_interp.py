import jax
import jax.numpy as jnp
import numpy as np
import pytest

from witch import utils as wu


def _reference(x, y, xp, yp, fp):
    """The original bilinear interpolation, using searchsorted."""
    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)
    f_xy1 = ((xp[ix] - x) * fp[ix - 1, iy - 1] + (x - xp[ix - 1]) * fp[ix, iy - 1]) / (
        xp[ix] - xp[ix - 1]
    )
    f_xy2 = ((xp[ix] - x) * fp[ix - 1, iy] + (x - xp[ix - 1]) * fp[ix, iy]) / (
        xp[ix] - xp[ix - 1]
    )
    f = ((yp[iy] - y) * f_xy1 + (y - yp[iy - 1]) * f_xy2) / (yp[iy] - yp[iy - 1])
    oob = (x < xp[0]) + (x > xp[-1]) + (y < yp[0]) + (y > yp[-1])
    return jnp.where(oob, 0.0, f)


@pytest.mark.parametrize("warp", [0.0, 0.4])
def test_bilinear_interp_matches_searchsorted(warp):
    rng = np.random.default_rng(0)
    n = 80
    u = np.linspace(-1, 1, n)
    # Non-uniform by less than one cell (like a projected WCS)
    xp = jnp.array(50 * u + warp * 50 / n * np.sin(3 * u))
    yp = jnp.array(40 * u - 0.3 + warp * 40 / n * u**2)
    fp = jnp.array(rng.normal(size=(n, n)))
    x = rng.uniform(-60, 60, 5000)
    y = rng.uniform(-50, 50, 5000)
    # Exact grid nodes and edges are the hard cases
    x[:n] = np.asarray(xp)
    y[:n] = np.asarray(yp)
    x[n : n + 2], y[n : n + 2] = (
        [float(xp[0]), float(xp[-1])],
        [float(yp[-1]), float(yp[0])],
    )
    x, y = jnp.array(x), jnp.array(y)

    np.testing.assert_allclose(
        wu.bilinear_interp(x, y, xp, yp, fp), _reference(x, y, xp, yp, fp), atol=1e-12
    )
    # Derivatives with respect to the map must match too
    g = jax.grad(lambda f: wu.bilinear_interp(x, y, xp, yp, f).sum())(fp)
    g_ref = jax.grad(lambda f: _reference(x, y, xp, yp, f).sum())(fp)
    np.testing.assert_allclose(g, g_ref, atol=1e-12)
