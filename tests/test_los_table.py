"""
Check that the 1D line of sight table used for spherical structures
agrees with the full 3D integration, for both the model and its gradient.
"""

import jax
import jax.numpy as jnp
import pytest

from witch import core, grid

A10 = [3.3, -2.7, 4.0, 0.3, 8.4, 1.18, 1.5e15, 0.31, 1.55, 5.49, 0.97]
CASES = {
    "a10": ({"a10": 1}, A10),
    "gnfw": (
        {"gnfw": 1},
        [3.3, -2.7, 4.0, 1.0, 8.4, 1.18, 1.5e15, 0.31, 1.55, 5.49, 0.97],
    ),
    "gnfw_rs": ({"gnfw_rs": 1}, [3.3, -2.7, 4.0, 8.4e-3, 60.0, 0.31, 1.55, 5.49, 0.97]),
    "sph_isobeta": ({"sph_isobeta": 1}, [3.3, -2.7, 4.0, 40.0, 0.3, 0.7, 1e-3]),
    # Non-spherical stage 0 structure is still integrated in 3D and summed
    "a10+isobeta": (
        {"isobeta": 1, "a10": 1},
        [1.0, 1.0, 4.0, 20.0, 30.0, 35.0, 0.4, 0.7, 1e-3] + A10,
    ),
}


def _run(counts, pars, fast, xyz):
    """
    Evaluate `core.model` and `core.model_grad` with the LOS table on or off.

    Parameters
    ----------
    counts : dict
        Number of each structure by name.
    pars : list[float]
        Parameters in the order of `core.ORDER`.
    fast : bool
        Whether to use the 1D LOS table.
    xyz : tuple
        Grid to evaluate on.

    Returns
    -------
    model : jax.Array
        Model on the grid.
    grad : jax.Array
        Gradient of the model with respect to every parameter.
    """
    n_structs = tuple(counts.get(name, 0) for name in core.ORDER)
    n_rbins = (0,) * len(core.ORDER)
    argnums = tuple(range(core.ARGNUM_SHIFT, core.ARGNUM_SHIFT + len(pars)))
    old = core.FAST_LOS
    core.FAST_LOS = fast
    try:
        jax.clear_caches()
        return core.model_grad.__wrapped__(
            xyz, n_structs, n_rbins, 1.0, (True,) * 4, argnums, *pars
        )
    finally:
        core.FAST_LOS = old
        jax.clear_caches()


@pytest.mark.parametrize("name", list(CASES))
def test_los_table_matches_3d(name):
    xyz = grid.make_grid(90.0, 1.0, 1.0, 1.0, 206.8, -11.8)
    counts, pars = CASES[name]
    ref, ref_grad = _run(counts, pars, False, xyz)
    fast, fast_grad = _run(counts, pars, True, xyz)

    assert jnp.abs(fast - ref).max() / jnp.abs(ref).max() < 1e-4
    scale = jnp.linalg.norm(ref_grad, axis=(1, 2)).max()
    for i in range(len(pars)):
        # Skip parameters with no sensitivity (e.g. theta of a spherical profile)
        if jnp.linalg.norm(ref_grad[i]) < 1e-6 * scale:
            continue
        err = jnp.linalg.norm(fast_grad[i] - ref_grad[i]) / jnp.linalg.norm(ref_grad[i])
        assert err < 2e-3, f"parameter {i}: {err}"
