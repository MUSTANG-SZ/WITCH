"""
model_on_skymap (the unfiltered "truth" map) must integrate the model along the fit grid's line of sight.

Regression test: it used to swap in a fixed 2" LOS axis while the LOS integral kept weighting samples by the fit's
coords.dz, so with dz = 4" every truth map was 2x too deep.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.wcs import WCS

pytest.importorskip("minkasi")
pytest.importorskip("jitkasi")

from witch import grid  # noqa: E402
from witch.external.minkasi.funcs import model_on_skymap  # noqa: E402

DZ = 4.0  # fit coords.dz, arcsec
R_MAP = 60.0  # fit grid half size, arcsec


def _skymap(pix: float, n: int) -> SimpleNamespace:
    """
    A minimal stand-in for a minkasi SkyMap: a CAR WCS and an empty map.

    Parameters
    ----------
    pix : float
        Pixel size, arcsec.
    n : int
        Number of pixels per side.

    Returns
    -------
    SimpleNamespace
        With `wcs` and `map` attributes.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    wcs.wcs.crval = [175.7, 15.45]
    wcs.wcs.crpix = [n / 2, n / 2]
    wcs.wcs.cdelt = [-pix / 3600, pix / 3600]
    return SimpleNamespace(wcs=wcs, map=np.empty((n, n)))


def _metamodel() -> SimpleNamespace:
    """
    A stub metamodel with one model on a fit grid (coords.dz = DZ) whose model_grid is the LOS integral of a unit
    pressure, weighted by DZ like core.model: the LOS length of the fit grid wherever it is evaluated.

    Returns
    -------
    SimpleNamespace
        With `models` and `model_grid`.
    """
    model = SimpleNamespace(
        xyz=grid.make_grid(R_MAP, 2.0, 2.0, DZ, 175.7 * 3600, 15.45 * 3600), dz=DZ
    )

    def model_grid(dataset_ind: int) -> jnp.ndarray:
        x, y, z = model.xyz[:3]
        pressure = jnp.ones((x.shape[0], y.shape[1], z.shape[2]))
        return jnp.trapezoid(pressure, dx=model.dz, axis=-1)

    return SimpleNamespace(models=[model], model_grid=model_grid)


@pytest.mark.parametrize("pix, dr", [(4.0, None), (3.0, 4.0)])
def test_truth_uses_fit_los(pix: float, dr: float | None):
    """
    The truth map equals the fit grid's LOS integral, directly and when resampled from dr onto other pixels, and
    the models get their fit grids back.
    """
    metamodel = _metamodel()
    fit_xyz = metamodel.models[0].xyz
    expected = float(metamodel.model_grid(0)[0, 0])
    truth = model_on_skymap(metamodel, 0, _skymap(pix, 32), dr)
    assert truth.shape == (32, 32)
    np.testing.assert_allclose(truth, expected, rtol=1e-6)
    for restored, orig in zip(metamodel.models[0].xyz, fit_xyz):
        np.testing.assert_array_equal(restored, orig)
