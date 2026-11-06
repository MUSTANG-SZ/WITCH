"""Tests for the CMB + white noise map model."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from witch import noise
from witch.noise import (
    NoiseCMBMap,
    NoiseSpectrum,
    cmb_cl,
    cmb_covariance,
    cmb_covariance_multiband,
    generate_lensed_dl,
)


def test_cmb_covariance_is_symmetric_positive() -> None:
    """The covariance is symmetric with a positive, decreasing variance profile."""
    cov = cmb_covariance((8, 8), 0.5)
    assert np.allclose(cov, cov.T)
    assert cov[0, 0] > 0
    assert cov[0, 0] > cov[0, 7]


def test_apply_noise_inverts_covariance() -> None:
    """N^-1 applied to a map matches a direct solve, and the model downweights smooth modes."""
    shape = (8, 8)
    ivar = jnp.full(shape, 0.01)
    noise = NoiseCMBMap.compute(jnp.zeros(shape), ivar, 0.5 / 60)
    rng = np.random.default_rng(0)
    dat = jnp.asarray(rng.normal(size=shape))
    cov = np.linalg.inv(np.asarray(noise.ninv))
    out = np.asarray(noise.apply_noise(dat)).ravel()
    assert np.allclose(cov @ out, np.asarray(dat).ravel(), rtol=1e-3, atol=1e-3)
    flat = jnp.ones(shape)
    assert float(jnp.sum(flat * noise.apply_noise(flat))) < 1e-2 * float(
        jnp.sum(flat * flat * ivar)
    )


def test_multiband_covariance_blocks() -> None:
    """Auto blocks match the single band covariance and the joint matrix is symmetric, PSD and correlated."""
    shape, fwhms = (6, 6), [2.1, 1.4, 1.0]
    cov = cmb_covariance_multiband(shape, 1.0, fwhms)
    n = 36
    assert np.allclose(cov, cov.T)
    for i, fwhm in enumerate(fwhms):
        blk = cov[i * n : (i + 1) * n, i * n : (i + 1) * n]
        assert np.allclose(blk, cmb_covariance(shape, 1.0, fwhm_arcmin=fwhm))
    assert np.linalg.eigvalsh(cov).min() > -1e-6 * cov.max()
    # Same sky, so the cross-band variance is close to the auto variances
    assert (
        0.5 * np.sqrt(cov[0, 0] * cov[n, n])
        < cov[0, n]
        < np.sqrt(cov[0, 0] * cov[n, n])
    )


def test_noise_spectrum_warns_interim(tmp_path) -> None:
    """The interim spectrum noise model warns that it is not the ACT noise model."""
    power = np.ones((4, 4))
    np.save(tmp_path / "p.npy", power)
    dat = np.zeros((4, 4))
    with pytest.warns(UserWarning, match="interim"):
        NoiseSpectrum.compute(dat, np.ones_like(dat), 1.0, str(tmp_path / "p.npy"))


def test_cmb_sample_properties() -> None:
    """The CMB sample has more variance than white noise (on a small patch, after mean removal), zero weighted mean and no noise in empty pixels."""
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


def test_stored_spectrum_is_planck_lcdm() -> None:
    """The stored lensed TT spectrum covers l=0..10000 with the right peak and amplitude."""
    dl = noise._lensed_dl()
    assert dl.shape == (10001,)
    assert 215 <= np.argmax(dl) <= 225
    assert 5500 < dl.max() < 6000
    assert 900 < dl[2] < 1100
    assert np.all(dl >= 0)


def test_default_spectrum_does_not_need_camb(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CAMB missing the default spectrum still loads, and generating one explains why."""
    monkeypatch.setitem(sys.modules, "camb", None)
    noise._lensed_dl.cache_clear()
    cl = cmb_cl(np.arange(2.0, 100.0))
    assert np.all(np.isfinite(cl)) and np.all(cl > 0)
    with pytest.raises(ImportError, match="camb"):
        generate_lensed_dl(100)


def test_lmax_above_stored_range_raises() -> None:
    """Asking for more multipoles than are stored points to `generate_lensed_dl`."""
    noise._lensed_dl.cache_clear()
    with pytest.raises(ValueError, match="generate_lensed_dl"):
        noise._lensed_dl(20000)


def test_custom_spectrum_is_used() -> None:
    """A spectrum passed as `dl` replaces the stored one in `cmb_cl`."""
    ell = np.arange(2.0, 50.0)
    default = cmb_cl(ell, fg_dl3000=0.0, fwhm_arcmin=0.0)
    scaled = cmb_cl(ell, fg_dl3000=0.0, fwhm_arcmin=0.0, dl=2 * noise._lensed_dl())
    np.testing.assert_allclose(scaled, 2 * default)


def test_stored_spectrum_matches_camb() -> None:
    """The stored table is what CAMB gives for the Planck 2018 parameters."""
    pytest.importorskip("camb")
    fresh = generate_lensed_dl(10000)
    np.testing.assert_allclose(noise._lensed_dl(), fresh, rtol=1e-3, atol=1e-6)


def test_other_cosmology_changes_the_spectrum() -> None:
    """Generating with a different cosmology gives a different spectrum."""
    pytest.importorskip("camb")
    base = generate_lensed_dl(500)
    other = generate_lensed_dl(500, H0=75.0)
    assert not np.allclose(base[100:], other[100:], rtol=1e-2)


def _sim_map(sim_noise_kwargs: dict | None) -> np.ndarray:
    """Simulate a small map with `process_maps` and return its data."""
    from types import SimpleNamespace

    from astropy.wcs import WCS
    from jitkasi.solutions import maps
    from jitkasi.wcs import JAXWCS
    from mpi4py import MPI

    from witch import fitter

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    wcs.wcs.crval = [44.6, 13.3]
    wcs.wcs.crpix = [4.5, 4.5]
    wcs.wcs.cdelt = [-0.5 / 60, 0.5 / 60]
    shape = (8, 8)
    imap = maps.WCSMap(
        "t",
        jnp.zeros(shape),
        MPI.COMM_WORLD,
        JAXWCS.from_astropy(wcs),
        ivar=jnp.full(shape, 0.0053),
    )
    dataset = SimpleNamespace(
        name="act",
        datavec=[imap],
        info={},
        noise_class=NoiseCMBMap,
        noise_args=("self.ivar", "self.wcs.cdelt[0]"),
        noise_kwargs={"fwhm_arcmin": 1.6},
    )
    metamodel = SimpleNamespace(
        get_dataset_ind=lambda name: 0,
        model_proj=lambda ind, i: jnp.zeros(shape),
    )
    cfg = {"sim": True, "wnoise": False, "sim_seed": 3}
    out = fitter.process_maps(cfg, dataset, metamodel, sim_noise_kwargs)
    return np.asarray(out[0].data)


def test_sim_noise_kwargs_override_the_sim_only() -> None:
    """`sim_noise_kwargs` changes the simulated noise, an empty one leaves it as it was."""
    base = _sim_map(None)
    assert np.array_equal(base, _sim_map({}))
    loud = _sim_map({"dl": 100 * noise._lensed_dl()})
    assert loud.std() > 3 * base.std()
