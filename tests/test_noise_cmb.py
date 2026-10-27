"""Tests for the CMB + white noise map model."""

import jax.numpy as jnp
import numpy as np
import pytest

from witch.noise import (
    NoiseCMBMap,
    NoiseSpectrum,
    cmb_covariance,
    cmb_covariance_multiband,
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
