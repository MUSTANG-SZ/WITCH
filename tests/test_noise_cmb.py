"""Tests for the CMB + white noise map model."""

import jax.numpy as jnp
import numpy as np

from witch.noise import NoiseCMBMap, cmb_covariance


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
