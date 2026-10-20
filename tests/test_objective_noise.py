import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jitkasi.noise import NoiseI, NoiseSmoothedSVD, NoiseWhite

from witch.objective import SYMMETRIC_NOISE, _tod_grad_curve

NDET, NSAMP, NFIT = 6, 128, 3


class _AsymmetricNoise:
    """A valid linear operator that is not symmetric and not in `SYMMETRIC_NOISE`."""

    def __init__(self, seed=0):
        self.mat = jnp.array(np.random.default_rng(seed).normal(size=(NSAMP, NSAMP)))

    def apply_noise(self, dat):
        return dat @ self.mat


def _noise_models():
    rng = np.random.default_rng(1)
    dat = jnp.array(rng.normal(size=(NDET, NSAMP)))
    return {
        "NoiseI": NoiseI(),
        "NoiseWhite": NoiseWhite.compute(dat),
        "NoiseSmoothedSVD": NoiseSmoothedSVD.compute(dat, 5.0),
    }


def _inputs():
    rng = np.random.default_rng(2)
    resid = jnp.array(rng.normal(size=(NDET, NSAMP)))
    grad_dat = jnp.array(rng.normal(size=(NFIT, NDET, NSAMP)))
    return resid, grad_dat


@pytest.mark.parametrize("name", list(_noise_models()))
def test_fast_path_noise_models_are_symmetric(name):
    noise = _noise_models()[name]
    assert isinstance(noise, SYMMETRIC_NOISE)
    rng = np.random.default_rng(3)
    x, y = (jnp.array(rng.normal(size=(NDET, NSAMP))) for _ in range(2))

    left = jnp.sum(x * noise.apply_noise(y))
    right = jnp.sum(noise.apply_noise(x) * y)

    np.testing.assert_allclose(left, right, rtol=1e-10)


@pytest.mark.parametrize("name", list(_noise_models()) + ["asymmetric"])
def test_grad_and_curve_match_general_formula(name):
    noise = _AsymmetricNoise() if name == "asymmetric" else _noise_models()[name]
    resid, grad_dat = _inputs()
    resid_filt = noise.apply_noise(resid)

    grad, curve = _tod_grad_curve(noise, resid, resid_filt, grad_dat, True, True)

    # Brute force (N J_i).r and J_i^T N J_j with no assumptions on N
    filt = jax.vmap(noise.apply_noise)(grad_dat).reshape(NFIT, -1)
    flat = grad_dat.reshape(NFIT, -1)
    np.testing.assert_allclose(grad, filt @ resid.ravel(), rtol=1e-9)
    np.testing.assert_allclose(curve, filt @ flat.T, rtol=1e-9)


def test_symmetric_shortcut_would_be_wrong_for_asymmetric_noise():
    """The guard matters: J.(N r) differs from (N J).r when N is not symmetric."""
    noise = _AsymmetricNoise()
    resid, grad_dat = _inputs()
    shortcut = grad_dat.reshape(NFIT, -1) @ noise.apply_noise(resid).ravel()
    general = jax.vmap(noise.apply_noise)(grad_dat).reshape(NFIT, -1) @ resid.ravel()

    assert not np.allclose(shortcut, general, rtol=1e-3)


def test_only_requested_outputs_are_computed():
    noise = _noise_models()["NoiseSmoothedSVD"]
    resid, grad_dat = _inputs()
    resid_filt = noise.apply_noise(resid)

    grad, curve = _tod_grad_curve(noise, resid, resid_filt, grad_dat, True, False)
    assert grad is not None and curve is None
    grad, curve = _tod_grad_curve(noise, resid, None, grad_dat, False, True)
    assert grad is None and curve is not None
