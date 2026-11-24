"""
Tests for the square-root noise filter: `noise.apply_noise = LᵀL` with `L = witch.noise.noise_sqrt`,
and the chi-squared, gradient and curvature computed from it in `chisq_objective`.
The noise models here are strongly correlated across detectors and in time, the square root
does not change the noise model.
"""

import jax
import jax.numpy as jnp
import jitkasi.noise as jn
import numpy as np
import pytest

from witch.fitting import _tod_chunked_objective
from witch.noise import has_noise_sqrt, noise_astype, noise_sqrt
from witch.objective import chisq_objective, with_noise_sqrt, without_noise_sqrt

from .conftest import correlated_noise


@pytest.mark.parametrize("nsamp", [301, 256])
def test_svd_sqrt_reproduces_apply_noise(nsamp):
    rng = np.random.default_rng(1)
    ndet = 12
    dat = jnp.asarray(correlated_noise(rng, ndet, nsamp))
    noise = jn.NoiseSmoothedSVD.compute(dat, 10.0)
    assert has_noise_sqrt(noise)

    model = jnp.asarray(rng.standard_normal((ndet, nsamp)))
    grads = jnp.asarray(rng.standard_normal((3, ndet, nsamp)).cumsum(-1))
    resid = dat - 1e-2 * model
    lres = noise_sqrt(noise, resid)
    assert lres.shape == resid.shape

    chisq = jnp.sum(resid * noise.apply_noise(resid))
    np.testing.assert_allclose(jnp.sum(lres**2), chisq, rtol=1e-11)

    ngrads = jax.vmap(noise.apply_noise)(grads)
    lgrads = jax.vmap(lambda g: noise_sqrt(noise, g))(grads)
    grad = jnp.einsum("ijk,jk->i", ngrads, resid)
    np.testing.assert_allclose(
        jnp.einsum("ijk,jk->i", lgrads, lres),
        grad,
        rtol=0,
        atol=1e-11 * abs(grad).max(),
    )
    curve = jnp.einsum("ijk,ljk->il", ngrads, grads)
    np.testing.assert_allclose(
        jnp.einsum("ijk,ljk->il", lgrads, lgrads), curve, rtol=1e-11
    )


def test_diagonal_sqrts():
    rng = np.random.default_rng(2)
    dat = jnp.asarray(rng.standard_normal((5, 40)))
    white = jn.NoiseWhite(jnp.asarray(rng.uniform(0.5, 2.0, 5)))
    for noise in (white, jn.NoiseI()):
        assert has_noise_sqrt(noise)
        np.testing.assert_allclose(
            jnp.sum(noise_sqrt(noise, dat) ** 2),
            jnp.sum(dat * noise.apply_noise(dat)),
            rtol=1e-13,
        )


def test_unsupported_noise():
    assert not has_noise_sqrt(jn.NoiseWhite(jnp.array([1.0, -1.0])))
    assert not has_noise_sqrt(object())
    with pytest.raises(TypeError):
        noise_sqrt(object(), jnp.zeros((2, 3)))


def _assert_objectives_match(full, sqrt):
    chisq, grad, curve = full
    np.testing.assert_allclose(sqrt[0], chisq, rtol=1e-11)
    np.testing.assert_allclose(sqrt[1], grad, rtol=0, atol=1e-10 * abs(grad).max())
    np.testing.assert_allclose(sqrt[2], curve, rtol=1e-11)


def test_objective_matches_full_filter(tod_metamodel):
    mm = tod_metamodel()
    mm_sqrt = with_noise_sqrt(mm)
    assert mm.datasets[0].ldata is None
    assert len(mm_sqrt.datasets[0].ldata) == 2
    full = chisq_objective(mm, 0, True, True, True)
    _assert_objectives_match(full, chisq_objective(mm_sqrt, 0, True, True, True))
    # MH (loglike only) and HMC (no curvature) modes
    np.testing.assert_allclose(
        chisq_objective(mm_sqrt, 0, True, False, False)[0], full[0], rtol=1e-11
    )
    np.testing.assert_allclose(
        chisq_objective(mm_sqrt, 0, False, True, False)[1],
        full[1],
        rtol=0,
        atol=1e-10 * abs(full[1]).max(),
    )
    assert without_noise_sqrt(mm_sqrt).datasets[0].ldata is None


def test_chunked_objective_uses_each_tods_ldata(tod_metamodel):
    # The chunked LM path splits datasets into one TOD each, `L d` must follow its TOD
    mm = tod_metamodel(ntod=3)
    _assert_objectives_match(
        _tod_chunked_objective(mm), _tod_chunked_objective(with_noise_sqrt(mm))
    )


def test_float32_sqrt():
    rng = np.random.default_rng(5)
    dat = jnp.asarray(correlated_noise(rng, 12, 301))
    noise = jn.NoiseSmoothedSVD.compute(dat, 10.0)
    noise32 = noise_astype(noise, jnp.float32)
    assert noise32.v.dtype == jnp.float32
    model = jnp.asarray(rng.standard_normal((12, 301)))
    lm = noise_sqrt(noise, model)
    lm32 = noise_sqrt(noise32, model.astype(jnp.float32))
    assert lm32.dtype == jnp.float32
    np.testing.assert_allclose(lm32, lm, rtol=0, atol=1e-5 * abs(lm).max())


def test_mixed_precision_objective(tod_metamodel):
    # Only the model term goes through float32, so the chisq error scales with |Lm|²
    mm = tod_metamodel()
    full = chisq_objective(with_noise_sqrt(mm), 0, True, True, True)
    mixed = chisq_objective(with_noise_sqrt(mm, True), 0, True, True, True)
    assert mixed[0].dtype == full[0].dtype == jnp.float64
    model_term = sum(
        float(jnp.sum(noise_sqrt(tod.noise, mm.model_proj(0, i)) ** 2))
        for i, tod in enumerate(mm.datasets[0].datavec)
    )
    assert abs(float(mixed[0] - full[0])) < 1e-6 * model_term
    np.testing.assert_allclose(
        mixed[1], full[1], rtol=0, atol=1e-6 * abs(full[1]).max()
    )
    np.testing.assert_allclose(mixed[2], full[2], rtol=1e-6)
