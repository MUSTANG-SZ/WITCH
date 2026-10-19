import jax
import jax.numpy as jnp
import numpy as np

from witch.fitting import hmc
from mpi4py import MPI


def _gauss(sigma):
    """Log prob and gradient of a zero mean gaussian with stds `sigma`."""

    @jax.jit
    def log_prob_and_grad(pars):
        return -0.5 * jnp.sum((pars / sigma) ** 2), -pars / sigma**2

    return log_prob_and_grad


def test_hmc_samples_target_distribution():
    sigma = jnp.array([1.0, 3.0])
    chain, accept = hmc(
        jnp.zeros(2),
        _gauss(sigma),
        num_steps=4000,
        num_leaps=10,
        step_size=jnp.array(0.3),
        comm=MPI.COMM_WORLD,
        key=jax.random.PRNGKey(1),
    )
    chain = np.asarray(chain)[500:]
    assert 0.3 < float(accept.mean()) <= 1.0
    np.testing.assert_allclose(chain.mean(axis=0), 0.0, atol=0.4)
    np.testing.assert_allclose(chain.std(axis=0), np.asarray(sigma), rtol=0.25)


def test_hmc_does_not_reuse_stale_sampler():
    """A new likelihood function must not pick up a previous one's compiled sampler."""
    kwargs = dict(
        num_steps=300,
        num_leaps=5,
        step_size=jnp.array(0.3),
        comm=MPI.COMM_WORLD,
        key=jax.random.PRNGKey(2),
    )
    narrow, _ = hmc(jnp.zeros(1), _gauss(jnp.array([0.5])), **kwargs)
    wide, _ = hmc(jnp.zeros(1), _gauss(jnp.array([5.0])), **kwargs)
    assert np.std(np.asarray(wide)) > 3 * np.std(np.asarray(narrow))
