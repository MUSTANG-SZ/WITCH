import jax.numpy as jnp
import numpy as np

from witch.fitting import _run_lmfit_tod_chunks, _tod_chunked_objective
from witch.utils import NullComm


class _DataVec:
    def __init__(self, tods):
        self.tods = list(tods)

    def __iter__(self):
        return iter(self.tods)

    def copy(self):
        return _DataVec(self.tods)


class _Dataset:
    mode = "tod"

    def __init__(self, tods):
        self.datavec = _DataVec(tods)

    def objective(self, metamodel, dataset_ind, do_loglike, do_grad, do_curve):
        del dataset_ind, do_loglike, do_grad, do_curve
        tods = metamodel.datasets[0].datavec.tods
        assert len(tods) == 1
        value = tods[0]
        return (
            jnp.asarray(value),
            jnp.asarray([value, 2 * value]),
            jnp.asarray([[value, 0], [0, value]]),
        )


class _MetaModel:
    def __init__(self, tods):
        self.parameters = jnp.zeros(1)
        self.errs = jnp.zeros(1)
        self.cov = jnp.zeros((1, 1))
        self.chisq = jnp.asarray(0.0)
        self.priors = (jnp.asarray([-jnp.inf]), jnp.asarray([jnp.inf]))
        self.to_fit = jnp.asarray([True])
        self.datasets = (_Dataset(tods),)
        self.global_comm = NullComm()

    def update(self, pars, errs, cov, chisq):
        self.parameters = pars
        self.errs = errs
        self.cov = cov
        self.chisq = chisq
        return self


class _QuadraticDataset(_Dataset):
    def objective(self, metamodel, dataset_ind, do_loglike, do_grad, do_curve):
        del dataset_ind, do_loglike, do_grad, do_curve
        target = metamodel.datasets[0].datavec.tods[0]
        residual = target - metamodel.parameters[0]
        return (
            0.5 * residual**2,
            jnp.asarray([residual]),
            jnp.ones((1, 1)),
        )


def test_tod_chunked_objective_sums_individual_tods():
    metamodel = _MetaModel([1.0, 2.0, 3.0])
    metamodel.parameters = jnp.zeros(2)
    chisq, grad, curve = _tod_chunked_objective(metamodel)

    assert float(chisq) == 6.0
    np.testing.assert_array_equal(grad, [6.0, 12.0])
    np.testing.assert_array_equal(curve, [[6.0, 0.0], [0.0, 6.0]])


def test_chunked_lm_fit_converges_across_multiple_tods():
    metamodel = _MetaModel([1.0, 2.0, 3.0])
    metamodel.datasets = (_QuadraticDataset([1.0, 2.0, 3.0]),)

    fitted, _, delta_chisq, _ = _run_lmfit_tod_chunks(metamodel, 5, 1e-5)

    np.testing.assert_allclose(fitted.parameters, [2.0], atol=1e-6)
    assert np.isfinite(delta_chisq)
