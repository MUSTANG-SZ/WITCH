import jax.numpy as jnp
import numpy as np
from jitkasi.tod import TOD, TODVec
from mpi4py import MPI

from witch.containers import MetaModel
from witch.dataset import DataSetData
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


def _mean_objective(metamodel, dataset_ind, do_loglike, do_grad, do_curve):
    """Quadratic objective that pulls the one parameter to the mean TOD value."""
    del do_loglike, do_grad, do_curve
    chisq, grad = 0.0, 0.0
    for tod in metamodel.datasets[dataset_ind].datavec.tods:
        residual = tod.data.mean() - metamodel.parameters[0]
        chisq += 0.5 * residual**2
        grad += residual
    return jnp.asarray(chisq), jnp.asarray([grad]), jnp.ones((1, 1))


def _real_metamodel(values):
    """
    A real (jitted code path) `MetaModel` with no models and one TOD per value.
    Each TOD carries a different array in `meta`, like real datasets do.
    """
    tods = [
        TOD(
            jnp.full((2, 3), v),
            jnp.zeros((2, 3)),
            jnp.zeros((2, 3)),
            meta={"calibration": np.array([v, 2 * v])},
        )
        for v in values
    ]
    dataset = DataSetData(
        "test", "tod", _mean_objective, TODVec(tods, MPI.COMM_WORLD), (), None, (), {}
    )
    return MetaModel(
        NullComm(),
        (),
        (dataset,),
        (),
        (),
        ((),),
        jnp.zeros(1),
        jnp.zeros(1),
        jnp.zeros((1, 1)),
        jnp.asarray(0.0),
    )


def test_chunked_objective_with_array_metadata_in_tods():
    """
    Regression test: jax compares the aux data (TOD.meta) of pytrees when looking up
    compiled functions, which raises for arrays if different TODs share a compiled function.
    Keep this the only test that builds a real `MetaModel` with array metadata: a second
    one built in the same process (distinct but equal TOD objects) trips jax's pytree
    comparison in the same way, independent of the code under test.
    """
    metamodel = _real_metamodel([1.0, 2.0, 3.0])
    assert isinstance(metamodel, MetaModel)

    chisq, grad, curve = _tod_chunked_objective(metamodel)

    np.testing.assert_allclose(chisq, 0.5 * (1 + 4 + 9))
    np.testing.assert_allclose(grad, [6.0])
    np.testing.assert_allclose(curve, [[3.0]])
