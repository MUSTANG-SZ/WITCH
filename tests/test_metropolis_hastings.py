import jax.numpy as jnp
import numpy as np

from witch import metropolis_hastings as mh


class _Dataset:
    def objective(self, metamodel, dataset_ind, **kwargs):
        chisq = jnp.sum(metamodel.parameters**2)
        return chisq, jnp.zeros(1), jnp.zeros((1, 1))


class _MetaModel:
    def __init__(self, parameters):
        self.parameters = jnp.asarray(parameters)
        self.errs = jnp.ones_like(self.parameters)
        self.priors = (
            jnp.full_like(self.parameters, -10),
            jnp.full_like(self.parameters, 10),
        )
        self.cov = jnp.zeros((len(parameters), len(parameters)))
        self.chisq = jnp.array(0.0)
        self.to_fit = jnp.ones_like(self.parameters, dtype=bool)
        self.datasets = [_Dataset()]

    def update(self, pars, errs, cov, chisq):
        self.parameters = pars
        self.errs = errs
        self.cov = cov
        self.chisq = chisq
        return self


def test_calc_like_dataset_converts_chisq_to_loglike():
    metamodel = _MetaModel([3.0])

    loglike = mh.calc_like_dataset.__wrapped__(metamodel, jnp.array([3.0]))

    assert float(loglike) == -4.5


def test_calc_like_joint_converts_chisq_to_loglike(monkeypatch):
    metamodel = _MetaModel([3.0])
    monkeypatch.setattr(
        mh,
        "joint_objective",
        lambda **kwargs: (jnp.array(9.0), jnp.zeros(1), jnp.zeros((1, 1))),
    )

    loglike = mh.calc_like_joint.__wrapped__(metamodel, jnp.array([3.0]))

    assert float(loglike) == -4.5


def test_proposals_are_centered_on_current_state(monkeypatch):
    metamodel = _MetaModel([0.0])
    centers = []

    def draw_from_current(metamodel, **kwargs):
        center = np.asarray(metamodel.parameters)
        centers.append(center.copy())
        return metamodel.parameters + 1

    monkeypatch.setattr(mh, "draw_samp", draw_from_current)
    monkeypatch.setattr(mh, "tqdm", lambda values, **kwargs: values)

    samples, acceptance_rate = mh.metropolis_hastings(
        metamodel,
        num_samples=3,
        calc_like_func=lambda **kwargs: 0.0,
    )

    np.testing.assert_array_equal(np.asarray(centers), [[0.0], [1.0], [2.0]])
    np.testing.assert_array_equal(samples, [[1.0], [2.0], [3.0]])
    assert acceptance_rate == 1.0


def test_uniform_proposal_is_local_to_current_state():
    first = _MetaModel([0.0])
    second = _MetaModel([3.0])
    key = jnp.array([0, 1], dtype=jnp.uint32)

    first_proposal = mh.draw_samp.__wrapped__(first, key, prior_type="uniform")
    second_proposal = mh.draw_samp.__wrapped__(second, key, prior_type="uniform")

    assert abs(float(first_proposal[0])) <= 1.0
    assert float(second_proposal[0] - first_proposal[0]) == 3.0


def test_uniform_proposal_has_minimum_width_for_tiny_fit_error():
    metamodel = _MetaModel([0.0])
    metamodel.errs = jnp.array([1e-12])

    proposal = mh.draw_samp.__wrapped__(
        metamodel, jnp.array([0, 1], dtype=jnp.uint32), prior_type="uniform"
    )

    assert 1e-6 < abs(float(proposal[0])) <= 0.2


def test_unbounded_uniform_proposal_scales_with_parameter_magnitude():
    metamodel = _MetaModel([1.5e15, 0.005])
    metamodel.errs = jnp.array([0.0, 0.0])
    metamodel.priors = (
        jnp.array([-jnp.inf, -jnp.inf]),
        jnp.array([jnp.inf, jnp.inf]),
    )

    proposal = mh.draw_samp.__wrapped__(
        metamodel,
        jnp.array([0, 1], dtype=jnp.uint32),
        bound=4,
        prior_type="uniform",
    )

    relative_steps = np.abs(
        np.asarray((proposal - metamodel.parameters) / metamodel.parameters)
    )
    assert np.all(relative_steps > 1e-3)
    assert np.all(relative_steps <= 0.25)


def test_chain_accepts_local_uniform_proposals(monkeypatch):
    metamodel = _MetaModel([0.0])
    draw_samp = mh.draw_samp.__wrapped__
    monkeypatch.setattr(mh, "draw_samp", lambda **kwargs: draw_samp(**kwargs))
    monkeypatch.setattr(mh, "tqdm", lambda values, **kwargs: values)

    _, acceptance_rate = mh.metropolis_hastings(
        metamodel,
        num_samples=100,
        seed=2,
        calc_like_func=lambda pars, **kwargs: -0.5 * jnp.sum(pars**2),
    )

    assert 0.0 < acceptance_rate < 1.0
