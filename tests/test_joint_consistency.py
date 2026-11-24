"""
Helpers for checking the datasets of a joint fit are consistent: profiling a linear parameter, the effective number
of constrained parameters, and Q_DMAP.
"""

import numpy as np
import pytest
from scipy import stats

from witch.model_comparison import n_eff, profile_linear, q_dmap


def _linear_problem(
    a0: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A two parameter model `m = b * f + a * g` with white noise, evaluated at `(b, a) = (1, a0)`, in the conventions
    of the WITCH objectives (grad = J^T N^-1 r, curve = J^T N^-1 J).

    Parameters
    ----------
    a0 : float
        Value of the linear parameter `a` to evaluate at.

    Returns
    -------
    chisq : float
        Chi-squared at `pars`.
    grad : np.ndarray
        `J^T N^-1 r`.
    curve : np.ndarray
        `J^T N^-1 J`.
    pars : np.ndarray
        `(1, a0)`.
    data : np.ndarray
        The data.
    templates : np.ndarray
        `(2, n)` templates `f` and `g`.
    """
    rng = np.random.default_rng(1)
    templates = rng.normal(size=(2, 200))
    data = templates[0] + 3.0 * templates[1] + rng.normal(size=200)
    pars = np.array([1.0, a0])
    resid = data - pars @ templates
    return (
        float(resid @ resid),
        templates @ resid,
        templates @ templates.T,
        pars,
        data,
        templates,
    )


@pytest.mark.parametrize("a0", [0.0, 2.5, 10.0])
def test_profile_linear_is_exact(a0: float):
    """One Newton step finds the exact least squares minimum over a linear parameter, from any starting value."""
    chisq, grad, curve, pars, data, templates = _linear_problem(a0)
    best_chisq, best = profile_linear(chisq, grad, curve, pars, 1)
    target = data - templates[0]
    exact = float(templates[1] @ target / (templates[1] @ templates[1]))
    assert best == pytest.approx(exact, rel=1e-10)
    assert best_chisq == pytest.approx(
        float(((target - exact * templates[1]) ** 2).sum()), rel=1e-10
    )
    # and nothing on a grid around it does better
    grid = np.linspace(exact - 1, exact + 1, 2001)
    assert (
        ((target - grid[:, None] * templates[1]) ** 2).sum(axis=1) >= best_chisq - 1e-9
    ).all()


def test_profile_linear_respects_prior():
    """When the unconstrained best value is outside the prior, the bound is returned with its chi-squared."""
    chisq, grad, curve, pars, data, templates = _linear_problem(0.0)
    best_chisq, best = profile_linear(chisq, grad, curve, pars, 1, lo=-1.0, hi=1.0)
    assert best == 1.0
    assert best_chisq == pytest.approx(
        float(((data - templates[0] - templates[1]) ** 2).sum())
    )


def test_n_eff_counts_constrained_parameters():
    """
    A parameter measured far better than its prior counts 1, one the prior dominates about 0, one the data don't
    touch is dropped, and one with an infinite prior counts 1.
    """
    fisher = np.diag([1e6, 1e-6, 0.0, 1.0])
    lower = np.array([0.0, 0.0, 0.0, -np.inf])
    upper = np.array([1.0, 1.0, 1.0, np.inf])
    mask = np.ones(4, dtype=bool)
    assert n_eff(fisher, mask, lower, upper) == pytest.approx(2.0, abs=1e-4)
    assert n_eff(
        fisher, np.array([True, True, True, False]), lower, upper
    ) == pytest.approx(1.0, abs=1e-4)


def test_n_eff_equal_data_and_prior():
    """Data as constraining as the prior count for half a parameter."""
    width = 2.0
    fisher = np.array([[12 / width**2]])
    assert n_eff(
        fisher, np.array([True]), np.array([0.0]), np.array([width])
    ) == pytest.approx(0.5)


def test_q_dmap():
    """Q is the sum of the costs, with N_eff(a) + N_eff(b) - N_eff(joint) dof and the matching p-value."""
    out = q_dmap(3.0, 4.5, 9.8, 6.3, 10.8)
    assert out["Q"] == pytest.approx(7.5)
    assert out["dof"] == pytest.approx(5.3)
    assert out["p"] == pytest.approx(stats.chi2.sf(7.5, 5.3))
    assert out["sigma"] == pytest.approx(stats.norm.isf(out["p"] / 2))
    assert np.isnan(q_dmap(1.0, 1.0, 2.0, 2.0, 4.0)["p"])
