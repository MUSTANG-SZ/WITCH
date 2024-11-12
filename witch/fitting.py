from functools import partial

import jax
import jax.numpy as jnp
import mpi4jax
from jitkasi.tod import TODVec
from mpi4py import MPI

from . import utils as wu
from .containers import Model


@jax.jit
def invsafe(matrix: jax.Array, thresh: float = 1e-14) -> jax.Array:
    """
    Safe SVD based psuedo-inversion of the matrix.
    This zeros out modes that are too small when inverting.
    Use with caution in cases where you really care about what the inverse is.

    Parameters
    ----------
    matrix : jax.Array
        The matrix to invert.
        Should be a `(n, n)` array.
    thresh : float, default: 1e-14
        Threshold at which to zero out a mode.

    Returns
    -------
    invmat: jax.Array
        The inverted matrix.
        Same shape as `matrix`.
    """
    u, s, v = jnp.linalg.svd(matrix, False)
    s_inv = jnp.array(jnp.where(jnp.abs(s) < thresh * jnp.max(s), 0, 1 / s))

    return jnp.dot(jnp.transpose(v), jnp.dot(jnp.diag(s_inv), jnp.transpose(u)))


@jax.jit
def invscale(matrix: jax.Array, thresh: float = 1e-14) -> jax.Array:
    """
    Invert and rescale a matrix by the diagonal.
    This uses `invsafe` for the inversion.

    Parameters
    ----------
    Parameters
    ----------
    matrix : jax.Array
        The matrix to invert and sxane.
        Should be a `(n, n)` array.
    thresh : float, default: 1e-14
        Threshold for `invsafe`.
        See that function for more info.

    Returns
    -------
    invmat: jax.Array
        The inverted and rescaled matrix.
        Same shape as `matrix`.
    """
    diag = jnp.diag(matrix)
    vec = jnp.array(jnp.where(diag != 0, 1.0 / jnp.sqrt(jnp.abs(diag)), 1e-10))
    mm = jnp.outer(vec, vec)

    return mm * invsafe(mm * matrix, thresh)


@jax.jit
def objective(
    pars: jax.Array, model: Model, todvec: TODVec, errs: jax.Array
) -> tuple[Model, jax.Array, jax.Array]:
    """
    Objective function to minimize when fitting.
    This is also responsible for updating our model with the current guess.
    This is an MPI aware function.

    Parameters
    ----------
    pars : jax.Array
        New parameters for our model.
    model : Model
        The model object we are using to fit.
    todvec : TODVec
        The TODs to fit against.
        This is what we use to compute our fit residuals.
    errs : jax.Array
        The error on `pars`, used to update the model state.

    Returns
    -------
    new_model : Model
        An updated model object.
        This contains the newly computed `chisq` for the input `pars`.
        Also is updated with input `pars` and `errs`.
    grad : jax.Array
        The gradient of the paramaters at there current values.
        This is a `(npar,)` array.
    curve : jax.Array
        The curvature of the parameter space at the current values.
        This is a `(npar, npar)` array.
    """
    npar = len(pars)
    new_model = model.update(pars, errs, model.chisq)
    chisq = jnp.array(0)
    grad = jnp.zeros(npar)
    curve = jnp.zeros((npar, npar))
    for tod in todvec:
        x = tod.x * wu.rad_to_arcsec
        y = tod.y * wu.rad_to_arcsec

        pred_tod, grad_tod = new_model.to_tod_grad(x, y)
        ndet, nsamp = tod.shape

        resid = tod.data - pred_tod
        resid_filt = tod.noise.apply_noise(resid)
        chisq += jnp.sum(resid * resid_filt)

        grad_filt = jnp.zeros_like(grad_tod)
        for i in range(npar):
            grad_filt = grad_filt.at[i].set(tod.noise.apply_noise(grad_tod.at[i].get()))
        grad_filt = jnp.reshape(grad_filt, (npar, ndet * nsamp))
        grad_tod = jnp.reshape(grad_tod, (npar, ndet * nsamp))
        resid = jnp.reshape(resid, (ndet * nsamp,))

        grad = grad.at[:].add(jnp.dot(grad_filt, jnp.transpose(resid)))
        curve = curve.at[:].add(jnp.dot(grad_filt, jnp.transpose(grad_tod)))

    chisq, _ = mpi4jax.allreduce(chisq, MPI.SUM, comm=todvec.comm)
    grad, _ = mpi4jax.allreduce(grad, MPI.SUM, comm=todvec.comm)
    curve, _ = mpi4jax.allreduce(curve, MPI.SUM, comm=todvec.comm)

    new_model = new_model.update(pars, errs, chisq)

    return new_model, grad, curve


@jax.jit
def _prior_pars_fit(
    priors: tuple[jax.Array, jax.Array], pars: jax.Array, to_fit: jax.Array
) -> tuple[jax.Array, jax.Array]:
    prior_l, prior_u = priors
    at_edge_l = pars <= prior_l
    at_edge_u = pars >= prior_u
    pars = jnp.where(at_edge_l, prior_l, pars)
    pars = jnp.where(at_edge_u, prior_u, pars)
    to_fit = jnp.where(at_edge_l + at_edge_u, False, to_fit)

    return pars, to_fit


def _failure(
    model,
    new_model,
    grad,
    new_grad,
    curve,
    new_curve,
    delta_chisq,
    new_delta_chisq,
    lmd,
):
    del new_model
    del new_grad
    del new_curve
    del new_delta_chisq
    lmd = jnp.where(lmd == 0, 1, 2 * lmd)
    return model, grad, curve, delta_chisq, lmd


def _success(
    model,
    new_model,
    grad,
    new_grad,
    curve,
    new_curve,
    delta_chisq,
    new_delta_chisq,
    lmd,
):
    del model
    del grad
    del curve
    del delta_chisq
    lmd = jnp.where(lmd < 0.2, 0, lmd / jnp.sqrt(2))
    return new_model, new_grad, new_curve, new_delta_chisq, lmd


@partial(jax.jit, static_argnums=(2, 3))
def fit_tods(
    model: Model, todvec: TODVec, maxiter: int = 10, chitol: float = 1e-5
) -> tuple[Model, int, float]:
    """
    Fit a model to TODs.
    This uses a modified Levenberg–Marquardt fitter with flat priors.
    This function is MPI aware.

    Parameters
    ----------
    model : Model
        The model object that defines the model and grid we are fitting with.
    todvec : TODVec
        The TODs to fit.
        The `todvec.comm` object is used to fit in an MPI aware way.
    maxiter : int, default: 10
        The maximum number of iterations to fit.
    chitol : float, default: 1e-5
        The delta chisq to use as the convergence criteria.

    Returns
    -------
    model : Model
        Model with the final set of fit parameters, errors, and chisq.
    final_iter : int
        The number of iterations the fitter ran for.
    delta_chisq : float
        The final delta chisq.
    """
    zero = jnp.array(0.0)

    def _cond_func(val):
        i, delta_chisq, lmd, *_ = val
        iterbool = jax.lax.lt(i, maxiter)
        chisqbool = jax.lax.ge(delta_chisq, chitol) + jax.lax.gt(lmd, zero)
        return iterbool * chisqbool

    def _body_func(val):
        i, delta_chisq, lmd, model, curve, grad = val
        curve_use = curve.at[:].add(lmd * jnp.diag(jnp.diag(curve)))
        # Get the step
        step = jnp.dot(invscale(curve_use), grad)
        new_pars, to_fit = _prior_pars_fit(
            model.priors, model.pars.at[:].add(step), jnp.array(model.to_fit)
        )
        # Get errs
        errs = jnp.where(to_fit, jnp.sqrt(jnp.diag(invscale(curve_use))), 0)
        # Now lets get an updated model
        new_model, new_grad, new_curve = objective(new_pars, model, todvec, errs)

        new_delta_chisq = model.chisq - new_model.chisq
        model, grad, curve, delta_chisq, lmd = jax.lax.cond(
            new_delta_chisq > 0,
            _success,
            _failure,
            model,
            new_model,
            grad,
            new_grad,
            curve,
            new_curve,
            delta_chisq,
            new_delta_chisq,
            lmd,
        )

        return (i + 1, delta_chisq, lmd, model, curve, grad)

    pars, _ = _prior_pars_fit(model.priors, model.pars, jnp.array(model.to_fit))
    model, grad, curve = objective(pars, model, todvec, model.errs)
    i, delta_chisq, _, model, *_ = jax.lax.while_loop(
        _cond_func, _body_func, (0, jnp.inf, zero, model, curve, grad)
    )

    return model, i, delta_chisq
