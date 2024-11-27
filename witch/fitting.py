from functools import partial

import emcee
import jax
import jax.numpy as jnp
import mpi4jax
import numpy as np
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
        The gradient of the parameters at there current values.
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

    chisq, token = mpi4jax.allreduce(chisq, MPI.SUM, comm=todvec.comm)
    grad, token = mpi4jax.allreduce(grad, MPI.SUM, comm=todvec.comm, token=token)
    curve, _ = mpi4jax.allreduce(curve, MPI.SUM, comm=todvec.comm, token=token)

    new_model = new_model.update(pars, errs, chisq)

    return new_model, grad, curve


@jax.jit
def get_chisq(model: Model, todvec: TODVec) -> jax.Array:
    """
    Get the chi-squared of a model given a set of TODs.
    This is an MPI aware function.

    Parameters
    ----------
    model : Model
        The model object we are using to fit.
    todvec : TODVec
        The TODs to fit against.
        This is what we use to compute our fit residuals.

    Returns
    -------
    chisq : jax.Array
    """
    token = mpi4jax.barrier(comm=todvec.comm)
    chisq = jnp.array(0)
    for tod in todvec:
        x = tod.x * wu.rad_to_arcsec
        y = tod.y * wu.rad_to_arcsec

        pred_tod = model.to_tod(x, y)

        resid = tod.data - pred_tod
        resid_filt = tod.noise.apply_noise(resid)
        chisq += jnp.sum(resid * resid_filt)

    chisq, token = mpi4jax.allreduce(chisq, MPI.SUM, comm=todvec.comm, token=token)
    _ = mpi4jax.barrier(comm=todvec.comm, token=token)

    return chisq


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


def run_mcmc(
    model: Model, todvec: TODVec, num_steps: int = 5000, num_walkers: int = 10
) -> tuple[Model, jax.Array]:
    """
    Run MCMC using the `emcee` package to estimate the posterior for our model.
    Currently this function only support flat priors, but more will be supported
    down the line. In order to ensure accuracy of the noise model used, it is
    reccomended that you run at least one round of `fit_tods` followed by noise
    reestimation before this function.

    This is MPI aware.
    Eventually this will be replaced with something more jaxy.

    Parameters
    ----------
    model : Model
        The model to run MCMC on.
        We expect that all parameters in this model have priors defined.
    todvec : TODVec
        The TODs to compute the likelihood of the model with.
    num_steps : int
        The number of steps to run MCMC for.
    num_walkers : int
        The number of walkers to use.
        If this is less than 2.1 times the number of
        parameters in the model then 2.1 times the number
        of parameters in the model is used.

    Returns
    -------
    model : Model
        The model with MCMC estimated parameters and errors.
        The parameters are estimated as the mean of the samples.
        The errors are estimated as the standard deviation.
        This also has the chi-squared of the estimated parameters.
    flat_samples : jax.Array
        Array of samples from running MCMC.
        This has some samples discarded to account for burn in,
        we either discard `int(2*tau_max)` samples where `tau_max` is
        the autocorrelation time for the variable with the longest
        autocorrelation time or `int(num_steps/25)` if there is an error
        when estimating autocorrelation. We also thin by a a factor of
        `int(tau_max/2)` or `int(num_steps/100)`.
        This array has shape `(npar, nsamps)` where the rows are in
        the same order as `model.pars`.

    """
    token = mpi4jax.barrier(comm=todvec.comm)
    rank = todvec.comm.Get_rank()

    init_pars = jnp.array(model.pars)
    init_errs = jnp.zeros_like(model.pars)
    to_fit_init = jnp.ones_like(jnp.array(model.to_fit), dtype=bool)

    num_walkers = max(num_walkers, int(2.1 * len(init_pars)))

    def _log_prob(pars):
        run = jnp.array(True)
        _, token = mpi4jax.bcast(run, 0, comm=todvec.comm)
        _, to_fit = _prior_pars_fit(model.priors, pars, to_fit_init)
        log_prior = jnp.sum(jnp.where(to_fit, 0, -1 * jnp.inf))
        if not jnp.isfinite(log_prior):
            return -1 * np.inf
        pars, token = mpi4jax.bcast(pars, 0, comm=todvec.comm, token=token)
        temp_model = model.update(pars, init_errs, model.chisq)
        log_like = -5.0 * get_chisq(temp_model, todvec)
        _ = mpi4jax.bcast(run, 0, comm=todvec.comm, token=token)
        return float(log_like + log_prior)

    run = jnp.array(True)
    final_pars = init_pars.copy()
    final_errs = init_errs.copy()
    if rank == 0:
        key1 = jax.random.PRNGKey(0)
        pos = 1e-4 * jax.random.normal(key1, shape=(num_walkers, len(init_pars)))
        pos = pos.at[:].add(init_pars)
        sampler = emcee.EnsembleSampler(
            num_walkers,
            len(init_pars),
            _log_prob,
            moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ],
        )

        # Burn in
        state = sampler.run_mcmc(
            pos, max(50, min(50, int(num_steps / 10))), progress=True
        )
        sampler.reset()

        # Run for real
        sampler.run_mcmc(state, num_steps, progress=True)
        run = jnp.array(False)
        run, token = mpi4jax.bcast(run, 0, comm=todvec.comm, token=token)
        tau = 0
        if num_steps > 100:
            try:
                tau = int(np.max(sampler.get_autocorr_time()))
            except emcee.autocorr.AutocorrError:
                tau = int(num_steps / 50)
        flat_samples = jnp.array(
            sampler.get_chain(discard=2 * tau, thin=int(tau / 2), flat=True)
        )
        final_pars = jnp.median(flat_samples, axis=0).ravel()
        final_errs = jnp.std(flat_samples, axis=0).ravel()
        final_pars, token = mpi4jax.bcast(final_pars, 0, comm=todvec.comm, token=token)
        final_errs, _ = mpi4jax.bcast(final_errs, 0, comm=todvec.comm, token=token)
    else:
        pars = init_pars.copy()
        flat_samples = jnp.zeros(1)
        while run:
            run, token = mpi4jax.bcast(run, 0, comm=todvec.comm, token=token)
            if not run:
                break
            pars, token = mpi4jax.bcast(pars, 0, comm=todvec.comm, token=token)
            temp_model = model.update(pars, init_errs, model.chisq)
            _ = get_chisq(temp_model, todvec)
            run, token = mpi4jax.bcast(run, 0, comm=todvec.comm, token=token)
        final_pars, token = mpi4jax.bcast(final_pars, 0, comm=todvec.comm, token=token)
        final_errs, _ = mpi4jax.bcast(final_errs, 0, comm=todvec.comm, token=token)

    model = model.update(final_pars, final_errs, model.chisq)
    chisq = get_chisq(model, todvec)
    model = model.update(final_pars, final_errs, chisq)

    return model, flat_samples
