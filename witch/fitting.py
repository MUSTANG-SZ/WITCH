import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import mpi4jax
from jitkasi.tod import TODVec
from mpi4py import MPI
from tqdm import tqdm

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
        The chi-squared of the model.
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
def get_grad(model: Model, todvec: TODVec) -> jax.Array:
    """
    Get the gradient of chi-squared of a model given a set of TODs.
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
    grad : jax.Array
        The gradient of the parameters at there current values.
        This is a `(npar,)` array.
    """
    token = mpi4jax.barrier(comm=todvec.comm)
    npar = len(model.pars)
    grad = jnp.zeros(npar)
    for tod in todvec:
        x = tod.x * wu.rad_to_arcsec
        y = tod.y * wu.rad_to_arcsec

        pred_tod, grad_tod = model.to_tod_grad(x, y)
        ndet, nsamp = tod.shape

        resid = tod.data - pred_tod

        grad_filt = jnp.zeros_like(grad_tod)
        for i in range(npar):
            grad_filt = grad_filt.at[i].set(tod.noise.apply_noise(grad_tod.at[i].get()))
        grad_filt = jnp.reshape(grad_filt, (npar, ndet * nsamp))
        grad_tod = jnp.reshape(grad_tod, (npar, ndet * nsamp))
        resid = jnp.reshape(resid, (ndet * nsamp,))

        grad = grad.at[:].add(jnp.dot(grad_filt, jnp.transpose(resid)))

    grad, token = mpi4jax.allreduce(grad, MPI.SUM, comm=todvec.comm, token=token)
    _ = mpi4jax.barrier(comm=todvec.comm, token=token)

    return grad


@jax.jit
def get_chisq_grad(model: Model, todvec: TODVec) -> jax.Array:
    """
    Get the chi-squared and gradient of a model given a set of TODs.
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
        The chi-squared of the model.
    grad : jax.Array
        The gradient of the parameters at there current values.
        This is a `(npar,)` array.
    """
    token = mpi4jax.barrier(comm=todvec.comm)
    npar = len(model.pars)
    chisq = jnp.array(0)
    grad = jnp.zeros(npar)
    for tod in todvec:
        x = tod.x * wu.rad_to_arcsec
        y = tod.y * wu.rad_to_arcsec

        pred_tod, grad_tod = model.to_tod_grad(x, y)
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

    chisq, token = mpi4jax.allreduce(chisq, MPI.SUM, comm=todvec.comm, token=token)
    grad, token = mpi4jax.allreduce(grad, MPI.SUM, comm=todvec.comm, token=token)
    _ = mpi4jax.barrier(comm=todvec.comm, token=token)

    return chisq, grad


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


def hmc(
    params: jax.Array,
    log_prob: Callable[[jax.Array], jax.Array],
    log_prob_grad: Callable[[jax.Array], jax.Array],
    num_steps: int,
    num_leaps: int,
    step_size: float,
    comm: MPI.Intracomm,
    key: jax.Array,
) -> jax.Array:
    """
    Runs Hamilonian Monte Carlo using a leapfrog integrator to approximate Hamilonian dynamics.
    This is a naive implementaion that will be replaced in the future.

    The parallelism model employed here is different that most samplers where each task runs
    a subset of the chain, instead since the rest of WITCH employs a model where the data is
    distributed across tasks we do that here as well.
    In this model the chain evolves simultaneously in all tasks,
    but only rank 0 actually stores the chain.

    Parameters
    ----------
    params : jax.Array
        The initial parameters to start the chain at.
    log_prob : Callable[[jax.Array], jax.Array]
        Function that returns the log probability of the model
        for a given set of params. This should take `params` as its
        first arguments, all other arguments should be fixed ahead of time
        (ie: using `functools.partial`).
    log_prob_grad : Callable[[jax.Array], jax.Array]
        Function that returns the gradient log probability of the model
        for a given set of params. This should take `params` as its
        first arguments, all other arguments should be fixed ahead of time
        (ie: using `functools.partial`). The returned gradient should have
        shape `(len(params),)`.
    num_steps : int
        The number of steps to run the chain for.
    num_leaps : int
        The number of leapfrog steps to run at each step of the chain.
    step_size : float
        The step size to use.
        At each leapfrog step the parameters will evolve by `step_size`*`momentum`.
    comm : MPI.Intracomm
        The MPI comm object to use.

    Returns
    -------
    chain : jax.Array
        The chain of samples.
        Will have shape `(num_steps, len(params))` in the rank 0 task.
        Note that on tasks with rank other than 0 the actual
        chain is not returned, instead a dummy array of size `(0,)` is
        returned.
    """
    rank = comm.Get_rank()
    vnorm = jax.vmap(
        partial(jax.random.normal, shape=params[0].shape, dtype=params.dtype)
    )
    npar = len(params)
    ones = jnp.ones(npar, dtype=bool)

    @jax.jit
    def _leap(_, args):
        params, momentum = args
        momentum = momentum.at[:].add(0.5 * step_size * log_prob_grad(params))  # kick
        params = params.at[:].add(step_size * momentum)  # drift
        momentum = momentum.at[:].add(0.5 * step_size * log_prob_grad(params))  # kick

        return params, momentum

    @jax.jit
    def _sample(i, key, params):
        token = mpi4jax.barrier(comm=comm)
        key, token = mpi4jax.bcast(key, 0, comm=comm, token=token)
        key, uniform_key = jax.random.split(key, 2)

        # generate random momentum
        momentum = vnorm(jax.random.split(key, npar))
        new_params, new_momentum = jax.lax.fori_loop(
            0, num_leaps, _leap, (params, momentum)
        )

        # MH correction
        dpe = log_prob(new_params) - log_prob(params)
        dke = -0.5 * (jnp.sum(new_momentum**2) - jnp.sum(momentum**2))
        log_accept = dke + dpe
        accept_prob = jnp.minimum(jnp.exp(log_accept), 1)
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = jax.lax.select(accept * ones, new_params, params)

        return key, params, accept_prob

    t0 = time.time()
    l_sample = _sample.lower(0, key, params)
    c_sample = l_sample.compile()
    t1 = time.time()
    if rank == 0:
        print(f"Compiled MC sample function in {t1-t0} s")

    chain = []
    accept_prob = []
    for i in tqdm(range(num_steps), disable=(rank != 0)):
        key, params, prob = c_sample(i, key, params)
        if rank == 0:
            chain += [params]
            accept_prob += [prob]
    if rank == 0:
        chain = jnp.vstack(chain)
        accept_prob = jnp.array(accept_prob)
        print(f"Accepted {accept_prob.mean():.2%} of samples")
    else:
        chain = jnp.zeros(0)
    return chain


def run_mcmc(
    model: Model,
    todvec: TODVec,
    num_steps: int = 5000,
    num_leaps: int = 10,
    step_size: float = 0.02,
    sample_which: int = -1,
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
    num_steps : int, default: 5000
        The number of steps to run MCMC for.
    num_leaps: int, default: 10
        The number of leapfrog steps to take at each sample.
    step_size, default: .02
        The step size to use in the leapfrog algorithm.
        This should be tuned to get an acceptance fraction of ~.65.
    sample_which : int, default: -1,
        Sets which parameters to sample.
        If this is >= 0 then we will sample which ever parameters were
        fit in that round of fitting.
        If this is -1 then we will sample which ever parameters were fit
        in the last round of fitting.
        If this is -2 then any parameters that were ever fit will be sampled.
        If this is <= -3 or >= `model.n_rounds` then all parameters are sampled.

    Returns
    -------
    model : Model
        The model with MCMC estimated parameters and errors.
        The parameters are estimated as the mean of the samples.
        The errors are estimated as the standard deviation.
        This also has the chi-squared of the estimated parameters.
    flat_samples : jax.Array
        Array of samples from running MCMC.

    """
    token = mpi4jax.barrier(comm=todvec.comm)
    rank = todvec.comm.Get_rank()

    if sample_which >= 0 and sample_which < model.n_rounds:
        model.cur_round = sample_which
        to_fit = model.to_fit
    elif sample_which == -1:
        to_fit = model.to_fit
    elif sample_which == -2:
        to_fit = model.to_fit_ever
    else:
        to_fit = jnp.ones_like(model.to_fit_ever, dtype=bool)
    to_fit = jnp.array(to_fit)
    model = model.add_round(jnp.array(to_fit))

    init_pars = jnp.array(model.pars)
    init_errs = jnp.zeros_like(model.pars)
    final_pars = init_pars.copy()
    final_errs = init_errs.copy()

    prior_l, prior_u = model.priors
    scale = (jnp.abs(prior_l) + jnp.abs(prior_u)) / 2.0
    scale = jnp.where(scale == 0, 1, scale)
    init_pars = init_pars.at[:].multiply(1.0 / scale)
    npar = jnp.sum(to_fit)

    def _is_inf(pars, model):
        _ = (pars, model)
        return -1 * jnp.inf

    def _not_inf(pars, model):
        pars, _ = mpi4jax.bcast(pars, 0, comm=todvec.comm)
        temp_model = model.update(pars, init_errs, model.chisq)
        log_like = -0.5 * get_chisq(temp_model, todvec)
        return log_like

    @jax.jit
    def _log_prob(pars, model=model, init_pars=init_pars, scale=scale):
        full_pars = init_pars.at[to_fit].set(pars)
        full_pars = full_pars.at[:].multiply(scale)
        _, in_bounds = _prior_pars_fit(model.priors, full_pars, jnp.array(model.to_fit))
        log_prior = jnp.sum(
            jnp.where(in_bounds.at[model.to_fit].get(), 0, -1 * jnp.inf)
        )
        return jax.lax.cond(
            jnp.isfinite(log_prior),
            _not_inf,
            _is_inf,
            full_pars,
            model,
        )

    def _is_inf_grad(pars, model, scale):
        _ = (pars, model, scale)
        return jnp.inf * jnp.ones(npar)

    def _not_inf_grad(pars, model, scale):
        pars, _ = mpi4jax.bcast(pars, 0, comm=todvec.comm)
        temp_model = model.update(pars, init_errs, model.chisq)
        grad = get_grad(temp_model, todvec)
        grad = grad.at[:].multiply(scale)
        return grad.at[to_fit].get().ravel()

    @jax.jit
    def _log_prob_grad(pars, model=model, init_pars=init_pars, scale=scale):
        full_pars = init_pars.at[to_fit].set(pars)
        full_pars = full_pars.at[:].multiply(scale)
        _, in_bounds = _prior_pars_fit(model.priors, full_pars, jnp.array(model.to_fit))
        log_prior = jnp.sum(jnp.where(in_bounds.at[to_fit].get(), 0, -1 * jnp.inf))
        return jax.lax.cond(
            jnp.isfinite(log_prior),
            _not_inf_grad,
            _is_inf_grad,
            full_pars,
            model,
            scale,
        )

    key = jax.random.PRNGKey(0)
    key, token = mpi4jax.bcast(key, 0, comm=todvec.comm, token=token)
    chain = hmc(
        init_pars.at[to_fit].get().ravel(),
        _log_prob,
        _log_prob_grad,
        num_steps=num_steps,
        num_leaps=num_leaps,
        step_size=step_size,
        comm=todvec.comm,
        key=key,
    )
    flat_samples = chain.at[:].multiply(scale.at[to_fit].get())
    if rank == 0:
        final_pars = final_pars.at[to_fit].set(jnp.median(flat_samples, axis=0).ravel())
        final_errs = final_errs.at[to_fit].set(jnp.std(flat_samples, axis=0).ravel())
    final_pars, token = mpi4jax.bcast(final_pars, 0, comm=todvec.comm, token=token)
    final_errs, _ = mpi4jax.bcast(final_errs, 0, comm=todvec.comm, token=token)
    model = model.update(
        final_pars.block_until_ready(), final_errs.block_until_ready(), model.chisq
    )
    chisq = get_chisq(model, todvec)
    model = model.update(final_pars, final_errs, chisq)

    return model, flat_samples
