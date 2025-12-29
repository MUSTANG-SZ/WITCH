import sys
import time
from copy import copy, deepcopy
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import mpi4jax
import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from .containers import MetaModel
from .objective import joint_objective
from .utils import NullComm

_cache = {}


@jax.jit
def invsafe(matrix: jax.Array, thresh: float = 1e-16) -> jax.Array:
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
    s_inv = jnp.array(jnp.where(jnp.abs(s) <= thresh * jnp.max(s), 0, 1 / s))

    return jnp.dot(jnp.transpose(v), jnp.dot(jnp.diag(s_inv), jnp.transpose(u)))


@partial(jax.jit, static_argnums=(2,))
def invscale(matrix: jax.Array, thresh: float = 1e-16, do_invsafe=False) -> jax.Array:
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
    vec = jnp.array(jnp.where(diag != 0, 1.0 / jnp.sqrt(jnp.abs(diag)), thresh))
    mm = jnp.outer(vec, vec)

    if do_invsafe:
        return mm * invsafe(mm * matrix, thresh)
    return mm * jnp.linalg.inv(mm * matrix)


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
    metamodel,
    new_metamodel,
    grad,
    new_grad,
    curve,
    new_curve,
    delta_chisq,
    new_delta_chisq,
    lmd,
):
    del new_metamodel
    del new_grad
    del new_curve
    del new_delta_chisq
    lmd = jnp.where(lmd == 0, 1, 2 * lmd)
    return metamodel, grad, curve, delta_chisq, lmd


def _success(
    metamodel,
    new_metamodel,
    grad,
    new_grad,
    curve,
    new_curve,
    delta_chisq,
    new_delta_chisq,
    lmd,
):
    del metamodel
    del grad
    del curve
    del delta_chisq
    lmd = jnp.where(lmd < 0.2, 0, lmd / jnp.sqrt(2))
    return new_metamodel, new_grad, new_curve, new_delta_chisq, lmd


def run_lmfit(
    metamodel: MetaModel,
    maxiter: int = 10,
    chitol: float = 1e-5,
) -> tuple[MetaModel, int, jax.Array]:
    """
    Fit a set of models to datasets jointly.
    This uses a modified Levenberg–Marquardt fitter with flat priors.
    This function is MPI aware.

    Parameters
    ----------
    metamodel : MetaModel
        The MetaModel that describes the models and datasets.
    maxiter : int, default: 10
        The maximum number of iterations to fit.
    chitol : float, default: 1e-5
        The delta chisq to use as the convergence criteria.

    Returns
    -------
    models : tuple[Model, ...]
        Model with the final set of fit parameters, errors, and chisq.
    final_iter : int
        The number of iterations the fitter ran for.
    delta_chisq : float
        The final delta chisq.
    """
    zero = jnp.array(0.0, jnp.float32)
    chitol = jnp.float32(chitol)
    tf = np.where(np.array(metamodel.to_fit))[0]

    @jax.jit
    def _cond_func(val):
        i, delta_chisq, lmd, *_ = val
        iterbool = jax.lax.lt(i, maxiter)
        chisqbool = jax.lax.ge(delta_chisq, chitol) + jax.lax.gt(lmd, zero)
        return iterbool * chisqbool

    @jax.jit
    def _body_func(val):
        i, delta_chisq, lmd, metamodel, curve, grad = val
        curve_use = curve.at[:].add(lmd * jnp.diag(jnp.diag(curve)))
        # Get the step
        step = jnp.dot(
            invscale(curve_use.at[tf, :].get().at[:, tf].get()), grad.at[tf].get()
        )
        new_pars, to_fit = _prior_pars_fit(
            metamodel.priors,
            metamodel.parameters.at[tf].add(step),
            jnp.array(metamodel.to_fit),
        )
        # Get errs
        errs = jnp.where(
            to_fit, jnp.sqrt(jnp.diag(invscale(curve_use, do_invsafe=True))), 0
        )
        # Now lets get an updated model
        new_metamodel = copy(metamodel).update(new_pars, errs, metamodel.chisq)
        new_chisq, new_grad, new_curve = joint_objective(
            new_metamodel, True, True, True
        )
        new_metamodel = copy(new_metamodel).update(new_pars, errs, new_chisq)

        new_delta_chisq = jnp.astype(metamodel.chisq - new_metamodel.chisq, jnp.float32)
        metamodel, grad, curve, delta_chisq, lmd = jax.lax.cond(
            new_delta_chisq > 0,
            _success,
            _failure,
            metamodel,
            new_metamodel,
            grad,
            new_grad,
            curve,
            new_curve,
            delta_chisq,
            new_delta_chisq,
            lmd,
        )

        return (i + 1, delta_chisq, lmd, metamodel, curve, grad)

    pars, _ = _prior_pars_fit(
        metamodel.priors, metamodel.parameters, jnp.array(metamodel.to_fit)
    )
    metamodel = metamodel.update(pars, metamodel.errors, metamodel.chisq)
    _, grad, curve = joint_objective(metamodel, True, True, True)
    i, delta_chisq, _, metamodel, *_ = jax.lax.while_loop(
        _cond_func,
        _body_func,
        (0, jnp.astype(jnp.inf, jnp.float32), zero.copy(), metamodel, curve, grad),
    )

    return metamodel, i, delta_chisq


def hmc(
    params: jax.Array,
    log_prob: Callable[[jax.Array], jax.Array],
    log_prob_grad: Callable[[jax.Array], jax.Array],
    num_steps: int,
    num_leaps: int,
    step_size: float,
    comm: MPI.Comm | MPI.Intracomm | NullComm,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
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
    comm : MPI.Comm | MPI.Intracomm | NullComm
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

    @partial(jax.jit, inline=True)
    def _leap(_, args):
        params, momentum, step_size = args
        momentum = momentum.at[:].add(0.5 * step_size * log_prob_grad(params))  # kick
        params = params.at[:].add(step_size * momentum)  # drift
        momentum = momentum.at[:].add(0.5 * step_size * log_prob_grad(params))  # kick

        return params, momentum, step_size

    @jax.jit
    def _sample(key, params, step_size):
        mpi4jax.barrier(comm=comm)
        key = mpi4jax.bcast(key, 0, comm=comm)
        key, uniform_key = jax.random.split(key, 2)

        # generate random momentum
        momentum = vnorm(jax.random.split(key, npar))
        new_params, new_momentum, _ = jax.lax.fori_loop(
            0, num_leaps, _leap, (params, momentum, step_size)
        )

        # MH correction
        dpe = log_prob(new_params) - log_prob(params)
        dke = -0.5 * (jnp.sum(new_momentum**2) - jnp.sum(momentum**2))
        log_accept = dke + dpe
        accept_prob = jnp.minimum(jnp.exp(log_accept), 1)
        accept = jax.random.uniform(uniform_key) < accept_prob
        params = jax.lax.select(accept * ones, new_params, params)

        return key, params, accept_prob

    @partial(jax.jit, donate_argnums=(0, 1))
    def _update(chain, accept_prob, params, prob, i):
        chain = chain.at[i].set(params)
        accept_prob = accept_prob.at[i].set(prob)
        return chain, accept_prob

    c_sample = _cache.get("hmc_sample", None)
    if c_sample is None:
        if rank == 0:
            print(f"Compiling MC sample function. This can take a few minutes!")
        t0 = time.time()
        t_sample = _sample.trace(key, params, step_size)
        l_sample = t_sample.lower()
        c_sample = l_sample.compile()
        t1 = time.time()
        if rank == 0:
            print(f"Compiled MC sample function in {t1-t0} s")
        _cache["hmc_sample"] = c_sample
    else:
        if rank == 0:
            print("Using cached sample function")

    if rank == 0:
        chain = jnp.zeros((num_steps, len(params)))
        accept_prob = jnp.zeros(num_steps)
    else:
        chain = jnp.zeros(0)
        accept_prob = jnp.zeros(0)
    for i in tqdm(range(num_steps), disable=(rank != 0)):
        key, params, prob = c_sample(key, params, step_size)
        if rank == 0:
            chain, accept_prob = _update(chain, accept_prob, params, prob, i)
    if rank == 0:
        print(f"Accepted {accept_prob.mean():.2%} of samples")
        sys.stdout.flush()
    return chain, accept_prob


def run_mcmc(
    metamodel: MetaModel,
    num_steps: int = 5000,
    num_leaps: int = 10,
    step_size: float = 0.02,
    sample_which: int = -1,
    burn_in: float = 0.1,
    max_tries: int = 20,
) -> tuple[MetaModel, jax.Array]:
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
    metamodel : MetaModel
        The MetaModel that describes the models and datasets.
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
    burn_in : float, default: 0.1
        Fractional burn-in period of samples to discard
    max_tries : int, default: 10
        Number of tries to tune step size that will be attemted.
        If 0 no tuning will be run.

    Returns
    -------
    metamodel : MetaModel
        The metamodel with MCMC estimated parameters and errors.
        The parameters are estimated as the mean of the samples.
        The errors are estimated as the standard deviation.
        This also has the chi-squared of the estimated parameters.
    flat_samples : jax.Array
        Array of samples from running MCMC.

    """
    global_comm = metamodel.global_comm
    mpi4jax.barrier(comm=global_comm)
    rank = global_comm.Get_rank()

    if burn_in >= 1.0 or burn_in < 0:
        raise ValueError("Error: burn_in must be in range [0, 1)")

    if sample_which >= 0 and sample_which < metamodel.n_rounds:
        metamodel.set_round(sample_which)
        to_fit = metamodel.to_fit
    elif sample_which == -1:
        to_fit = metamodel.to_fit
    elif sample_which == -2:
        to_fit = metamodel.to_fit_ever
    else:
        to_fit = jnp.ones_like(metamodel.to_fit_ever, dtype=bool)
    to_fit = jnp.array(to_fit)
    metamodel = metamodel.add_round(jnp.array(to_fit))

    init_pars = jnp.array(metamodel.parameters)
    init_errs = jnp.zeros_like(init_pars)
    final_pars = init_pars.copy()
    final_errs = init_errs.copy()

    prior_l, prior_u = metamodel.priors
    scale = (jnp.abs(prior_l) + jnp.abs(prior_u)) / 2.0
    scale = jnp.where(jnp.isfinite(scale), scale, init_pars)
    scale = jnp.where(scale == 0, 1, scale)
    init_pars = init_pars.at[:].multiply(1.0 / scale)

    @jax.jit
    def _log_prob(pars, metamodel=metamodel, init_pars=init_pars):
        full_pars = init_pars.at[to_fit].set(pars)
        full_pars = full_pars.at[:].multiply(scale)
        _, in_bounds = _prior_pars_fit(
            metamodel.priors, full_pars, jnp.array(metamodel.to_fit)
        )
        log_prior = jnp.sum(
            jnp.where(
                jnp.array(jnp.where(metamodel.to_fit, in_bounds, True)), 0, -1 * jnp.inf
            )
        )
        temp_metamodel = copy(metamodel).update(full_pars, init_errs, jnp.array(0))
        chisq, *_ = joint_objective(temp_metamodel, True, False, False)
        log_like = -0.5 * chisq
        log_like = log_like + log_prior
        return log_like

    @jax.jit
    def _log_prob_grad(pars, metamodel=metamodel, init_pars=init_pars):
        full_pars = init_pars.at[to_fit].set(pars)
        full_pars = full_pars.at[:].multiply(scale)
        _, in_bounds = _prior_pars_fit(
            metamodel.priors, full_pars, jnp.array(metamodel.to_fit)
        )
        log_prior = jnp.sum(
            jnp.where(
                jnp.array(jnp.where(metamodel.to_fit, in_bounds, True)), 0, -1 * jnp.inf
            )
        )
        temp_metamodel = copy(metamodel).update(full_pars, init_errs, jnp.array(0))
        _, grad, _ = joint_objective(temp_metamodel, False, True, False)
        grad = grad.at[:].multiply(1.0 / scale)
        log_like_grad = grad.at[to_fit].get().ravel()
        log_like_grad = log_like_grad.at[:].add(log_prior)
        return log_like_grad

    def _get_step_size(step_size, key, comm, max_tries):
        mpi4jax.barrier(comm=global_comm)

        def _can_interp(probs):
            probs = jnp.array(probs)
            msk = jnp.isfinite(probs) * probs > 0
            if jnp.sum(msk) < 2:
                return False
            dprob = (probs[msk] - 0.63) / 0.13
            for n in range(1, 4):
                upper, lower = (
                    jnp.sum((dprob <= n) * (dprob >= 0)),
                    jnp.sum((dprob >= -1 * n) * (dprob < 0)),
                )
                if upper >= 1 and lower >= 1 and upper + lower >= n + 1:
                    return True
            return False

        def _hmc(n_steps, step_size):
            _, chain_probs = hmc(
                init_pars.at[to_fit].get().ravel(),
                _log_prob,
                _log_prob_grad,
                num_steps=n_steps,
                num_leaps=num_leaps,
                step_size=step_size,
                comm=comm,
                key=key,
            )
            prob = 0.0
            if rank == 0:
                prob = jnp.mean(chain_probs)
            prob = jnp.array(mpi4jax.bcast(prob, 0, comm=comm))

            return prob

        if max_tries == 0:
            return step_size
        step_fac = 1
        n_steps = 1
        i = 0
        step_sizes = []
        probs = []
        for i in range(max_tries):
            mpi4jax.barrier(comm=global_comm)
            step_size *= step_fac
            if rank == 0:
                print(f"Trying step size {step_size}")
            prob = _hmc(n_steps, step_size)
            step_sizes += [step_size]
            probs += [prob]

            if prob == 0 or not jnp.isfinite(prob):
                step_fac = 1e-1
                n_steps = 1
            elif prob > 0.6 and prob < 0.66:
                break
            elif _can_interp(probs):
                break
            elif prob <= 0.1:
                step_fac = 1 / 5
                n_steps = 3
            elif prob >= 0.9:
                step_fac = 5
                n_steps = 3
            elif prob <= 0.5:
                step_fac = 1 / 3
                n_steps = 5
            elif prob >= 0.76:
                step_fac = 3
                n_steps = 5
            elif prob > 0.5 and prob < 0.63:
                step_fac = 1 / 1.5
                n_steps = 10
            elif prob < 0.76 and prob > 0.63:
                step_fac = 1 / 1.5
                n_steps = 10
            else:
                raise ValueError(f"Unclear what to do with probability {prob}")
        if probs[-1] > 0.60 and probs[-1] < 0.66:
            if rank == 0:
                print(
                    f"Final step size of {step_size} with an accept probability of {probs[-1]}, took {i} tries to find."
                )
                sys.stdout.flush()
            return step_sizes[-1]
        mpi4jax.barrier(comm=global_comm)
        probs = jnp.array(probs)
        step_sizes = jnp.array(step_sizes)
        msk = jnp.isfinite(probs) + (probs > 0)
        srt = jnp.argsort(probs[msk])
        if jnp.sum(msk) == 0:
            raise ValueError(
                "No finite non-zero step sizes found! Manual intervention is needed!"
            )
        step_size_interp = jnp.interp(
            jnp.array([0.63]), probs[msk][srt], step_sizes[msk][srt]
        )
        step_size = float(step_size_interp[0].item())
        prob = _hmc(5, step_size)
        if rank == 0:
            print(
                f"Interpolated to get a step size of {step_size} with an acceptance probability {prob:.2%}"
            )
            sys.stdout.flush()
        return step_size

    key = jax.random.key(0)
    key = jnp.array(mpi4jax.bcast(key, 0, comm=global_comm))

    step_size = _get_step_size(
        step_size=step_size,
        key=key,
        comm=global_comm,
        max_tries=max_tries,
    )

    chain, _ = hmc(
        init_pars.at[to_fit].get().ravel(),
        _log_prob,
        _log_prob_grad,
        num_steps=num_steps,
        num_leaps=num_leaps,
        step_size=step_size,
        comm=global_comm,
        key=key,
    )
    flat_samples = chain.at[:].multiply(scale.at[to_fit].get())

    if rank == 0:
        burn_in = int(num_steps * burn_in)
        flat_samples = flat_samples[burn_in:]
        final_pars = final_pars.at[to_fit].set(jnp.median(flat_samples, axis=0).ravel())
        final_errs = final_errs.at[to_fit].set(jnp.std(flat_samples, axis=0).ravel())
    final_pars = jnp.array(mpi4jax.bcast(final_pars, 0, comm=global_comm))
    final_errs = jnp.array(mpi4jax.bcast(final_errs, 0, comm=global_comm))
    metamodel = metamodel.update(
        final_pars.block_until_ready(),
        final_errs.block_until_ready(),
        jnp.array(0).block_until_ready(),
    )
    chisq, *_ = joint_objective(metamodel, True, False, False)
    metamodel = metamodel.update(final_pars, final_errs, chisq)
    jax.block_until_ready(metamodel)

    return metamodel, flat_samples
