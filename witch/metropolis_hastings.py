"""Metropolis-Hastings sampling utilities for WITCH models.

This module provides likelihood evaluators, proposal generation, and two
chain-running modes. The joint likelihood uses MPI collectives and must run
chains serially, while the single-dataset likelihood can run independent
chains concurrently in threads.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from copy import copy


from .fitter import joint_objective
from .containers import MetaModel
from tqdm import tqdm


def _updated_metamodel(metamodel: MetaModel, pars: np.ndarray):
    """
    Simple helper function which updates a metamodel with new pars.

    Parameters
    ----------
    metamodel : witch.containers.MetaModel
        Metamodel to update
    pars : np.ndarray
        Numpy array containing parameters to update metamodel with.

    Returns
    -------
    copy(metamodel) : witch.containers.MetaModel
        Copy of metamodel with updated parameters.
    """
    cur_pars = copy(metamodel.parameters).at[:].set(pars)
    return copy(metamodel).update(
        pars=cur_pars, errs=metamodel.errs, cov=metamodel.cov, chisq=metamodel.chisq
    )


@jax.jit
def calc_like_joint(metamodel: MetaModel, pars: np.ndarray, dataset_ind: int = 0):
    """
    Calculate the joint likelihood for the metamodel to the parameters pars.
    Note the dummy variable since this function needs to have the same
    signature as calc_like_dataset.

    Parameters
    ----------
    metamodel : witch.containers.MetaModel
        Metamodel to compute likelihood for.
    pars : np.ndarray
        Numpy array containing parameters to update metamodel with.
    dataset_ind : int
        Dummy variable.

    Returns
    -------
    loglike : the log likelihood of the data to the parameters.
    """
    del dataset_ind
    cur_meta = _updated_metamodel(metamodel, pars)
    loglike, _, _ = joint_objective(
        metamodel=cur_meta,
        do_loglike=True,
        do_grad=False,
        do_curve=False,
    )
    return loglike


@partial(jax.jit, static_argnames=("dataset_ind",))
def calc_like_dataset(metamodel, pars, dataset_ind=0):
    """
    Calculate the joint likelihood for one dataset in the metamodel (i.e. M2, X-ray) to the parameters pars.
    Note the dummy variable since this function needs to have the same
    signature as calc_like_dataset.

    Parameters
    ----------
    metamodel : witch.containers.MetaModel
        Metamodel to compute likelihood for.
    pars : np.ndarray
        Numpy array containing parameters to update metamodel with.
    dataset_ind : int
        Which dataset to compute the log like for.

    Returns
    -------
    loglike : the log likelihood of the data to the parameters.
    """
    cur_meta = _updated_metamodel(metamodel, pars)
    loglike, _, _ = cur_meta.datasets[dataset_ind].objective(
        cur_meta,
        dataset_ind,
        do_loglike=True,
        do_grad=False,
        do_curve=False,
    )

    return loglike


@partial(jax.jit, static_argnames=("prior_type",))
def draw_samp(metamodel, key, bound=2, prior_type="uniform", err_scale=1.0):
    """
    Draw a masked proposal from a uniform or normal distribution.

    Uniform proposals use the model's prior bounds, replacing unbounded
    limits with values scaled from the current parameters. Normal proposals
    are centered on the current parameters and use ``metamodel.errs`` as
    standard deviations. Parameters outside ``metamodel.to_fit`` remain
    unchanged.

    Parameters
    ----------
    metamodel : witch.containers.MetaModel
        Model containing parameters, priors, errors, and the fit mask.
    key : jax.Array
        JAX PRNG key used for the proposal.
    bound : float, default=2
        Scale factor used to replace unbounded uniform limits.
    prior_type : {"uniform", "normal"}, default="uniform"
        Proposal distribution to use.

    Returns
    -------
    jax.Array
        Proposed full parameter vector.
    """
    pars = jnp.asarray(metamodel.parameters)
    if prior_type == "uniform":
        lower = jnp.asarray(metamodel.priors[0])
        upper = jnp.asarray(metamodel.priors[1])
        unbounded = jnp.isinf(lower)
        lower = jnp.where(unbounded, pars / bound, lower)
        upper = jnp.where(unbounded, pars * bound, upper)
        new_pars = jax.random.uniform(key, shape=pars.shape, minval=lower, maxval=upper)
    elif prior_type == "normal":
        # For normal proposals, sample around current pars using metamodel.errs
        # scaled by `err_scale` (passed from the chain runner/CLI).
        errs = jnp.asarray(metamodel.errs)
        new_pars = pars + errs * err_scale * jax.random.normal(key, shape=pars.shape)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    return jnp.where(metamodel.to_fit, new_pars, pars)


def metropolis_hastings(
    metamodel,
    num_samples,
    bound=2,
    seed=0,
    chain_id=0,
    calc_like_func=calc_like_dataset,
    dataset_ind=0,
    prior_type="uniform",
    prior_err_scale=1.0,
):
    """
    Run one Metropolis-Hastings chain using log-likelihoods.

    The likelihood function must return a log likelihood. Since the proposal
    distribution matches the selected uniform or normal prior, the proposal
    and prior terms cancel in the acceptance ratio.

    Parameters
    ----------
    metamodel : witch.containers.MetaModel
        Model used to evaluate the likelihood.
    num_samples : int
        Number of samples to store.
    bound : float, default=2
        Scale factor for unbounded uniform proposal limits.
    seed : int, default=0
        Seed for this chain's random number generators.
    chain_id : int, default=0
        Identifier displayed by the progress bar.
    calc_like_func : callable, default=calc_like_dataset
        Function returning the log likelihood for a parameter vector.
    dataset_ind : int, default=0
        Dataset index passed to ``calc_like_func``.
    prior_type : {"uniform", "normal"}, default="uniform"
        Proposal/prior distribution.

    Returns
    -------
    samples : numpy.ndarray
        Samples with shape ``(num_samples, num_parameters)``.
    acceptance_rate : float
        Fraction of proposals accepted.
    """
    samples = np.zeros((num_samples, len(metamodel.parameters)))
    current_state = metamodel.parameters
    accepted_count = 0
    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)
    # Compute current log-posterior (log likelihood + log prior when gaussian priors used)
    p_like_current = calc_like_func(
        metamodel=metamodel,
        pars=current_state,
        dataset_ind=dataset_ind,
    )
    # helper to compute log prior (supports gaussian via metamodel.errs and uniform bounds)
    def _log_prior(meta, pars, err_scale=1.0):
        lower = jnp.asarray(meta.priors[0])
        upper = jnp.asarray(meta.priors[1])
        # If any prior is finite, treat as uniform (already handled by bounds)
        # For gaussian-style priors we assume metamodel.errs provides 1-sigma and
        # that gaussian priors are desired when prior_type == 'normal'.
        if prior_type == "normal":
            errs = jnp.asarray(meta.errs) * err_scale
            # avoid division by zero
            errs = jnp.where(errs <= 0, jnp.inf, errs)
            # Gaussian log-prior (up to additive constant)
            return -0.5 * jnp.sum(((pars - jnp.asarray(meta.parameters)) / errs) ** 2)
        else:
            # Uniform priors: return 0 if inside bounds else -inf
            inside = jnp.logical_and(pars >= lower, pars <= upper)
            return jnp.where(jnp.all(inside), 0.0, -jnp.inf)

    p_current = p_like_current + _log_prior(metamodel, current_state, prior_err_scale)

    for i in tqdm(
        range(num_samples),
        desc=f"Chain {chain_id + 1}",
        unit="step",
    ):
        # Propose a candidate state using a symmetric Normal distribution
        key, sample_key = jax.random.split(key)
        candidate = draw_samp(
            metamodel=metamodel,
            key=sample_key,
            bound=bound,
            prior_type=prior_type,
            err_scale=prior_err_scale,
        )

        # Compare log acceptance probabilities to avoid exponentiating likelihoods.
        p_like_candidate = calc_like_func(
            metamodel=metamodel,
            pars=candidate,
            dataset_ind=dataset_ind,
        )
        p_candidate = p_like_candidate + _log_prior(metamodel, candidate, prior_err_scale)

        current_logpost = float(p_current)
        candidate_logpost = float(p_candidate)
        if np.isneginf(current_logpost):
            accept = False
        elif np.isfinite(candidate_logpost):
            log_alpha = min(0.0, candidate_logpost - current_logpost)
            accept = np.log(rng.random()) < log_alpha
        else:
            accept = False

        # Accept or reject the candidate
        if accept:
            current_state = candidate
            p_current = p_candidate
            accepted_count += 1

        samples[i] = current_state

    acceptance_rate = accepted_count / num_samples
    return samples, acceptance_rate


def run_chains_serial(
    metamodel, num_samples, num_chains, bound=2, seed=0, prior_type="uniform", prior_err_scale=1.0
):
    """
    Run multiple chains serially with the MPI-aware joint likelihood.

    This mode is required for ``joint_objective`` because its MPI collectives
    must be called in the same order by all ranks.
    """
    jax.block_until_ready(calc_like_joint(metamodel, metamodel.parameters))
    jax.block_until_ready(draw_samp(metamodel, jax.random.key(seed), bound, prior_type, prior_err_scale))

    chains = []
    for chain_id in range(num_chains):
        chains.append(
            metropolis_hastings(
                metamodel,
                num_samples,
                bound,
                seed + chain_id,
                chain_id,
                calc_like_joint,
                0,
                prior_type,
                prior_err_scale,
            )
        )
    return chains


def run_chains_parallel(
    metamodel,
    num_samples,
    num_chains,
    bound=2,
    seed=0,
    dataset_ind=0,
    prior_type="uniform",
    prior_err_scale=1.0,
):
    """
    Run independent chains concurrently for one dataset.

    This mode uses threads and ``dataset[dataset_ind].objective``. It avoids
    interleaving MPI collectives and is intended for independent local
    dataset likelihood evaluations.
    """
    # Compile shared JAX kernels before concurrent execution.
    jax.block_until_ready(
        calc_like_dataset(metamodel, metamodel.parameters, dataset_ind)
    )
    jax.block_until_ready(draw_samp(metamodel, jax.random.key(seed), bound, prior_type, prior_err_scale))

    with ThreadPoolExecutor(max_workers=num_chains) as executor:
        futures = [
            executor.submit(
                metropolis_hastings,
                metamodel,
                num_samples,
                bound,
                seed + chain_id,
                chain_id,
                calc_like_dataset,
                dataset_ind,
                prior_type,
                prior_err_scale,
            )
            for chain_id in range(num_chains)
        ]
        chains = [future.result() for future in futures]

    return chains
