"""Metropolis-Hastings sampling utilities for WITCH models.

This module provides likelihood evaluators, proposal generation, and two
chain-running modes. The joint likelihood uses MPI collectives and must run
chains serially, while the single-dataset likelihood can run independent
chains concurrently in threads.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .containers import MetaModel
from .fitter import joint_objective


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
    chisq, _, _ = joint_objective(
        metamodel=cur_meta,
        do_loglike=True,
        do_grad=False,
        do_curve=False,
    )
    return -0.5 * chisq


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
    chisq, _, _ = cur_meta.datasets[dataset_ind].objective(
        cur_meta,
        dataset_ind,
        do_loglike=True,
        do_grad=False,
        do_curve=False,
    )

    return -0.5 * chisq


@partial(jax.jit, static_argnames=("prior_type",))
def draw_samp(
    metamodel,
    key,
    bound=2,
    prior_type="uniform",
    err_scale=1.0,
    parameter_scale=None,
):
    """
    Draw a masked proposal from a uniform or normal distribution.

    Uniform and normal proposals are centered on the current parameters and
    use ``metamodel.errs`` to set their step widths. Uniform-prior bounds are
    enforced by the acceptance test. Parameters outside ``metamodel.to_fit``
    remain unchanged.

    Parameters
    ----------
    metamodel : witch.containers.MetaModel
        Model containing parameters, priors, errors, and the fit mask.
    key : jax.Array
        JAX PRNG key used for the proposal.
    bound : float, default=2
        Fallback scale for proposal widths when fitted errors or prior bounds
        cannot provide a finite width.
    prior_type : {"uniform", "normal"}, default="uniform"
        Prior and symmetric proposal distribution to use.
    err_scale : float, default=1.0
        Scale factor for proposal widths derived from ``metamodel.errs``.
    parameter_scale : jax.Array, optional
        Fixed scale for normalized coordinates. If omitted, derive it from the
        current parameters and errors.

    Returns
    -------
    jax.Array
        Proposed full parameter vector.
    """
    pars = jnp.asarray(metamodel.parameters)
    if parameter_scale is None:
        errs_for_scale = jnp.abs(jnp.asarray(metamodel.errs))
        parameter_scale = jnp.where(
            pars != 0,
            jnp.abs(pars),
            jnp.where(errs_for_scale > 0, errs_for_scale, jnp.ones_like(pars)),
        )
    else:
        parameter_scale = jnp.asarray(parameter_scale)
    normalized_pars = pars / parameter_scale
    if prior_type == "uniform":
        lower = jnp.asarray(metamodel.priors[0]) / parameter_scale
        upper = jnp.asarray(metamodel.priors[1]) / parameter_scale
        errs = jnp.asarray(metamodel.errs) * err_scale / parameter_scale
        fallback = jnp.where(
            jnp.isfinite(lower) & jnp.isfinite(upper),
            (upper - lower) / 10.0,
            jnp.ones_like(normalized_pars) / bound,
        )
        valid_errs = jnp.isfinite(errs) & (errs > 0)
        step_width = jnp.where(
            valid_errs,
            jnp.clip(errs, fallback * 0.1, fallback),
            fallback,
        )
        normalized_new_pars = normalized_pars + step_width * (
            2 * jax.random.uniform(key, shape=pars.shape) - 1
        )
    elif prior_type == "normal":
        # For normal proposals, sample around current pars using metamodel.errs
        # scaled by `err_scale` (passed from the chain runner/CLI).
        errs = jnp.asarray(metamodel.errs) * err_scale / parameter_scale
        normalized_new_pars = normalized_pars + errs * jax.random.normal(
            key, shape=pars.shape
        )
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    new_pars = normalized_new_pars * parameter_scale
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

    The likelihood function must return a log likelihood. Proposals are
    symmetric random walks, so their forward and reverse proposal terms cancel
    in the acceptance ratio. The selected prior is included separately.

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
        Prior and symmetric proposal distribution.

    Returns
    -------
    samples : numpy.ndarray
        Samples with shape ``(num_samples, num_parameters)``.
    acceptance_rate : float
        Fraction of proposals accepted.
    """
    samples = np.zeros((num_samples, len(metamodel.parameters)))
    current_state = metamodel.parameters
    initial_errs = jnp.abs(jnp.asarray(metamodel.errs))
    parameter_scale = jnp.where(
        current_state != 0,
        jnp.abs(current_state),
        jnp.where(initial_errs > 0, initial_errs, jnp.ones_like(current_state)),
    )
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
        # Propose a candidate state using a symmetric random walk.
        key, sample_key = jax.random.split(key)
        proposal_metamodel = _updated_metamodel(metamodel, current_state)
        candidate = draw_samp(
            metamodel=proposal_metamodel,
            key=sample_key,
            bound=bound,
            prior_type=prior_type,
            err_scale=prior_err_scale,
            parameter_scale=parameter_scale,
        )

        # Compare log acceptance probabilities to avoid exponentiating likelihoods.
        p_like_candidate = calc_like_func(
            metamodel=metamodel,
            pars=candidate,
            dataset_ind=dataset_ind,
        )
        p_candidate = p_like_candidate + _log_prior(
            metamodel, candidate, prior_err_scale
        )

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
    metamodel,
    num_samples,
    num_chains,
    bound=2,
    seed=0,
    prior_type="uniform",
    prior_err_scale=1.0,
):
    """
    Run multiple chains serially with the MPI-aware joint likelihood.

    This mode is required for ``joint_objective`` because its MPI collectives
    must be called in the same order by all ranks.
    """
    jax.block_until_ready(calc_like_joint(metamodel, metamodel.parameters))
    jax.block_until_ready(
        draw_samp(metamodel, jax.random.key(seed), bound, prior_type, prior_err_scale)
    )

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
    jax.block_until_ready(
        draw_samp(metamodel, jax.random.key(seed), bound, prior_type, prior_err_scale)
    )

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
