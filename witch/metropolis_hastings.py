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
from .objective import joint_objective


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


@partial(jax.jit, static_argnames=("prior_type", "proposal"))
def draw_samp(
    metamodel,
    key,
    bound=2,
    prior_type="uniform",
    err_scale=1.0,
    parameter_scale=None,
    proposal="diag",
    pars=None,
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
    proposal : {"diag", "cov"}, default="diag"
        ``diag`` uses independent per-parameter steps. ``cov`` draws a
        correlated gaussian step from ``metamodel.cov`` restricted to the
        fitted parameters and scaled by ``2.38 / sqrt(n_fit) * err_scale``
        (the optimal random-walk scaling). It falls back to ``diag`` with
        the ``prior_type`` diagonal step if the covariance is not positive definite.
    pars : jax.Array, optional
        Parameters to center the proposal on. Defaults to ``metamodel.parameters``.
        Passing the chain state here avoids building an updated metamodel.

    Returns
    -------
    jax.Array
        Proposed full parameter vector.
    """
    pars = jnp.asarray(metamodel.parameters if pars is None else pars)
    if proposal == "cov":
        mask = jnp.asarray(metamodel.to_fit)
        n_fit = jnp.maximum(jnp.sum(mask), 1)
        cov = jnp.asarray(metamodel.cov)
        sub = jnp.where(mask[:, None] & mask[None, :], cov, 0.0)
        diag = jnp.diag(sub)
        # Unit variance for unfit params keeps the cholesky well defined
        sub = sub + jnp.diag(jnp.where(mask, 1e-10 * diag, 1.0))
        chol = jnp.linalg.cholesky(sub)
        step = chol @ jax.random.normal(key, shape=pars.shape)
        step = step * 2.38 / jnp.sqrt(n_fit) * err_scale
        ok = jnp.all(jnp.isfinite(chol)) & jnp.all(jnp.where(mask, diag > 0, True))
        return jax.lax.cond(
            ok,
            lambda: jnp.where(mask, pars + step, pars),
            lambda: draw_samp.__wrapped__(
                metamodel,
                key,
                bound,
                prior_type,
                err_scale,
                parameter_scale,
                "diag",
                pars,
            ),
        )
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


@partial(jax.jit, static_argnames=("prior_type",))
def _log_prior(pars, lower, upper, center, errs, prior_type="uniform"):
    """
    Log prior (up to a constant) of a parameter vector.

    Parameters
    ----------
    pars : jax.Array
        Parameters to evaluate the prior at.
    lower, upper : jax.Array
        Uniform prior bounds, used when ``prior_type`` is "uniform".
    center : jax.Array
        Mean of the gaussian prior, used when ``prior_type`` is "normal".
    errs : jax.Array
        Standard deviations (already scaled) of the gaussian prior.
    prior_type : {"uniform", "normal"}, default="uniform"
        Which prior to use.

    Returns
    -------
    log_prior : jax.Array
        Scalar log prior, ``-inf`` outside uniform bounds.
    """
    if prior_type == "normal":
        errs = jnp.where(errs <= 0, jnp.inf, errs)
        return -0.5 * jnp.sum(((pars - center) / errs) ** 2)
    inside = jnp.logical_and(pars >= lower, pars <= upper)
    return jnp.where(jnp.all(inside), 0.0, -jnp.inf)


@jax.jit
def _accept(key, state, logpost, candidate, candidate_logpost):
    """
    Metropolis accept/reject on device, so the chain never syncs with the host.

    Parameters
    ----------
    key : jax.Array
        PRNG key for the uniform draw.
    state, candidate : jax.Array
        Current and proposed parameters.
    logpost, candidate_logpost : jax.Array
        Log posteriors of ``state`` and ``candidate``.

    Returns
    -------
    state : jax.Array
        New chain state.
    logpost : jax.Array
        Log posterior of the new state.
    accept : jax.Array
        Whether the candidate was accepted.
    """
    valid = jnp.isfinite(candidate_logpost) & ~jnp.isneginf(logpost)
    log_alpha = jnp.minimum(0.0, jnp.where(valid, candidate_logpost - logpost, 0.0))
    accept = valid & (jnp.log(jax.random.uniform(key)) < log_alpha)
    return (
        jnp.where(accept, candidate, state),
        jnp.where(accept, candidate_logpost, logpost),
        accept,
    )


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
    progress=True,
    proposal="diag",
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
    prior_err_scale : float, default=1.0
        Multiplier on ``metamodel.errs`` for proposal width and normal prior.
    progress : bool, default=True
        Whether to show a progress bar.
    proposal : {"diag", "cov"}, default="diag"
        Proposal type, see ``draw_samp``.

    Returns
    -------
    samples : numpy.ndarray
        Samples with shape ``(num_samples, num_parameters)``.
    acceptance_rate : float
        Fraction of proposals accepted.
    """
    current_state = jnp.asarray(metamodel.parameters)
    initial_errs = jnp.abs(jnp.asarray(metamodel.errs))
    parameter_scale = jnp.where(
        current_state != 0,
        jnp.abs(current_state),
        jnp.where(initial_errs > 0, initial_errs, jnp.ones_like(current_state)),
    )
    key = jax.random.key(seed)
    prior_args = (
        jnp.asarray(metamodel.priors[0]),
        jnp.asarray(metamodel.priors[1]),
        current_state,
        jnp.asarray(metamodel.errs) * prior_err_scale,
    )

    def log_prior(pars):
        return _log_prior(pars, *prior_args, prior_type=prior_type)

    # Compute current log-posterior (log likelihood + log prior)
    p_current = calc_like_func(
        metamodel=metamodel,
        pars=current_state,
        dataset_ind=dataset_ind,
    ) + log_prior(current_state)

    states = []
    accepted = []
    for i in tqdm(
        range(num_samples),
        desc=f"Chain {chain_id + 1}",
        unit="step",
        disable=not progress,
    ):
        # Propose a candidate state using a symmetric random walk.
        key, sample_key, accept_key = jax.random.split(key, 3)
        candidate = draw_samp(
            metamodel=metamodel,
            key=sample_key,
            bound=bound,
            prior_type=prior_type,
            err_scale=prior_err_scale,
            parameter_scale=parameter_scale,
            proposal=proposal,
            pars=current_state,
        )

        # Compare log posteriors to avoid exponentiating likelihoods.
        p_candidate = calc_like_func(
            metamodel=metamodel,
            pars=candidate,
            dataset_ind=dataset_ind,
        ) + log_prior(candidate)
        current_state, p_current, accept = _accept(
            accept_key, current_state, p_current, candidate, p_candidate
        )
        states.append(current_state)
        accepted.append(accept)
        # Dispatch is async, sync now and then so the progress bar is honest
        if i % 64 == 63:
            p_current.block_until_ready()

    samples = np.asarray(jnp.stack(states))
    accepted_count = int(jnp.sum(jnp.stack(accepted)))
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
    progress=True,
    proposal="diag",
):
    """
    Run multiple chains serially with the MPI-aware joint likelihood.

    This mode is required for ``joint_objective`` because its MPI collectives
    must be called in the same order by all ranks.
    """
    jax.block_until_ready(calc_like_joint(metamodel, metamodel.parameters))
    jax.block_until_ready(
        draw_samp(
            metamodel,
            jax.random.key(seed),
            bound,
            prior_type,
            prior_err_scale,
            proposal=proposal,
        )
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
                progress,
                proposal,
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
    proposal="diag",
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
        draw_samp(
            metamodel,
            jax.random.key(seed),
            bound,
            prior_type,
            prior_err_scale,
            proposal=proposal,
        )
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
                False,
                proposal,
            )
            for chain_id in range(num_chains)
        ]
        chains = [future.result() for future in futures]

    return chains


def run_mh(
    metamodel,
    num_samples=10000,
    num_chains=4,
    burn_in=0.1,
    bound=2,
    seed=0,
    prior_type="uniform",
    prior_err_scale=1.0,
    joint=True,
    dataset_ind=0,
    progress=True,
    proposal="cov",
):
    """
    Run Metropolis-Hastings chains and summarize them into the metamodel.

    The chains start at the current ``metamodel.parameters`` (e.g. the
    Levenberg-Marquardt best fit) and only parameters with ``to_fit`` set are
    varied. Mirrors the interface of ``witch.fitting.run_mcmc``.

    Parameters
    ----------
    metamodel : witch.containers.MetaModel
        Model to sample.
    num_samples : int, default=10000
        Samples per chain.
    num_chains : int, default=4
        Number of chains.
    burn_in : float, default=0.1
        Fraction of each chain discarded.
    bound : float, default=2
        Scale for unbounded uniform proposals.
    seed : int, default=0
        Base seed; chain ``i`` uses ``seed + i``.
    prior_type : {"uniform", "normal"}, default="uniform"
        Prior and proposal type.
    prior_err_scale : float, default=1.0
        Multiplier on ``metamodel.errs`` for proposal width and normal prior.
    joint : bool, default=True
        Use the MPI-aware joint likelihood (chains run serially). If False
        chains run in threads on a single dataset.
    dataset_ind : int, default=0
        Dataset used when ``joint`` is False.
    progress : bool, default=True
        Show progress bars.
    proposal : {"cov", "diag"}, default="cov"
        ``cov`` proposes steps from the LM covariance in ``metamodel.cov``
        (falls back to ``diag`` with the chosen prior_type if unusable); ``diag`` uses independent steps.

    Returns
    -------
    metamodel : witch.containers.MetaModel
        Metamodel with median parameters, std errors, covariance and chisq.
    flat_samples : np.ndarray
        Post burn-in samples of the fitted parameters, ``(nsamp, nfit)``.
    chains : np.ndarray
        All chains, ``(num_chains, num_samples, num_parameters)``.
    acceptance_rates : np.ndarray
        Acceptance rate of each chain.
    """
    if joint:
        results = run_chains_serial(
            metamodel,
            num_samples,
            num_chains,
            bound,
            seed,
            prior_type,
            prior_err_scale,
            progress,
            proposal,
        )
    else:
        results = run_chains_parallel(
            metamodel,
            num_samples,
            num_chains,
            bound,
            seed,
            dataset_ind,
            prior_type,
            prior_err_scale,
            proposal,
        )
    chains = np.array([r[0] for r in results])
    acceptance_rates = np.array([r[1] for r in results])

    to_fit = np.array(metamodel.to_fit)
    start = int(burn_in * num_samples)
    flat_samples = chains[:, start:, :][:, :, to_fit].reshape(-1, int(to_fit.sum()))

    pars = np.array(metamodel.parameters)
    errs = np.zeros_like(pars)
    cov = np.zeros((len(pars), len(pars)))
    pars[to_fit] = np.median(flat_samples, axis=0)
    errs[to_fit] = np.std(flat_samples, axis=0)
    cov[np.ix_(to_fit, to_fit)] = np.atleast_2d(np.cov(flat_samples.T))
    chisq = -2 * calc_like_joint(metamodel, jnp.array(pars))
    metamodel = metamodel.update(
        pars=jnp.array(pars), errs=jnp.array(errs), cov=jnp.array(cov), chisq=chisq
    )
    return metamodel, flat_samples, chains, acceptance_rates
