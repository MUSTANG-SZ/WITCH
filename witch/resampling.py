import numpy as np
import jax.numpy as jnp
import jax
from copy import copy


from .containers import MetaModel
from .objective import joint_objective


def MC_resample(
    metamodel: MetaModel,
    nresamps: int = 3000,
) -> (np.ndarray, np.ndarray):
    """
    Use Monte Carlo resampling to estimate whether ML parameters are accurate or not.
    See section 5 here for details:
    https://github.com/sievers/phys512-2025/blob/main/notes/mcmc.pdf

    Parameters
    ----------
    metamodel : MetaModel
        Metamodel containing our fit model/parameters
    nresamps : int, default; 3000
        Number of points to resample at.

    Returns
    -------
    samps : np.ndarray
        Array of resampled parameters
    likelihood_chain : np.ndarray
        Array of likelihood ratios
    """
    if not np.any(metamodel.cov):
        raise RuntimeError(
            "Error: cov is not computed. You must fit data before doing resampling"
        )
    likelihood_chain = np.zeros(
        nresamps
    )  # Contains the likelihood_ratio, L_exact/L_gaussian
    samps = draw_samps(
        par_means=metamodel.parameters[metamodel.to_fit],
        cov=metamodel.cov[metamodel.to_fit].T[metamodel.to_fit],
        nsamps=nresamps,
    ).T
    inv_cov = np.linalg.inv(metamodel.cov[metamodel.to_fit].T[metamodel.to_fit])
    base_loglike, _, _ = joint_objective(
        metamodel=metamodel, do_loglike=True, do_grad=False, do_curve=False
    )
    for i in range(nresamps):
        cur_pars = copy(metamodel.parameters).at[metamodel.to_fit].set(samps[i])
        cur_meta = copy(metamodel).update(
            pars=cur_pars, errs=metamodel.errs, cov=metamodel.cov, chisq=metamodel.chisq
        )
        loglike, _, _ = joint_objective(
            metamodel=cur_meta, do_loglike=True, do_grad=False, do_curve=False
        )
        l_exact = np.exp(-1 / 2 * (loglike - base_loglike))
        delta_p = samps[i] - metamodel.parameters[metamodel.to_fit]
        l_gauss = np.exp(-1 / 2 * np.dot(delta_p.T, np.dot(inv_cov, delta_p)))
        likelihood_chain[i] = l_exact / l_gauss

    return samps, likelihood_chain


def draw_samps(par_means: jax.Array, cov: jax.Array, nsamps: int = 3000) -> jax.Array:
    """
    Given a maximum likelihood computed parameter means and covariance,
    draw a sample assuming gaussian distribution under that mean, cov.
    See this stack exchange on how to generate samples from a given
    cov, p: https://stats.stackexchange.com/questions/120179/generating-data-with-a-given-sample-covariance-matrix

    Parameters
    ----------
    par_means : jax.Array
        Array of parameters means
    cov : jax.Array
        Covariance of parameters
    nsamps : int, default; 3000
        Number of samples to draw

    Returns
    -------
    X : jax.Array
        Sample of p drawn from par_means, cov.
    """
    key = jax.random.key(seed=42)
    X = jax.random.multivariate_normal(key=key, mean=par_means, cov=cov, shape=nsamps).T
    for n in range(X.shape[0]):
        X.at[n].set(X[n] - X[n].mean())

    # Make each variable in X orthogonal to one another
    L_inv = jnp.linalg.cholesky(jnp.cov(X))
    L_inv = jnp.linalg.inv(L_inv)
    X = jnp.dot(L_inv, X)

    # Rescale X to exactly match Sigma
    L = jnp.linalg.cholesky(cov)
    X = jnp.dot(L, X)

    # Add the mean back into each variable
    for n in range(X.shape[0]):
        X.at[n].set(X[n] + par_means[n])

    return X
