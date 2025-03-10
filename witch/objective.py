"""
Module for the objective functions used by LM fitting and MCMC.
All objective functions should return a log-likelihood (modulo a DC offset)
as well as the gradient and curvature of the log-likelihood with respect to
the model parameters.

Note that everything in done in analogy to chi-squared so there is a factor of -2
applied as needed to the non chi-squared distributions.
"""

from functools import partial
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import mpi4jax
from jax.scipy.special import factorial
from jitkasi.solutions import SolutionSet
from jitkasi.tod import TODVec
from mpi4py import MPI

from . import utils as wu
from .containers import Model


@runtime_checkable
class ObjectiveFunc(Protocol):
    def __call__(
        self,
        model: Model,
        datavec: TODVec | SolutionSet,
        mode: str = ...,
        do_chisq: bool = ...,
        do_grad: bool = ...,
        do_loglike: bool = ...,
    ) -> tuple[jax.Array, jax.Array, jax.Array]: ...


@partial(jax.jit, static_argnames=("mode", "do_loglike", "do_grad", "do_curve"))
def chisq_objective(
    model: Model,
    datavec: TODVec | SolutionSet,
    mode: str = "tod",
    do_loglike: bool = True,
    do_grad: bool = True,
    do_curve: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Objective function to minimize when fitting a dataset where a Gaussian distribution is reasonible.
    This is an MPI aware function.

    Parameters
    ----------
    model : Model
        The model object we are using to fit.
    datavec: TODVec | SolutionSet
        The data to fit against.
        This is what we use to compute our fit residuals.
    mode : str, default: "tod"
        The type of data we compile this function for.
        Should be either "tod" or "map".
    do_loglike : bool, default: True
        If True then we will compute the chi-squared between
        the model and the data.
    do_grad : bool, default: True
        If True then compute the gradient of chi-squared with
        respect to the model parameters.
    do_curve : bool, default: True
        If True than compute the curvature of chi-squared with
        respect to the model parameters.

    Returns
    -------
    chisq : jax.Array
        The chi-squared between the model and data.
        If `do_loglike` is `False` then this is `jnp.array(0)`.
    grad : jax.Array
        The gradient of the parameters at there current values.
        If `do_grad` is `False` then this is an array of zeros.
        This is a `(npar,)` array.
    curve : jax.Array
        The curvature of the parameter space at the current values.
        If `do_curve` is `False` then this is an array of zeros.
        This is a `(npar, npar)` array.
    """
    if mode not in ["tod", "map"]:
        raise ValueError("Invalid mode")
    npar = len(model.pars)
    chisq = jnp.array(0)
    grad = jnp.zeros(npar)
    curve = jnp.zeros((npar, npar))

    zero = jnp.zeros((1, 1))
    only_chisq = not (do_grad or do_curve)

    for data in datavec:
        if mode == "tod":
            x = data.x * wu.rad_to_arcsec
            y = data.y * wu.rad_to_arcsec
            if only_chisq:
                pred_dat = model.to_tod(x, y)
                grad_dat = zero
            else:
                pred_dat, grad_dat = model.to_tod_grad(x, y)
        else:
            x, y = data.xy
            if only_chisq:
                pred_dat = model.to_map(x * wu.rad_to_arcsec, y * wu.rad_to_arcsec)
                grad_dat = zero
            else:
                pred_dat, grad_dat = model.to_map_grad(
                    x * wu.rad_to_arcsec, y * wu.rad_to_arcsec
                )

        resid = data.data - pred_dat
        resid_filt = data.noise.apply_noise(resid)
        if do_loglike:
            chisq += jnp.sum(resid * resid_filt)

        if only_chisq:
            continue

        grad_filt = jnp.zeros_like(grad_dat)
        for i in range(npar):
            grad_filt = grad_filt.at[i].set(
                data.noise.apply_noise(grad_dat.at[i].get())
            )
        grad_filt = jnp.reshape(grad_filt, (npar, -1))
        grad_dat = jnp.reshape(grad_dat, (npar, -1))
        resid = resid.ravel()

        if do_grad:
            grad = grad.at[:].add(jnp.dot(grad_filt, jnp.transpose(resid)))
        if do_curve:
            curve = curve.at[:].add(jnp.dot(grad_filt, jnp.transpose(grad_dat)))

    token = mpi4jax.barrier(comm=datavec.comm)
    if do_loglike:
        chisq, token = mpi4jax.allreduce(chisq, MPI.SUM, comm=datavec.comm, token=token)
    if do_grad:
        grad, token = mpi4jax.allreduce(grad, MPI.SUM, comm=datavec.comm, token=token)
    if do_curve:
        curve, token = mpi4jax.allreduce(curve, MPI.SUM, comm=datavec.comm, token=token)
    _ = token

    return chisq, grad, curve


@partial(jax.jit, static_argnames=("mode", "do_loglike", "do_grad", "do_curve"))
def poisson_objective(
    model: Model,
    datavec: TODVec | SolutionSet,
    mode: str = "tod",
    do_loglike: bool = True,
    do_grad: bool = True,
    do_curve: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Objective function to minimize when fitting a dataset where a Poisson distribution is reasonible.
    This is an MPI aware function.

    Parameters
    ----------
    model : Model
        The model object we are using to fit.
    datavec: TODVec | SolutionSet
        The data to fit against.
        This is what we use to compute our fit residuals.
    mode : str, default: "tod"
        The type of data we compile this function for.
        Should be either "tod" or "map".
    do_loglike : bool, default: True
        If True then we will compute the log-likelihood between
        the model and the data.
    do_grad : bool, default: True
        If True then compute the gradient of chi-squared with
        respect to the model parameters.
    do_curve : bool, default: True
        If True than compute the curvature of chi-squared with
        respect to the model parameters.

    Returns
    -------
    loglike : jax.Array
        The log-likelihood between the model and data.
        If `do_loglike` is `False` then this is `jnp.array(0)`.
        Note that there is a factor of -2 here to make it add with chi-squared.
    grad : jax.Array
        The gradient of the parameters at there current values.
        If `do_grad` is `False` then this is an array of zeros.
        This is a `(npar,)` array.
        Note that there is a factor of -2 here to make it add with the gradient of chi-squared.
    curve : jax.Array
        The curvature of the parameter space at the current values.
        If `do_curve` is `False` then this is an array of zeros.
        This is a `(npar, npar)` array.
        Note that there is a factor of -2 here to make it add with the cruvature of chi-squared.
    """
    if mode not in ["tod", "map"]:
        raise ValueError("Invalid mode")
    npar = len(model.pars)
    loglike = jnp.array(0)
    grad = jnp.zeros(npar)
    curve = jnp.zeros((npar, npar))

    zero = jnp.zeros((1, 1))
    only_loglike = not (do_grad or do_curve)

    for data in datavec:
        if mode == "tod":
            x = data.x * wu.rad_to_arcsec
            y = data.y * wu.rad_to_arcsec
            if only_loglike:
                pred_dat = model.to_tod(x, y)
                grad_dat = zero
            else:
                pred_dat, grad_dat = model.to_tod_grad(x, y)
        else:
            x, y = data.xy
            if only_loglike:
                pred_dat = model.to_map(x * wu.rad_to_arcsec, y * wu.rad_to_arcsec)
                grad_dat = zero
            else:
                pred_dat, grad_dat = model.to_map_grad(
                    x * wu.rad_to_arcsec, y * wu.rad_to_arcsec
                )

        resid = (data.data / pred_dat) - 1
        if do_loglike:
            loglike += jnp.sum(
                data.data * jnp.log(pred_dat) - pred_dat - jnp.log(factorial(data.data))
            )

        if only_loglike:
            continue

        grad_filt = jnp.zeros_like(grad_dat)
        for i in range(npar):
            grad_filt = grad_filt.at[i].set(
                data.noise.apply_noise(grad_dat.at[i].get())
            )
        grad_filt = jnp.reshape(grad_filt, (npar, -1))
        grad_dat = jnp.reshape(grad_dat, (npar, -1))
        resid = resid.ravel()

        if do_grad:
            grad = grad.at[:].add(jnp.dot(grad_dat, jnp.transpose(resid)))
        if do_curve:
            # Dropping the second term here, so Jon note for justification
            curve = curve.at[:].add(
                jnp.dot(
                    -1 * grad_dat * (data.data / (pred_dat**2)).ravel(),
                    jnp.transpose(grad_dat),
                )
            )

    token = mpi4jax.barrier(comm=datavec.comm)
    if do_loglike:
        loglike, token = mpi4jax.allreduce(
            loglike, MPI.SUM, comm=datavec.comm, token=token
        )
    if do_grad:
        grad, token = mpi4jax.allreduce(grad, MPI.SUM, comm=datavec.comm, token=token)
    if do_curve:
        curve, token = mpi4jax.allreduce(curve, MPI.SUM, comm=datavec.comm, token=token)
    _ = token

    return -2 * loglike, -2 * grad, -2 * curve
