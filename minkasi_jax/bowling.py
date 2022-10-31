"""
Tools for dealing with bowling in maps
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax.config import config
import jax.scipy.optimize as sopt

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@partial(jax.jit, static_argnums=(1, 2, 3))
def poly(x, c0, c1, c2):
    return c0 + c1 * x + c2 * x**2


# Jax atmospheric fitter
@jax.jit
def poly_sub(x, y):
    # Function which fits a 2nd degree polynomial to TOD data and then returns the best fit parameters
    # Inputs: designed to work with x = tod.info['apix'][j], y = tod.info['dat_calib'][j] - tod.info['cm']
    # which fits apix vs data-common mode, but in principle you can fit whatever
    # Outputs: res.x, the best fit parameters in the order c0, c1, c2 for y = c0 + c1*x + c2*x**2
    # TODO: this can be improved to accept arbitrary functions to fit, but I think for now
    # leaving it is fine as I suspect that would incur a performance penalty

    # compute the residual function for a 2nd degree poly, using res**2
    poly_resid = lambda p, x, y: jnp.sum(((p[0] + p[1] * x + p[2] * x**2) - y) ** 2)
    p0 = jnp.array([0.1, 0.1, 0.1])
    # minimize the residual
    res = sopt.minimize(poly_resid, p0, args=(x, y), method="BFGS")

    return res.x


@jax.jit
def potato_chip(p, xi, yi):
    # A shitty potato chip (hyperbolic parabaloid) model for removing scan signal from maps
    # Inputs: p, a parameter vector with entries A, the overall amplitude, c0, an overall offset
    # c1 and c2, linear slopes, c3 and c4, parabolic amplitudes, and theta, a rotation angle.
    # xi, yi are x and y vectors
    #
    # Outputs: a vector of values for this model given p at the xi, yi

    # A, c0, c1, c2, c3, c4, theta = p
    # A, c1, c2, c3, c4, theta = p
    c1, theta = p
    x1, x2 = (
        jnp.cos(theta) * xi + yi * jnp.sin(theta),
        -1 * jnp.sin(theta) * xi + jnp.cos(theta) * yi,
    )

    # return A*(c0 + c1*x1 + c2*x2 + x1**2/c3 - x2**2/c4)
    # return A*(1+c1*x1 + c2*x2 + x1**2/c3 - x2**2/c4)
    # return A*(c1*x1 + x1**2/c3 - x2**2/c4)
    return c1 * x1


# Todo: an analytic expression for the potato gradient definitely exists, just needs to be derived + implemented
@jax.jit
def jac_potato_grad(p, tods):
    return jax.jacfwd(potato_chip, argnums=0)(p, tods[0], tods[1])


@jax.jit
def jit_potato_full(p, tods):
    pred = potato_chip(p, tods[0], tods[1])
    grad = jax.jacfwd(potato_chip, argnums=0)(p, tods[0], tods[1])

    return pred, grad
