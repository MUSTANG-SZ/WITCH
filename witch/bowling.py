"""
Tools for dealing with bowling in maps
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.optimize as sopt
from jax.config import config

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")


@partial(jax.jit, static_argnums=(1, 2, 3))
def poly(x, c0, c1, c2):
    """
    JITed second order polynomial

    Arguments:

        x: X value(s) of polynomial

        c0: 0th order constant

        c1: 1st order constant

        c2: 2nd order constant

    Returns:

        poly: Polynomial values at x
    """
    return c0 + c1 * x + c2 * x**2


# Jax atmospheric fitter
# TODO: this can be improved to accept arbitrary functions to fit, but I think for now
# leaving it is fine as I suspect that would incur a performance penalty
@jax.jit
def poly_sub(x, y):
    """
    Fit a second order TOD to data.
    Nominally used to fit out atmosphere.

    # Inputs: designed to work with
    Arguments:
        x: X values, nominally tod.info['apix'][j]

        y: Y values, nominally tod.info['dat_calib'][j] - tod.info['cm']

    Returns:

        res.x: The best fit parameters in the order c0, c1, c2 for y = c0 + c1*x + c2*x**2
    """
    # compute the residual function for a 2nd degree poly, using res**2
    poly_resid = lambda p, x, y: jnp.sum(((p[0] + p[1] * x + p[2] * x**2) - y) ** 2)
    p0 = jnp.array([0.1, 0.1, 0.1])
    # minimize the residual
    res = sopt.minimize(poly_resid, p0, args=(x, y), method="BFGS")

    return res.x


@partial(jax.jit, static_argnums=(0))
def potato_chip(pn, p, xi, yi):
    """
    A shitty potato chip (hyperbolic parabaloid) model for removing scan signal from maps
    Arguments:

        pn: Which version of p is passed in, see below for details

        p: Parameter vector with entries:

           * A, the overall amplitude
           * c0, an overall offset
           * c1 and c2, linear slopes
           * c3 and c4, parabolic amplitudes
           * theta, a rotation angle

           Acceptable subsets are:

           * (A, c0, c1, c2, c3, c4, theta) for pn = 0
           * (A, c1, c2, c3, c4, theta) for pn = 1
           * (A, c1, c3, c4, theta) for pn = 2
           * (c1, theta) for pn = 3

        xi: X vector

        yi: Y vectors

    Returns:

        chip: A vector of values for this model given p at the xi, yi
    """
    if pn == 0:
        A, c0, c1, c2, c3, c4, theta = p
    elif pn == 1:
        A, c1, c2, c3, c4, theta = p
    elif pn == 2:
        A, c1, c3, c4, theta = p
    elif pn == 3:
        c1, theta = p
    else:
        return 0

    x1, x2 = (
        jnp.cos(theta) * xi + yi * jnp.sin(theta),
        -1 * jnp.sin(theta) * xi + jnp.cos(theta) * yi,
    )

    if pn == 0:
        return A * (c0 + c1 * x1 + c2 * x2 + x1**2 / c3 - x2**2 / c4)
    elif pn == 1:
        return A * (1 + c1 * x1 + c2 * x2 + x1**2 / c3 - x2**2 / c4)
    elif pn == 2:
        return A * (c1 * x1 + x1**2 / c3 - x2**2 / c4)
    elif pn == 3:
        return c1 * x1


# TODO: an analytic expression for the potato gradient definitely exists, just needs to be derived + implemented
@partial(jax.jit, static_argnums=(0))
def jac_potato_grad(pn, p, tods):
    """
    Gradient of shitty potato chip (hyperbolic parabaloid) model for removing scan signal from maps
    Arguments:

        pn: Which version of p is passed in, see below for details

        p: Parameter vector with entries:

           * A, the overall amplitude
           * c0, an overall offset
           * c1 and c2, linear slopes
           * c3 and c4, parabolic amplitudes
           * theta, a rotation angle

           Acceptable subsets are:

           * (A, c0, c1, c2, c3, c4, theta) for pn = 0
           * (A, c1, c2, c3, c4, theta) for pn = 1
           * (A, c1, c3, c4, theta) for pn = 2
           * (c1, theta) for pn = 3

        tods: (x, y) vectors

    Returns:

        grad: Gradient for this model with respect to p
    """
    return jax.jacfwd(potato_chip, argnums=0)(pn, p, tods[0], tods[1])


@partial(jax.jit, static_argnums=(0))
def jit_potato_full(pn, p, tods):
    """
    Shitty potato chip and its gradient (hyperbolic parabaloid) model for removing scan signal from maps
    Arguments:

        pn: Which version of p is passed in, see below for details

        p: Parameter vector with entries:

           * A, the overall amplitude
           * c0, an overall offset
           * c1 and c2, linear slopes
           * c3 and c4, parabolic amplitudes
           * theta, a rotation angle

           Acceptable subsets are:

           * (A, c0, c1, c2, c3, c4, theta) for pn = 0
           * (A, c1, c2, c3, c4, theta) for pn = 1
           * (A, c1, c3, c4, theta) for pn = 2
           * (c1, theta) for pn = 3

        tods: (x, y) vectors

    Returns:

        pred: A vector of values for this model given p at the xi, yi

        grad: Gradient for this model with respect to p
    """
    pred = potato_chip(pn, p, tods[0], tods[1])
    grad = jax.jacfwd(potato_chip, argnums=0)(pn, p, tods[0], tods[1])

    return pred, grad
