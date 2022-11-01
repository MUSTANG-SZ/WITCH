"""
Gaussian stuff
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.config import config

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def elliptical_gauss(p, x, y):
    """
    Gives the value of an eliptical gaussian.

    Arguments:

        p: Parameter array (x0, y0, theta1, theta2, psi, amp)

           * x0: Center in x
           * y0: Center in y
           * theta1: FWHM along x
           * theta2: FWHM along y
           * psi: Rotation angle
           * amp: Amplitude of gaussian

        x: X values to evaluate the gaussian at

        y: Y values to evaluate the gaussian at

    Returns:

        gauss: Elliptical gaussian evaluated at x and y
    """
    x0, y0, theta1, theta2, psi, amp = p

    theta1_inv = 1 / theta1
    theta2_inv = 1 / theta2
    theta1_inv_sqr = theta1_inv**2
    theta2_inv_sqr = theta2_inv**2
    cosdec = jnp.cos(y0)
    cospsi = jnp.cos(psi)
    sinpsi = jnp.sin(psi)

    delx = (x - x0) * cosdec
    dely = y - y0
    xx = delx * cospsi + dely * sinpsi
    yy = dely * cospsi - delx * sinpsi
    xfac = theta1_inv_sqr * xx * xx
    yfac = theta2_inv_sqr * yy * yy
    rr = xfac + yfac
    rrpow = jnp.exp(-0.5 * rr)

    return amp * rrpow


@jax.jit
def gauss(p, x, y):
    """
    Gives the value of an eliptical gaussian.

    Arguments:

        p: Parameter array (x0, y0, sigma, amp)

           * x0: Center in x
           * y0: Center in y
           * sigma: Sigma of gaussian
           * amp: Amplitude of gaussian

        x: X values to evaluate the gaussian at

        y: Y values to evaluate the gaussian at

    Returns:

        gauss: Elliptical gaussian evaluated at x and y
    """
    x0, y0, sigma, amp = p

    sigma_inv = 1 / sigma
    sigma_inv_sqr = sigma_inv**2
    cosdec = jnp.cos(y0)
    cospsi = 1
    sinpsi = 0

    delx = (x - x0) * cosdec
    dely = y - y0
    xx = delx * cospsi + dely * sinpsi
    yy = dely * cospsi - delx * sinpsi
    xfac = sigma_inv_sqr * xx * xx
    yfac = sigma_inv_sqr * yy * yy
    rr = xfac + yfac
    rrpow = jnp.exp(-0.5 * rr)

    return amp * rrpow
