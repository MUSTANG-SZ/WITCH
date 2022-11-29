"""
Module for generating isobeta profiles with substructure and their gradients.
"""
# TODO: Move some coordinate operations out of main funcs
# TODO: Move unit conversions out of main funcs
# TODO: Make unit agnostic? Already sorta is in parts
# TODO: One function to rule them all

from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from utils import (
    dzline,
    daline,
    Mparsec,
    Xthom,
    me,
    y2K_RJ,
    fft_conv,
    make_grid,
    add_shock,
    add_bubble,
)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@jax.jit
def _isobeta_elliptical(r_1, r_2, r_3, theta, beta, amp, xyz):
    """
    Elliptical isobeta pressure profile in 3d
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Arguments:

        r_1: Amount to scale along x-axis

        r_2: Amount to scale along y-axis

        r_3: Amount to scale along z-axis

        theta: Angle to rotate in xy-plane

        beta: Beta value of isobeta model

        amp: Amplitude of isobeta model

        xyz: Coordinte grid to calculate model on

    Returns:

        model: The isobeta model centered on the origin of xyz
    """
    # Rotate
    xx = xyz[0] * jnp.cos(theta) + xyz[1] * jnp.sin(theta)
    yy = xyz[1] * jnp.cos(theta) - xyz[0] * jnp.sin(theta)
    zz = xyz[2]

    # Apply ellipticity
    xfac = (xx / r_1) ** 2
    yfac = (yy / r_2) ** 2
    zfac = (zz / r_3) ** 2

    # Calculate pressure profile
    rr = 1 + xfac + yfac + zfac
    power = -1.5 * beta
    rrpow = rr**power

    return amp * rrpow


@partial(jax.jit, static_argnums=(0, 1, 3, 5, 7, 8, 9, 10))
def isobeta(xyz, n_profiles, profiles, n_shocks, shocks, n_bubbles, bubbles, dx, beam, idx, idy):
    """
    Generically create isobeta models with substructure.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_profiles: Number of isobeta profiles to add.

        profile: 2d array of profile parameters.
                 Each row is an isobeta profile with parameters:
                 (r_1, r_2, r_3, theta, beta, amp)

        n_shocks: Number of shocks to add.

        shocks: 2d array of shock parameters.
                Each row is a shock with parameters:
                (sr_1, sr_2, sr_3, s_theta, shock)

        n_bubbles: Number of bubbles to add.

        bubbles: 2d array of bubble parameters.
                 Each row is a bubble with parameters:
                 (xb, yb, zb, rb, sup, z)

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        idx: RA TOD in units of pixels.
             Should have Dec stretch applied.

        idy: Dec TOD in units of pixels.

    Returns:

        model: The isobeta model with the specified substructure.
    """
    da = jnp.interp(z, dzline, daline)
    XMpc = Xthom * Mparsec

    pressure = jnp.zeros_like(xyz)
    for i in range(n_profiles):
        pressure = jnp.add(pressure, _isobeta_elliptical(*profiles[i], xyz))

    for i in range(n_shocks):
        pressure = add_shock(pressure, xyz, *shocks[i])

    for i in range(n_bubbles):
        pressure = add_bubble(pressure, xyz, *bubbles[i])

    # Integrate along line of site
    ip = jnp.trapz(pressure, dx=dx, axis=-1)

    bound0, bound1 = int((ip.shape[0] - beam.shape[0]) / 2), int(
        (ip.shape[1] - beam.shape[1]) / 2
    )
    beam = jnp.pad(
        beam,
        (
            (bound0, ip.shape[0] - beam.shape[0] - bound0),
            (bound1, ip.shape[1] - beam.shape[1] - bound1),
        ),
    )

    ip = fft_conv(ip, beam)

    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    return jsp.ndimage.map_coordinates(ip, (idy, idx), order=0)


# TODO: figure out how to skip parameters
@partial(jax.jit, static_argnums=(0, 1, 3, 5, 7, 8, 9, 10))
def isobeta_grad(xyz, n_profiles, profiles, n_shocks, shocks, n_bubbles, bubbles, dx, beam, idx, idy):
    """
    Generically create isobeta models with substructure and get their gradients.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_profiles: Number of isobeta profiles to add.

        profile: 2d array of profile parameters.
                 Each row is an isobeta profile with parameters:
                 (r_1, r_2, r_3, theta, beta, amp)

        n_shocks: Number of shocks to add.

        shocks: 2d array of shock parameters.
                Each row is a shock with parameters:
                (sr_1, sr_2, sr_3, s_theta, shock)

        n_bubbles: Number of bubbles to add.

        bubbles: 2d array of bubble parameters.
                 Each row is a bubble with parameters:
                 (xb, yb, zb, rb, sup, z)

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        idx: RA TOD in units of pixels.
             Should have Dec stretch applied.

        idy: Dec TOD in units of pixels.

    Returns:

        model: The isobeta model with the specified substructure.

        grad: The gradient of the model with respect to the model parameters.
    """
    pred = isobeta(xyz, n_profiles, profiles, n_shocks, shocks, n_bubbles, bubbles, dx, beam, idx, idy)

    grad = jax.jacfwd(isobeta, argnums=(2, 4, 6))(xyz, n_profiles, profiles, n_shocks, shocks, n_bubbles, bubbles, dx, beam, idx, idy)
    grad = jnp.array(grad)

    return pred, grad
