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


@partial(
    jax.jit,
    static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
)
def jit_conv_int_isobeta_elliptical_two_bubbles(
    p,
    tods,
    z,
    xb1,
    yb1,
    zb1,
    rb1,
    xb2,
    yb2,
    zb2,
    rb2,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
):
    """
    Compute elliptical isobeta with two bubbles and its gradients.

    Arguments:

        p: Model parameters (x0, y0, r_1, r_2, r_3, theta, beta, amp, sup1, sup2).
           See conv_int_isobeta_elliptical_two_bubbles for details.

        tods: Ra and Dec TODs.

        z: Cluster redshift

        xb1: Ra of first bubble's center relative to cluster center

        yb1: Dec of first bubble's center relative to cluster center

        zb1: Line of site offset of first bubble's center relative to cluster center

        rb1: Radius of first bubble

        xb2: Ra of second bubble's center relative to cluster center

        yb2: Dec of second bubble's center relative to cluster center

        zb2: Line of site offset of second bubble's center relative to cluster center

        rb2: Radius of second bubble

        fwhm: Full width half max of beam used for smoothing

        freq: Frequency of observation in Hz

        T_electron: Electron temperature

        r_map: Size of map to model on

        dr: Map grid size

        argnums: Parameters in conv_int_isobeta_elliptical_two_bubbles to compute the gradent with respect to

    Returns:

        pred: The result of conv_int_isobeta_elliptical_two_bubbles

        derivs: The gradent of pred with respect to all model parameters.
                Will always be padded to have 18 rows with each one corresponding to
                one of the first 18 parameters of conv_int_isobeta_elliptical_two_bubbles.
                Parameters not in argnums will have all their values set to 1e-30.
    """
    x0, y0, r_1, r_2, r_3, theta, beta, amp, sup1, sup2 = p

    pred = conv_int_isobeta_elliptical_two_bubbles(
        x0,
        y0,
        r_1,
        r_2,
        r_3,
        theta,
        beta,
        amp,
        xb1,
        yb1,
        zb1,
        rb1,
        sup1,
        xb2,
        yb2,
        zb2,
        rb2,
        sup2,
        tods[0],
        tods[1],
        z,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    if len(argnums) == 0:
        return pred, jnp.zeros((len(p) + 8,) + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_isobeta_elliptical_two_bubbles, argnums=argnums)(
        x0,
        y0,
        r_1,
        r_2,
        r_3,
        theta,
        beta,
        amp,
        xb1,
        yb1,
        zb1,
        rb1,
        sup1,
        xb2,
        yb2,
        zb2,
        rb2,
        sup2,
        tods[0],
        tods[1],
        z,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )
    grad = jnp.array(grad)

    padded_grad = jnp.zeros((len(p) + 8,) + grad[0].shape) + 1e-30
    argnums = jnp.array(argnums)
    grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad


@partial(
    jax.jit,
    static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
)
def jit_conv_int_double_isobeta_elliptical_two_bubbles(
    p,
    tods,
    z,
    xb1,
    yb1,
    zb1,
    rb1,
    xb2,
    yb2,
    zb2,
    rb2,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
):
    """
    Compute elliptical double isobeta with two bubbles and its gradients.

    Arguments:

        p: Model parameters (x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1,
                             r_4, r_5, r_6, theta_2, beta_2, amp_2, sup1, sup2).
           See conv_int_double_isobeta_elliptical_two_bubbles for details.

        tods: Ra and Dec TODs.

        z: Cluster redshift

        xb1: Ra of first bubble's center relative to cluster center

        yb1: Dec of first bubble's center relative to cluster center

        zb1: Line of site offset of first bubble's center relative to cluster center

        rb1: Radius of first bubble

        xb2: Ra of second bubble's center relative to cluster center

        yb2: Dec of second bubble's center relative to cluster center

        zb2: Line of site offset of second bubble's center relative to cluster center

        rb2: Radius of second bubble

        fwhm: Full width half max of beam used for smoothing

        freq: Frequency of observation in Hz

        T_electron: Electron temperature

        r_map: Size of map to model on

        dr: Map grid size

        argnums: Parameters in conv_int_double_isobeta_elliptical_two_bubbles to compute the gradent with respect to

    Returns:

        pred: The result of conv_int_double_isobeta_elliptical_two_bubbles

        derivs: The gradient of pred with respect to all model parameters.
                Will always be padded to have 24 rows with each one corresponding to
                one of the first 24 parameters of conv_int_double_isobeta_elliptical_two_bubbles.
                Parameters not in argnums will have all their values set to 1e-30.
    """
    (
        x0,
        y0,
        r_1,
        r_2,
        r_3,
        theta_1,
        beta_1,
        amp_1,
        r_4,
        r_5,
        r_6,
        theta_2,
        beta_2,
        amp_2,
        sup1,
        sup2,
    ) = p

    pred = conv_int_double_isobeta_elliptical_two_bubbles(
        x0,
        y0,
        r_1,
        r_2,
        r_3,
        theta_1,
        beta_1,
        amp_1,
        r_4,
        r_5,
        r_6,
        theta_2,
        beta_2,
        amp_2,
        xb1,
        yb1,
        zb1,
        rb1,
        sup1,
        xb2,
        yb2,
        zb2,
        rb2,
        sup2,
        tods[0],
        tods[1],
        z,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    if len(argnums) == 0:
        return pred, jnp.zeros((len(p) + 8,) + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_double_isobeta_elliptical_two_bubbles, argnums=argnums)(
        x0,
        y0,
        r_1,
        r_2,
        r_3,
        theta_1,
        beta_1,
        amp_1,
        r_4,
        r_5,
        r_6,
        theta_2,
        beta_2,
        amp_2,
        xb1,
        yb1,
        zb1,
        rb1,
        sup1,
        xb2,
        yb2,
        zb2,
        rb2,
        sup2,
        tods[0],
        tods[1],
        z,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )
    grad = jnp.array(grad)

    padded_grad = jnp.zeros((len(p) + 8,) + grad[0].shape) + 1e-30
    argnums = jnp.array(argnums)
    grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad


@partial(
    jax.jit,
    static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
)
def jit_conv_int_double_isobeta_elliptical_two_bubbles_shock(
    p,
    tods,
    z,
    xb1,
    yb1,
    zb1,
    rb1,
    xb2,
    yb2,
    zb2,
    rb2,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
):
    """
    Compute elliptical double isobeta with two bubbles plus a shock and its gradients.

    Arguments:

        p: Model parameters (x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1,
                             r_4, r_5, r_6, theta_2, beta_2, amp_2, sup1, sup2,
                             sr_1, sr_2, sr_3, s_theta, shock).
           See conv_int_double_isobeta_elliptical_two_bubbles_shock for details.

        tods: Ra and Dec TODs.

        z: Cluster redshift

        xb1: Ra of first bubble's center relative to cluster center

        yb1: Dec of first bubble's center relative to cluster center

        zb1: Line of site offset of first bubble's center relative to cluster center

        rb1: Radius of first bubble

        xb2: Ra of second bubble's center relative to cluster center

        yb2: Dec of second bubble's center relative to cluster center

        zb2: Line of site offset of second bubble's center relative to cluster center

        rb2: Radius of second bubble

        fwhm: Full width half max of beam used for smoothing

        freq: Frequency of observation in Hz

        T_electron: Electron temperature

        r_map: Size of map to model on

        dr: Map grid size

        argnums: Parameters in conv_int_double_isobeta_elliptical_two_bubbles_shock to compute the gradent with respect to

    Returns:

        pred: The result of conv_int_double_isobeta_elliptical_two_bubbles_shock

        derivs: The gradient of pred with respect to all model parameters.
                Will always be padded to have 29 rows with each one corresponding to
                one of the first 29 parameters of conv_int_double_isobeta_elliptical_two_bubbles_shock.
                Parameters not in argnums will have all their values set to 1e-30.
    """
    (
        x0,
        y0,
        r_1,
        r_2,
        r_3,
        theta_1,
        beta_1,
        amp_1,
        r_4,
        r_5,
        r_6,
        theta_2,
        beta_2,
        amp_2,
        sup1,
        sup2,
        sr_1,
        sr_2,
        sr_3,
        s_theta,
        shock,
    ) = p

    pred = conv_int_double_isobeta_elliptical_two_bubbles_shock(
        x0,
        y0,
        r_1,
        r_2,
        r_3,
        theta_1,
        beta_1,
        amp_1,
        r_4,
        r_5,
        r_6,
        theta_2,
        beta_2,
        amp_2,
        xb1,
        yb1,
        zb1,
        rb1,
        sup1,
        xb2,
        yb2,
        zb2,
        rb2,
        sup2,
        sr_1,
        sr_2,
        sr_3,
        s_theta,
        shock,
        tods[0],
        tods[1],
        z,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    if len(argnums) == 0:
        return pred, jnp.zeros((len(p) + 8,) + pred.shape) + 1e-30

    grad = jax.jacfwd(
        conv_int_double_isobeta_elliptical_two_bubbles_shock, argnums=argnums
    )(
        x0,
        y0,
        r_1,
        r_2,
        r_3,
        theta_1,
        beta_1,
        amp_1,
        r_4,
        r_5,
        r_6,
        theta_2,
        beta_2,
        amp_2,
        xb1,
        yb1,
        zb1,
        rb1,
        sup1,
        xb2,
        yb2,
        zb2,
        rb2,
        sup2,
        sr_1,
        sr_2,
        sr_3,
        s_theta,
        shock,
        tods[0],
        tods[1],
        z,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )
    grad = jnp.array(grad)

    padded_grad = jnp.zeros((len(p) + 8,) + grad[0].shape) + 1e-30
    argnums = jnp.array(argnums)
    grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad
