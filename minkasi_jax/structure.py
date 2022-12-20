"""
Functions for generating structure.
This includes both cluster profiles and substructure.
"""
import jax
import jax.numpy as jnp


@jax.jit
def isobeta(r_1, r_2, r_3, theta, beta, amp, xyz):
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


@jax.jit
def add_bubble(pressure, xyz, xb, yb, zb, rb, sup):
    """
    Add bubble to 3d pressure profile.

    Arguments:

        pressure: The pressure profile

        xyz: Coordinate grids, see make_grid for details

        xb: Ra of bubble's center relative to cluster center

        yb: Dec of bubble's center relative to cluster center

        zb: Line of site offset of bubble's center relative to cluster center

        rb: Radius of bubble

        sup: Supression factor of bubble

    Returns:

        pressure_b: Pressure profile with bubble added
    """
    # Recenter grid on bubble center
    x = xyz[0] - xb
    y = xyz[1] - yb
    z = xyz[2] - zb

    # Supress points inside bubble
    pressure_b = jnp.where(
        jnp.sqrt(x**2 + y**2 + z**2) >= rb, pressure, (1 - sup) * pressure
    )
    return pressure_b


@jax.jit
def add_shock(pressure, xyz, sr_1, sr_2, sr_3, s_theta, shock):
    """
    Add bubble to 3d pressure profile.

    Arguments:

        pressure: The pressure profile

        xyz: Coordinate grids, see make_grid for details

        sr_1: Amount to scale shock along x-axis

        sr_2: Amount to scale shock along y-axis

        sr_3: Amount to scale shock along z-axis

        s_theta: Angle to rotate shock in xy-plane

        shock: Factor by which pressure is enhanced within shock

    Returns:

        pressure_s: Pressure profile with shock added
    """
    # Rotate
    xx = xyz[0] * jnp.cos(s_theta) + xyz[1] * jnp.sin(s_theta)
    yy = xyz[1] * jnp.cos(s_theta) - xyz[0] * jnp.sin(s_theta)
    zz = xyz[2]

    # Apply ellipticity
    xfac = (xx / sr_1) ** 2
    yfac = (yy / sr_2) ** 2
    zfac = (zz / sr_3) ** 2

    # Enhance points inside shock
    pressure_s = jnp.where(
        jnp.sqrt(xfac + yfac + zfac) > 1, pressure, (1 + shock) * pressure
    )
    return pressure_s
