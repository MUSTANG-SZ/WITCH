"""
Functions for generating structure.
This includes both cluster profiles and substructure.
"""
import jax
import jax.numpy as jnp


@jax.jit
def isobeta(dx, dy, dz, r_1, r_2, r_3, theta, beta, amp, xyz):
    """
    Elliptical isobeta pressure profile in 3d
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Arguments:

        dx: RA of cluster center relative to grid origin

        dy: Dec of cluster center relative to grid origin

        dz: Line of sight offset of cluster center relative to grid origin

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
    # Shift origin
    x = xyz[0] - dx
    y = xyz[1] - dy
    z = xyz[2] - dz

    # Rotate
    xx = x * jnp.cos(theta) + y * jnp.sin(theta)
    yy = y * jnp.cos(theta) - x * jnp.sin(theta)

    # Apply ellipticity
    xfac = (xx / r_1) ** 2
    yfac = (yy / r_2) ** 2
    zfac = (z / r_3) ** 2

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

        xb: Ra of bubble's center relative to grid origin

        yb: Dec of bubble's center relative to grid origin

        zb: Line of site offset of bubble's center relative to grid origin

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
def add_shock(pressure, xyz, xs, ys, zs, sr_1, sr_2, sr_3, s_theta, shock):
    """
    Add bubble to 3d pressure profile.

    Arguments:

        pressure: The pressure profile

        xyz: Coordinate grids, see make_grid for details

        xs: RA of cluster center relative to grid origin

        ys: Dec of cluster center relative to grid origin

        zs: Line of sight offset of cluster center relative to grid origin

        sr_1: Amount to scale shock along x-axis

        sr_2: Amount to scale shock along y-axis

        sr_3: Amount to scale shock along z-axis

        s_theta: Angle to rotate shock in xy-plane

        shock: Factor by which pressure is enhanced within shock

    Returns:

        pressure_s: Pressure profile with shock added
    """
    # Recenter grid on bubble center
    x = xyz[0] - xs
    y = xyz[1] - ys
    z = xyz[2] - zs

    # Rotate
    xx = x * jnp.cos(s_theta) + y * jnp.sin(s_theta)
    yy = y * jnp.cos(s_theta) - x * jnp.sin(s_theta)

    # Apply ellipticity
    xfac = (xx / sr_1) ** 2
    yfac = (yy / sr_2) ** 2
    zfac = (z / sr_3) ** 2

    # Enhance points inside shock
    pressure_s = jnp.where(
        jnp.sqrt(xfac + yfac + zfac) > 1, pressure, (1 + shock) * pressure
    )
    return pressure_s
