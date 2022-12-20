"""
Functions for generating structure.
This includes both cluster profiles and substructure.
"""
import jax
import jax.numpy as jnp
from .utils import transform_grid, ap, h70, get_nz, get_hz


@jax.jit
def gnfw(dx, dy, dz, r_1, r_2, r_3, theta, P0, c500, m500, gamma, alpha, beta, z, xyz):
    """
    Elliptical gNFW pressure profile in 3d.
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

        P0: Amplitude of the pressure profile

        c500: Concentration parameter at a density contrast of 500

        m500: Mass at a density contrast of 500

        gamma: The central slope

        alpha: The intermediate slope

        beta: The outer slope

        z: Redshift of cluster

        xyz: Coordinte grid to calculate model on

    Returns:

        model: The gnfw model
    """
    nz = get_nz(z)
    hz = get_hz(z)

    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    r500 = (m500 / (4.00 * jnp.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    r = c500 * jnp.sqrt(x**2 + y**2 + z**2) / r500
    denominator = (r**gamma) * (1 + r**alpha) ** ((beta - gamma) / alpha)

    P500 = (
        1.65e-03
        * (m500 / (3.00e14 / h70)) ** (2.00 / 3.00 + ap)
        * hz ** (8.00 / 3.00)
        * h70**2
    )

    return P500 * P0 / denominator


@jax.jit
def isobeta(dx, dy, dz, r_1, r_2, r_3, theta, beta, amp, xyz):
    """
    Elliptical isobeta pressure profile in 3d.
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

        model: The isobeta model
    """
    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    rr = 1 + x**2 + y**2 + z**2
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
    x, y, z = transform_grid(xs, ys, zs, sr_1, sr_2, sr_3, s_theta, xyz)

    pressure_s = jnp.where(
        jnp.sqrt(x**2 + y**2 + z**2) > 1, pressure, (1 + shock) * pressure
    )
    return pressure_s
