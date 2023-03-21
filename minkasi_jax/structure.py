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
def gaussian(dx, dy, dz, r_1, r_2, r_3, theta, sigma, amp, xyz):
    """
    Elliptical gaussian profile in 3d.
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Arguments:

        dx: RA of gaussian center relative to grid origin

        dy: Dec of gaussian center relative to grid origin

        dz: Line of sight offset of gaussian center relative to grid origin

        r_1: Amount to scale along x-axis

        r_2: Amount to scale along y-axis

        r_3: Amount to scale along z-axis

        theta: Angle to rotate in xy-plane

        sigma: Sigma of the gaussian

        amp: Amplitude of the gaussian

        xyz: Coordinte grid to calculate model on

    Returns:

        model: The gaussian
    """
    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    rr = x**2 + y**2 + z**2
    power = -1 * rr / (2 * sigma**2)

    return amp * jnp.exp(power)


@jax.jit
def add_uniform(pressure, xyz, dx, dy, dz, r_1, r_2, r_3, theta, amp):
    """
    Add ellipsoid with uniform structure to 3d pressure profile.

    Arguments:

        pressure: The pressure profile

        xyz: Coordinate grids, see make_grid for details

        dx: RA of ellipsoid center relative to grid origin

        dy: Dec of ellipsoid center relative to grid origin

        dz: Line of sight offset of ellipsoid center relative to grid origin

        r_1: Amount to scale ellipsoid along x-axis

        r_2: Amount to scale ellipsoid along y-axis

        r_3: Amount to scale ellipsoid along z-axis

        theta: Angle to rotate ellipsoid in xy-plane

        amp: Factor by which pressure is enhanced at peak of exponential

    Returns:

        new_pressure: Pressure profile with ellipsoid added
    """
    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    new_pressure = jnp.where(
        jnp.sqrt(x**2 + y**2 + z**2) > 1, pressure, (1 + amp) * pressure
    )
    return new_pressure


@jax.jit
def add_exponential(
    pressure, xyz, dx, dy, dz, r_1, r_2, r_3, theta, amp, xk, x0, yk, y0, zk, z0
):
    """
    Add ellipsoid with exponential structure to 3d pressure profile.

    Arguments:

        pressure: The pressure profile

        xyz: Coordinate grids, see make_grid for details

        dx: RA of ellipsoid center relative to grid origin

        dy: Dec of ellipsoid center relative to grid origin

        dz: Line of sight offset of ellipsoid center relative to grid origin

        r_1: Amount to scale ellipsoid along x-axis

        r_2: Amount to scale ellipsoid along y-axis

        r_3: Amount to scale ellipsoid along z-axis

        theta: Angle to rotate ellipsoid in xy-plane

        amp: Factor by which pressure is enhanced at peak of exponential

        xk: Power of exponential in RA direction

        x0: RA offset of exponential.
            Note that this is in transformed coordinates so x0=1 is at xs + sr_1.

        yk: Power of exponential in Dec direction

        y0: Dec offset of exponential.
            Note that this is in transformed coordinates so y0=1 is at ys + sr_2.

        zk: Power of exponential along the line of sight

        z0: Line of sight offset of exponential.
            Note that this is in transformed coordinates so z0=1 is at zs + sr_3.

    Returns:

        new_pressure: Pressure profile with ellipsoid added
    """
    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    exponential = amp * jnp.exp(((x - x0) * xk) + ((y - y0) * yk) + ((z - z0) * zk))

    new_pressure = jnp.where(
        jnp.sqrt(x**2 + y**2 + z**2) > 1, pressure, (1 + exponential) * pressure
    )
    return new_pressure


@jax.jit
def add_powerlaw(
    pressure, xyz, dx, dy, dz, r_1, r_2, r_3, theta, amp, k, xa, x0, ya, y0, za, z0
):
    """
    Add ellipsoid with power law structure to 3d pressure profile.

    Arguments:

        pressure: The pressure profile

        xyz: Coordinate grids, see make_grid for details

        dx: RA of ellipsoid center relative to grid origin

        dy: Dec of ellipsoid center relative to grid origin

        dz: Line of sight offset of ellipsoid center relative to grid origin

        r_1: Amount to scale ellipsoid along x-axis

        r_2: Amount to scale ellipsoid along y-axis

        r_3: Amount to scale ellipsoid along z-axis

        theta: Angle to rotate ellipsoid in xy-plane

        amp: Factor by which pressure is enhanced at peak of power law

        k: Power of power law

        xa: Relative amplitude of power law in RA direction

        x0: RA offset of power law.
            Note that this is in transformed coordinates so x0=1 is at xs + sr_1.

        ya: Relative amplitude of power law in Dec direction

        y0: Dec offset of power law.
            Note that this is in transformed coordinates so y0=1 is at ys + sr_2.

        za: Relative amplitude of power law along the line of sight

        z0: Line of sight offset of power law.
            Note that this is in transformed coordinates so z0=1 is at zs + sr_3.

    Returns:

        new_pressure: Pressure profile with ellipsoid added
    """
    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    x = jnp.where(
        x <= x0,
        jnp.where(
            jnp.abs(x) > 1,
            0.0,
            jnp.interp(x, jnp.array((-2.0 - x0, x0)), jnp.array((-1.0, 1.0))),
        ),
        jnp.where(
            jnp.abs(x) > 1,
            0.0,
            jnp.interp(x, jnp.array((x0, 2 - x0)), jnp.array((1.0, -1.0))),
        ),
    )
    y = jnp.where(
        y <= y0,
        jnp.where(
            jnp.abs(y) > 1,
            0.0,
            jnp.interp(y, jnp.array((-2.0 - y0, y0)), jnp.array((-1.0, 1.0))),
        ),
        jnp.where(
            jnp.abs(y) > 1,
            0.0,
            jnp.interp(y, jnp.array((y0, 2 - y0)), jnp.array((1.0, -1.0))),
        ),
    )
    z = jnp.where(
        z <= z0,
        jnp.where(
            jnp.abs(z) > 1,
            0.0,
            jnp.interp(z, jnp.array((-2.0 - z0, z0)), jnp.array((-1.0, 1.0))),
        ),
        jnp.where(
            jnp.abs(z) > 1,
            0.0,
            jnp.interp(z, jnp.array((z0, 2 - z0)), jnp.array((1.0, -1.0))),
        ),
    )

    powerlaw = (
        xa * jnp.float_power(x, k)
        + ya * jnp.float_power(y, k)
        + za * jnp.float_power(z, k)
    )
    powerlaw *= amp / (xa + ya + za)
    powerlaw = jnp.where(jnp.isinf(powerlaw), amp, powerlaw)
    powerlaw = jnp.where(jnp.isnan(powerlaw), 0.0, powerlaw)

    new_pressure = jnp.where(
        jnp.sqrt(x**2 + y**2 + z**2) > 1, pressure, (1 + powerlaw) * pressure
    )
    return new_pressure
