"""
Functions for generating structure.
This includes both cluster profiles and substructure.
"""

import inspect
from functools import partial

import jax
import jax.numpy as jnp

from .utils import ap, get_da, get_hz, get_nz, h70, transform_grid


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
def a10(dx, dy, dz, theta, P0, c500, m500, gamma, alpha, beta, z, xyz):
    """
    gNFW pressure profile in 3d based on Arnaud2010.
    Compared to the function gnfw, this function fixes r1/r2/r3 to r500.
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Arguments:

        dx: RA of cluster center relative to grid origin

        dy: Dec of cluster center relative to grid origin

        dz: Line of sight offset of cluster center relative to grid origin

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
    da = get_da(z)  # TODO pass these arguments rather than recompute them everytime???

    r500 = (m500 / (4.00 * jnp.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    r_1, r_2, r_3 = r500 / da, r500 / da, r500 / da

    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    r = c500 * jnp.sqrt(x**2 + y**2 + z**2)
    denominator = (r**gamma) * (1 + r**alpha) ** ((beta - gamma) / alpha)

    P500 = (
        1.65e-03
        * (m500 / (3.00e14 / h70)) ** (2.00 / 3.00 + ap)
        * hz ** (8.00 / 3.00)
        * h70**2
    )

    return P500 * P0 / denominator


@jax.jit
def ea10(dx, dy, dz, r_1, r_2, r_3, theta, P0, c500, m500, gamma, alpha, beta, z, xyz):
    """
    Eliptical gNFW pressure profile in 3d based on Arnaud2010.
    r_ell is computed in the usual way for an a10 profile, then the axes are
    scaled according to r_1, r_2, r_3, with a normalization applied.
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Arguments:

        dx: RA of cluster center relative to grid origin

        dy: Dec of cluster center relative to grid origin

        dz: Line of sight offset of cluster center relative to grid origin

        r_1: X-axis scaling. Units are arbitrary, only radio of r_1/r_2, r_1/r_3, r_2/r_3 matters

        r_2: Y-axis scaling.

        r_3: Z-axis scaling.

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
    da = get_da(z)  # TODO pass these arguments rather than recompute them everytime???

    r500 = (m500 / (4.00 * jnp.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    r_ell = r500 / da
    r_norm = jnp.sqrt(r_1**2 + r_2**2 + r_3**2)

    r_1 *= r_ell / r_norm
    r_2 *= r_ell / r_norm
    r_3 *= r_ell / r_norm

    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    r = c500 * jnp.sqrt(x**2 + y**2 + z**2)
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
def cylindrical_beta(dx, dy, dz, L, theta, P0, r_c, beta, xyz):
    """
    
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Arguments:

        dx: RA of cluster center relative to grid origin

        dy: Dec of cluster center relative to grid origin

        dz: Line of sight offset of cluster center relative to grid origin

   





        theta: Angle to rotate in xy-plane

        P0: Amplitude of the pressure profile









        beta: The outer slope



        xyz: Coordinte grid to calculate model on

    Returns:

        model: The gnfw model
    """

    x, y, z = transform_grid(dx, dy, dz, 1.0, 1.0, 1.0, theta, xyz)
    r = jnp.sqrt(y**2 + z**2)
    powerlaw = P0 / (1.0 + (r / r_c) ** 2) ** (3.0 / 2.0 * beta)

    pressure = jnp.where(jnp.abs(x) >= L / 2.0, 0, powerlaw)

    return pressure


@jax.jit
def egaussian(dx, dy, dz, r_1, r_2, r_3, theta, sigma, amp, xyz):
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
def gaussian(dx, dy, sigma, amp, xyz):
    """
    Standard gaussian profile in 3d.
    This function does not include smoothing or declination stretch
    which should be applied at the end. The transform_grid call is
    awkward and can probably be removed/worked around. Function exists
    to match existing guassian interfaces.

    Arguments:

        dx: RA of gaussian center relative to grid origin

        dy: Dec of gaussian center relative to grid origin

        sigma: The effective, beam-convolved half-width of the point source.

        amp: Amplitude of the gaussian

        xyz: Coordinte grid to calculate model on

    Returns:

        model: The gaussian
    """
    x, y, z = transform_grid(dx, dy, 0, 1, 1, 1, 0, xyz)
    rr = x[..., 0] ** 2 + y[..., 0] ** 2
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
    pressure, xyz, dx, dy, dz, r_1, r_2, r_3, theta, amp, phi0, k_r, k_phi
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

        phi0: Polar angle of nose of power law

        k_r: Slope of power law in radial dirction

        k_phi: Slope of power law in polar direction

    Returns:

        new_pressure: Pressure profile with ellipsoid added
    """
    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)
    r = jnp.sqrt(x**2 + y**2 + z**2)
    phi = abs((jnp.arctan2(y, x) - phi0) % (2 * jnp.pi) - jnp.pi) / jnp.pi

    powerlaw = (
        amp
        * (1 - jnp.float_power(1 + r, -1.0 * k_r))
        * (1 - jnp.float_power(1 + phi, -1 * k_phi))
    )
    new_pressure = jnp.where(r > 1, pressure, (1 + powerlaw) * pressure)
    return new_pressure


@jax.jit
def add_powerlaw_cos(
    pressure, xyz, dx, dy, dz, r_1, r_2, r_3, theta, amp, phi0, k_r, omega
):
    """
    Add ellipsoid with radial power law and angular cos dependant structure to 3d pressure profile.

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

        phi0: Polar angle of nose of power law

        k_r: Slope of power law in radial dirction

        omega: Angular freqency if cos term

    Returns:

        new_pressure: Pressure profile with ellipsoid added
    """
    x, y, z = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)
    r = jnp.sqrt(x**2 + y**2 + z**2)
    phi = (jnp.arctan2(y, x) - phi0) % (2 * jnp.pi)

    powerlaw = amp * jnp.abs(jnp.cos(omega * phi)) * jnp.float_power(r, k_r)
    new_pressure = jnp.where(r > 1, pressure, (1 + powerlaw) * pressure)
    return new_pressure


# Get number of parameters for each structure
# The -1 is because xyz doesn't count
# -2 for Uniform, expo, and power as they also take a pressure arg that doesn't count
# For now a line needs to be added for each new model but this could be more magic down the line
N_PAR_ISOBETA = len(inspect.signature(isobeta).parameters) - 1
N_PAR_GNFW = len(inspect.signature(gnfw).parameters) - 1
N_PAR_A10 = len(inspect.signature(a10).parameters) - 1
N_PAR_EA10 = len(inspect.signature(ea10).parameters) - 1
N_PAR_CYLINDRICAL = len(inspect.signature(cylindrical_beta).parameters) - 1
N_PAR_GAUSSIAN = len(inspect.signature(gaussian).parameters) - 1
N_PAR_EGAUSSIAN = len(inspect.signature(egaussian).parameters) - 1
N_PAR_UNIFORM = len(inspect.signature(add_uniform).parameters) - 2
N_PAR_EXPONENTIAL = len(inspect.signature(add_exponential).parameters) - 2
N_PAR_POWERLAW = len(inspect.signature(add_powerlaw).parameters) - 2

# Make a convenience mapping
STRUCT_FUNCS = {
    "a10": a10,
    "ea10": ea10,
    "cylindrical": cylindrical_beta,
    "exponential": add_exponential,
    "powerlaw": add_powerlaw,
    "powerlaw_cos": add_powerlaw_cos,
    "uniform": add_uniform,
    "egaussian": egaussian,
    "gaussian": gaussian,
    "gnfw": gnfw,
    "isobeta": isobeta,
}
STRUCT_N_PAR = {
    "a10": N_PAR_A10,
    "ea10": N_PAR_EA10,
    "cylindrical": N_PAR_CYLINDRICAL,
    "exponential": N_PAR_EXPONENTIAL,
    "powerlaw": N_PAR_POWERLAW,
    "powerlaw_cos": N_PAR_POWERLAW,
    "uniform": N_PAR_UNIFORM,
    "egaussian": N_PAR_EGAUSSIAN,
    "gaussian": N_PAR_GAUSSIAN,
    "gnfw": N_PAR_GNFW,
    "isobeta": N_PAR_ISOBETA,
}
STRUCT_STAGE = {
    "a10": 0,
    "ea10": 0,
    "cylindrical": 0,
    "exponential": 1,
    "powerlaw": 1,
    "powerlaw_cos": 1,
    "uniform": 1,
    "egaussian": 0,
    "gaussian": 2,
    "gnfw": 0,
    "isobeta": 0,
}
