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
def gnfw(
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    P0: float,
    c500: float,
    m500: float,
    gamma: float,
    alpha: float,
    beta: float,
    z: float,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
) -> jax.Array:
    r"""
    Elliptical gNFW pressure profile in 3d.
    This function does not include smoothing or declination stretch
    which should be applied at the end.
    TODO: Add units to add parameters!

    Once the grid is transformed the profile is computed as:

    $$
    \dfrac{P_{500} * P_{0}}{{\left( r^{\gamma}\left( 1 + r^{\alpha} \right) \right)}^{\dfrac{\beta - \gamma}{\alpha}}}
    $$

    where:

    $$
    r = c_{500} \sqrt{x^2 + y^2 + z^2} {\frac{3m_{500}}{2000 \pi n_z}}^{-\frac{1}{3}}
    $$

    $$
    P_{500} = 1.65 \times 10^{-3} {\frac{m_{500}*h_{70}}{3 \times 10^{14}}}^{\frac{2}{3} + ap}{h_z}^{\frac{8}{3}}{h_{70}}^2
    $$

    $n_z$ is the critical density at the cluster redshift and $h_z$ is the Hubble constant at the cluster redshift.

    Parameters
    ----------
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    r_1 : float
        Amount to scale along x-axis.
        Passed to `utils.transform_grid`.
    r_2 : float
        Amount to scale along y-axis.
        Passed to `utils.transform_grid`.
    r_3 : float
        Amount to scale along z-axis.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    P0 : float
        Amplitude of the pressure profile.
    c500 : float
        Concentration parameter at a density contrast of 500.
    m500 : float
        Mass at a density contrast of 500.
    gamma : float
        The central slope.
    alpha : float
        The intermediate slope.
    beta : float
        The outer slope.
    z : float
        Redshift of cluster.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.

    Returns
    -------
    model : jax.Array
        The gnfw model evaluated on the grid.
    """
    nz = get_nz(z)
    hz = get_hz(z)

    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

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
def a10(
    dx: float,
    dy: float,
    dz: float,
    theta: float,
    P0: float,
    c500: float,
    m500: float,
    gamma: float,
    alpha: float,
    beta: float,
    z: float,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
) -> jax.Array:
    r"""
    gNFW pressure profile in 3d based on [Arnaud2010](https://ui.adsabs.harvard.edu/abs/2010A%26A...517A..92A/).
    Compared to the function gnfw, this function fixes r1/r2/r3 to r500.
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    See the docstring for `gnfw` for more details.

    Parameters
    ----------
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    P0 : float
        Amplitude of the pressure profile
    c500 : float
        Concentration parameter at a density contrast of 500
    m500 : float
        Mass at a density contrast of 500
    gamma : float
        The central slope
    alpha : float
        The intermediate slope
    beta : float
        The outer slope
    z : float
        Redshift of cluster
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.

    Returns
    -------
    model : jax.Array
        The gnfw model evaluated on the grid.
    """

    nz = get_nz(z)
    hz = get_hz(z)
    da = get_da(z)  # TODO pass these arguments rather than recompute them everytime???

    r500 = (m500 / (4.00 * jnp.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    r_1, r_2, r_3 = r500 / da, r500 / da, r500 / da

    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

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
def ea10(
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    P0: float,
    c500: float,
    m500: float,
    gamma: float,
    alpha: float,
    beta: float,
    z: float,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
) -> jax.Array:
    r"""
    Eliptical gNFW pressure profile in 3d based on Arnaud2010.
    r_ell is computed in the usual way for an a10 profile, then the axes are
    scaled according to r_1, r_2, r_3, with a normalization applied.
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    See the docstring for `gnfw` for more details.

    Parameters
    ----------
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    r_1 : float
        Amount to scale along x-axis.
        Units are arbitrary, only ratio of r_1/r_2, r_1/r_3, r_2/r_3 matters
        Passed to `utils.transform_grid`.
    r_2 : float
        Amount to scale along y-axis.
        Passed to `utils.transform_grid`.
    r_3 : float
        Amount to scale along z-axis.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    P0 : float
        Amplitude of the pressure profile
    c500 : float
        Concentration parameter at a density contrast of 500
    m500 : float
        Mass at a density contrast of 500
    gamma : float
        The central slope
    alpha : float
        The intermediate slope
    beta : float
        The outer slope
    z : float
        Redshift of cluster
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.

    Returns
    -------
    model : jax.Array
        The gnfw model evaluated on the grid.
    """
    nz = get_nz(z)
    hz = get_hz(z)
    da = get_da(z)  # TODO pass these arguments rather than recompute them everytime???

    r500 = (m500 / (4.00 * jnp.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    r_ell = r500 / da
    r_norm = (r_1 * r_2 * r_3) ** (1 / 3)

    r_1 *= r_ell / r_norm
    r_2 *= r_ell / r_norm
    r_3 *= r_ell / r_norm

    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

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
def isobeta(
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    beta: float,
    amp: float,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
) -> jax.Array:
    r"""
    Elliptical isobeta pressure profile in 3d.
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Once the grid is transformed the profile is computed as:

    $$
    P_{0}\left( 1 + x**2 + y**2 + z**2 \right)^{-1.5\beta}
    $$

    where $P_{0}$ is `amp`.

    Parameters
    ----------
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    r_1 : float
        Amount to scale along x-axis.
        Passed to `utils.transform_grid`.
    r_2 : float
        Amount to scale along y-axis.
        Passed to `utils.transform_grid`.
    r_3 : float
        Amount to scale along z-axis.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    beta : float
        Beta value of isobeta model.
    amp : float
        Amplitude of isobeta model.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.

    Returns
    -------
    model : Array
        The jax.isobeta model evaluated on the grid.
    """
    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    rr = 1 + x**2 + y**2 + z**2
    power = -1.5 * beta
    rrpow = rr**power

    return amp * rrpow


@jax.jit
def cylindrical_beta(
    dx: float,
    dy: float,
    dz: float,
    L: float,
    theta: float,
    P0: float,
    r_c: float,
    beta: float,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
) -> jax.Array:
    r"""

    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Once the grid is transformed the profile is computed as:

    $$
    P_{0}\left( 1 + \frac{y^2 + z^2}{{r_c}^2} \right)^{-1.5\beta}
    $$

    Parameters
    ----------
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    L : float
        Length of the cylinder.
        Aligned with the x-axis.
        Note that we consider anything where $\left| x \right| \\leq L$
        to be in the profile, so the actual length is $2L$.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    P0 : float
        Amplitude of the pressure profile.
    r_c : float
        The critical radius of the cylindrical profile.
    beta : float
        Beta value of isobeta model.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.

    Returns
    -------
    model : jax.Array
        The cylindrical beta model evaluated on the grid.
    """
    x, y, z, *_ = transform_grid(dx, dy, dz, 1.0, 1.0, 1.0, theta, xyz)
    r = jnp.sqrt(y**2 + z**2)
    powerlaw = P0 / (1.0 + (r / r_c) ** 2) ** (3.0 / 2.0 * beta)

    pressure = jnp.where(jnp.abs(x) >= L / 2.0, 0, powerlaw)

    return pressure


@jax.jit
def egaussian(
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    sigma: float,
    amp: float,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
) -> jax.Array:
    r"""
    Elliptical gaussian profile in 3d.
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Once the grid is transformed the profile is computed as:

    $$
    P_{0} e^{-\frac{x^2 + y^2 + z^2}{2\sigma^2}}
    $$

    where $P_{0}$ is `amp`.

    Parameters
    ----------
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    r_1 : float
        Amount to scale along x-axis.
        Passed to `utils.transform_grid`.
    r_2 : float
        Amount to scale along y-axis.
        Passed to `utils.transform_grid`.
    r_3 : float
        Amount to scale along z-axis.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    sigma : float
        Sigma value of gaussian model.
    amp : float
        Amplitude of gaussian model.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.

    Returns
    -------
    model : jax.Array
        The gaussain model evaluated on the grid.
    """
    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    rr = x**2 + y**2 + z**2
    power = -1 * rr / (2 * sigma**2)

    return amp * jnp.exp(power)


@jax.jit
def gaussian(
    dx: float,
    dy: float,
    sigma: float,
    amp: float,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
) -> jax.Array:
    r"""
    Standard gaussian profile in 2d.
    This function does not include smoothing or declination stretch
    which should be applied at the end. The transform_grid call is
    awkward and can probably be removed/worked around. Function exists
    to match existing guassian interfaces.

    Once the grid is transformed the profile is computed as:

    $$
    P_{0} e^{-\frac{x^2 + y^2}{2\sigma^2}}
    $$

    where $P_{0}$ is `amp`.

    Parameters
    ----------
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    sigma : float
        Sigma value of gaussian model.
    amp : float
        Amplitude of gaussian model.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        We only care about x and y here.
        See `containers.Model.xyz` for details.

    Returns
    -------
    model : jax.Array
        The gaussian model evaluated on only the 2d xy grid.
    """
    x, y, *_ = transform_grid(dx, dy, 0, 1, 1, 1, 0, xyz)
    rr = x[..., 0] ** 2 + y[..., 0] ** 2
    power = -1 * rr / (2 * sigma**2)

    return amp * jnp.exp(power)


@jax.jit
def add_uniform(
    pressure: jax.Array,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    amp: float,
) -> jax.Array:
    r"""
    Add ellipsoid with uniform structure to 3d pressure profile.

    After transforming the grid the region where $\sqrt{x^2 + y^2 + z^2} \leq 1$
    will be multiplied by a factor of $1 + P_{0}$ where $P_{0}$ is `amp`.

    Parameters
    ----------
    pressure : jax.Array
        The pressure profile to modify with this ellipsoid.
        Should be evaluated on the same grid as `xyz`.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    r_1 : float
        Amount to scale along x-axis.
        Passed to `utils.transform_grid`.
    r_2 : float
        Amount to scale along y-axis.
        Passed to `utils.transform_grid`.
    r_3 : float
        Amount to scale along z-axis.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    amp : float
        Factor by which pressure is enhanced within the ellipsoid.

    Returns
    -------
    new_pressure : Array
        Pressure profile with ellipsoid added.
    """
    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    new_pressure = jnp.where(
        jnp.sqrt(x**2 + y**2 + z**2) > 1, pressure, (1 + amp) * pressure
    )
    return new_pressure


@jax.jit
def add_exponential(
    pressure: jax.Array,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    amp: float,
    xk: float,
    x0: float,
    yk: float,
    y0: float,
    zk: float,
    z0: float,
) -> jax.Array:
    r"""
    Add ellipsoid with exponential structure to 3d pressure profile.

    After transforming the grid the region where $\sqrt{x^2 + y^2 + z^2} \leq 1$
    will be multiplied by a factor of $1 + P_{0} e^{x_k(x-x_0) + y_k(y-y_0) + z_k(z-z_0)}$
    where $P_{0}$ is `amp`.

    Parameters
    ----------
    pressure : jax.Array
        The pressure profile to modify with this ellipsoid.
        Should be evaluated on the same grid as `xyz`.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    r_1 : float
        Amount to scale along x-axis.
        Passed to `utils.transform_grid`.
    r_2 : float
        Amount to scale along y-axis.
        Passed to `utils.transform_grid`.
    r_3 : float
        Amount to scale along z-axis.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    amp : float
        Factor by which pressure is enhanced at the peak of ellipsoid.
    xk : float
        Power of exponential in RA direction
    x0 : float
        RA offset of exponential.
        Note that this is in transformed coordinates so x0=1 is at xs + sr_1.
    yk : float
        Power of exponential in Dec direction
    y0 : float
        Dec offset of exponential.
        Note that this is in transformed coordinates so y0=1 is at ys + sr_2.
    zk : float
        Power of exponential along the line of sight
    z0 : float
        Line of sight offset of exponential.
        Note that this is in transformed coordinates so z0=1 is at zs + sr_3.

    Returns
    -------
    new_pressure : Array
        Pressure profile with ellipsoid added.
    """
    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)

    exponential = amp * jnp.exp(((x - x0) * xk) + ((y - y0) * yk) + ((z - z0) * zk))

    new_pressure = jnp.where(
        jnp.sqrt(x**2 + y**2 + z**2) > 1, pressure, (1 + exponential) * pressure
    )
    return new_pressure


@jax.jit
def add_powerlaw(
    pressure: jax.Array,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    amp: float,
    phi0: float,
    k_r: float,
    k_phi: float,
) -> jax.Array:
    r"""
    Add ellipsoid with power law structure to 3d pressure profile.

    After transforming the grid the region where $\sqrt{x^2 + y^2 + z^2} \leq 1$
    will be multiplied by a factor of $1 + P_{0}(1 - {1 + r}^{-k_r})(1 - {1 + \phi}^{-k_{\phi}})$.
    Where $r$ and $\phi$ are the usual polar coordinates and $P_{0}$ is `amp`.

    Parameters
    ----------
    pressure : jax.Array
        The pressure profile to modify with this ellipsoid.
        Should be evaluated on the same grid as `xyz`.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    r_1 : float
        Amount to scale along x-axis.
        Passed to `utils.transform_grid`.
    r_2 : float
        Amount to scale along y-axis.
        Passed to `utils.transform_grid`.
    r_3 : float
        Amount to scale along z-axis.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    amp : float
        Factor by which pressure is enhanced within the ellipsoid.
    phi0 : float
        Polar angle of nose of power law.
    k_r : float
        Slope of power law in radial direction.
    k_phi : float
        Slope of power law in polar direction.

    Returns
    -------
    new_pressure : Array
        Pressure profile with ellipsoid added.
    """
    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)
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
    pressure: jax.Array,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    amp: float,
    phi0: float,
    k_r: float,
    omega: float,
) -> jax.Array:
    r"""
    Add ellipsoid with radial power law and angular cosine dependant structure to 3d pressure profile.

    After transforming the grid the region where $\sqrt{x^2 + y^2 + z^2} \leq 1$
    will be multiplied by a factor of $1 + P_{0} ({1 + r}^{-k_r}) \left| cos(\omega\phi) \right|$.
    Where $r$ and $\phi$ are the usual polar coordinates and $P_{0}$ is `amp`.

    Parameters
    ----------
    pressure : jax.Array
        The pressure profile to modify with this ellipsoid.
        Should be evaluated on the same grid as `xyz`.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Coordinte grid to calculate model on.
        See `containers.Model.xyz` for details.
    dx : float
        RA of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dy : float
        Dec of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    dz : float
        Line of sight offset of cluster center relative to grid origin.
        Passed to `utils.transform_grid`.
    r_1 : float
        Amount to scale along x-axis.
        Passed to `utils.transform_grid`.
    r_2 : float
        Amount to scale along y-axis.
        Passed to `utils.transform_grid`.
    r_3 : float
        Amount to scale along z-axis.
        Passed to `utils.transform_grid`.
    theta : float
        Angle to rotate in xy-plane.
        Passed to `utils.transform_grid`.
    amp : float
        Factor by which pressure is enhanced within the ellipsoid.
    phi0 : float
        Polar angle of nose of power law.
    k_r : float
        Slope of power law in radial direction.
    omega : float
        Angular freqency of the cosine term.

    Returns
    -------
    new_pressure : Array
        Pressure profile with ellipsoid added.
    """
    x, y, z, *_ = transform_grid(dx, dy, dz, r_1, r_2, r_3, theta, xyz)
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
