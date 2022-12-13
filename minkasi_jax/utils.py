"""
A set of utility functions and constants used for unit conversions
and adding generic structure common to multiple models.
"""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# Constants
# --------------------------------------------------------

h70 = cosmo.H0.value / 7.00e01

Tcmb = 2.7255
kb = const.k_B.value
me = ((const.m_e * const.c**2).to(u.keV)).value
h = const.h.value
Xthom = const.sigma_T.to(u.cm**2).value

Mparsec = u.Mpc.to(u.cm)
XMpc = Xthom * Mparsec

# Cosmology
# --------------------------------------------------------
dzline = np.linspace(0.00, 5.00, 1000)
daline = cosmo.angular_diameter_distance(dzline) / u.radian
nzline = cosmo.critical_density(dzline)
hzline = cosmo.H(dzline) / cosmo.H0

daline = daline.to(u.Mpc / u.arcsec)
nzline = nzline.to(u.Msun / u.Mpc**3)

dzline = jnp.array(dzline)
hzline = jnp.array(hzline.value)
nzline = jnp.array(nzline.value)
daline = jnp.array(daline.value)

# Unit conversions
# --------------------------------------------------------
@partial(jax.jit, static_argnums=(0, 1))
def y2K_CMB(freq, Te):
    """
    Convert from compton y to K_CMB.

    Arguments:

        freq: The observing frequency in Hz.

        Te: Electron temperature

    Returns:

        y2K_CMB: Conversion factor from compton y to K_CMB.
    """
    x = freq * h / kb / Tcmb
    xt = x / jnp.tanh(0.5 * x)
    st = x / jnp.sinh(0.5 * x)
    # fmt:off
    Y0 = -4.0 + xt
    Y1 = (-10.0
        + ((47.0 / 2.0) + (-(42.0 / 5.0) + (7.0 / 10.0) * xt) * xt) * xt
        + st * st * (-(21.0 / 5.0) + (7.0 / 5.0) * xt)
    )
    Y2 = ((-15.0 / 2.0)
        + ((1023.0 / 8.0) + ((-868.0 / 5.0) + ((329.0 / 5.0) + ((-44.0 / 5.0) + (11.0 / 30.0) * xt) * xt) * xt) * xt) * xt
        + ((-434.0 / 5.0) + ((658.0 / 5.0) + ((-242.0 / 5.0) + (143.0 / 30.0) * xt) * xt) * xt
        + (-(44.0 / 5.0) + (187.0 / 60.0) * xt) * (st * st)) * st * st
    )
    Y3 = ((15.0 / 2.0)
        + ((2505.0 / 8.0) + ((-7098.0 / 5.0) + ((14253.0 / 10.0) + ((-18594.0 / 35.0) 
         + ((12059.0 / 140.0) + ((-128.0 / 21.0) + (16.0 / 105.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt
        + (((-7098.0 / 10.0) + ((14253.0 / 5.0) + ((-102267.0 / 35.0) + ((156767.0 / 140.0)
         + ((-1216.0 / 7.0) + (64.0 / 7.0) * xt) * xt) * xt) * xt) * xt)
         + (((-18594.0 / 35.0) + ((205003.0 / 280.0) + ((-1920.0 / 7.0) + (1024.0 / 35.0) * xt) * xt) * xt)
          + ((-544.0 / 21.0) + (992.0 / 105.0) * xt) * st * st) * st * st) * st * st
    )
    Y4 = ((-135.0 / 32.0)
        + ((30375.0 / 128.0) + ((-62391.0 / 10.0) + ((614727.0 / 40.0) + ((-124389.0 / 10.0) + ((355703.0 / 80.0) + ((-16568.0 / 21.0)
         + ((7516.0 / 105.0) + ((-22.0 / 7.0) + (11.0 / 210.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt) * xt) * xt
        + ((-62391.0 / 20.0) + ((614727.0 / 20.0) + ((-1368279.0 / 20.0) + ((4624139.0 / 80.0) + ((-157396.0 / 7.0) + ((30064.0 / 7.0)
         + ((-2717.0 / 7.0) + (2761.0 / 210.0) * xt) * xt) * xt) * xt) * xt) * xt) * xt
         + ((-124389.0 / 10.0)
          + ((6046951.0 / 160.0) + ((-248520.0 / 7.0) + ((481024.0 / 35.0) + ((-15972.0 / 7.0) + (18689.0 / 140.0) * xt) * xt) * xt) * xt) * xt
          + ((-70414.0 / 21.0) + ((465992.0 / 105.0) + ((-11792.0 / 7.0) + (19778.0 / 105.0) * xt) * xt) * xt
           + ((-682.0 / 7.0) + (7601.0 / 210.0) * xt) * st * st) * st * st) * st * st) * st * st
    )
    # fmt:on
    factor = Y0 + (Te / me) * (
        Y1 + (Te / me) * (Y2 + (Te / me) * (Y3 + (Te / me) * Y4))
    )
    return factor * Tcmb


@partial(jax.jit, static_argnums=(0,))
def K_CMB2K_RJ(freq):
    """
    Convert from K_CMB to K_RJ.

    Arguments:

        freq: The observing frequency in Hz.

    Returns:

        K_CMB2K_RJ: Conversion factor from K_CMB to K_RJ.
    """
    x = freq * h / kb / Tcmb
    return jnp.exp(x) * x * x / jnp.expm1(x) ** 2


@partial(jax.jit, static_argnums=(0, 1))
def y2K_RJ(freq, Te):
    """
    Convert from compton y to K_RJ.

    Arguments:

        freq: The observing frequency in Hz.

        Te: Electron temperature

    Returns:

        y2K_RJ: Conversion factor from compton y to K_RJ.
    """
    factor = y2K_CMB(freq, Te)
    return factor * K_CMB2K_RJ(freq)


def get_da(z):
    """
    Get factor to convert from arcseconds to MPc.

    Arguments:

        z: The redshift at which to compute the  factor.

    Returns:

        da: Conversion factor from arcseconds to MPc
    """
    return jnp.interp(z, dzline, daline)


# FFT Operations
# -----------------------------------------------------------
@jax.jit
def fft_conv(image, kernel):
    """
    Perform a convolution using FFTs for speed.

    Arguments:

        image: Data to be convolved

        kernel: Convolution kernel

    Returns:

        convolved_map: Image convolved with kernel.
    """
    Fmap = jnp.fft.fft2(jnp.fft.fftshift(image))
    Fkernel = jnp.fft.fft2(jnp.fft.fftshift(kernel))
    convolved_map = jnp.fft.fftshift(jnp.real(jnp.fft.ifft2(Fmap * Fkernel)))

    return convolved_map


@partial(jax.jit, static_argnums=(1,))
def tod_hi_pass(tod, N_filt):
    """
    High pass a tod with a tophat

    Arguments:

        tod: TOD to high pass

        N_filt: N_filt of tophat


    Returns:

        tod_filtered: Filtered TOD
    """
    mask = jnp.ones(tod.shape)
    mask = jax.ops.index_update(mask, jax.ops.index[..., :N_filt], 0.0)

    ## apply the filter in fourier space
    Ftod = jnp.fft.fft(tod)
    Ftod_filtered = Ftod * mask
    tod_filtered = jnp.fft.ifft(Ftod_filtered).real
    return tod_filtered


# Model building tools
# -----------------------------------------------------------
@partial(jax.jit, static_argnums=(1, 2))
def make_grid(r_map, dr):
    """
    Make coordinate grids to build models in.
    All grids are sparse and are int(2*r_map / dr) in each dimension.

    Arguments:

        r_map: Size of grid radially.

        dr: Grid resolution, should be in same units as r_map.

    Returns:

        x: Grid of x coordinates in same units as r_map.

        y: Grid of y coordinates in same units as r_map

        z: Grid of z coordinates in same units as r_map
    """
    # Make grid with resolution dr and size r_map
    x = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dr))
    y = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dr))
    z = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dr))

    return jnp.meshgrid(x, y, z, sparse=True, indexing="xy")


@jax.jit
def add_bubble(pressure, xyz, xb, yb, zb, rb, sup, z):
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


def tod_to_index(xi, yi, x0, y0, r_map, dr, conv_factor):
    """
    Convert RA/Dec TODs to index space.

    Arguments:

        xi: RA TOD

        yi: Dec TOD

        x0: RA at center of model. Nominally the cluster center.

        y0: Dec at center of model. Nominally the cluster center.

        r_map: Radial size of grid

        dr: Pixel size

        conv_factor: Conversion factor to put RA and Dec in same units as r_map.
                     Nominally (da * 180 * 3600) / pi

    Returns:

        idx: The RA TOD in index space

        idy: The Dec TOD in index space.
    """
    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    dx *= conv_factor
    dy *= conv_factor
    full_rmap = jnp.arange(-1 * r_map, r_map, dr)

    idx, idy = (dx + r_map) / (2 * r_map) * len(full_rmap), (-dy + r_map) / (
        2 * r_map
    ) * len(full_rmap)

    return idx, idy


def beam_double_gauss(dr, fwhm1=9.735, amp1=0.9808, fwhm2=32.627, amp2=0.0192):
    """
    Helper function to generate a double gaussian beam.

    Arguments:

        dr: Pixel size.

        fwhm1: Full width half max of the primary gaussian.

        amp1: Amplitude of the primary gaussian.

        fwhm2: Full width half max of the secondairy gaussian.

        amp2: Amplitude of the secondairy gaussian.

    Returns:

        beam: Double gaussian beam.
    """
    x = jnp.arange(-1.5 * fwhm1 // (dr), 1.5 * fwhm1 // (dr)) * (dr)
    beam_xx, beam_yy = jnp.meshgrid(x, x)
    beam_rr = jnp.sqrt(beam_xx**2 + beam_yy**2)
    beam = amp1 * jnp.exp(-4 * jnp.log(2) * beam_rr**2 / fwhm1**2) + amp2 * jnp.exp(
        -4 * jnp.log(2) * beam_rr**2 / fwhm2**2
    )
    return beam / jnp.sum(beam)
