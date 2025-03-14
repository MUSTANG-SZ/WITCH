"""
A set of utility functions and constants used for unit conversions
and cosmology as well as some generically useful math functions.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)

# TODO: Move generically useful math stuff to a new library for reuse

# Constants
# --------------------------------------------------------
ap = 0.12
h70 = cosmo.H0.value / 7.00e01
Tcmb = 2.7255  # K
kb = const.k_B.value
me = ((const.m_e * const.c**2).to(u.keV)).value
h = const.h.value
Xthom = const.sigma_T.to(u.cm**2).value
XMpc = Xthom * u.Mpc.to(u.cm)
rad_to_arcsec = (180 * 3600) / np.pi
rad_to_arcmin = (180 * 60) / np.pi
rad_to_deg = 180 / np.pi


# Cosmology
# --------------------------------------------------------
zline = jnp.linspace(0, 10, 100)
daline = jnp.array(
    (cosmo.angular_diameter_distance(zline) / u.radian).to(u.Mpc / u.arcsec).value
)
nzline = jnp.array(cosmo.critical_density(zline).to(u.Msun / u.Mpc**3).value)
hzline = jnp.array((cosmo.H(zline) / cosmo.H0).value)


# Unit conversions
# --------------------------------------------------------
@partial(jax.jit, static_argnums=(0, 1))
def y2K_CMB(freq: float, Te: float) -> float:
    """
    Convert from compton y to K_CMB.

    Parameters
    ----------
    freq : float
        The observing frequency in Hz.
    Te : float
        Electron temperature

    Returns
    -------
    y2K_CMB : float
        Conversion factor from compton y to K_CMB.
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
def K_CMB2K_RJ(freq: float) -> float:
    """
    Convert from K_CMB to K_RJ.

    Parameters
    ----------
    freq : float
        The observing frequency in Hz.

    Returns
    -------
    K_CMB2K_RJ : float
        Conversion factor from K_CMB to K_RJ.
    """
    x = freq * h / kb / Tcmb
    return jnp.exp(x) * x * x / jnp.expm1(x) ** 2


@partial(jax.jit, static_argnums=(0, 1))
def y2K_RJ(freq: float, Te: float) -> float:
    """
    Convert from compton y to K_RJ.

    Parameters
    ----------
    freq : float
        The observing frequency in Hz.
    Te : float
        Electron temperature

    Returns
    -------
    y2K_RJ : float
        Conversion factor from compton y to K_RJ.
    """
    factor = y2K_CMB(freq, Te)
    return factor * K_CMB2K_RJ(freq)


@jax.jit
def get_da(z: ArrayLike) -> jax.Array:
    """
    Get factor to convert from arcseconds to MPc.

    Parameters
    ----------
    z : ArrayLike
        The redshift at which to compute the factor.

    Returns
    -------
    da : jax.Array
        Conversion factor from arcseconds to MPc.
    """
    return jnp.interp(z, zline, daline)


@jax.jit
def get_nz(z: ArrayLike) -> jax.Array:
    """
    Get the critical density at a given redshift.

    Parameters
    ----------
    z : ArrayLike
        The redshift at which to compute the critical density.

    Returns
    -------
    nz : jax.Array
        Critical density at the given z.
        This is in units of solar masses per cubic Mpc.
    """
    return jnp.interp(z, zline, nzline)


@jax.jit
def get_hz(z: ArrayLike) -> jax.Array:
    """
    Get the dimensionless hubble constant, h, at a given redshift.

    Parameters
    ----------
    z : ArrayLike
        The redshift at which to compute h.

    Returns
    -------
    hz : jax.Array
        h at the given z.
    """
    return jnp.interp(z, zline, hzline)


# FFT Operations
# -----------------------------------------------------------
@jax.jit
def fft_conv(image: ArrayLike, kernel: ArrayLike) -> jax.Array:
    """
    Perform a convolution using FFTs for speed with jax.

    Parameters
    ----------
    image : ArrayLike
        Data to be convolved.
    kernel : ArrayLike
        Convolution kernel.

    Returns
    -------
    convolved_map : jax.Array
        Image convolved with kernel.
    """
    Fmap = jnp.fft.fft2(jnp.fft.fftshift(image))
    Fkernel = jnp.fft.fft2(jnp.fft.fftshift(kernel))
    convolved_map = jnp.fft.fftshift(jnp.real(jnp.fft.ifft2(Fmap * Fkernel)))

    return convolved_map


@partial(jax.jit, static_argnums=(1,))
def tod_hi_pass(tod: jax.Array, N_filt: int) -> jax.Array:
    """
    High pass a tod with a tophat

    Parameters
    ----------
    tod : jax.Array
        TOD to high pass.
    N_filt : int
        N_filt of tophat.

    Returns
    -------
    tod_filtered : jax.Array
        High pass filtered TOD
    """
    mask = jnp.ones(tod.shape)
    mask = mask.at[..., :N_filt].set(0.0)

    ## apply the filter in fourier space
    Ftod = jnp.fft.fft(tod)
    Ftod_filtered = Ftod * mask
    tod_filtered = jnp.fft.ifft(Ftod_filtered).real
    return tod_filtered


# Model building tools
# -----------------------------------------------------------
@jax.jit
def bilinear_interp(
    x: jax.Array, y: jax.Array, xp: jax.Array, yp: jax.Array, fp: jax.Array
) -> jax.Array:
    """
    JAX implementation of bilinear interpolation.
    Out of bounds values are set to 0.
    Using the repeated linear interpolation method here,
    see https://en.wikipedia.org/wiki/Bilinear_interpolation#Repeated_linear_interpolation.

    Parameters
    ----------
    x : jax.Array
        X values to return interpolated values at.
    y : jax.Array
        Y values to return interpolated values at.
    xp : jax.Array
        X values to interpolate with, should be 1D.
        Assumed to be sorted.
    yp : jax.Array
        Y values to interpolate with, should be 1D.
        Assumed to be sorted.
    fp : jax.Array
        Functon values at `(xp, yp)`, should have shape `(len(xp), len(yp))`.
        Note that if you are using meshgrid, we assume `'ij'` indexing.

    Returns
    -------
    f : jax.Array
        The interpolated values.
    """
    if len(xp.shape) != 1:
        raise ValueError("xp must be 1D")
    if len(yp.shape) != 1:
        raise ValueError("yp must be 1D")
    if fp.shape != xp.shape + yp.shape:
        raise ValueError(
            "Incompatible shapes for fp, xp, yp: %s, %s, %s",
            fp.shape,
            xp.shape,
            yp.shape,
        )

    # Figure out bounds and mapping
    # This breaks if xp, yp is not sorted
    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)
    q_11 = fp[ix - 1, iy - 1]
    q_21 = fp[ix, iy - 1]
    q_12 = fp[ix - 1, iy]
    q_22 = fp[ix, iy]

    # Interpolate in x to start
    denom_x = xp[ix] - xp[ix - 1]
    dx_1 = x - xp[ix - 1]
    dx_2 = xp[ix] - x
    f_xy1 = (dx_2 * q_11 + dx_1 * q_21) / denom_x
    f_xy2 = (dx_2 * q_12 + dx_1 * q_22) / denom_x

    # Now do y as well
    denom_y = yp[iy] - yp[iy - 1]
    dy_1 = y - yp[iy - 1]
    dy_2 = yp[iy] - y
    f = (dy_2 * f_xy1 + dy_1 * f_xy2) / denom_y

    # Zero out the out of bounds values
    f = jnp.where((x < xp[0]) + (x > xp[-1]) + (y < yp[0]) + (y > yp[-1]), 0.0, f)

    return f


def beam_double_gauss(
    dr: float, fwhm1: float, amp1: float, fwhm2: float, amp2: float
) -> jax.Array:
    """
    Helper function to generate a double gaussian beam.

    Parameters
    ----------
    dr : float
        Pixel size.
    fwhm1 : float
        Full width half max of the primary gaussian in the same units as `dr`.
    amp1 : float
        Amplitude of the primary gaussian.
    fwhm2 : float
        Full width half max of the secondairy gaussian in the same units as `dr`.
    amp2 : float
        Amplitude of the secondairy gaussian.

    Returns
    -------
        beam: Double gaussian beam.
    """
    x = jnp.arange(-1.5 * fwhm1 // (dr), 1.5 * fwhm1 // (dr)) * (dr)
    beam_xx, beam_yy = jnp.meshgrid(x, x)
    beam_rr = jnp.sqrt(beam_xx**2 + beam_yy**2)
    beam = amp1 * jnp.exp(-4 * jnp.log(2) * beam_rr**2 / fwhm1**2) + amp2 * jnp.exp(
        -4 * jnp.log(2) * beam_rr**2 / fwhm2**2
    )
    return beam / jnp.sum(beam)


def get_radial_mask(data, pix_size, radius):
    x0, y0 = int(data.shape[0] / 2), int(data.shape[1] / 2)  # TODO: subpixel alignment
    X, Y = np.arange(data.shape[0]), np.arange(data.shape[1])
    XX, YY = np.meshgrid(Y, X)

    dist = np.sqrt((XX - x0) ** 2 + (YY - y0) ** 2) * pix_size

    return dist < radius


def bin_map(data: ArrayLike, pixsize: float) -> tuple[np.array, np.array, np.array]:
    """
    Bins data radially.

    Parameters
    ----------
    data : ArrayLike
        Data to be radially binned
    pixsize : float
        Pixel spacing for data

    Returns
    -------
    rs : np.array
        Left bin edges 
    bin1d : np.array
        Mean of pixels in bin
    var1d : np.array
        Variance of pixels in bin
    """
    x = np.linspace(
        -data.shape[1] / 2 * pixsize, data.shape[1] / 2 * pixsize, data.shape[1]
    )
    y = np.linspace(
        -data.shape[0] / 2 * pixsize, data.shape[0] / 2 * pixsize, data.shape[0]
    )

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)  # TODO: miscentering?

    rs = np.arange(0, np.amax(R), pixsize)
    rs = np.append(rs, 999999)
    bin1d = np.zeros(len(rs) - 1)
    var1d = np.zeros(len(rs) - 1)

    for k in range(len(rs) - 1):
        pixels = [
            data[i, j]
            for i in range(len(y))
            for j in range(len(x))
            if rs[k] < R[i, j] <= rs[k + 1]
        ]
        if len(pixels) == 0:
            bin1d[k] = 0
            var1d[k] = 0
        else:
            bin1d[k] = np.mean(pixels)
            var1d[k] = np.var(pixels)
    rs = rs[:-1]

    return rs, bin1d, var1d
