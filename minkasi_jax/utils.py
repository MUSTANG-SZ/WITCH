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
from astropy.cosmology import planck15 as cosmo

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

# Compton y to Kcmb
# --------------------------------------------------------
@partial(jax.jit, static_argnums=(0, 1))
def y2K_CMB(freq, Te):
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


# FFT Convolutions
# -----------------------------------------------------------
@jax.jit
def fft_conv(image, kernel):
    Fmap = jnp.fft.fft2(jnp.fft.fftshift(image))
    Fkernel = jnp.fft.fft2(jnp.fft.fftshift(kernel))
    convolved_map = jnp.fft.fftshift(jnp.real(jnp.fft.ifft2(Fmap * Fkernel)))

    return convolved_map


@partial(jax.jit, static_argnums=(0,))
def K_CMB2K_RJ(freq):
    x = freq * h / kb / Tcmb
    return jnp.exp(x) * x * x / jnp.expm1(x) ** 2


@partial(jax.jit, static_argnums=(0, 1))
def y2K_RJ(freq, Te):
    factor = y2K_CMB(freq, Te)
    return factor * K_CMB2K_RJ(freq)


# Bowling
# -----------------------------------------------------------


@jax.jit
def bowl(x0, y0, c0, c1, c2, xi, yi):
    # A function which returns predictions and gradients for a simple eliptical bowl
    # Inputs:
    #    x0,y0, the center of the bowl
    #    c0, c1, c2 the polynomial coefficients
    #    xi, yi, the xi and yi to evaluate at

    # Outputs:
    # pred the value f(x0, y0, c0, c1, c2)(xi, yi)

    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0
    dr = jnp.sqrt(dx * dx + dy * dy) * 180.0 / np.pi * 3600.0

    pred = 0


def get_rmap(r_map, r_1, r_2, r_3, z, beta, amp):
    if beta == 0 or amp == 0:
        return r_map
    da = np.interp(z, dzline, daline)
    r = np.max(jnp.array([r_1, r_2, r_3]))
    rmap = ((1e-10 / np.abs(amp)) ** (-1 / (1.5 * beta)) - 1) * (r / da)
    return np.nanmin(np.array([rmap, r_map]))


@partial(jax.jit, static_argnums=(1, 2))
def make_grid(z, r_map, dr):
    da = jnp.interp(z, dzline, daline)

    # Make grid with resolution dr and size r_map and convert to Mpc
    x = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dr)) * da
    y = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dr)) * da
    z = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dr)) * da

    return jnp.meshgrid(x, y, z, sparse=True, indexing="xy")


@jax.jit
def add_bubble(pressure, xyz, xb, yb, zb, rb, sup, z):
    da = jnp.interp(z, dzline, daline)

    # Recenter grid on bubble center
    x = xyz[0] - (xb * da)
    y = xyz[1] - (yb * da)
    z = xyz[2] - (zb * da)

    # Supress points inside bubble
    pressure_b = jnp.where(
        jnp.sqrt(x**2 + y**2 + z**2) >= (rb * da), pressure, (1 - sup) * pressure
    )
    return pressure_b


@jax.jit
def add_shock(pressure, xyz, sr_1, sr_2, sr_3, s_theta, shock):
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
