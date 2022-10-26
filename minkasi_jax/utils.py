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
me = ((const.m_e * const.c ** 2).to(u.keV)).value
h = const.h.value
Xthom = const.sigma_T.to(u.cm ** 2).value

Mparsec = u.Mpc.to(u.cm)

# Cosmology
# --------------------------------------------------------
dzline = np.linspace(0.00, 5.00, 1000)
daline = cosmo.angular_diameter_distance(dzline) / u.radian
nzline = cosmo.critical_density(dzline)
hzline = cosmo.H(dzline) / cosmo.H0

daline = daline.to(u.Mpc / u.arcsec)
nzline = nzline.to(u.Msun / u.Mpc ** 3)

dzline = jnp.array(dzline)
hzline = jnp.array(hzline.value)
nzline = jnp.array(nzline.value)
daline = jnp.array(daline.value)

# Compton y to Kcmb
# --------------------------------------------------------
@partial(jax.jit, static_argnums=(0, 1))
def y2K_CMB(freq,Te):
    x = freq*h/kb/Tcmb
    xt = x/jnp.tanh(0.5*x)
    st = x/jnp.sinh(0.5*x)
    Y0 = -4.0+xt
    Y1 = -10.+((47./2.)+(-(42./5.)+(7./10.)*xt)*xt)*xt+st*st*(-(21./5.)+(7./5.)*xt)
    Y2 = (-15./2.)+((1023./8.)+((-868./5.)+((329./5.)+((-44./5.)+(11./30.)*xt)*xt)*xt)*xt)*xt+ \
         ((-434./5.)+((658./5.)+((-242./5.)+(143./30.)*xt)*xt)*xt+(-(44./5.)+(187./60.)*xt)*(st*st))*st*st
    Y3 = (15./2.)+((2505./8.)+((-7098./5.)+((14253./10.)+((-18594./35.)+((12059./140.)+((-128./21.)+(16./105.)*xt)*xt)*xt)*xt)*xt)*xt)*xt+ \
         (((-7098./10.)+((14253./5.)+((-102267./35.)+((156767./140.)+((-1216./7.)+(64./7.)*xt)*xt)*xt)*xt)*xt) +
         (((-18594./35.)+((205003./280.)+((-1920./7.)+(1024./35.)*xt)*xt)*xt) +((-544./21.)+(992./105.)*xt)*st*st)*st*st)*st*st
    Y4 = (-135./32.)+((30375./128.)+((-62391./10.)+((614727./40.)+((-124389./10.)+((355703./80.)+((-16568./21.)+((7516./105.)+((-22./7.)+(11./210.)*xt)*xt)*xt)*xt)*xt)*xt)*xt)*xt)*xt + \
         ((-62391./20.)+((614727./20.)+((-1368279./20.)+((4624139./80.)+((-157396./7.)+((30064./7.)+((-2717./7.)+(2761./210.)*xt)*xt)*xt)*xt)*xt)*xt)*xt + \
         ((-124389./10.)+((6046951./160.)+((-248520./7.)+((481024./35.)+((-15972./7.)+(18689./140.)*xt)*xt)*xt)*xt)*xt +\
         ((-70414./21.)+((465992./105.)+((-11792./7.)+(19778./105.)*xt)*xt)*xt+((-682./7.)+(7601./210.)*xt)*st*st)*st*st)*st*st)*st*st
    factor = Y0+(Te/me)*(Y1+(Te/me)*(Y2+(Te/me)*(Y3+(Te/me)*Y4)))
    return factor*Tcmb

# FFT Convolutions
#-----------------------------------------------------------
@jax.jit
def fft_conv(image, kernel):
    Fmap = jnp.fft.fft2(jnp.fft.fftshift(image))
    Fkernel = jnp.fft.fft2(jnp.fft.fftshift(kernel))
    convolved_map = jnp.fft.fftshift(jnp.real(jnp.fft.ifft2(Fmap*Fkernel)))

    return convolved_map



@partial(jax.jit, static_argnums=(0,))
def K_CMB2K_RJ(freq):
    x = freq * h / kb / Tcmb
    return jnp.exp(x) * x * x / jnp.expm1(x) ** 2


@partial(jax.jit, static_argnums=(0, 1))
def y2K_RJ(freq, Te):
    factor = y2K_CMB(freq, Te)
    return factor * K_CMB2K_RJ(freq)

#Bowling
#-----------------------------------------------------------

@jax.jit
def bowl(
    x0, y0, c0, c1, c2,
    xi, 
    yi
):
    #A function which returns predictions and gradients for a simple eliptical bowl
    #Inputs:
    #    x0,y0, the center of the bowl
    #    c0, c1, c2 the polynomial coefficients
    #    xi, yi, the xi and yi to evaluate at

    #Outputs:
    #pred the value f(x0, y0, c0, c1, c2)(xi, yi)

    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0
    dr = jnp.sqrt(dx * dx + dy * dy) * 180.0 / np.pi * 3600.0

    pred = 0


def get_rmap(r_map, r_1, r_2, r_3, z, beta, amp):
    if beta == 0 or amp == 0:
        return r_map
    da = np.interp(z, dzline, daline)
    r = np.max(jnp.array([r_1, r_2, r_3]))
    rmap = ((1e-10/np.abs(amp))**(-1/(1.5*beta)) - 1) * (r / da)
    return np.nanmin(np.array([rmap, r_map]))

@partial(jax.jit, static_argnums=(1, 2))
def make_grid(z, r_map, dr):
    da = jnp.interp(z, dzline, daline)

    # Make grid with resolution dr and size r_map and convert to Mpc
    x = jnp.linspace(-1*r_map, r_map, 2*int(r_map/dr)) * da
    y = jnp.linspace(-1*r_map, r_map, 2*int(r_map/dr)) * da
    z = jnp.linspace(-1*r_map, r_map, 2*int(r_map/dr)) * da

    return jnp.meshgrid(x, y, z, sparse=True, indexing='xy')


@jax.jit
def add_bubble(pressure, xyz, xb, yb, zb, rb, sup, z):
    da = jnp.interp(z, dzline, daline)

    # Recenter grid on bubble center
    x = xyz[0] - (xb * da)
    y = xyz[1] - (yb * da)
    z = xyz[2] - (zb * da)

    # Supress points inside bubble
    pressure_b = jnp.where(jnp.sqrt(x**2 + y**2 + z**2) >= (rb * da), pressure, (1 - sup)*pressure)
    return pressure_b

@jax.jit
def add_shock(pressure, xyz, sr_1, sr_2, sr_3, s_theta, shock):
    # Rotate
    xx = xyz[0]*jnp.cos(s_theta) + xyz[1]*jnp.sin(s_theta)
    yy = xyz[1]*jnp.cos(s_theta) - xyz[0]*jnp.sin(s_theta)
    zz = xyz[2]

    # Apply ellipticity
    xfac = (xx/sr_1)**2
    yfac = (yy/sr_2)**2
    zfac = (zz/sr_3)**2

    # Enhance points inside shock
    pressure_s = jnp.where(jnp.sqrt(xfac + yfac + zfac) > 1, pressure, (1 + shock)*pressure)
    return pressure_s
