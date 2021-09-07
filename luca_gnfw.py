import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from astropy import units as u

import numpy as np
import timeit
import time

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
@jax.partial(jax.jit, static_argnums=(0, 1))
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


@jax.partial(jax.jit, static_argnums=(0,))
def K_CMB2K_RJ(freq):
    x = freq * h / kb / Tcmb
    return jnp.exp(x) * x * x / jnp.expm1(x) ** 2


@jax.partial(jax.jit, static_argnums=(0, 1))
def y2K_RJ(freq, Te):
    factor = y2K_CMB(freq, Te)
    return factor * K_CMB2K_RJ(freq)


# Beam-convolved gNFW profiel
# --------------------------------------------------------
@jax.partial(jax.jit, static_argnums=(11, 12, 13, 14, 15, 16))
def _conv_int_gnfw(
    x0, y0, P0, c500, alpha, beta, gamma, m500,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.5,
):
    hz = jnp.interp(z, dzline, hzline)
    nz = jnp.interp(z, dzline, nzline)

    ap = 0.12

    r500 = (m500 / (4.00 * jnp.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    P500 = (
        1.65e-03
        * (m500 / (3.00e14 / h70)) ** (2.00 / 3.00 + ap)
        * hz ** (8.00 / 3.00)
        * h70 ** 2
    )

    dR = max_R / 2e3
    r = jnp.arange(0.00, max_R, dR) + dR / 2.00

    x = r / r500
    pressure = (
        P500
        * P0
        / (
            (c500 * x) ** gamma
            * (1.00 + (c500 * x) ** alpha) ** ((beta - gamma) / alpha)
        )
    )

    rmap = jnp.arange(1e-10, r_map, dr)
    r_in_Mpc = rmap * (jnp.interp(z, dzline, daline))
    rr = jnp.meshgrid(r_in_Mpc, r_in_Mpc)
    rr = jnp.sqrt(rr[0] ** 2 + rr[1] ** 2)
    yy = jnp.interp(rr, r, pressure, right=0.0)

    XMpc = Xthom * Mparsec

    ip = jnp.sum(yy, axis=1) * 2.0 * XMpc / (me * 1000)

    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam = jnp.exp(-4 * np.log(2) * x ** 2 / fwhm ** 2)
    beam = beam / jnp.sum(beam)

    nx = x.shape[0] // 2 + 1

    ipp = jnp.concatenate((ip[0:nx][::-1], ip))
    ip = jnp.convolve(ipp, beam, mode="same")[nx:]

    ip = ip * y2K_RJ(freq=freq, Te=T_electron)

    return rmap, ip


def conv_int_gnfw(
    x0, y0, P0, c500, alpha, beta, gamma, m500,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.5,
):
    rmap, ip = _conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500,
        xi,
        yi,
        z,
        max_R=max_R,
        fwhm=fwhm,
        freq=freq,
        T_electron=T_electron,
        r_map=r_map,
        dr=dr,
    )

    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0
    dr = jnp.sqrt(dx * dx + dy * dy) * 180.0 / np.pi * 3600.0

    return jnp.interp(dr, rmap, ip, right=0.0)


def conv_int_gnfw_elliptical(
    x_scale, y_scale, theta, x0, y0, P0, c500, alpha, beta, gamma, m500,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.5,
):
    """
    Modification of conv_int_gnfw that adds ellipticity
    This is a somewhat crude implementation that could be improved in the future

    Arguments:
       Same as conv_int_gnfw except first 3 values of p are now:
           x_scale: Amount to scale along the x-axis (dx = dx/x_scale)

           y_scale: Amount to scale along the y-axis (dy = dy/y_scale)

           theta: Angle to rotate profile by in radians

       remaining args are the same as conv_int_gnfw

    Returns:
       Elliptical gnfw profile
    """
    rmap, ip = _conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500,
        xi,
        yi,
        z,
        max_R=max_R,
        fwhm=fwhm,
        freq=freq,
        T_electron=T_electron,
        r_map=r_map,
        dr=dr,
    )
    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    dx = dx / x_scale
    dy = dy / y_scale
    dx = dx * jnp.cos(theta) + dy * jnp.sin(theta)
    dy = -1 * dx * jnp.sin(theta) + dy * jnp.cos(theta)

    dr = jnp.sqrt(dx * dx + dy * dy) * 180.0 / np.pi * 3600.0
    return jnp.interp(dr, rmap, ip, right=0.0)


# ---------------------------------------------------------------

pars = jnp.array([0, 0, 1.0, 1.0, 1.5, 4.3, 0.7, 3e14])
pars_elliptical = jnp.array([1, 1, 0, 0, 0, 1.0, 1.0, 1.5, 4.3, 0.7, 3e14])
tods = jnp.array(np.random.rand(2, int(1e4)))


@jax.partial(
    jax.jit,
    static_argnums=(
        3,
        4,
        5,
        6,
        7,
        8,
    ),
)
def val_conv_int_gnfw(
    p,
    tods,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.5,
):
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p
    return conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )


@jax.partial(
    jax.jit,
    static_argnums=(
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ),
)
def jac_conv_int_gnfw_fwd(
    p,
    tods,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.5,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7,)
):
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p
    grad = jax.jacfwd(conv_int_gnfw, argnums=argnums)(
        x0, y0, P0, c500, alpha, beta, gamma, m500, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )

    if len(argnums) != len(p):
        padded_grad = jnp.zeros(p.shape + grad[0].shape)
        grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return grad


@jax.partial(
    jax.jit,
    static_argnums=(
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ),
)
def jit_conv_int_gnfw(
    p,
    tods,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.5,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7,)
    ):
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p
    pred = conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jax.jacfwd(conv_int_gnfw, argnums=argnums)(
        x0, y0, P0, c500, alpha, beta, gamma, m500, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )

    if len(argnums) != len(p):
        padded_grad = jnp.zeros(p.shape + grad[0].shape)
        grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad


@jax.partial(
    jax.jit,
    static_argnums=(
        6,
        7,
        8,
        9,
        10,
        11,
        12
    ),
)
def jit_conv_int_gnfw_elliptical(
    p,
    tods,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.5,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,)
):
    x_scale, y_scale, theta, x0, y0, P0, c500, alpha, beta, gamma, m500 = p
    pred = conv_int_gnfw_elliptical(
        x_scale, y_scale, theta, x0, y0, P0, c500, alpha, beta, gamma, m500,
        tods[0],
        tods[1],
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )
    grad = jax.jacfwd(conv_int_gnfw_elliptical, argnums=argnums)(
        x_scale, y_scale, theta, x0, y0, P0, c500, alpha, beta, gamma, m500,
        tods[0],
        tods[1],
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    if len(argnums) != len(p):
        padded_grad = jnp.zeros(p.shape + grad[0].shape)
        grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad


def helper():
    return jit_conv_int_gnfw(pars, tods, 1.00)[0].block_until_ready()


if __name__ == "__main__":
    toc = time.time()
    val_conv_int_gnfw(pars, tods, 1.00)
    tic = time.time()
    print("1", tic - toc)
    toc = time.time()
    val_conv_int_gnfw(pars, tods, 1.00)
    tic = time.time()
    print("1", tic - toc)

    toc = time.time()
    jac_conv_int_gnfw_fwd(pars, tods, 1.00)
    tic = time.time()
    print("2", tic - toc)
    toc = time.time()
    jac_conv_int_gnfw_fwd(pars, tods, 1.00)
    tic = time.time()
    print("2", tic - toc)

    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00)
    tic = time.time()
    print("3", tic - toc)
    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00)
    tic = time.time()
    print("3", tic - toc)

    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00, argnums=(0,))
    tic = time.time()
    print("3.1", tic - toc)
    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00, argnums=(0,))
    tic = time.time()
    print("3.1", tic - toc)
    
    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00, argnums=(0, 1,))
    tic = time.time()
    print("3.2", tic - toc)
    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00, argnums=(0, 1,))
    tic = time.time()
    print("3.2", tic - toc)
    
    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00, argnums=(0, 1, 2, 3,))
    tic = time.time()
    print("3.4", tic - toc)
    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00, argnums=(0, 1, 2, 3,))
    tic = time.time()
    print("3.4", tic - toc)
    
    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00, argnums=(0, 1, 2, 3, 4, 5, 6,))
    tic = time.time()
    print("3.6", tic - toc)
    toc = time.time()
    jit_conv_int_gnfw(pars, tods, 1.00, argnums=(0, 1, 2, 3, 4, 5, 6,))
    tic = time.time()
    print("3.6", tic - toc)
    
    toc = time.time()
    jit_conv_int_gnfw_elliptical(pars_elliptical, tods, 1.00)
    tic = time.time()
    print("4", tic - toc)
    toc = time.time()
    jit_conv_int_gnfw_elliptical(pars_elliptical, tods, 1.00)
    tic = time.time()
    print("4", tic - toc)

    pars = jnp.array([0, 0, 1.0, 1.0, 1.5, 4.3, 0.7, 3e14])
    tods = jnp.array(np.random.rand(2, int(1e4)))

    print(timeit.timeit(helper, number=10) / 10)
