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
def y2K_CMB(freq, Te):
    x = freq * h / kb / Tcmb
    xt = x / jnp.tanh(0.5 * x)
    st = x / jnp.sinh(0.5 * x)
    Y0 = -4.0 + xt
    Y1 = (
        -10.0
        + ((47.0 / 2.0) + (-(42.0 / 5.0) + (7.0 / 10.0) * xt) * xt) * xt
        + st * st * (-(21.0 / 5.0) + (7.0 / 5.0) * xt)
    )
    Y2 = (
        (-15.0 / 2.0)
        + (
            (1023.0 / 8.0)
            + (
                (-868.0 / 5.0)
                + ((329.0 / 5.0) + ((-44.0 / 5.0) + (11.0 / 30.0) * xt) * xt) * xt
            )
            * xt
        )
        * xt
        + (
            (-434.0 / 5.0)
            + ((658.0 / 5.0) + ((-242.0 / 5.0) + (143.0 / 30.0) * xt) * xt) * xt
            + (-(44.0 / 5.0) + (187.0 / 60.0) * xt) * (st * st)
        )
        * st
        * st
    )
    Y3 = (
        (15.0 / 2.0)
        + (
            (2505.0 / 8.0)
            + (
                (-7098.0 / 5.0)
                + (
                    (14253.0 / 10.0)
                    + (
                        (-18594.0 / 35.0)
                        + (
                            (12059.0 / 140.0)
                            + ((-128.0 / 21.0) + (16.0 / 105.0) * xt) * xt
                        )
                        * xt
                    )
                    * xt
                )
                * xt
            )
            * xt
        )
        * xt
        + (
            (
                (-7098.0 / 10.0)
                + (
                    (14253.0 / 5.0)
                    + (
                        (-102267.0 / 35.0)
                        + (
                            (156767.0 / 140.0)
                            + ((-1216.0 / 7.0) + (64.0 / 7.0) * xt) * xt
                        )
                        * xt
                    )
                    * xt
                )
                * xt
            )
            + (
                (
                    (-18594.0 / 35.0)
                    + (
                        (205003.0 / 280.0)
                        + ((-1920.0 / 7.0) + (1024.0 / 35.0) * xt) * xt
                    )
                    * xt
                )
                + ((-544.0 / 21.0) + (992.0 / 105.0) * xt) * st * st
            )
            * st
            * st
        )
        * st
        * st
    )
    Y4 = (
        (-135.0 / 32.0)
        + (
            (30375.0 / 128.0)
            + (
                (-62391.0 / 10.0)
                + (
                    (614727.0 / 40.0)
                    + (
                        (-124389.0 / 10.0)
                        + (
                            (355703.0 / 80.0)
                            + (
                                (-16568.0 / 21.0)
                                + (
                                    (7516.0 / 105.0)
                                    + ((-22.0 / 7.0) + (11.0 / 210.0) * xt) * xt
                                )
                                * xt
                            )
                            * xt
                        )
                        * xt
                    )
                    * xt
                )
                * xt
            )
            * xt
        )
        * xt
        + (
            (-62391.0 / 20.0)
            + (
                (614727.0 / 20.0)
                + (
                    (-1368279.0 / 20.0)
                    + (
                        (4624139.0 / 80.0)
                        + (
                            (-157396.0 / 7.0)
                            + (
                                (30064.0 / 7.0)
                                + ((-2717.0 / 7.0) + (2761.0 / 210.0) * xt) * xt
                            )
                            * xt
                        )
                        * xt
                    )
                    * xt
                )
                * xt
            )
            * xt
            + (
                (-124389.0 / 10.0)
                + (
                    (6046951.0 / 160.0)
                    + (
                        (-248520.0 / 7.0)
                        + (
                            (481024.0 / 35.0)
                            + ((-15972.0 / 7.0) + (18689.0 / 140.0) * xt) * xt
                        )
                        * xt
                    )
                    * xt
                )
                * xt
                + (
                    (-70414.0 / 21.0)
                    + (
                        (465992.0 / 105.0)
                        + ((-11792.0 / 7.0) + (19778.0 / 105.0) * xt) * xt
                    )
                    * xt
                    + ((-682.0 / 7.0) + (7601.0 / 210.0) * xt) * st * st
                )
                * st
                * st
            )
            * st
            * st
        )
        * st
        * st
    )
    factor = Y0 + (Te / me) * (
        Y1 + (Te / me) * (Y2 + (Te / me) * (Y3 + (Te / me) * Y4))
    )
    return factor * Tcmb


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
@jax.partial(jax.jit, static_argnums=(4, 5, 6, 7, 8, 9))
def _conv_int_gnfw(
    p,
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
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p

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
    p,
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
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p

    rmap, ip = _conv_int_gnfw(
        p,
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
    x_scale,
    y_scale,
    theta,
    p,
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
       x_scale: Amount to scale along the x-axis (dx = dx/x_scale)

       y_scale: Amount to scale along the y-axis (dy = dy/y_scale)

       theta: Angle to rotate profile by in radians

       remaining args are the same as conv_int_gnfw

    Returns:
       Elliptical gnfw profile
    """
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p

    rmap, ip = _conv_int_gnfw(
        p,
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
    return conv_int_gnfw(
        p, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
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
):
    return jax.jacfwd(conv_int_gnfw, argnums=0)(
        p, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
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
):
    pred = conv_int_gnfw(
        p, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jax.jacfwd(conv_int_gnfw, argnums=0)(
        p, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )

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
    ),
)
def jit_conv_int_gnfw_elliptical(
    x_scale,
    y_scale,
    theta,
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
    pred = conv_int_gnfw_elliptical(
        x_scale,
        y_scale,
        theta,
        p,
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
    grad = jax.jacfwd(conv_int_gnfw_elliptical, argnums=3)(
        x_scale,
        y_scale,
        theta,
        p,
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
    jit_conv_int_gnfw_elliptical(1, 1, 0, pars, tods, 1.00)
    tic = time.time()
    print("4", tic - toc)
    toc = time.time()
    jit_conv_int_gnfw_elliptical(1, 1, 0, pars, tods, 1.00)
    tic = time.time()
    print("4", tic - toc)

    pars = jnp.array([0, 0, 1.0, 1.0, 1.5, 4.3, 0.7, 3e14])
    tods = jnp.array(np.random.rand(2, int(1e4)))

    print(timeit.timeit(helper, number=10) / 10)
