import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import jax.scipy as jsp

from astropy.cosmology import planck15 as cosmo
from astropy import constants as const
from astropy import units as u

import scipy as sp
import numpy as np
import timeit
import time

import matplotlib.pyplot as plt

from functools import partial
from utils import dzline, hzline, nzline, daline, h70, Mparsec, Xthom, me, y2K_RJ, fft_conv, make_grid, add_shock, add_bubble 

@jax.jit
def _isobeta_elliptical(
    x0, y0, r_1, r_2, r_3, theta, beta, amp,
    xi,
    yi,
    xyz
):
    """
    Elliptical isobeta pressure profile
    This function does not include smoothing or declination stretch
    which should be applied at the end
    """
    # Rotate
    xx = xyz[0]*jnp.cos(theta) + xyz[1]*jnp.sin(theta)
    yy = xyz[1]*jnp.cos(theta) - xyz[0]*jnp.sin(theta)
    zz = xyz[2]

    # Apply ellipticity
    xfac = (xx/r_1)**2
    yfac = (yy/r_2)**2
    zfac = (zz/r_3)**2

    # Calculate pressure profile
    rr = 1 + xfac + yfac + zfac
    power = -1.5*beta
    rrpow = rr**power

    return amp*rrpow


@partial(jax.jit, static_argnums=(11, 12, 13, 14, 15, 16))
def _int_isobeta_elliptical(
    x0, y0, r_1, r_2, r_3, theta, beta, amp,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    """
    Elliptical isobeta
    This function does not include smoothing or declination stretch
    which should be applied at the end
    """
    da = jnp.interp(z, dzline, daline)
    XMpc = Xthom * Mparsec

    pressure, _ = _isobeta_elliptical(
        x0, y0, r_1, r_2, r_3, theta, beta, amp,
        xi,
        yi,
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    # Integrate line of sight pressure
    return jnp.trapz(pressure, dx=dr*da, axis=-1) * XMpc / me


def conv_int_isobeta_elliptical_two_bubbles(
    x0, y0, r_1, r_2, r_3, theta, beta, amp,
    xb1, yb1, zb1, rb1, sup1,
    xb2, yb2, zb2, rb2, sup2,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    da = jnp.interp(z, dzline, daline)
    XMpc = Xthom * Mparsec

    # Get xyz grid
    xyz = make_grid(z, r_map, dr)

    # Get pressure
    pressure = _isobeta_elliptical(
        x0*(180*3600)/jnp.pi, y0*(180*3600)/jnp.pi,
        r_1, r_2, r_3, theta, beta, amp,
        xi,
        yi,
        xyz
    )

    # Add first bubble
    pressure = add_bubble(pressure, xyz, xb1, yb1, zb1, rb1, sup1, z)

    # Add second bubble
    pressure = add_bubble(pressure, xyz, xb2, yb2, zb2, rb2, sup2, z)

    # Integrate along line of site
    ip = jnp.trapz(pressure, dx=dr*da, axis=-1) * XMpc / me

    # Sum of two gaussians with amp1, fwhm1, amp2, fwhm2
    amp1, fwhm1, amp2, fwhm2 = 9.735, 0.9808, 32.627, 0.0192
    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam_xx, beam_yy = jnp.meshgrid(x,x)
    beam_rr = jnp.sqrt(beam_xx**2 + beam_yy**2)
    beam = amp1*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm1 ** 2) + amp2*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm2 ** 2)
    beam = beam / jnp.sum(beam)
    
    bound0, bound1 = int((ip.shape[0]-beam.shape[0])/2), int((ip.shape[1] - beam.shape[1])/2)
    beam = jnp.pad(beam, ((bound0, ip.shape[0]-beam.shape[0]-bound0), (bound1, ip.shape[1] - beam.shape[1] - bound1)))

    ip = fft_conv(ip, beam)
    ip = ip * y2K_RJ(freq=freq, Te=T_electron)
    
    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    dx *= (180*3600)/jnp.pi
    dy *= (180*3600)/jnp.pi
    full_rmap = jnp.arange(-1*r_map, r_map, dr) * da

    idx, idy = (dx + r_map)/(2*r_map)*len(full_rmap), (-dy + r_map)/(2*r_map)*len(full_rmap)
    return jsp.ndimage.map_coordinates(ip, (idy, idx), order = 0)#, ip

def conv_int_double_isobeta_elliptical_two_bubbles(
    x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1,
    r_4, r_5, r_6, theta_2, beta_2, amp_2,
    xb1, yb1, zb1, rb1, sup1,
    xb2, yb2, zb2, rb2, sup2,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    da = jnp.interp(z, dzline, daline)
    XMpc = Xthom * Mparsec

    # Get xyz grid
    xyz = make_grid(z, r_map, dr)

    # Get first pressure
    pressure_1 = _isobeta_elliptical(
        x0*(180*3600)/jnp.pi, y0*(180*3600)/jnp.pi,
        r_1, r_2, r_3, theta_1, beta_1, amp_1,
        xi,
        yi,
        xyz
    )
    
    # Get second pressure
    pressure_2 = _isobeta_elliptical(
        x0*(180*3600)/jnp.pi, y0*(180*3600)/jnp.pi,
        r_4, r_5, r_6, theta_2, beta_2, amp_2,
        xi,
        yi,
        xyz
    )

    # Add profiles
    pressure = pressure_1 + pressure_2

    # Add first bubble
    pressure = add_bubble(pressure, xyz, xb1, yb1, zb1, rb1, sup1, z)

    # Add second bubble
    pressure = add_bubble(pressure, xyz, xb2, yb2, zb2, rb2, sup2, z)

    # Integrate along line of site
    ip = jnp.trapz(pressure, dx=dr*da, axis=-1) * XMpc / me

    # Sum of two gaussians with amp1, fwhm1, amp2, fwhm2
    amp1, fwhm1, amp2, fwhm2 = 9.735, 0.9808, 32.627, 0.0192
    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam_xx, beam_yy = jnp.meshgrid(x,x)
    beam_rr = jnp.sqrt(beam_xx**2 + beam_yy**2)
    beam = amp1*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm1 ** 2) + amp2*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm2 ** 2)
    beam = beam / jnp.sum(beam)

    bound0, bound1 = int((ip.shape[0]-beam.shape[0])/2), int((ip.shape[1] - beam.shape[1])/2)
    beam = jnp.pad(beam, ((bound0, ip.shape[0]-beam.shape[0]-bound0), (bound1, ip.shape[1] - beam.shape[1] - bound1)))

    ip = fft_conv(ip, beam)
    ip = ip * y2K_RJ(freq=freq, Te=T_electron)
    
    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    dx *= (180*3600)/jnp.pi
    dy *= (180*3600)/jnp.pi
    full_rmap = jnp.arange(-1*r_map, r_map, dr) * da

    idx, idy = (dx + r_map)/(2*r_map)*len(full_rmap), (-dy + r_map)/(2*r_map)*len(full_rmap)
    return jsp.ndimage.map_coordinates(ip, (idy, idx), order = 0)#, ip


def conv_int_double_isobeta_elliptical_two_bubbles_shock(
    x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1,
    r_4, r_5, r_6, theta_2, beta_2, amp_2,
    xb1, yb1, zb1, rb1, sup1,
    xb2, yb2, zb2, rb2, sup2,
    sr_1, sr_2, sr_3, s_theta, shock, 
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    da = jnp.interp(z, dzline, daline)
    XMpc = Xthom * Mparsec

    # Get xyz grid
    xyz = make_grid(z, r_map, dr)

    # Get first pressure
    pressure_1 = _isobeta_elliptical(
        x0*(180*3600)/jnp.pi, y0*(180*3600)/jnp.pi,
        r_1, r_2, r_3, theta_1, beta_1, amp_1,
        xi,
        yi,
        xyz
    )

    # Get second pressure
    pressure_2 = _isobeta_elliptical(
        x0*(180*3600)/jnp.pi, y0*(180*3600)/jnp.pi,
        r_4, r_5, r_6, theta_2, beta_2, amp_2,
        xi,
        yi,
        xyz
    )

    # Add profiles
    pressure = pressure_1 + pressure_2

    # Add shock
    pressure = add_shock(pressure, xyz, sr_1, sr_2, sr_3, s_theta, shock)

    # Add first bubble
    pressure = add_bubble(pressure, xyz, xb1, yb1, zb1, rb1, sup1, z)

    # Add second bubble
    pressure = add_bubble(pressure, xyz, xb2, yb2, zb2, rb2, sup2, z)

    # Integrate along line of site
    ip = jnp.trapz(pressure, dx=dr*da, axis=-1) * XMpc / me

    # Sum of two gaussians with amp1, fwhm1, amp2, fwhm2
    amp1, fwhm1, amp2, fwhm2 = 9.735, 0.9808, 32.627, 0.0192
    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam_xx, beam_yy = jnp.meshgrid(x, x)
    beam_rr = jnp.sqrt(beam_xx**2 + beam_yy**2)
    beam = amp1*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm1 ** 2) + amp2*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm2 ** 2)
    beam = beam / jnp.sum(beam)

    bound0, bound1 = int((ip.shape[0]-beam.shape[0])/2), int((ip.shape[1] - beam.shape[1])/2)
    beam = jnp.pad(beam, ((bound0, ip.shape[0]-beam.shape[0]-bound0), (bound1, ip.shape[1] - beam.shape[1] - bound1)))

    ip = fft_conv(ip, beam)
    ip = ip * y2K_RJ(freq=freq, Te=T_electron)

    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    dx *= (180*3600)/jnp.pi
    dy *= (180*3600)/jnp.pi
    full_rmap = jnp.arange(-1*r_map, r_map, dr) * da

    idx, idy = (dx + r_map)/(2*r_map)*len(full_rmap), (-dy + r_map)/(2*r_map)*len(full_rmap)
    return jsp.ndimage.map_coordinates(ip, (idy, idx), order = 0)#, ip

@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17
    ),
)
def jit_conv_int_isobeta_elliptical_two_bubbles(
    p,
    tods,
    z,
    xb1, yb1, zb1, rb1,
    xb2, yb2, zb2, rb2,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    ):
    x0, y0, r_1, r_2, r_3, theta, beta, amp, sup1, sup2 = p
    
    pred = conv_int_isobeta_elliptical_two_bubbles(
         x0, y0, r_1, r_2, r_3, theta, beta, amp, xb1, yb1, zb1, rb1, sup1, xb2, yb2, zb2, rb2, sup2, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
  
    if len(argnums) == 0:
        return pred, jnp.zeros((len(p)+8,) + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_isobeta_elliptical_two_bubbles, argnums=argnums)(
         x0, y0, r_1, r_2, r_3, theta, beta, amp, xb1, yb1, zb1, rb1, sup1, xb2, yb2, zb2, rb2, sup2, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jnp.array(grad)
 
    padded_grad = jnp.zeros((len(p)+8,) + grad[0].shape) + 1e-30
    argnums = jnp.array(argnums)
    grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad


@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17
    ),
)
def jit_conv_int_double_isobeta_elliptical_two_bubbles(
    p,
    tods,
    z,
    xb1, yb1, zb1, rb1,
    xb2, yb2, zb2, rb2,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    ):
    x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1, r_4, r_5, r_6, theta_2, beta_2, amp_2, sup1, sup2 = p
    
    pred = conv_int_double_isobeta_elliptical_two_bubbles(
         x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1, r_4, r_5, r_6, theta_2, beta_2, amp_2, xb1, yb1, zb1, rb1, sup1, xb2, yb2, zb2, rb2, sup2, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    
    if len(argnums) == 0:
        return pred, jnp.zeros((len(p)+8,) + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_double_isobeta_elliptical_two_bubbles, argnums=argnums)(
         x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1, r_4, r_5, r_6, theta_2, beta_2, amp_2, xb1, yb1, zb1, rb1, sup1, xb2, yb2, zb2, rb2, sup2, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jnp.array(grad)
 
    padded_grad = jnp.zeros((len(p)+8,) + grad[0].shape) + 1e-30
    argnums = jnp.array(argnums)
    grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad


@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17
    ),
)
def jit_conv_int_double_isobeta_elliptical_two_bubbles_shock(
    p,
    tods,
    z,
    xb1, yb1, zb1, rb1,
    xb2, yb2, zb2, rb2,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    ):
    x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1, r_4, r_5, r_6, theta_2, beta_2, amp_2, sup1, sup2, sr_1, sr_2, sr_3, s_theta, shock = p

    pred = conv_int_double_isobeta_elliptical_two_bubbles_shock(
         x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1, r_4, r_5, r_6, theta_2, beta_2, amp_2, xb1, yb1, zb1, rb1, sup1, xb2, yb2, zb2, rb2, sup2, sr_1, sr_2, sr_3, s_theta, shock, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )

    if len(argnums) == 0:
        return pred, jnp.zeros((len(p)+8,) + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_double_isobeta_elliptical_two_bubbles_shock, argnums=argnums)(
         x0, y0, r_1, r_2, r_3, theta_1, beta_1, amp_1, r_4, r_5, r_6, theta_2, beta_2, amp_2, xb1, yb1, zb1, rb1, sup1, xb2, yb2, zb2, rb2, sup2, sr_1, sr_2, sr_3, s_theta, shock, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jnp.array(grad)

    padded_grad = jnp.zeros((len(p)+8,) + grad[0].shape) + 1e-30
    argnums = jnp.array(argnums)
    grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad
