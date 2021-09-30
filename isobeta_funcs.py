import time, datetime, glob, sys, copy
import presets_by_source as pbs

import jax
import jax.numpy as jnp

import minkasi
import minkasi_nb

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

def helper(params, tod):
    x = tod.info['dx']
    y = tod.info['dy']
    
    xy = [x, y]
    xy = jnp.asarray(xy)

    pred, derivs = jit_conv_isobeta(params, xy)

    derivs = jnp.moveaxis(derivs, 2, 0)

    return derivs, pred

def jit_conv_isobeta(
    p,
    tod,
    z,
    max_R=10.00,
    fwhm=9.0,
    r_map=15.0 * 60,
    dr=0.5
):
    pred = conv_isobeta(
        p,
        tod[0],
        tod[1],
        z,
        max_R,
        fwhm,
        r_map,
        dr
    )
    grad = jax.jacfwd(conv_isobeta, argnums=0)(
        p,
        tod[0],
        tod[1],
        z,
        max_R,
        fwhm,
        r_map,
        dr
    )

    return pred, grad

def conv_isobeta(
    p,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    r_map=15.0 * 60,
    dr=0.5
):
    x0, y0, theta, beta, amp = p
    rmap, iff = _conv_isobeta(
        p,
        xi,
        yi,
        z,
        max_R=10.00,
        fwhm=9.0,
        r_map=15.0 * 60,
        dr=0.5
    )
    
    cosdec = np.cos(y0)
    
    delx = (x0-xi)*cosdec
    dely = y0-yi
    
    dr = jnp.sqrt(delx*delx + dely*dely)*180.0 / np.pi * 3600

    return jnp.interp(dr, rmap, iff, right=0.0)

def _conv_isobeta(
    p,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    r_map=15.0 * 60,
    dr=0.5
):
    x0, y0, theta, beta, amp = p
    theta_inv = 1./theta
    theta_inv_sqr = theta_inv*theta_inv
    cosdec = np.cos(y0)
    cosdec_inv = 1./cosdec
    sindec = np.sin(y0)
    power = 0.5 - 1.5*beta
    
    dR = max_R / 2e3
    
    r = jnp.arange(0.00, max_R, dR) + dR / 2.00
    gg = theta_inv_sqr * r**2
    flux = amp*(gg + 1)**power
    
        
    
    #-------
    
    rmap = jnp.arange(1e-10, r_map, dr)
    r_in_Mpc = rmap * (jnp.interp(z, dzline, daline))
    #rr = jnp.meshgrid(r_in_Mpc, r_in_Mpc)
    #rr = jnp.sqrt(rr[0] ** 2 + rr[1] ** 2)
    ff = jnp.interp(r_in_Mpc, r, flux, right=0.0)

    XMpc = Xthom * Mparsec

    #mustang beam 
    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam = jnp.exp(-4 * np.log(2) * x ** 2 / fwhm ** 2)
    beam = beam / jnp.sum(beam)

    nx = x.shape[0] // 2 + 1

    iff = jnp.concatenate((ff[0:nx][::-1], ff))
    iff_conv = jnp.convolve(iff, beam, mode="same")[nx:]

    return rmap, iff_conv


