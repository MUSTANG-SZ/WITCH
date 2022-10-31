import numpy as np

import jax
import jax.numpy as jnp
JAX_DEBUG_NANS = True
from jax.config import config
import jax.scipy.optimize as sopt

from functools import partial

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name','cpu')

from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from astropy import units as u



def eliptical_gauss(p, x, y):
    #Gives the value of an eliptical gaussian with its center at x0, y0, evaluated at x,y, where theta1 and theta2 are the
    #FWHM of the two axes and psi is the roation angle. Amp is the amplitude
    #Note x,y are at the end to make the gradient indicies make more sense i.e. grad(numarg = 0) is grad wrt x0

    x0,y0,theta1,theta2,psi,amp = p

    theta1_inv=1/theta1
    theta2_inv=1/theta2
    theta1_inv_sqr=theta1_inv**2
    theta2_inv_sqr=theta2_inv**2
    cosdec=jnp.cos(y0)
    sindec=jnp.sin(y0)/jnp.cos(y0)
    cospsi=jnp.cos(psi)
    cc=cospsi**2
    sinpsi=jnp.sin(psi)
    ss=sinpsi**2
    cs=cospsi*sinpsi

    delx=(x-x0)*cosdec
    dely=y-y0
    xx=delx*cospsi+dely*sinpsi
    yy=dely*cospsi-delx*sinpsi
    xfac=theta1_inv_sqr*xx*xx
    yfac=theta2_inv_sqr*yy*yy
    rr=xfac+yfac
    rrpow=jnp.exp(-0.5*rr)

    return amp*rrpow

@jax.jit
def gauss(p, x, y):
    #Gives the value of an eliptical gaussian with its center at x0, y0, evaluated at x,y, where theta1 and theta2 are the
    #FWHM of the two axes and psi is the roation angle. Amp is the amplitude
    #Note x,y are at the end to make the gradient indicies make more sense i.e. grad(numarg = 0) is grad wrt x0

    x0,y0,sigma,amp = p

    sigma_inv=1/sigma
    sigma_inv_sqr=sigma_inv**2
    cosdec=jnp.cos(y0)
    sindec=jnp.sin(y0)/jnp.cos(y0)
    cospsi = 1
    cc=1
    sinpsi=0
    ss=sinpsi**2
    cs=cospsi*sinpsi

    delx=(x-x0)*cosdec
    dely=y-y0
    xx=delx*cospsi+dely*sinpsi
    yy=dely*cospsi-delx*sinpsi
    xfac=sigma_inv_sqr*xx*xx
    yfac=sigma_inv_sqr*yy*yy
    rr=xfac+yfac
    rrpow=jnp.exp(-0.5*rr)

    return amp*rrpow
