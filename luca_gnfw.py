import jax
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_platform_name','cpu')

import jax.numpy as jnp

from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from astropy import units as u

import numpy as np
import timeit
import time

h70 = cosmo.H0.value/7.00E+01

dzline = np.linspace(0.00,5.00,1000)
daline = cosmo.angular_diameter_distance(dzline)/u.radian
nzline = cosmo.critical_density(dzline)
hzline = cosmo.H(dzline)/cosmo.H0

daline = daline.to(u.Mpc/u.arcsec)
nzline = nzline.to(u.Msun/u.Mpc**3)

dzline = jnp.array(dzline)
hzline = jnp.array(hzline.value)
nzline = jnp.array(nzline.value)
daline = jnp.array(daline.value)

@jax.partial(jax.jit, static_argnums = (0,1))
def y2K(freq=90e9,T_e=5.):
    k_b=1.3806e-16 ; T_cmb=2.725 ; h=6.626e-27
    me=511.0 # keV/c^2
    x=freq*h/k_b/T_cmb
    coth = lambda x : (jnp.exp(x)+jnp.exp(-x))/(jnp.exp(x)-jnp.exp(-x))
    rel_corr=(T_e/me)*(-10. + 23.5*x*coth(x/2) - 8.4*x**2*coth(x/2)**2 +0.7*x**3*coth(x/2)**3 + 1.4*x**2*(x*coth(x/2)-3.)/(jnp.sinh(x/2)**2))
    y2K=x*coth(x/2.)-4 + rel_corr
    return y2K

def conv_int_gnfw(p,xi,yi,z,max_R=10.00,fwhm=9.,freq=90e9,T_electron=5.0,r_map=15.0*60,dr=0.5):
  x0, y0, P0, c500, alpha, beta, gamma, m500 = p

  hz = jnp.interp(z,dzline,hzline)
  nz = jnp.interp(z,dzline,nzline)

  ap = 0.12

  r500 = (m500/(4.00*jnp.pi/3.00)/5.00E+02/nz)**(1.00/3.00)
  P500 = 1.65E-03*(m500/(3.00E+14/h70))**(2.00/3.00+ap)*hz**(8.00/3.00)*h70**2

  dR = max_R/2e3
  r = jnp.arange(0.00,max_R,dR)+dR/2.00

  x = r/r500
  pressure = P500*P0/((c500*x)**gamma*(1.00+(c500*x)**alpha)**((beta-gamma)/alpha))

  rmap = jnp.arange(1e-10,r_map,dr) 
  r_in_Mpc = rmap*(jnp.interp(z,dzline,daline))
  rr = jnp.meshgrid(r_in_Mpc,r_in_Mpc)
  rr = jnp.sqrt(rr[0]**2+rr[1]**2)
  yy = jnp.interp(rr,r,pressure,right=0.)

  Mparsec = 3.08568025e24 # centimeters in a megaparsec
  Xthom = 6.6524586e-25   # Thomson cross section (cm^2)
  mev = 5.11e2            # Electron mass (keV)
  XMpc = Xthom*Mparsec

  ip = jnp.sum(yy,axis=1)*2.*XMpc/(mev*1000)

  x = jnp.arange(-1.5*fwhm//(dr),1.5*fwhm//(dr))*(dr)
  beam = jnp.exp(-4*np.log(2)*x**2/fwhm**2)
  beam = beam/jnp.sum(beam)

  nx = x.shape[0]//2+1

  ipp = jnp.concatenate((ip[0:nx][::-1],ip))
  ip = jnp.convolve(ipp,beam,mode='same')[nx:]

  y2K_CMB=y2K(freq=freq,T_e=T_electron)
  T_cmb=2.725 ; CMB2RJ=1.23
  ip = ip*y2K_CMB*T_cmb/CMB2RJ

  dx = (xi-x0)*jnp.cos(yi)
  dy  = yi-y0
  dr = jnp.sqrt(dx*dx + dy*dy)*180./np.pi*3600. 

  return jnp.interp(dr,rmap,ip,right=0.)

# ---------------------------------------------------------------

pars = jnp.array([0,0,1.,1.,1.5,4.3,0.7,3e14])
tods = jnp.array(np.random.rand(2,int(1e4)))

@jax.partial(jax.jit,static_argnums=(3,4,5,6,7,8,))
def val_conv_int_gnfw(p,tods,z,max_R=10.00,fwhm=9.,freq=90e9,T_electron=5.0,r_map=15.0*60,dr=0.5):
  return conv_int_gnfw(p,tods[0],tods[1],z,max_R,fwhm,freq,T_electron,r_map,dr)

@jax.partial(jax.jit,static_argnums=(3,4,5,6,7,8,))
def jac_conv_int_gnfw_fwd(p,tods,z,max_R=10.00,fwhm=9.,freq=90e9,T_electron=5.0,r_map=15.0*60,dr=0.5):
  return jax.jacfwd(conv_int_gnfw,argnums=0)(p,tods[0],tods[1],z,max_R,fwhm,freq,T_electron,r_map,dr)

@jax.partial(jax.jit,static_argnums=(3,4,5,6,7,8,))
def jit_conv_int_gnfw(p,tods,z,max_R=10.00,fwhm=9.,freq=90e9,T_electron=5.0,r_map=15.0*60,dr=0.5):
  pred = conv_int_gnfw(p,tods[0],tods[1],z,max_R,fwhm,freq,T_electron,r_map,dr)
  grad = jax.jacfwd(conv_int_gnfw,argnums=0)(p,tods[0],tods[1],z,max_R,fwhm,freq,T_electron,r_map,dr)

  return pred, grad

def helper():
  return jit_conv_int_gnfw(pars,tods,1.00)[0].block_until_ready()


if __name__ == '__main__':
    toc = time.time(); val_conv_int_gnfw(pars,tods,1.00); tic = time.time(); print('1',tic-toc)
    toc = time.time(); val_conv_int_gnfw(pars,tods,1.00); tic = time.time(); print('1',tic-toc)

    toc = time.time(); jac_conv_int_gnfw_fwd(pars,tods,1.00); tic = time.time(); print('2',tic-toc)
    toc = time.time(); jac_conv_int_gnfw_fwd(pars,tods,1.00); tic = time.time(); print('2',tic-toc)

    toc = time.time(); jit_conv_int_gnfw(pars,tods,1.00); tic = time.time(); print('3',tic-toc)
    toc = time.time(); jit_conv_int_gnfw(pars,tods,1.00); tic = time.time(); print('3',tic-toc)

    pars = jnp.array([0,0,1.,1.,1.5,4.3,0.7,3e14])
    tods = jnp.array(np.random.rand(2,int(1e4)))

    print(timeit.timeit(helper,number=10)/10)
