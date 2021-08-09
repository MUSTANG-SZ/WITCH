import minkasi
import minkasi_nb
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, value_and_grad, partial
import timeit
import jax
import numpy as np
from astropy.cosmology import Planck15 as cosmo
JAX_DEBUG_NANS = True
from jax.config import config

config.update("jax_debug_nans", True)
config.update("jax_enable_x64", True)

@partial(jit, static_argnums = (0,1))
def angular_diameter_distance(z, H0 = 67.4):
    #Calculate angular diameter distance assuming flat cosmology with Planck18 parameters
    #Currently not working but it seems astropy angular diameter distance works with jax
    c = 299792.458 #km/s to match H0
    dh = c/H0
    print(dh)
    omega_m = 0.315
    omega_l = 1-omega_m
    zs = jnp.linspace(0., z)
    E_z = lambda x: 1/jnp.sqrt(omega_m*(1+x**3)+omega_l*(1+x))
    d_C = dh*jnp.trapz(E_z(zs))
    
    return d_C/(1+z)

@partial(jit, static_argnums = 0)
def jit_ang_dia_dis(z):
    return cosmo.angular_diameter_distance(z)    

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


@jit
def y2K(freq=90e9,T_e=5.):
    k_b=1.3806e-16 ; T_cmb=2.725 ; h=6.626e-27
    me=511.0 # keV/c^2
    x=freq*h/k_b/T_cmb
    coth = lambda x : (jnp.exp(x)+jnp.exp(-x))/(jnp.exp(x)-jnp.exp(-x))
    rel_corr=(T_e/me)*(-10. + 23.5*x*coth(x/2) - 8.4*x**2*coth(x/2)**2 +0.7*x**3*coth(x/2)**3 + 1.4*x**2*(x*coth(x/2)-3.)/(jnp.sinh(x/2)**2))
    y2K=x*coth(x/2.)-4 + rel_corr
    return y2K


@partial(jit, static_argnums=0)
def gnfw(max_R, p, r500, P500):
    '''returns pressure profile with r in MPc'''
    P0, c500,alpha,beta,gamma, m500 = p
    dR=max_R / 2e3
    radius=(jnp.arange(0,max_R,dR) + dR/2.0)

    x = radius / r500
    pressure=P500*P0/((c500*x)**gamma * (1.+(c500*x)**alpha)**((beta-gamma)/alpha) )
    return pressure,radius

def np_gnfw(max_R, p, r500, P500):
    '''returns pressure profile with r in MPc'''
    P0, c500,alpha,beta,gamma = p
    dR=max_R / 2e3
    radius=(np.arange(0,max_R,dR) + dR/2.0)
    print('radius ', radius)
    x = radius / r500
    pressure=P500*P0/((c500*x)**gamma * (1.+(c500*x)**alpha)**((beta-gamma)/alpha) )
    print('pressure ', pressure)
    return pressure,radius


@partial(jit, static_argnums = (2,4))
def int_gnfw(profile, radius, z, r_map = 15.0*60, dr = 0.5):
    r_in_arcsec = jnp.arange(1e-10,r_map,dr) 
    r_in_Mpc = r_in_arcsec*(jit_ang_dia_dis(z)*(jnp.pi/180./3600.))
    xx,zz = jnp.meshgrid(r_in_Mpc,r_in_Mpc)
    rr = jnp.sqrt(xx*xx+zz*zz)
    yy = jnp.interp(rr,radius,profile,right=0.)#2d pressue crosssection thru cluster
    Mparsec = 3.08568025e24 # centimeters in a megaparsec
    Xthom = 6.6524586e-25   # Thomson cross section (cm^2)
    mev = 5.11e2            # Electron mass (keV)
    #kev = 1.602e-9          # 1 keV in ergs.
    XMpc = Xthom*Mparsec
    y = jnp.sum(yy,axis=1)*2.*XMpc/(mev*1000)
    return y,r_in_arcsec

def np_int_gnfw(profile, radius, z, r_map = 15.0*60, dr = 0.5):
    r_in_arcsec = np.arange(0,r_map,dr)
    r_in_Mpc = r_in_arcsec*(cosmo.angular_diameter_distance(z).value*(np.pi/180./3600.))
    xx,zz = np.meshgrid(r_in_Mpc,r_in_Mpc)
    rr = np.sqrt(xx*xx+zz*zz)
    yy = np.interp(rr,radius,profile,right=0.)#2d pressue crosssection thru cluster
    Mparsec = 3.08568025e24 # centimeters in a megaparsec
    Xthom = 6.6524586e-25   # Thomson cross section (cm^2)
    mev = 5.11e2            # Electron mass (keV)
    #kev = 1.602e-9          # 1 keV in ergs.
    XMpc = Xthom*Mparsec
    y = np.sum(yy,axis=1)*2.*XMpc/(mev*1000)
    return y,r_in_arcsec


@partial(jit, static_argnums = (3,4,5,8))
def conv_int_gnfw(p, xi, yi, max_R, z, fwhm = 9., freq = 90e9, T_electron = 5.0, r_map = 15.0*60, dr = 0.5):
    x0, y0, P0, c500,alpha,beta,gamma,m500 = p
    max_R_in_Mpc = 10.
    rho_crit = (cosmo.critical_density(z)).value
    rho_crit *= 3.086e24**3 / 1.989e33 #in M_sol / Mpc^3
    r500 = ((m500)/(500.*4.0/3.0*jnp.pi*rho_crit))**(1.0/3.0)#in Mpc
    ap=0.12 #eq 13 & 7 of A10 - break from self similarity
    h_z=(cosmo.H(z)/cosmo.H0).value
    h_70=(cosmo.H0/70.).value
    P500 = 1.65e-3*(m500/(3e14/h_70))**(2.0/3.0+ap)*h_z**(8./3.)*h_70**2
    
    pressure,r=gnfw(max_R_in_Mpc,p[2:], r500, P500)

    ip, rmap = int_gnfw(pressure, r, z, dr = dr)

    #plt.plot(rmap,ip)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlabel('r (Mpc)')
    #plt.ylabel('y')
    #plt.savefig('plots/comp_int_gnfw.png')
    #plt.close()

    x = jnp.arange(-1.5*fwhm//(dr),1.5*fwhm//(dr))*(dr)
    beam = jnp.exp(-4*np.log(2)*x**2/fwhm**2) ; beam = beam / jnp.sum(beam)
    nx = x.shape[0]//2+1
    ipp = jnp.concatenate((ip[0:nx][::-1],ip))#extend array to stop edge effects at cluster center
    ip = jnp.convolve(ipp,beam,mode='same')[nx:]
    #print(ip) 
    #ip = jnp.concatenate((ip[:1], ip))
    #rmap = jnp.concatenate((rmap[:1]-dr, rmap))

    profile = (ip, rmap)

    y2K_CMB=y2K(freq=freq,T_e=T_electron)
    T_cmb=2.725 ; CMB2RJ=1.23
    ip*=y2K_CMB*T_cmb/CMB2RJ #conver to brightness temp

    #FIX HERE
    #when working in 'flat' arcsec, you need to remove the factor of cos(y) and the radian to arcsec conversion
    #Add back in if working with TODs
    dx = (xi - x0)*jnp.cos(yi)
    dy  = yi - y0
    dr = jnp.sqrt(dx*dx + dy*dy)*180./np.pi*3600. 
     
    pred=jnp.interp(dr,rmap,ip,right=0.)
    
    return pred

def np_conv_int_gnfw(p, xi, yi, max_R, z, m500, fwhm = 9., freq = 90e9, T_electron = 5.0, r_map = 15.0*60, dr = 0.5):
    x0, y0, P0, c500,alpha,beta,gamma = p
    max_R_in_Mpc = 10.
    rho_crit = (cosmo.critical_density(z)).value
    rho_crit *= 3.086e24**3 / 1.989e33 #in M_sol / Mpc^3
    r500 = ((m500)/(500.*4.0/3.0*np.pi*rho_crit))**(1.0/3.0)#in Mpc
    ap=0.12 #eq 13 & 7 of A10 - break from self similarity
    h_z=(cosmo.H(z)/cosmo.H0).value
    h_70=(cosmo.H0/70.).value
    P500 = 1.65e-3*(m500/(3e14/h_70))**(2.0/3.0+ap)*h_z**(8./3.)*h_70**2

    pressure,r=np_gnfw(max_R_in_Mpc,p[2:], r500, P500)

    ip, rmap = np_int_gnfw(pressure, r, z, dr = dr)
    
    #plt.plot(rmap,ip)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlabel('r (Mpc)')
    #plt.ylabel('y')
    #plt.savefig('plots/comp_int_gnfw.png')
    #plt.close()

    x = np.arange(-1.5*fwhm//(dr),1.5*fwhm//(dr))*(dr)
    beam = np.exp(-4*np.log(2)*x**2/fwhm**2) ; beam = beam / np.sum(beam)
    nx = x.shape[0]//2+1
    ipp = np.concatenate((ip[0:nx][::-1],ip))#extend array to stop edge effects at cluster center
    ip = np.convolve(ipp,beam,mode='same')[nx:]
    #print(ip)
    #ip = jnp.concatenate((ip[:1], ip))
    #rmap = jnp.concatenate((rmap[:1]-dr, rmap))

    profile = (ip, rmap)

    y2K_CMB=y2K(freq=freq,T_e=T_electron)
    T_cmb=2.725 ; CMB2RJ=1.23
    ip*=y2K_CMB*T_cmb/CMB2RJ #conver to brightness temp

    #FIX HERE
    #when working in 'flat' arcsec, you need to remove the factor of cos(y) and the radian to arcsec conversion
    #Add back in if working with TODs
    dx = (xi - x0)#*jnp.cos(yi)
    dy  = yi - y0
    dr = np.sqrt(dx*dx + dy*dy)#*180./np.pi*3600.
    
    pred=np.interp(dr,rmap,ip,right=0.)
    pred = jnp.moveaxis(pred, 2, 0)
    return pred

	
#A grid of points we'll use a lot
bound = 10*jnp.pi/(180*3600)
x = jnp.linspace(-1*bound, bound, 20)
y = jnp.linspace(-1*bound, bound, 20)
xx, yy = jnp.meshgrid(x, y, sparce=True)

#Gaussian parameters
pars = jnp.array([0., 0., 3.5, 2.5, 0., 2.])

#Below block of code is for plotting the gaussian
"""
z = jit_elip_gauss(xx,yy, 0,0, 2.5, 2.5, 0, 2)
h = plt.contourf(x,y,z)
plt.axis('scaled')
plt.savefig('gauss.png')
"""

#Jit-ize the profile generator
jit_elip_gauss = jit(eliptical_gauss)

#
gnfw_pars = jnp.array([0., 0., 1., 1., 1.3, 4.3, 0.7, 3e14])
gnfw_labels = ['ra', 'dec', 'P500', 'c500', 'alpha', 'beta', 'delta', 'm500']

#y, r = conv_int_gnfw(gnfw_pars, 10., 0.5, 3e14)
#plt.plot(r, y)
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel('r (arcsec)')
#plt.ylabel('y')
#plt.savefig('plots/conv_int_gnfw.png')
#plt.close()


#We need non-sparce xx and yy now
xx, yy = jnp.meshgrid(x, y)


#Calculating the Jacobian: fast
jit_ell = jax.jacfwd(eliptical_gauss,argnums=0)

jit_gnfw_deriv = jax.jacfwd(conv_int_gnfw, argnums = 0)

out = jit_gnfw_deriv(gnfw_pars, xx, yy, 10., 0.5)
#print(out[:,:,4])
print(out.shape)
for i in range(len(gnfw_pars)): 
    #print(out[:,:,i])
    #print(np.amax(out[:,:,i]))
    plt.imshow(out[i,:,:])
    plt.title('Grad wrt {}'.format(gnfw_labels[i]))
    plt.xlabel('ra (arcsec)')
    plt.ylabel('dec (arcsec)')
    plt.savefig('plots/derivs_{}.png'.format(i))
    plt.close()

#pred = np.zeros(xx.shape)
#for i in range(xx.shape[0]):
#    for j in range(xx.shape[1]):
#        pred[i,j] = conv_int_gnfw(gnfw_pars,xx[i,j], yy[i,j], 10., 0.5, 3e14)

pred = conv_int_gnfw(gnfw_pars,xx, yy, 10., 0.5)    

plt.imshow(pred)
plt.savefig('plots/pred.png')
plt.close()
















