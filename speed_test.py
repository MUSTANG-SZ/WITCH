from minkasi_jax import conv_int_gnfw, gauss
import jax
import jax.numpy as jnp
import numpy as np
import timeit
import minkasi
import glob
import time
from astropy.cosmology import Planck15 as cosmo #choose your cosmology here
from jax import jit

#compile jit gradient function
jit_gnfw_deriv = jax.jacfwd(conv_int_gnfw, argnums = 0)
gauss_grad = jax.jacfwd(gauss, argnums = 0)

def helper(params, tod):
    x, y = tod
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    
    
    pred = conv_int_gnfw(params, x, y, 10., 0.5)

    derivs = jit_gnfw_deriv(params, x, y, 10., 0.5)
    
    #derivs = np.moveaxis(derivs, 2, 0)

    return derivs, pred

#compile vectorized functions
vmap_conv_int_gnfw = jax.vmap(conv_int_gnfw, in_axes = (None, 0, 0, None, None))
vmap_jit_gnfw_deriv = jax.jacfwd(vmap_conv_int_gnfw)

def vmap_helper(params, tod):
    x, y = tod
    x = jnp.asarray(x)
    y = jnp.asarray(y)


    pred = vmap_conv_int_gnfw(params, x, y, 10., 0.5)

    derivs = vmap_jit_gnfw_deriv(params, x, y, 10., 0.5)
    
    #derivs = np.moveaxis(derivs, 2, 0)

    return derivs, pred

test_tod = np.random.rand(2, int(1e4))
pars = np.array([0, 0, 1., 1., 1.5, 4.3, 0.7,3e14])

#Run both once to compile
#_, __ = helper(pars, test_tod)
#_,__ = vmap_helper(pars, test_tod)

def timeit_helper():
    return helper(pars, test_tod)

def vmap_timeit_helper():
    return vmap_helper(pars, test_tod)

#print(timeit.timeit(timeit_helper, number = 100))
#print(timeit.timeit(vmap_timeit_helper, number = 100))

#Gauss speed comparison

#compile jit gradient function


gauss_grad = jax.jacfwd(gauss, argnums = 0)
gauss_grad = jit(gauss_grad)

#vmap func
@jit
def vmap_gauss(pars, x, y):
    return vmap(gauss, in_axes = (None, 0, 0))(pars, x, y)


@jit
def vmap_gauss_grad(pars, x, y):
    return vmap(gauss_grad, in_axes = (None, 0, 0))(pars, x, y)


def helper(params, tod):
    x = tod.info['dx']
    y = tod.info['dy']
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    #print(np.amin((x-params[0])*3600*180/np.pi))
    #print(np.amin((y-params[1])*3600*180/np.pi))
    pred = vmap_gauss(params, x, y)

    #pred = conv_int_gnfw(params, x, y, 10., 0.5)
    derivs = vmap_gauss_grad(params, x, y)

    #derivs = jit_gnfw_deriv(params, x, y, 10., 0.5)

    derivs = jnp.moveaxis(derivs, 2, 0)

    return derivs, pred

#find tod files we want to map
outroot='/users/jscherer/Python/Minkasi/maps/zw3146'
dir='/home/scratch/cromero/MUSTANG2/Reductions/Zw3146/best_tods/'
#dir='../data/moo1110/'
#dir='../data/moo1046/'
tod_names=glob.glob(dir+'Sig*.fits')
tod_names.sort() #make sure everyone agrees on the order of the file names
tod_names=tod_names[:1] #you can cut the number of TODs here for testing purposes


#if running MPI, you would want to split up files between processes
#one easy way is to say to this:
tod_names=tod_names[minkasi.myrank::minkasi.nproc]
#NB - minkasi checks to see if MPI is around, if not
#it sets rank to 0 an nproc to 1, so this would still
#run in a non-MPI environment



todvec=minkasi.TodVec()
#loop over each file, and read it.
for fname in tod_names:
    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
                              #the length happy for ffts
    #minkasi.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
    #                              #this fixes that.  You don't need to, though.
    #minkasi.truncate_tod(dat)    #since our length changed, make sure we have a happy length#
    #

    #figure out a guess at common mode #and (assumed) linear detector drifts/offset
    #drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd=minkasi.fit_cm_plus_poly(dat['dat_calib'])

    dat['dat_calib']=dd
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)


for tod in todvec.tods:
    #ipix=map.get_pix(tod) #don't need pixellization for a timestream fit...
    #tod.info['ipix']=ipix
    #tod.set_noise_smoothed_svd()
    tod.set_noise(minkasi.NoiseSmoothedSVD)

pars = np.array([0., 0., 1., 1.])





#Jack vs Simon
#same as above for jax

def gnfw(max_R,P0,c500,alpha,beta,gamma,r500,P500,z,dR=None):
    '''returns pressure profile with r in MPc'''
    if dR == None : dR=max_R / 2e3
    radius=(np.arange(0,max_R,dR) + dR/2.0)

    x = radius / r500
    pressure=P500*P0/((c500*x)**gamma * (1.+(c500*x)**alpha)**((beta-gamma)/alpha) )
    return pressure,radius

def intProfile(profile,radius,z,r_map=15.0*60,dr=0.5):
    '''turn 1D pressure profile into surface brighness profile'''
    '''Input radius is in kpc returns y and radius in arcseconds'''
    '''r_map and dr are the radius you want to profile projected onto in "'''
    r_in_arcsec=np.arange(0,r_map,dr)
    r_in_Mpc=r_in_arcsec*(cosmo.angular_diameter_distance(z)*(np.pi/180./3600.)).value
    xx,zz=np.meshgrid(r_in_Mpc,r_in_Mpc)
    rr=np.sqrt(xx*xx+zz*zz)
    yy=np.interp(rr,radius,profile,right=0.)#2d pressue crosssection thru cluster
    Mparsec = 3.08568025e24 # centimeters in a megaparsec
    Xthom = 6.6524586e-25   # Thomson cross section (cm^2)
    mev = 5.11e2            # Electron mass (keV)
    #kev = 1.602e-9          # 1 keV in ergs.
    XMpc = Xthom*Mparsec
    y=np.sum(yy,axis=1)*2.*XMpc/(mev*1000)
    return y,r_in_arcsec

def y2K(freq=90e9,T_e=5.):
    k_b=1.3806e-16 ; T_cmb=2.725 ; h=6.626e-27
    me=511.0 # keV/c^2
    x=freq*h/k_b/T_cmb
    coth = lambda x : (np.exp(x)+np.exp(-x))/(np.exp(x)-np.exp(-x))
    rel_corr=(T_e/me)*(-10. + 23.5*x*coth(x/2) - 8.4*x**2*coth(x/2)**2 +0.7*x**3*coth(x/2)**3 + 1.4*x**2*(x*coth(x/2)-3.)/(np.sinh(x/2)**2))
    y2K=x*coth(x/2.)-4 + rel_corr
    return y2K



def derivs_from_gnfw(params,tod,z=None,profile=None,fwhm=9.,max_R=10.,freq=90e9,T_electron=5.0,pd=[(1.*np.pi)/(60*180),0.01],*args,**kwargs):
    '''for use with fit many functions etc
    integrates p(r)=P0/(c500*r)^gamma*(1+(c500*r)^alpha)^(beta-gamma)/alpha
    max_R is the radius to integrate the cluster profile out to in arcmin on the sky
    TBD - connect to faster python, add reuse of calculated profile for multiple calls
    freq is used to convert into T_RJ
    pd=steps to use for calc derivs (offsets, other parameters as fraction)
    '''
    #print('Entering')
    npar=8
    x0,y0,P0,c500,alpha,beta,gamma,m500=params
    #m500 *= 1e14
    sz_deriv=np.append(npar,tod.info['dat_calib'].shape)
    if z == None : #don't put this in function definition so user gets warning
        z=0.3
        print('Warning - redshift not given, assuming 0.3, type c to continue or go back and fix your code')
        #import pdb ; pdb.set_trace()
    #if Te == None : #don't put this in function definition so user gets warning
    #    Te=3e14
    #    print('Warning - Te not given, assuming 3e14, type c to continue or go back and fix your code')
    #    import pdb ; pdb.set_trace()
    if profile == None :
        MpcInArcmin=(cosmo.angular_diameter_distance(z)*(np.pi/180./60.)).value
        max_R_in_MPc = 10 #max_R * MpcInArcmin - integrate out to 16Mpc
        rho_crit=(cosmo.critical_density(z)).value#in g/cc
        rho_crit *= 3.086e24**3 / 1.989e33 #in M_sol / Mpc^3
        r500 = ((m500)/(500.*4.0/3.0*np.pi*rho_crit))**(1.0/3.0)#in Mpc
        ap=0.12 #eq 13 & 7 of A10 - break from self similarity
        h_z=(cosmo.H(z)/cosmo.H0).value
        h_70=(cosmo.H0/70.).value
        P500 = 1.65e-3*(m500/(3e14/h_70))**(2.0/3.0+ap)*h_z**(8./3.)*h_70**2
        print('r500 in Mpc=',r500,'  P500 =',P500)

        pressure,r=gnfw(max_R_in_MPc,P0,c500,alpha,beta,gamma,r500,P500,z)
        #import pylab as pl ; pl.plot(r,pressure) ; pl.show()
        ip,rmap=intProfile(pressure,r,z)#default out to r=16'
        #import pylab as pl ; pl.plot(rmap,ip) ; pl.show()
        #convolve with beam
        x=np.arange(-1.5*fwhm//(rmap[1]-rmap[0]),1.5*fwhm//(rmap[1]-rmap[0]))*(rmap[1]-rmap[0])
        beam=np.exp(-4*np.log(2)*x**2/fwhm**2) ; beam = beam / np.sum(beam)
        nx=x.shape[0]//2+1
        ipp=np.concatenate((ip[0:nx][::-1],ip))#extend array to stop edge effects at cluster center
        ip[:]=np.convolve(ipp,beam,mode='same')[nx:]
        profile=(ip,rmap)
    else :
        ip,rmap=profile

    #conversion factors from pressure to T_RJ
    y2K_CMB=y2K(freq=freq,T_e=T_electron)
    T_cmb=2.725 ; CMB2RJ=1.23
    ip*=y2K_CMB*T_cmb/CMB2RJ #conver to brightness temp

    dx=(tod.info['dx']-x0)*np.cos(tod.info['dy'])
    dy=tod.info['dy']-y0
    dr=np.sqrt(dx*dx + dy*dy)*180./np.pi*3600.
    pred=np.interp(dr,rmap,ip,right=0.)
    #should really use +-delta/2 but this would mean twice the calculations....
    sz_deriv=np.append(npar,tod.info['dat_calib'].shape)
    derivs=np.zeros(sz_deriv) #partual derivatives....
    ddx=(tod.info['dx']-x0-pd[0])*np.cos(tod.info['dy'])
    ddy=dy-pd[0]
    ddr_x=np.sqrt(ddx*ddx + dy*dy)*180./np.pi*3600.
    #print(ddr_x)
    derivs[0,:,:]=(np.interp(ddr_x,rmap,ip,right=0.)-pred)/pd[0]
    ddr_y=np.sqrt(dx*dx + ddy*ddy)*180./np.pi*3600.
    #print(dy)
    derivs[1,:,:]=(np.interp(ddr_y,rmap,ip,right=0.)-pred)/pd[0]
    derivs[2,:,:]=pred/P0
    print(derivs[2,:,:])
    pressure,r=gnfw(max_R_in_MPc,P0,c500*(1+pd[1]),alpha,beta,gamma,r500,P500,z)
    ip,rmap=intProfile(pressure,r,z)#default out to r=16'
    ipp=np.concatenate((ip[0:nx][::-1],ip))#extend array to stop edge effects at cluster center
    ip[:]=np.convolve(ipp,beam,mode='same')[nx:]
    ip*=y2K_CMB*T_cmb/CMB2RJ #convert to brightness temp
    derivs[3,:,:]=(np.interp(dr,rmap,ip,right=0.)-pred)/(pd[1]*c500)
    #####THIS IS GETTING NASTY - USE JAX OR SIMILAR???################

    pressure,r=gnfw(max_R_in_MPc,P0,c500,alpha*(1+pd[1]),beta,gamma,r500,P500,z)
    ip,rmap=intProfile(pressure,r,z)#default out to r=16'
    ipp=np.concatenate((ip[0:nx][::-1],ip))#extend array to stop edge effects at cluster center
    ip[:]=np.convolve(ipp,beam,mode='same')[nx:]
    ip*=y2K_CMB*T_cmb/CMB2RJ #convert to brightness temp
    derivs[4,:,:]=(np.interp(dr,rmap,ip,right=0.)-pred)/(pd[1]*alpha)

    pressure,r=gnfw(max_R_in_MPc,P0,c500,alpha,beta*(1+pd[1]),gamma,r500,P500,z)
    ip,rmap=intProfile(pressure,r,z)#default out to r=16'
    ipp=np.concatenate((ip[0:nx][::-1],ip))#extend array to stop edge effects at cluster center
    ip[:]=np.convolve(ipp,beam,mode='same')[nx:]
    ip*=y2K_CMB*T_cmb/CMB2RJ #convert to brightness temp
    derivs[5,:,:]=(np.interp(dr,rmap,ip,right=0.)-pred)/(pd[1]*beta)

    pressure,r=gnfw(max_R_in_MPc,P0,c500,alpha,beta,gamma*(1+pd[1]),r500,P500,z)
    ip,rmap=intProfile(pressure,r,z)#default out to r=16'
    ipp=np.concatenate((ip[0:nx][::-1],ip))#extend array to stop edge effects at cluster center
    ip[:]=np.convolve(ipp,beam,mode='same')[nx:]
    ip*=y2K_CMB*T_cmb/CMB2RJ #convert to brightness temp
    derivs[6,:,:]=(np.interp(dr,rmap,ip,right=0.)-pred)/(pd[1]*gamma)

    dr500 = ((m500*(1+pd[1]))/(500.*4.0/3.0*np.pi*rho_crit))**(1.0/3.0)#in Mpc
    dP500 =  1.65e-3*(m500/(3e14/h_70))**(2.0/3.0+ap)*h_z**(8./3.)*h_70**2
    pressure,r=gnfw(max_R_in_MPc,P0,c500,alpha,beta,gamma,dr500,dP500,z)
    ip,rmap=intProfile(pressure,r,z)#default out to r=16'
    ipp=np.concatenate((ip[0:nx][::-1],ip))#extend array to stop edge effects at cluster center
    ip[:]=np.convolve(ipp,beam,mode='same')[nx:]
    ip*=y2K_CMB*T_cmb/CMB2RJ #convert to brightness temp
    derivs[7,:,:]=(np.interp(dr,rmap,ip,right=0.)-pred)/(pd[1]*m500)
    #print('Derivs: ',derivs)
    #import pdb ; pdb.set_trace()
    #print('Exiting')
    return derivs, pred

#compile jit gradient function
gnfw_grad = jax.jacfwd(conv_int_gnfw, argnums = 0)
gauss_grad = jax.jacfwd(gauss, argnums = 0)

jit_gnfw_grad = jit(gnfw_grad)

def helper(params, tod):
    x, y = tod
    x = jnp.asarray(x)
    y = jnp.asarray(y)


    pred = conv_int_gnfw(params, x, y, 10., 0.5)

    derivs = jit_gnfw_grad(params, x, y, 10., 0.5)

    #derivs = np.moveaxis(derivs, 2, 0)

    return derivs, pred

#compile vectorized functions
vmap_conv_int_gnfw = jax.vmap(jit_gnfw_grad, in_axes = (None, 0, 0, None, None))
vmap_jit_gnfw_deriv = jax.jacfwd(vmap_conv_int_gnfw)

def vmap_helper(params, tod):
    x, y = tod
    x = jnp.asarray(x)
    y = jnp.asarray(y)


    pred = vmap_conv_int_gnfw(params, x, y, 10., 0.5)

    derivs = vmap_jit_gnfw_deriv(params, x, y, 10., 0.5)

    #derivs = np.moveaxis(derivs, 2, 0)

    return derivs, pred

test_tod = np.random.rand(2, int(1e4))
pars = np.array([0, 0, 1., 1., 1.5, 4.3, 0.7,3e14])

#Run both once to compile
_, __ = helper(pars, test_tod)
_,__ = vmap_helper(pars, test_tod)





