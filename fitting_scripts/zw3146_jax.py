#This is a template script to show how to fit multi-component models
#directly to timestreams.  The initial part (where the TODs, noise model etc.)
#are set up is the same as general mapping scripts, although we don't
#need to bother with setting a map/pixellization in general (although if
#your timestream model used a map you might).  This script works under
#MPI as well.

import numpy
import numpy as np
from matplotlib import pyplot as plt
import minkasi
import time
import glob
from importlib import reload
reload(minkasi)
plt.ion()
from minkasi_jax import conv_int_gnfw, eliptical_gauss, gauss
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, partial
import timeit
import jax
from astropy.coordinates import Angle
from astropy import units as u

jit_ell = jax.jacfwd(eliptical_gauss,argnums=0)
jit_gauss = jax.jacfwd(gauss, argnums = 0)


#compile jit gradient function
jit_gnfw_deriv = jax.jacfwd(conv_int_gnfw, argnums = 0)

def helper(params, tod):
    x = tod.info['dx']
    y = tod.info['dy']
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    #print(np.amin((x-params[0])*3600*180/np.pi))
    #print(np.amin((y-params[1])*3600*180/np.pi))
    pred = gauss(params, x, y)

    #pred = conv_int_gnfw(params, x, y, 10., 0.5)
    derivs = jit_gauss(params, x, y)

    #derivs = jit_gnfw_deriv(params, x, y, 10., 0.5)

    derivs = np.moveaxis(derivs, 2, 0)

    return derivs, pred


#find tod files we want to map
outroot='/users/jscherer/Python/Minkasi/maps/zw3146'
dir='/home/scratch/cromero/MUSTANG2/Reductions/Zw3146/best_tods/'
#dir='../data/moo1110/'
#dir='../data/moo1046/'
tod_names=glob.glob(dir+'Sig*.fits')  
tod_names.sort() #make sure everyone agrees on the order of the file names
tod_names=tod_names[:6] #you can cut the number of TODs here for testing purposes


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

#make a template map with desired pixel size an limits that cover the data
#todvec.lims() is MPI-aware and will return global limits, not just
#the ones from private TODs
lims=todvec.lims()
pixsize=2.0/3600*numpy.pi/180
#map=minkasi.SkyMap(lims,pixsize) #we don't need a map when fitting timestreams

#once we have a map, we can figure out the pixellization of the data.  Save that
#so we don't have to recompute.  Also calculate a noise model.  The one here
#is to rotate the data into SVD space, then smooth the power spectrum of each mode. 
# The smoothing could do with a bit of tuning..


for tod in todvec.tods:
    #ipix=map.get_pix(tod) #don't need pixellization for a timestream fit...
    #tod.info['ipix']=ipix
    #tod.set_noise_smoothed_svd()
    tod.set_noise(minkasi.NoiseSmoothedSVD)


#we need an initial guess since this fitting routine is
#for nonlinear models.  This guess came from looking
#at a map/some initial fits.  The better the guess, the
#faster the convergence.
d2r=np.pi/180
sig=9/2.35/3600*d2r
theta_0=40/3600*d2r

#Original par guesses
#beta_pars=np.asarray([155.91355*d2r,4.1877*d2r,theta_0,0.7,-8.2e-4])
#src1_pars=np.asarray([155.9374*d2r,4.1775*d2r,3.1e-5,  9.15e-4])
#src2_pars=np.asarray([155.90447*d2r,4.1516*d2r,2.6e-5,  5.1e-4])

#Further off
beta_pars=np.asarray([155.91255*d2r,4.1877*d2r,theta_0,0.7,-7.2e-4])
src1_pars=np.asarray([155.9364*d2r,4.1775*d2r,2.3e-5,  8.25e-4])
src2_pars=np.asarray([155.90347*d2r,4.1516*d2r,3.8e-5,  4.2e-4])

par_names = ['ra beta', 'dec beta', 'theta0', 'gamma', 'beta scale', 'ra src 1', 'dec src 1', 'amp src 1', 'sigma src 1', 'ra src 2', 'dec src 2', 'amp src 2', 'sigma src 2']


pars=np.hstack([beta_pars,src1_pars,src2_pars])  #we need to combine parameters into a single vector
npar=np.hstack([len(beta_pars),len(src1_pars),len(src2_pars)]) #and the fitter needs to know how many per function

#NFW Params
#x0,y0,P0,c500,alpha,beta,gamma,m500
#ra, dec = 155.91355*d2r,4.1877*d2r
#ra, dec = 155.94355*d2r,4.1677*d2r
#gnfw_pars = np.array([ra, dec, 1., 1., 1.3, 4.3, 0.7,3e14])
#gnfw_labels = np.array(['ra', 'dec', 'P500', 'c500', 'alpha', 'beta', 'gamma', 'm500'])

#pars = np.hstack([gnfw_pars])
#npar = np.hstack([len(gnfw_pars)])
#this array of functions needs to return the model timestreams and derivatives w.r.t. parameters
#of the timestreams.  
funs=[minkasi.derivs_from_isobeta_c,minkasi.derivs_from_gauss_c,minkasi.derivs_from_gauss_c] 
#funs=[minkasi.derivs_from_isobeta_c,helper, helper]
#funs = [helper]
#we can keep some parameters fixed at their input values if so desired.
to_fit=np.ones(len(pars),dtype='bool')
#to_fit[[2,3,5,6,7]]=False  #C500 fixed
to_fit[3] = False

t1=time.time()
pars_fit_mink,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit)
t2=time.time()
if minkasi.myrank==0:
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit_mink)):
        print('minkasi parameter ',i,' is ',pars_fit_mink[i],' with error ',errs[i])





funs=[minkasi.derivs_from_isobeta_c,helper, helper]

t1=time.time()
pars_fit_jax,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit)
t2=time.time()
if minkasi.myrank==0:
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit_jax)):
        print('jax parameter ',i,' is ',pars_fit_jax[i],' with error ',errs[i])


#minkasi.comm.barrier()

for i in range(len(pars_fit_mink)):
    print('diff par ', par_names[i],' is ', pars_fit_jax[i]-pars_fit_mink[i])



