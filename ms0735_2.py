import time, datetime, glob, sys, copy
import presets_by_source as pbs

import sys
sys.path.insert(0,'/users/ksarmien/jax')

import jax
import jax.numpy as jnp
sys.path.insert(0,'/users/ksarmien/Documents/clusters_substructure/minkasi/')

import minkasi
import minkasi_nb

from minkasi_jax import  val_conv_int_gnfw, jit_potato_full
from luca_gnfw import jit_conv_int_gnfw_elliptical
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

##########
z=0.216
###########

def helper(params, tod):
    x = tod.info['dx']
    y = tod.info['dy']
    
    xy = [x, y]
    xy = jnp.asarray(xy)

    pred, derivs = jit_conv_int_gnfw_elliptical(params, xy, z)

    derivs = jnp.moveaxis(derivs, 2, 0)

    return derivs, pred

def potato_helper(params, tod):
    x = tod.info['dx']
    y = tod.info['dy']

    xy = [x, y]
    xy = jnp.asarray(xy)

    pred, derivs = jit_potato_full(params, xy)

    derivs = jnp.moveaxis(derivs, 2, 0)

    return derivs, pred

def load_MS0735_and_downsample():
    name = 'MS0735'
    myadj=None # If None, then it'll select the most recently made folder with string "TS_*"
    mydir='/home/scratch/cromero/mustang/MUSTANG2/Reductions/'+name+'/'

    #Some presets
    elxel     = False
    projstr='-'
    #projstr='-AGBT18A_175_07-'
    #projstr='-AGBT18A_175_06-'
    gpdir = '/home/scratch/cromero/mustang/MUSTANG2/Reductions/'+name+'/Minkasi/'
    tenc      = True     # Is the inner ring 10" in radius? (If False, then 5")
    widebins  = True    # Wide bins
    ultrawide = False     # Wide bins
    maxbins   = False    # How far out to go? If True, then max extent is 330"
    medbins   = False     # If neither maxbins nor medbins, then maximum radial extent is 180"; If yes to medbins, then 240"
    ndo       = False    # New Data Only
    odo       = False    # Old Data Only
    n_atm     = 2        # Polynomial order for fitting an atmospheric term
    chisq     = 3.0      # Roughly a factor by which the noise needs to be adjusted
    #find tod files we want to map
    outroot='/home/scratch/jscherer/mustang/MUSTANG2/Reductions/MS0735/num_derv_'

    #Load the most resent made folder starting with 'TS_'
    if myadj is None:
        dirs = glob.glob(mydir+'TS_*')
        mydates = []
        for adir in dirs:
            txtspl = adir.split("_")
            month  = time.strptime(txtspl[-2],'%b').tm_mon
            adate  = datetime.date(int(txtspl[-1]),month,int(txtspl[-3]))
            mydates.append(adate)
        mysorted = [x for _, x in sorted(zip(mydates,dirs), key=lambda pair: pair[0])]
        mytsdir  = mysorted[-1]
        dirspl   = mytsdir.split("/")
        myadj    = dirspl[-1]
    else:
        mytsdir = mydir+myadj

    tod_files=mytsdir+'/Signal_TOD'+projstr+'*.fits'

    tod_names=glob.glob(tod_files)
    
    #cut bat tods
    bad_tod,addtag = pbs.get_bad_tods(name,ndo=ndo,odo=odo)
    tod_names=minkasi.cut_blacklist(tod_names,bad_tod)
    
    #if running MPI, you would want to split up files between processes
    #one easy way is to say to this:
    tod_names=tod_names[minkasi.myrank::minkasi.nproc]
    #NB - minkasi checks to see if MPI is around, if not
    #it sets rank to 0 an nproc to 1, so this would still
    #run in a non-MPI environment
    #only look at first 25 tods here
    tod_names.sort()
    tod_names=tod_names
    todvec=minkasi.TodVec()
    ###Downsampling

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
    return todvec

def mapmaking1(todvec):
    lims=todvec.lims()
    pixsize=2.0/3600*np.pi/180
    mapD=minkasi.SkyMap(lims,pixsize)
    for tod in todvec.tods:
        ipix=mapD.get_pix(tod)
        tod.info['ipix']=ipix
        tod.set_noise(minkasi.NoiseSmoothedSVD)
    print("set noise all tods")
    return todvec, mapD

def mapmaking2(todvec,mapD): 
    hits=minkasi.make_hits(todvec,mapD)
    print("finished hits")
    mapset = minkasi.Mapset()
    mapset.add_map(mapD)
    rhs=mapset.copy()
    todvec.make_rhs(rhs)
    x0=rhs.copy()
    x0.clear()
    precon=mapset.copy()
    tmp=hits.map.copy()
    ii=tmp>0
    tmp[ii]=1.0/tmp[ii]
    #precon.maps[0].map[:]=numpy.sqrt(tmp)
    precon.maps[0].map[:]=tmp[:]
    return rhs,x0,todvec,precon

def mapmaking3(rhs,x0,todvec,precon,filename_for_map):
    #run PCG!
    mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=50)
    if minkasi.myrank==0:
        mapset_out.maps[0].write(filename_for_map+'.fits') #and write out the map as a FITS file
    else:
        print('not writing map on process ',minkasi.myrank)
        
##### Only creating cluster model (no point source)
def only_MS0735_cluster_sim(todvec):
    """
    creates cluster model from GNFW parameters obtained by [add reference]
    """
    lims=todvec.lims()
    pixsize=2.0/3600*np.pi/180
    mapD=minkasi.SkyMap(lims,pixsize)
    
    #ra = Angle('07 41 44.8 hours')
    #dec = Angle('74:14:52 degrees')
    #ra, dec = 2.014669635618214, 1.295835411299298
    gnfw_pars = np.array([1.0e+00, 1.0e+00, 0.0e+00, 2.01466968e+00,
       1.29583543e+00, 8.403e+00, 1.177e+00, 1.2223e+00,
       5.49e+00, 7.29816574e-02, 1.59608409e+15])
    gnfw_labels = np.array(['x_scale', 'y_scale', 'theta', 'ra', 'dec', 'P500', 'c500', 'alpha', 'beta', 'gamma', 'm500'])
    d2r=np.pi/180
    sig=9/2.35/3600*d2r
    theta_0=40/3600*d2r
    scale = 1
    for i,tod in enumerate(todvec.tods):
        temp_tod = tod.copy()
        pred = helper(gnfw_pars, temp_tod)[1]
        ipix=mapD.get_pix(tod)
        tod.info['ipix']=ipix
        if (i % 2) == 0:
            tod.info['dat_calib'] = -1*scale*tod.info['dat_calib']
        else:
            tod.info['dat_calib'] = scale*tod.info['dat_calib']
        
        tod.info['dat_calib'] = tod.info['dat_calib']-pred
        print('mean is ',np.mean(tod.info['dat_calib']))
        tod.set_noise(minkasi.NoiseSmoothedSVD)
        
        
def fitting_GNFW_and_gauss(todvec):
    ra = Angle('07 41 44.8 hours')
    dec = Angle('74:14:52 degrees')
    z=0.216
    ra, dec = ra.to(u.radian).value, dec.to(u.radian).value
    
    gnfw_pars = np.array([1.0e+00, 1.0e+00, 0.0e+00, 2.01466968e+00,
       1.29583543e+00, 8.403e+00, 1.177e+00, 1.2223e+00,
       5.49e+00, 7.29816574e-02, 1.59608409e+15]) 
    gnfw_labels = np.array(['x_scale', 'y_scale', 'theta', 'ra', 'dec', 'P500', 'c500', 'alpha', 'beta', 'gamma', 'm500'])
    
    ps_pars = np.array([ 2.01497784e+00,1.29594886e+00, 6.19164039e-05, 8.00856812e-05]) 
    ps_labels = np.array(['ra','dec','sigma','amp'])
    
    d2r=np.pi/180
    sig=9/2.35/3600*d2r
    theta_0=40/3600*d2r
    scale = 1
    pars=np.hstack([gnfw_pars,ps_pars])
    npar=np.hstack([len(gnfw_pars),len(ps_pars)])
    labels = np.hstack([gnfw_labels,ps_labels])
    funs=[helper,minkasi.derivs_from_gauss_c]
    
    to_fit=np.ones(len(pars),dtype='bool')
    to_fit[[0,1,2,5,6,7,8]]=False # for cluster: fitting ra, fitting dec, fitting gamma and fitting m500
                                  # for point source: fitting ra, dec, sigma, amp
    t1=time.time()
    pars_fit,chisq,curve,errs = minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit,maxiter=50)
    t2=time.time()
    return pars_fit,chisq,curve,errs

def set_noise(todvec):
    for i,tod in enumerate(todvec.tods):
        tod.set_noise(minkasi.NoiseSmoothedSVD)
