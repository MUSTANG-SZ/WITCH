import datetime
import glob
import sys
import time
import presets_by_source as pbs
import minkasi
import jax.numpy as jnp
from minkasi_jax import poly_sub
from luca_gnfw import jit_conv_int_gnfw_elliptical_two_bubbles
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u
from functools import partial
import scipy
import os


def helper(params, tod, z, to_fit):
    x = tod.info['dx']
    y = tod.info['dy']

    xy = [x, y]
    xy = jnp.asarray(xy)
    
    e, theta = params[:2]    

    xb1, yb1, rb1 = params[10:13]

    xb2, yb2, rb2 = params[14:17]

    
    argnums = np.array([i for i, p in enumerate(to_fit[:len(params)]) if p])

    params = np.delete(params, [[0,1,10,11,12, 14,15,16]])

    

    pred, derivs = jit_conv_int_gnfw_elliptical_two_bubbles(e, theta, params, xy, z, xb1, yb1, rb1, xb2, yb2, rb2, r_map = 7.0*60., dr = 0.25, argnums = tuple(argnums))
    
    
    return derivs, pred


def poly(x, c0, c1, c2):
    temp = 0
    #for i in range(len(p)):
    #    temp += p[i]*x**i
    temp += c0+c1*x+c2*x**2
    return temp
# name = 'MS0735'
name = 'Bridge1' 
myadj=None # If None, then it'll select the most recently made folder with string "TS_*"
# mydir='/scratch/r/rbond/jorlo/'+name+'/'
mydir = os.environ['SCRATCH']+'/tods/'+name+'/'
#Some presets
projstr='-'
#projstr='-AGBT18A_175_07-'
#projstr='-AGBT18A_175_06-'
ndo       = False    # New Data Only
odo       = False    # Old Data Only
n_atm     = 2        # Polynomial order for fitting an atmospheric term
chisq     = 3.0      # Roughly a factor by which the noise needs to be adjusted
z         = 0.216    #Redshift of MS0735
svdfwhm   = 10


#find tod files we want to map
outroot=os.environ['SCRATCH']+'/Reductions/MS0735/'

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
tod_names.sort()
# print(tod_names[147::160])
# sys.stdout.flush()
# tod_names=tod_names[:120]

#if running MPI, you would want to split up files between processes
#one easy way is to say to this:
tod_names=tod_names[minkasi.myrank::minkasi.nproc]
#NB - minkasi checks to see if MPI is around, if not
#it sets rank to 0 an nproc to 1, so this would still
#run in a non-MPI environment
# print('nproc: ', minkasi.nproc)
# sys.stdout.flush()
#only look at first 25 tods here


minkasi.barrier()
session_id = False
if session_id:
    tod_names = [name for name in tod_names if name[79:81] == session_id]


#at a map/some initial fits.  The better the guess, the
#faster the convergence.
d2r=np.pi/180
sig=9/2.35/3600*d2r
theta_0=40/3600*d2r
r2arcsec = 3600*180/np.pi


#NFW Params
#x0,y0,P0,c500,alpha,beta,gamma,m500
ra = Angle('07 41 44.5 hours')
dec = Angle('74:14:38.7 degrees')
ra, dec = ra.to(u.radian).value, dec.to(u.radian).value

gnfw_labels = np.array(['e', 'theta','ra', 'dec', 'P500', 'c500', 'alpha', 'beta', 'gamma', 'm500'])
#Label nums for ref:     0     1       2      3       4      5        6       7       8       9                 
model_type = 'best'
PS = True

#Cluster Center
# ra = Angle('07 41 44.5 hours')
# dec = Angle('74:14:38.7 degrees')
# ra, dec = ra.to(u.radian).value, dec.to(u.radian).value

ra, dec = np.mean((0.7840614914894104, 0.7755526304244995)), np.mean((0.2399413138628006, 0.2287592887878418))

if model_type == 'cc':

    #Cool Core
    gnfw_pars = np.array([1.38, 7*d2r, ra, dec, 8.403, 1.177, 1.2223, 5.49, 0.7736,3.2e14])
    #gnfw_pars = np.array([dec, ra, 8.403, 1.177, 1.2223, 5.49, 0.7736,3.2e14])

if model_type == 'A10':

    #A10
    gnfw_pars = np.array([0.6835, 7*d2r, ra, dec, 8.403, 1.177, 1.05, 5.49, 0.31,3.2e14])

if model_type == 'simon':

    #Simon sims
    gnfw_pars = np.array([0.6835, 83*d2r,ra, dec, 8.403, 1.177, 1.4063, 5.49, 0.3798,3.2e14])

if model_type == 'best':
    #Best fit
    gnfw_pars = np.array([6.00664516e-01,  1.81718202e+00, ra, dec, 
        8.403,  1.177,  1.4063,  5.49,
        0.3798,  2.27264989e+14])

if model_type == 'beta_ish':
    gnfw_pars = np.array([.60066, 83*d2r, ra, dec, 8.403, 1.177, 1.4063, 5.49, 0.1, 4.2e14])


#southwest bubble pars
ra_sw = Angle('07 41 49 hours')
dec_sw = Angle('74:15:22 degrees')
ra_sw, dec_sw = ra_sw.to(u.radian).value, dec_sw.to(u.radian).value

sw_labels = np.array(['sw ra', 'sw dec', 'radius', 'sup'])
#Label nums:            10         11         12     13
sw_pars = np.array([7, -44, 30, 0.5]) 

#North east bubble
ra_ne = Angle('07 41 39 hours')
dec_ne = Angle('74:13:51 degrees')       
ra_ne, dec_ne = ra_ne.to(u.radian).value, dec_ne.to(u.radian).value

ne_labels = np.array(['ne ra', 'ne dec', 'radius', 'sup'])
#Label nums:            14        15        16       17
ne_pars = np.array([7, 44, 30, 0.5])


#In case we want to later add more functions to the model
pars = np.hstack([gnfw_pars,sw_pars, ne_pars])
npar = np.hstack([len(gnfw_pars)+len(sw_pars)+len(ne_pars)])
labels = np.hstack([gnfw_labels, sw_labels, ne_labels])
#this array of functions needs to return the model timestreams and derivatives w.r.t. parameters
#of the timestreams.

fits = np.zeros((8, len(pars)), dtype='bool') 
fits[0][[7, 8, 9, 13, 17]] = True
fits[1][[8, 9, 13, 17]] = True
fits[2][[7, 9, 13, 17]] = True
fits[3][[7, 8, 13, 17]] = True
fits[4][[7, 13, 17]] = True
fits[5][[8, 13, 17]] = True
fits[6][[9, 13, 17]] = True
fits[7][[13, 17]] = True

noise = np.zeros(pars.shape)
add_noise = False 
if add_noise:
    to_noise = ~fits[0]
    noise = np.random.random(pars.shape) - .5
    noise *= pars/10
e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500, xb1, yb1, rb1, sup1, xb2, yb2, rb2, sup2 = pars[:int(npar[0])]


todvec=minkasi.TodVec()

#loop over each file, and read it.
for fname in tod_names:
    t1=time.time()
    dat=minkasi.read_tod_from_fits(fname)
    t2=time.time()
    minkasi.truncate_tod(dat) #truncate_tod chops samples from the end to make
                              #the length happy for ffts

    #figure out a guess at common mode #and (assumed) linear detector drifts/offset
    #drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd, pred2, cm = minkasi.fit_cm_plus_poly(dat['dat_calib'],cm_ord = 3, full_out = True)

    dat['dat_calib']=dd
    dat['pred2'] = pred2
    dat['cm'] = cm
    t3=time.time()
    tod=minkasi.Tod(dat)
    todvec.add_tod(tod)
    # print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

#make a template map with desired pixel size an limits that cover the data
#todvec.lims() is MPI-aware and will return global limits, not just
#the ones from private TODs
lims=todvec.lims()
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)


sim = True 
#If true, fit a polynomial to tods and remove
sub_poly = False 
method = 'pred2'
scale = 1
for i, tod in enumerate(todvec.tods):

    temp_tod = tod.copy()
    if sub_poly:
        tod.set_apix()
        nbin = 10
        #Fit a simple poly model to tods to remove atmosphere
        for j in range(tod.info['dat_calib'].shape[0]):
            x, y =tod.info['apix'][j], tod.info['dat_calib'][j] - tod.info[method][j]
            
            res, res_er = scipy.optimize.curve_fit(poly, x, y)
            
            #Using jax, not actually faster than scipy but maybe with parallelization or tcu's it is?
            #res = poly_sub(x, y) 
                      
            tod.info['dat_calib'][j] -= poly(x, *res)            

    #print(tod.info['dat_calib'].shape)
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    if sim:
        pred = helper(pars, temp_tod, z = z, to_fit = np.zeros(len(pars),dtype ='bool'))[1]
        #Flip alternate TODs and add simulated profile on top
        # if ((i*minkasi.nproc+minkasi.myrank) % 2) == 0:
        if (i % 2) == 0:
            tod.info['dat_calib'] = -1*scale*tod.info['dat_calib']
        else:
            tod.info['dat_calib'] = scale*tod.info['dat_calib']

        tod.info['dat_calib'] = tod.info['dat_calib'] + pred

    # print('mean is ',np.mean(tod.info['dat_calib']))
    tod.set_noise(minkasi.NoiseSmoothedSVD,fwhm=svdfwhm);tag='svd' 

for f in fits:
    funs = [partial(helper, z = z, to_fit = f)]
    t1=time.time()
    _pars = pars.copy()
    _pars[f] *= 1.1
    pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,_pars,npar,todvec,f, maxiter = 20)
    t2=time.time()
    if minkasi.myrank==0:
        print('took ',t2-t1,' seconds to fit timestreams')
        print('Fitting with parameters', labels[f], 'freed')
        for i in range(len(labels)):
            print('parameter ',labels[i],' is ', pars_fit[i],' with error ',errs[i])
        sys.stdout.flush()

minkasi.comm.barrier()
