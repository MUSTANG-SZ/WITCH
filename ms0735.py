import time, datetime, glob, sys, copy
import presets_by_source as pbs
import minkasi
import jax
import jax.numpy as jnp
from luca_gnfw import jit_conv_int_gnfw
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u

def helper(params, tod):
    x = tod.info['dx']
    y = tod.info['dy']
    
    xy = [x, y]
    xy = jnp.asarray(xy)

    pred, derivs = jit_conv_int_gnfw(params, xy, z)

    derivs = jnp.moveaxis(derivs, 2, 0)

    return derivs, pred

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
z         = 0.216    #Redshift of MS0735



#find tod files we want to map
outroot='/home/scratch/jscherer/mustang/MUSTANG2/Reductions/MS0735/num_derv'

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
tod_names=tod_names[:6]

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
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)

#NFW Params
#x0,y0,P0,c500,alpha,beta,gamma,m500
ra = Angle('07 41 44.8 hours')
dec = Angle('74:14:52 degrees')
ra, dec = ra.to(u.radian).value, dec.to(u.radian).value
gnfw_pars = np.array([ra, dec, 1., 1., 1.3, 4.3, 0.7,3e14])
gnfw_labels = np.array(['ra', 'dec', 'P500', 'c500', 'alpha', 'beta', 'gamma', 'm500'])

d2r=np.pi/180
sig=9/2.35/3600*d2r
theta_0=40/3600*d2r


iso_pars = np.array([ra, dec, theta_0,0.7,-7.2e-4])

scale = 1
#If sim = Ture, then the tods will be 'replaced' with the pure model output from helper
sim = False
for i, tod in enumerate(todvec.tods):

    temp_tod = tod.copy()
    if sim:
        ignore, pred = helper(gnfw_pars, temp_tod)



    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    if sim:
        #Flip alternate TODs and add simulated profile on top
        if (i % 2) == 0:
            tod.info['dat_calib'] = -1*scale*tod.info['dat_calib']
        else:
            tod.info['dat_calib'] = scale*tod.info['dat_calib']

        tod.info['dat_calib'] = tod.info['dat_calib'] + pred


    tod.set_noise(minkasi.NoiseSmoothedSVD)



#at a map/some initial fits.  The better the guess, the
#faster the convergence.
d2r=np.pi/180
sig=9/2.35/3600*d2r
theta_0=40/3600*d2r


#NFW Params
#x0,y0,P0,c500,alpha,beta,gamma,m500
ra = Angle('07 41 44.8 hours')
dec = Angle('74:14:52 degrees')
ra, dec = ra.to(u.radian).value, dec.to(u.radian).value

gnfw_labels = np.array(['ra', 'dec', 'P500', 'c500', 'alpha', 'beta', 'gamma', 'm500'])
#Label nums for ref:     0     1        2      3        4       5       6        7
model_type = 'A10'

if model_type == 'cc':

    #Cool Core
    gnfw_pars = np.array([ra, dec, 8.403, 1.177, 1.2223, 5.49, 0.7736,3.2e14])

if model_type == 'A10':

    #A10
    gnfw_pars = np.array([ra, dec, 8.403, 1.177, 1.05, 5.49, 0.31,3.2e14])

if model_type == 'simon':

    #Simon sims
    gnfw_pars = np.array([ra, dec, 8.403, 1.177, 1.4063, 5.49, 0.3798,3.2e14])

#In case we want to later add more functions to the model
pars = np.hstack([gnfw_pars])
npar = np.hstack([len(gnfw_pars)])

#this array of functions needs to return the model timestreams and derivatives w.r.t. parameters
#of the timestreams.
funs = [helper]


#we can keep some parameters fixed at their input values if so desired.
to_fit=np.ones(len(pars),dtype='bool')
to_fit[[2,3]]=False  #C500, beta fixed


t1=time.time()
pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit, maxiter = 30)
t2=time.time()
if minkasi.myrank==0:
    print('took ',t2-t1,' seconds to fit timestreams')
    for i in range(len(pars_fit)):
        print('parameter ',gnfw_labels[i],' is ',pars_fit[i],' with error ',errs[i])














