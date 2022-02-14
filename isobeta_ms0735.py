import time, datetime, glob, sys, copy
import presets_by_source as pbs
import minkasi
import jax
import jax.numpy as jnp
from minkasi_jax import  val_conv_int_gnfw, jit_potato_full, poly_sub
from luca_gnfw import jit_conv_int_isobeta_elliptical_two_bubbles, get_rmap
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial
from functools import partial
import scipy
import os
import resource
# import dill as pk
# jax.config.update("jax_traceback_filtering", "off")

def helper(params, tod, z, to_fit):
    x = tod.info['dx']
    y = tod.info['dy']

    xy = [x, y]
    xy = jnp.asarray(xy)
    
    xb1, yb1, rb1 = params[8:11]
    xb2, yb2, rb2 = params[13:16]
    
    argnums = np.array([i for i, p in enumerate(to_fit[:len(params)]) if p])
    params = np.delete(params, [[8, 9, 10, 13, 14, 15]])

    r_map = 7.0*60
    r_map = float(get_rmap(r_map, params[2], params[3], params[4], z, params[6], params[7]))
    pred, derivs = jit_conv_int_isobeta_elliptical_two_bubbles(params, xy, z, xb1, yb1, rb1, xb2, yb2, rb2, r_map = r_map, dr = 0.25, argnums = tuple(argnums))
    
    return derivs, pred


def poly(x, c0, c1, c2):
    temp = 0
    #for i in range(len(p)):
    #    temp += p[i]*x**i
    temp += c0+c1*x+c2*x**2
    return temp

name = 'MS0735'
myadj=None # If None, then it'll select the most recently made folder with string "TS_*"
mydir='/scratch/r/rbond/jorlo/'+name+'/'
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
svdfwhm   = 10
nfft      = 1


#find tod files we want to map
outroot=os.environ['SCRATCH']+'/Reductions/MS0735/isobeta/'

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

#cut bad tods
bad_tod,addtag = pbs.get_bad_tods(name,ndo=ndo,odo=odo)
tod_names=minkasi.cut_blacklist(tod_names,bad_tod)
tod_names.sort()

tod_names = tod_names[:5]
#if running MPI, you would want to split up files between processes
tod_names=tod_names[minkasi.myrank::minkasi.nproc]


minkasi.barrier()

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


d2r=np.pi/180
sig=9/2.35/3600*d2r
theta_0=40/3600*d2r

sim = False
#If true, fit a polynomial to tods and remove
sub_poly = True 
method = 'pred2'
for i, tod in enumerate(todvec.tods):

    temp_tod = tod.copy()
    # if sim:
    #      pred = helper(gnfw_pars, temp_tod, z = z, to_fit = np.ones(len(gnfw_pars),dtype ='bool'))[1] + minkasi.derivs_from_gauss_c(ps_pars, temp_tod)[1]


    #print(tod.info['dat_calib'].shape)
    ipix=map.get_pix(tod)
    tod.info['ipix']=ipix
    # if sim:
    #     #Flip alternate TODs and add simulated profile on top
    #     if (i % 2) == 0:
    #         tod.info['dat_calib'] = -1*scale*tod.info['dat_calib']
        # else:
        #     tod.info['dat_calib'] = scale*tod.info['dat_calib']
        # 
        # tod.info['dat_calib'] = tod.info['dat_calib'] + pred
        
    if sub_poly:
        tod.set_apix()
        nbin = 10
        #Fit a simple poly model to tods to remove atmosphere
        for j in range(tod.info['dat_calib'].shape[0]):
            x, y =tod.info['apix'][j], tod.info['dat_calib'][j] - tod.info[method][j]
            
            res, res_er = scipy.optimize.curve_fit(poly, x, y)
            
            #Using jax, not actually faster than scipy but maybe with parallelization or tcu's it is?
            tod.info['dat_calib'][j] -= poly(x, *res)            

    tod.set_noise(minkasi.NoiseSmoothedSVD,fwhm=svdfwhm);tag='svd' 



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

#First fit just the central ps
#PS pars, currently just central
ps_labels = np.array(['ra', 'dec', 'sigma', 'amp'])
#                       16     17     18      19
ps_pars = np.array([ra, dec, 1.37e-5,  1.7e-4])




isobeta_labels = np.array(['ra', 'dec', 'r_1', 'r_2', 'r_3', 'theta', 'beta', 'amp'])
#Label nums for ref:       0       1       2      3     4       5        6      7
PS = True

#Cluster Center
ra = Angle('07 41 44.5 hours')
dec = Angle('74:14:38.7 degrees')
ra, dec = ra.to(u.radian).value, dec.to(u.radian).value


isobeta_pars = np.array([ra, dec, .12, .16, .12, 0.0, 9.0, .3])

#southwest bubble pars
ra_sw = Angle('07 41 49 hours')
dec_sw = Angle('74:15:22 degrees')
ra_sw, dec_sw = ra_sw.to(u.radian).value, dec_sw.to(u.radian).value

print((dec-dec_sw)*r2arcsec)

sw_labels = np.array(['sw ra', 'sw dec', 'radius', 'sup'])
#Label nums:            8         9         10       11
sw_pars = np.array([-7, 44, 30, 0.5]) 

#North east bubble
ra_ne = Angle('07 41 39 hours')
dec_ne = Angle('74:13:51 degrees')       
ra_ne, dec_ne = ra_ne.to(u.radian).value, dec_ne.to(u.radian).value

ne_labels = np.array(['ne ra', 'ne dec', 'radius', 'sup'])
#Label nums:            12        13        14       15
ne_pars = np.array([7,-47, 30, 0.5])


#In case we want to later add more functions to the model
pars = np.hstack([isobeta_pars,sw_pars, ne_pars, ps_pars])
npar = np.hstack([len(isobeta_pars)+len(sw_pars)+len(ne_pars), len(ps_pars)])
labels = np.hstack([isobeta_labels, sw_labels, ne_labels, ps_labels])

to_fit=np.ones(len(pars),dtype='bool')
to_fit[[0, 1, 8, 9, 10, 12, 13, 14, 16, 17]]=False 

funs = [partial(helper, z = z, to_fit = to_fit), minkasi.derivs_from_gauss_c]

fit = True 
if fit:
    t1=time.time()
    print('starting actual fitting')
    sys.stdout.flush()
    # jax.profiler.start_trace('./tmp/tensorboard')
    pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit, maxiter = 20)
    # jax.profiler.stop_trace()
    t2=time.time()
    if minkasi.myrank==0:
        print('took ',t2-t1,' seconds to fit timestreams')
        for i in range(len(labels)):
            print('parameter ',labels[i],' is ', pars_fit[i],' with error ',errs[i])
        sys.stdout.flush()
    rs = np.linspace(0, 8, 1000)
    fake_tod = np.zeros((2, len(rs)))
    for i in range(len(rs)):
        temp = np.sqrt(rs[i])*np.pi/(60*180)
        fake_tod[0][i], fake_tod[1][i] = temp+pars_fit[0], temp+pars_fit[1]

    profile = val_conv_int_gnfw(pars_fit[:8], fake_tod,z) 

else:
    pars_fit = pars

minkasi.comm.barrier()


if fit:
    dic = {'pars':pars_fit, 'chisq':chisq, 'curve':curve, 'errs':errs}

remove_bowl = False
if remove_bowl:
    #Remove model from TODs
    for i, tod in enumerate(todvec.tods):

        temp_tod = tod.copy()
        pred = helper(pars_fit[:18], temp_tod, z = z, to_fit = to_fit)[1] + minkasi.derivs_from_gauss_c(pars_fit[18:], temp_tod)[1]
        tod.info['dat_calib'] = tod.info['dat_calib'] - np.array(pred)

        #Unclear if we need to reset the noise
        tod.set_noise(minkasi.NoiseSmoothedSVD,fwhm=svdfwhm);tag='svd'

    #Fit bowl to TODs
    t1=time.time()
    print('removing bowl')
    sys.stdout.flush()
   
    pars_fit,chisq,curve,errs=minkasi.fit_timestreams_with_derivs_manyfun(funs,pars,npar,todvec,to_fit, maxiter = 20)
  
    t2=time.time()
    if minkasi.myrank==0:
        print('took ',t2-t1,' seconds to fit timestreams')
        for i in range(len(labels)):
            print('parameter ',labels[i],' is ', pars_fit[i],' with error ',errs[i])
        sys.stdout.flush()


################################################################################################
#                                 Begin map making                                            #
###############################################################################################

#Subtract off model from ToDs

for i in range(len(labels)):
    if to_fit[i]:
        outroot += labels[i] + '_'

# if session_id:
#     outroot += session_id + '_'


#delete trailing _
outroot = outroot [:-1] 
outroot += '/'

minkasi.barrier()

if minkasi.myrank == 0:
    if not os.path.exists(outroot):
        os.mkdir(outroot)

minkasi.barrier()

resid = True 
fitting = 'charles'

for i, tod in enumerate(todvec.tods):

    temp_tod = tod.copy()
    if resid:  
        pred = helper(pars_fit[:npar[0]], temp_tod, z = z, to_fit = np.zeros(npar[0], dtype=bool)[1] + minkasi.derivs_from_gauss_c(pars_fit[npar[0]:], temp_tod)[1])

        tod.info['dat_calib'] = tod.info['dat_calib'] - np.array(pred)
        
    #Unclear if we need to reset the noise
    tod.set_noise(minkasi.NoiseSmoothedSVD,fwhm=svdfwhm);tag='svd' 

# if session_id:
#     outroot += session_id

for i in range(len(labels)):
    if to_fit[i]:
        outroot += labels[i] + '_'
if sim:
    outroot += 'sim_'
if sub_poly:
    outroot += 'poly_sub_'+str(method)+'_'
if resid:
    outroot += 'resid_'
else:
    outroot += 'data_'

# if fit:
#     outroot_pickle = outroot[:-1]+'.p'
#     pk.dump(dic, open(outroot_pickle, 'wb'))

if fitting == 'charles':
    npass=5
    dograd = False
    #get the hit count map.  We use this as a preconditioner
    #which helps small-scale convergence quite a bit.
    print('starting hits')
    hits=minkasi.make_hits(todvec,map)
    print('finished hits.')
    naive=map.copy()
    naive.clear()
    for tod in todvec.tods:
        tmp=tod.info['dat_calib'].copy()
        u,s,v=np.linalg.svd(tmp,0)
        pred=np.outer(u[:,0],s[0]*v[0,:])
        tmp=tmp-pred
    
        #cm=np.median(tmp,axis=0)
        #for i in range(tmp.shape[0]):
        #    tmp[i,:]=tmp[i,:]-cm
        naive.tod2map(tod,tmp)
    naive.mpi_reduce()
    naive.map[hits.map>0]=naive.map[hits.map>0]/hits.map[hits.map>0]
    if minkasi.myrank==0:
        naive.write(outroot+'naive.fits')
        hits.write(outroot+'hits.fits')
    hits_org=hits.copy()
    hits.invert()
    
    #assert(1==0)
    
    #setup the mapset.  In general this can have many things
    #in addition to map(s) of the sky, but for now we'll just
    #use a single skymap.
    
    #for tod in todvec.tods:
    #     tod.set_noise(minkasi.NoiseSmoothedSVD)
    weightmap=minkasi.make_hits(todvec,map,do_weights=True)
    mask=weightmap.map>0
    tmp=weightmap.map.copy()
    tmp[mask]=1./np.sqrt(tmp[mask])
    noisemap=weightmap.copy()
    noisemap.map[:]=tmp
    if minkasi.myrank==0:
        noisemap.write(outroot+'noise.fits')
        weightmap.write(outroot+'weights.fits')
    
    
    
    
    mapset=minkasi.Mapset()
    mapset.add_map(map)
    
    #make A^T N^1 d.  TODs need to understand what to do with maps
    #but maps don't necessarily need to understand what to do with TODs,
    #hence putting make_rhs in the vector of TODs.
    #Again, make_rhs is MPI-aware, so this should do the right thing
    #if you run with many processes.
    rhs=mapset.copy()
    todvec.make_rhs(rhs)
    
    #this is our starting guess.  Default to starting at 0,
    #but you could start with a better guess if you have one.
    x0=rhs.copy()
    x0.clear()
    
    #preconditioner is 1/ hit count map.  helps a lot for
    #convergence.
    precon=mapset.copy()
    #tmp=hits.map.copy()
    #ii=tmp>0
    #tmp[ii]=1.0/tmp[ii]
    #precon.maps[0].map[:]=np.sqrt(tmp)
    precon.maps[0].map[:]=hits.map[:]
    #for tod in todvec.tods:
    #    cc=precon.maps[1].data[tod.info['fname']]
    #    cc.map[:]=1.0
    
    
    
    
    #run PCG
    #iters=[5,10,15,20,25,50,75,100]
    iters=[5,25,100]
    
    mapset_out=minkasi.run_pcg_wprior(rhs,x0,todvec,None,precon,maxiter=26,outroot=outroot+"_noprior",save_iters=iters)
    if minkasi.myrank==0:
        mapset_out.maps[0].write(outroot+'_initial_'+tag+'.fits') #and write out the map as a FITS file
    else:
        print('not writing map on process ',minkasi.myrank)
    
    
    
    #noise_iter=4
    for niter in range(npass):
        maxiter=26+25*(niter+1)
        #first, re-do the noise with the current best-guess map
        for tod in todvec.tods:
            mat=0*tod.info['dat_calib']
            for mm in mapset_out.maps:
                mm.map2tod(tod,mat)
            #tod.set_noise(minkasi.NoiseSmoothedSVD,tod.info['dat_calib']-mat)
            tod.set_noise(minkasi.NoiseSmoothedSVD,tod.info['dat_calib']-mat,fwhm=svdfwhm);tag='svd'
    
    
        gradmap=hits.copy()
        gradmap=hits.copy()
    
        if dograd:
            gradmap.map[:]=minkasi.get_grad_mask_2d(mapset_out.maps[0],todvec,thresh=1.8)
            prior=minkasi.tsModel(todvec,minkasi.CutsCompact)
            for tod in todvec.tods:
                prior.data[tod.info['fname']]=tod.prior_from_skymap(gradmap)
                print('prior on tod ' + tod.info['fname']+ ' length is ' + repr(prior.data[tod.info['fname']].map.size))
    
            mapset=minkasi.Mapset()
            mapset.add_map(mapset_out.maps[0])
            pp=prior.copy()
            pp.clear()
            mapset.add_map(pp)
    
            priorset=minkasi.Mapset()
            priorset.add_map(map)
            priorset.add_map(prior)
            priorset.maps[0]=None
    
        else:
            priorset = None
    
        rhs=mapset.copy()
        todvec.make_rhs(rhs)
    
        precon=mapset.copy()
        precon.maps[0].map[:]=hits.map[:]
        #for tod in todvec.tods:
        #    cc=precon.maps[1].data[tod.info['fname']]
        #    cc.map[:]=1.0
        mapset_out=minkasi.run_pcg_wprior(rhs,mapset,todvec,priorset,precon,maxiter=maxiter,outroot=outroot+'_niter_'+repr(niter+1),save_iters=iters)
        if minkasi.myrank==0:
            mapset_out.maps[0].write(outroot+'niter_'+repr(niter+1)+'.fits')
    
    minkasi.barrier()
    









