#OLD minkasi mapmaking script which does not make use of the new features....
import numpy as np
import presets_by_source as pbs
import minkasi
import glob
import sys, datetime, time
from astropy.io import fits as pyfits
import astropy
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo #choose your cosmology here
if sys.version_info.major == 3:
    from importlib import reload

from minkasi_jax import conv_int_gnfw 
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, value_and_grad, partial
import timeit
import jax

reload(minkasi)

#compile jit gradient function
jit_gnfw_deriv = jax.jacfwd(conv_int_gnfw, argnums = 0)

def helper(params, tod):
    x = tod.info['dx']
    y = tod.info['dy'] 
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    pred = conv_int_gnfw(params, x, y, 10., 0.3)
    derivs = jit_gnfw_deriv(params, x, y, 10., 0.3)
    
    derivs = np.moveaxis(derivs, 2, 0)

    return derivs, pred


def gnfw(max_R,P0,c500,alpha,beta,gamma,r500,P500,z,dR=None):
    '''returns pressure profile with r in MPc'''
    if dR == None : dR=max_R / 2e3
    radius=(np.arange(0,max_R,dR) + dR/2.0)

    x = radius / r500
    pressure=P500*P0/((c500*x)**gamma * (1.+(c500*x)**alpha)**((beta-gamma)/alpha) )
    return pressure,radius

#@jit(nopython = nopython)
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

def derivs_from_gnfw(params,tod,z=None,profile=None,fwhm=9.,max_R=10.,freq=90e9,T_electron=5.0,pd=[1.,0.01],*args,**kwargs):
    '''for use with fit many functions etc
    integrates p(r)=P0/(c500*r)^gamma*(1+(c500*r)^alpha)^(beta-gamma)/alpha
    max_R is the radius to integrate the cluster profile out to in arcmin on the sky
    TBD - connect to faster python, add reuse of calculated profile for multiple calls
    freq is used to convert into T_RJ
    pd=steps to use for calc derivs (offsets, other parameters as fraction)
    '''
    npar=8
    x0,y0,P0,c500,alpha,beta,gamma,m500=params
    m500 *= 1e14
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
        #print('r500 in Mpc=',r500,'  P500 =',P500)

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
    ddr=np.sqrt(ddx*ddx + dy*dy)*180./np.pi*3600.
    derivs[0,:,:]=(np.interp(ddr,rmap,ip,right=0.)-pred)/pd[0]
    ddr=np.sqrt(dx*dx + ddy*ddy)*180./np.pi*3600.
    derivs[1,:,:]=(np.interp(ddr,rmap,ip,right=0.)-pred)/pd[0]

    derivs[2,:,:]=pred/P0

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




#find tod files we want to map
outroot='/home/scratch/jscherer/mustang/MUSTANG2/Reductions/MS0735/maps/'

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

tod_names=tod_names[minkasi.myrank::minkasi.nproc]
#NB - minkasi checks to see if MPI is around, if not
#it sets rank to 0 an nproc to 1, so this would still
#run in a non-MPI environment

#only look at first 25 tods here
tod_names.sort()
tod_names=tod_names[:6]

todvec=minkasi.TodVec()
fit_todvec = minkasi.TodVec()
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
    fit_todvec.add_tod(tod)
    print('took ',t2-t1,' ',t3-t2,' seconds to read and downsample file ',fname)

#make a template map with desired pixel size an limits that cover the data
#todvec.lims() is MPI-aware and will return global limits, not just
#the ones from private TODs
lims=todvec.lims()
pixsize=2.0/3600*np.pi/180
map=minkasi.SkyMap(lims,pixsize)
mm = map.copy()


#NFW Params
#x0,y0,P0,c500,alpha,beta,gamma,m500
ra = Angle('07 41 44.8 hours')
dec = Angle('74:14:52 degrees')
params = np.array([ra.to(u.radian).value, dec.to(u.radian).value, 1000., 1., 1.3, 4.3, 0.7,3e14])
gnfw_labels = ['ra', 'dec', 'P500', 'c500', 'alpha', 'beta', 'gamma', 'm500']

params2 = np.array([ra.to(u.radian).value, dec.to(u.radian).value, 1000., 1., 1.3, 4.3, 0.7,3])

#A grid of points we'll use a lot
bound = 10*jnp.pi/(180*3600)
x = jnp.linspace(-1*bound, bound, 20)
y = jnp.linspace(-1*bound, bound, 20)




scale = 1

sim = True
fit = True

if sim:
    for i, tod in enumerate(todvec.tods):
        temp_tod = tod.copy()
       
        ignore, pred = derivs_from_gnfw(params2, temp_tod, z = 0.3)
        #print('Tod: ', np.amax(tod.info['dat_calib']))
        #print('derivs: ', np.amax(pred))
        ignore, pred2 = helper(params, temp_tod)
        print(pred-pred2)
        #pred, ignore = minkasi.derivs_from_elliptical_gauss(params, temp_tod)
        ipix=map.get_pix(tod)
        tod.info['ipix']=ipix
        #tod.set_noise_smoothed_svd()
        #Flip alternate TODs and add simulated profile on top
        if (i % 2) == 0:
            tod.info['dat_calib'] = -1*scale*tod.info['dat_calib']
        else:
            tod.info['dat_calib'] = scale*tod.info['dat_calib']

        tod.info['dat_calib'] = tod.info['dat_calib'] + pred2
        
        tod.set_noise(minkasi.NoiseSmoothedSVD)
    prefix = 'sim'

else:
    for tod in enumerate(todvec.tods):
        ipix = map.get_pix(tod)
        tod.info['ipix'] = ipix
        tod.set_noise(minkasi.NoiseSmoothedSVD)
    prefix = 'data'

#get the hit count map.  We use this as a preconditioner
#which helps small-scale convergence quite a bit.
hits=minkasi.make_hits(todvec,map)
if minkasi.myrank==0:
    hits.write(outroot+prefix+'_hits.fits')
#setup the mapset.  In general this can have many things
#in addition to map(s) of the sky, but for now we'll just 
#use a single skymap.
mapset=minkasi.Mapset()
mapset.add_map(map)

#make A^T N^1 d.  TODs need to understand what to do with maps
#but maps don't necessarily need to understand what to do with TODs, 
#hence putting make_rhs in the vector of TODs. 
#Again, make_rhs is MPI-aware, so this should do the right thing
#if you run with many processes.
rhs=mapset.copy()
todvec.make_rhs(rhs)
print(rhs)
#this is our starting guess.  Default to starting at 0,
#but you could start with a better guess if you have one.
x0=rhs.copy()
x0.clear()

#preconditioner is 1/ hit count map.  helps a lot for 
#convergence.
precon=mapset.copy()
tmp=hits.map.copy()
ii=tmp>0
tmp[ii]=1.0/tmp[ii]
precon.maps[0].map[:]=tmp[:]

print('here')
#run PCG!
mapset_out=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=40)

#mapset_out.maps[0].smooth(hits.map,fwhm=4,ng=13)
#mapset_out.maps[0].median()

mapset_out.maps[0].write(outroot+prefix+'_itter40.fits')

#########################################################################
m_con=mapset_out.copy()
m_con.maps[0].map -= 1.0#0.2 #0.1 #0.1 for three pass 1.0 for 4 pass
#m_con.maps[0].trim(0.0)
for tod in todvec.tods:
    dat=tod.info['dat_calib'].copy()
    tmp=np.zeros(dat.shape)
    m_con.maps[0].map2tod(tod,tmp)
    tod.info['dat_calib'][:]=dat-tmp
    tod.set_noise_smoothed_svd()
rhs=mapset.copy()
todvec.make_rhs(rhs)
x0=rhs.copy()
x0.clear()
precon=mapset.copy()
tmp=hits.map.copy()
ii=tmp>0
tmp[ii]=1.0/tmp[ii]
precon.maps[0].map[:]=tmp[:]

 
#run PCG!
mapset_out2=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=30)
#mapset_out2.maps[0].smooth(hits.map,fwhm=4,ng=13)
#mapset_out2.maps[0].median()
if minkasi.myrank==0:
    mapset_out2.maps[0].write(outroot+prefix+'_pass2.fits') #and write out the map as a FITS file

m_con2=mapset_out2.copy()
m_con2.maps[0].map -= 0.2 # 0.05 three pass 0.2 4 pass
#m_con2.maps[0].trim(0.0)
for tod in todvec.tods:
    dat=tod.info['dat_calib'].copy()
    tmp=np.zeros(dat.shape)
    m_con2.maps[0].map2tod(tod,tmp)
    tod.info['dat_calib'][:]=dat-tmp
    tod.set_noise_smoothed_svd()#,nfft=1) #3 pass
rhs=mapset.copy()
todvec.make_rhs(rhs)
x0=rhs.copy()
x0.clear()
precon=mapset.copy()
tmp=hits.map.copy()
ii=tmp>0
tmp[ii]=1.0/tmp[ii]
precon.maps[0].map[:]=tmp[:]
#run PCG!
mapset_out3=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=30)
#mapset_out3.maps[0].smooth(hits.map,fwhm=4,ng=13)
#mapset_out3.maps[0].median()
if minkasi.myrank==0:
    mapset_out3.maps[0].write(outroot+prefix+'_pass3.fits') #and write out the map as a FITS file

########maps 3 and 4 were very similar#########################
m_con3=mapset_out3.copy() # this time round use full tods as noise
m_con3.maps[0].map -= 0.01 #0.05
#m_con3.maps[0].trim(0.00)
for tod in todvec.tods:
    dat=tod.info['dat_calib'].copy()
    tmp=np.zeros(dat.shape)
    m_con3.maps[0].map2tod(tod,tmp)
    tod.info['dat_calib'][:]=dat-tmp
    tod.set_noise_smoothed_svd()
rhs=mapset.copy()
todvec.make_rhs(rhs)
x0=rhs.copy()
x0.clear()
precon=mapset.copy()
tmp=hits.map.copy()
ii=tmp>0
tmp[ii]=1.0/tmp[ii]
precon.maps[0].map[:]=tmp[:]
#run PCG!
mapset_out4=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=30)
#mapset_out4.maps[0].median()
#mapset_out4.maps[0].smooth(hits.map,fwhm=4,ng=13)
if minkasi.myrank==0:
    mapset_out4.maps[0].write(outroot+prefix+'_pass4.fits') #and write out the map as a FITS file

#################################################################
m_con4=mapset_out4.copy() # this time round use full tods as noise
m_con4.maps[0].map -= 0.01 #0.05
#m_con4.maps[0].trim(0.0)
for tod in todvec.tods:
    dat=tod.info['dat_calib'].copy()
    tmp=np.zeros(dat.shape)
    m_con4.maps[0].map2tod(tod,tmp)
    tod.info['dat_calib'][:]=dat-tmp
    tod.set_noise_smoothed_svd()
rhs=mapset.copy()
todvec.make_rhs(rhs)
x0=rhs.copy()
x0.clear()
precon=mapset.copy()
tmp=hits.map.copy()
ii=tmp>0
tmp[ii]=1.0/tmp[ii]
precon.maps[0].map[:]=tmp[:]
#run PCG!
mapset_out5=minkasi.run_pcg(rhs,x0,todvec,precon,maxiter=30)
#mapset_out5.maps[0].median()
#mapset_out5.maps[0].smooth(hits.map,fwhm=4,ng=13)
if minkasi.myrank==0:
    mapset_out5.maps[0].write(outroot+prefix+'_pass5.fits') #and write out the map as a FITS file

outmap=m_con+m_con2+m_con3+m_con4+mapset_out5
outmap.maps[0].write(outroot+prefix+'_final.fits')
#outmap.maps[0].smooth(hits.map,fwhm=4)
#outmap.maps[0].write(outroot+'_final_smooth4.fits')

print('Done fitting')
if fit:
    chisq, grad, curve = minkasi.get_ts_curve_derivs_many_funcs(fit_todvec, pars = params, npar_fun = [len(params)], funcs = [helper])
    print(chisq)




