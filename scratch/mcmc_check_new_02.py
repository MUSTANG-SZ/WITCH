import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

import emcee
import corner

import minkasi

from minkasi_jax.fitter import make_parser, load_tods, process_tods, load_config
from minkasi_jax.forward_modeling import *
from minkasi_jax.containers import Model
from minkasi_jax import core as mjcore

np.random.seed(6958476)

# ---------------------------------------------------------------------------

parser = make_parser()
args = parser.parse_args()

# TODO: Serialize cfg to a data class (pydantic?)
cfg = load_config({},args.config)
cfg["fit"] = cfg.get("fit","model" in cfg)
cfg["sim"] = cfg.get("sim",False)
cfg["map"] = cfg.get("map",True)
cfg["sub"] = cfg.get("sub",True)
if args.nosub:
    cfg["sub"] = False
if args.nofit:
    cfg["fit"] = False

# Get TODs
todvec = load_tods(cfg)

# make a template map with desired pixel size an limits that cover the data
# todvec.lims() is MPI-aware and will return global limits, not just
# the ones from private TODs
lims = todvec.lims()
pixsize = np.deg2rad(1.00
                     /3600)
skymap = minkasi.maps.SkyMap(lims, pixsize)

# Define the model and get stuff setup for minkasi
model = Model.from_cfg(cfg)
params = np.array(model.pars)

gridmx, gridmy = np.meshgrid(np.arange(skymap.nx),np.arange(skymap.ny))
gridwx, gridwy = skymap.wcs.all_pix2world(gridmx,gridmy,0)
gridwx = np.fliplr(gridwx)
gridwz = np.linspace(-1 * cfg["coords"]["r_map"], cfg["coords"]["r_map"], 2 * int(cfg["coords"]["r_map"] / cfg["coords"]["dr"]), dtype=float)

# xyz = [(gridwy[:,0][...,None,None]-np.rad2deg(model.y0))*60*60,
#        (gridwx[0,:][None,...,None]-np.rad2deg(model.x0))*60*60*np.cos(model.y0),
#         gridwz[None,None,...]]

xyz = [(gridwx[0,:][...,None,None]-np.rad2deg(model.x0))*60*60*np.cos(model.y0),
       (gridwy[:,0][None,...,None]-np.rad2deg(model.y0))*60*60,
        gridwz[None,None,...]]

model.xyz = xyz

# Deal with bowling and simming in TODs and setup noise
noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
bowl_str = process_tods(cfg,todvec,skymap,noise_class,noise_args,noise_kwargs,model)

# ---------------------------------------------------------------------------

priors = [[-200,200], # dx
          [-200,200], # dy
          [   1, 50], # sigma
          [   0, 10]] # amplitude

priors = [scipy.stats.uniform(loc=p[0],scale=p[1]-p[0]) for p in priors]

if False: # full likelihood
    @jax.jit
    def getlike(theta,x,y,v,weight,d_tod):
        m = mjcore.model(model.xyz,*model.n_struct,model.dz,model.beam,*theta)
        r_tod = d_tod-bilinear_interp(x,y,model.xyz[0].ravel(),model.xyz[1].ravel(),m)
        r_rot = jnp.dot(v,r_tod)
    
        r_tmp = jnp.hstack([r_rot,jnp.fliplr(r_rot[:,1:-1])])
        r_rft = jnp.real(jnp.fft.rfft(r_tmp,axis=1))
        return -0.50*jnp.sum(weight*r_rft**2)
elif False: # (m^t A^t N^-1 d) likelihood
    @jax.jit
    def getlike(theta,x,y,d_tod):
        m = mjcore.model(model.xyz,*model.n_struct,model.dz,model.beam,*theta)
        m_tod = bilinear_interp(x,y,model.xyz[0].ravel(),model.xyz[1].ravel(),m)

        m_rot = jnp.dot(v,m_tod)

        m_tmp = jnp.hstack([m_rot,jnp.fliplr(m_rot[:,1:-1])])
        m_rft = jnp.real(jnp.fft.rfft(m_tmp,axis=1))

        m_ift = jnp.fft.irfft(weight*m_rft,axis=1,norm='forward')[:,:x.shape[1]]
        m_irt = jnp.dot(v.T,m_ift)
        m_irt = m_irt.at[:, 0].multiply(0.50)
        m_irt = m_irt.at[:,-1].multiply(0.50)
        return jnp.sum(m_irt*d_tod)-0.50*jnp.sum(m_irt*m_tod)
elif False: # rhs likelihood
    @jax.jit
    def getlike(theta,x,y,rhs):
        m = mjcore.model(model.xyz,*model.n_struct,model.dz,model.beam,*theta)
        m_tod = bilinear_interp(x,y,model.xyz[0].ravel(),model.xyz[1].ravel(),m)

        m_rot = jnp.dot(v,m_tod)

        m_tmp = jnp.hstack([m_rot,jnp.fliplr(m_rot[:,1:-1])])
        m_rft = jnp.real(jnp.fft.rfft(m_tmp,axis=1))

        m_ift = jnp.fft.irfft(weight*m_rft,axis=1,norm='forward')[:,:x.shape[1]]
        m_irt = jnp.dot(v.T,m_ift)
        m_irt = m_irt.at[:, 0].multiply(0.50)
        m_irt = m_irt.at[:,-1].multiply(0.50)
        return jnp.sum(rhs*m)-0.50*jnp.sum(m_irt*m_tod)
else:
    @jax.jit
    def getlike(theta,tods):
        return sample(model,theta,tods)


def getprob(theta):
    prior = np.array([p.logpdf(theta[pi]) for pi, p in enumerate(priors)])
    return np.sum(prior)

def getpost(theta,*args):
    lp = getprob(theta)
    if np.isfinite(lp):
        return lp+getlike(theta,*args)
    else: return -np.inf

nwalk, ndims = 3*len(params), len(params)

steps = 2
for step in range(steps):
    tods = make_tod_stuff(todvec,skymap,x0=model.x0,y0=model.y0)

    pinit = params[None,:]*(1.00+0.01*np.random.rand(nwalk,ndims))

    sampler = emcee.EnsembleSampler(nwalk,ndims,getpost,args=(tods,))
    sampler.run_mcmc(pinit,500,skip_initial_state_check=True,progress=True)

    samples = sampler.get_chain(discard=100,thin=10,flat=True)

    for p in range(samples.shape[1]):
        print(corner.quantile(samples[:,p],[0.16,0.50,0.84]))

    if step==steps-1 or steps==1:
        edges = [corner.quantile(samples[:,i],[0.16,0.50,0.84]) for i in range(samples.shape[1])]
        edges = [[edges[i][1]-5.00*(edges[i][1]-edges[i][0]),
                  edges[i][1]+5.00*(edges[i][2]-edges[i][1])] for i in range(samples.shape[1])]

        for i in range(samples.shape[1]):
            if edges[i][0]<priors[i].support()[0]: edges[i][0] = priors[i].support()[0]
            if edges[i][1]>priors[i].support()[1]: edges[i][1] = priors[i].support()[1]

            if edges[i][0]==edges[i][1]: 
                edges[i][0] = edges[i][0]-0.10*edges[i][0]
                edges[i][1] = edges[i][1]+0.10*edges[i][1]
            
        fig = corner.corner(samples,labels=model.par_names,truths=params,quantiles=[0.16,0.50,0.84],show_titles=True,title_fmt='.2e')
        plt.show(); plt.close()
    else:
        pars_new = np.array([corner.quantile(samples[:,p],0.50)[0] for p in range(samples.shape[1])])
        pred_new = mjcore.model(model.xyz,*model.n_struct,model.dz,model.beam,*pars_new)
        
        for ti, tod in enumerate(tods):
            
            x, y, rhs, v, weight, norm, dd = tod
            pred_tod = bilinear_interp(x,y,model.xyz[0].ravel(),model.xyz[1].ravel(),pred_new)

            todvec.tods[ti].set_noise(noise_class,todvec.tods[ti].info["dat_calib"]-pred_tod,*noise_args,**noise_kwargs)
            del pred_tod
