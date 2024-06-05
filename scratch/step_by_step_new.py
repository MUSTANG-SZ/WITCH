import os
import sys
import time
import glob
import shutil
import argparse as argp
from functools import partial
import yaml
import numpy as np
import scipy.stats

import minkasi.tods.io as io
import minkasi.parallel as parallel
import minkasi.tods.core as mtods
import minkasi.tods.processing as tod_processing
from minkasi.maps.skymap import SkyMap
from minkasi.fitting import models
from minkasi.mapmaking import noise
import minkasi

from astropy.coordinates import Angle
from astropy import units as u

import minkasi_jax.presets_by_source as pbs
from minkasi_jax.utils import *
from minkasi_jax.core import model as mink_model
from minkasi_jax.forward_modeling import make_tod_stuff, sample, get_chis
from minkasi_jax.forward_modeling import sampler as my_sampler
from minkasi_jax.fitter import get_outdir, make_parser, load_tods, process_tods, load_config
from minkasi_jax.containers import Model

import emcee
import corner
import functools

import pickle as pk

from matplotlib import pyplot as plt

parser = make_parser()
args = parser.parse_args()

# TODO: Serialize cfg to a data class (pydantic?)
cfg = load_config({}, args.config)
cfg["fit"] = cfg.get("fit", "model" in cfg)
cfg["sim"] = cfg.get("sim", False)
cfg["map"] = cfg.get("map", True)
cfg["sub"] = cfg.get("sub", True)
if args.nosub:
    cfg["sub"] = False
if args.nofit:
    cfg["fit"] = False

# Get TODs
todvec = load_tods(cfg)

lims = todvec.lims()
pixsize = 2.00 / 3600 * np.pi / 180
skymap = SkyMap(lims, pixsize) 

model  = Model.from_cfg(cfg)
params = np.array(model.pars)

gridmx, gridmy = np.meshgrid(np.arange(skymap.nx),np.arange(skymap.ny))
gridwx, gridwy = skymap.wcs.all_pix2world(gridmx,gridmy,0)
gridwx = np.fliplr(gridwx)
gridwz = np.linspace(-1 * cfg["coords"]["r_map"], cfg["coords"]["r_map"], 2 * int(cfg["coords"]["r_map"] / cfg["coords"]["dr"]), dtype=float)

xyz = [(gridwy[:,0][...,None,None]-np.rad2deg(model.y0))*60*60,
       (gridwx[0,:][None,...,None]-np.rad2deg(model.x0))*60*60*np.cos(model.y0),
        gridwz[None,None,...]]   

model.xyz = xyz

sigma = 1.00E-04

vis_model = mink_model(model.xyz,*model.n_struct,model.dz,model.beam,*params)
vis_noise = scipy.stats.norm.rvs(size=vis_model.shape,loc=0.00,scale=sigma) 
dat_model  = vis_model+vis_noise

# plt.imshow(dat_model)
# plt.show(); plt.close()

priors = [scipy.stats.uniform(loc=-30,scale=60),
          scipy.stats.uniform(loc=-30,scale=60),
          scipy.stats.uniform(loc=  1,scale=19),
          scipy.stats.uniform(loc=-10,scale=20)]

if True:
    print('* Test 1: map-space model with TOD shape')
    dx = (gridwx.ravel()[None,...]-np.rad2deg(model.x0))*60*60*np.cos(model.y0)
    dy = (gridwy.ravel()[None,...]-np.rad2deg(model.y0))*60*60
    v = np.eye(dx.shape[0])

    w = np.full(dx.shape,1.00/sigma**2) #*2.00/dx.shape[1]
    rhs = dat_model*w[0].reshape(dat_model.shape)

    @jax.jit
    def log_like(theta):
        m = mink_model(model.xyz,*model.n_struct,model.dz,model.beam,*theta)   
    # 1 return -0.50*get_chis(m,dx,dy,model.xyz,rhs,v,w)
        model_tod = bilinear_interp(dx, dy, xyz[0].ravel(), xyz[1].ravel(), dat_model-m)
        model_rot = jnp.dot(v, model_tod)
        model_tmp = jnp.hstack([model_rot, jnp.fliplr(model_rot[:, 1:-1])])
        model_rft = jnp.real(jnp.fft.rfft(model_tmp, axis=1))
        return -0.50*jnp.sum(w*model_rft**2)
    
    def log_prior(theta):
        logprior = 0.00
        for pi in range(len(theta)):
            logprior += priors[pi].logpdf(theta[pi])
        return logprior

    def log_probability(theta):
        lp = log_prior(theta)    
        if not np.isfinite(lp):
            return -np.inf
        else:
            ll = log_like(theta)
            return lp + ll

    truths = params

    fixed_pars_ids = []
    fixed_params = params[fixed_pars_ids]
    params = params[[0,1,2,3]]
    params2 = params*(1.00 + 1e-1*np.random.randn(5*len(params), len(params)))
    params2[:,2] = np.abs(params2[:,2]) #Force sigma positive

    nwalkers, ndim = params2.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

    sampler.run_mcmc(params2, 500, skip_initial_state_check = True, progress=True)

    flat_samples = sampler.get_chain(discard=100,thin=15,flat=True)

    for p in range(flat_samples.shape[1]):
        print(corner.quantile(flat_samples[:,p],[0.16,0.50,0.84]))

    fig = corner.corner(flat_samples,labels=model.par_names,truths=truths,range=[prior.support() for prior in priors])
    plt.savefig('1gauss_corner.pdf')
    plt.close()