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

from schwimmbad import MPIPool
from astropy.coordinates import Angle
from astropy import units as u

import minkasi_jax.presets_by_source as pbs
from minkasi_jax.utils import *
from minkasi_jax.core import model as mink_model
from minkasi_jax.forward_modeling import make_tod_stuff, sample
from minkasi_jax.forward_modeling import sampler as my_sampler
from minkasi_jax.fitter import get_outdir, make_parser, load_tods, process_tods, load_config
from minkasi_jax.containers import Model

import emcee
import corner

import functools
import dill

from matplotlib import pyplot as plt

_priors = [scipy.stats.uniform(loc=-30,scale=60),
          scipy.stats.uniform(loc=-30,scale=60),
          scipy.stats.uniform(loc=  1,scale=49),
          scipy.stats.uniform(loc=-10,scale=20)]

def log_prior(theta):
    logprior = 0.00
    for pi in range(len(theta)):
        logprior += _priors[pi].logpdf(theta[pi])
    return logprior

def log_probability(theta, tods, jsample, fixed_params, fixed_pars_ids):
    lp = log_prior(theta)    
    if not np.isfinite(lp):
        return -np.inf
    return lp + my_sampler(theta, tods, jsample, fixed_params, fixed_pars_ids)

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

# make a template map with desired pixel size an limits that cover the data
# todvec.lims() is MPI-aware and will return global limits, not just
# the ones from private TODs
lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = minkasi.maps.SkyMap(lims, pixsize)

# Define the model and get stuff setup for minkasi
model = Model.from_cfg(cfg)
funs = [model.minkasi_helper]
params = np.array(model.pars)
npars = np.array([len(params)])
prior_vals = model.priors

lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = SkyMap(lims, pixsize)

gridmx, gridmy = np.meshgrid(np.arange(skymap.nx),np.arange(skymap.ny))
gridwx, gridwy = skymap.wcs.all_pix2world(gridmx,gridmy,0)
gridwx = np.fliplr(gridwx)
gridwz = np.linspace(-1 * cfg["coords"]["r_map"], cfg["coords"]["r_map"], 2 * int(cfg["coords"]["r_map"] / cfg["coords"]["dr"]), dtype=float)

xyz = [(gridwy[:,0][...,None,None]-np.rad2deg(model.y0))*60*60,
       (gridwx[0,:][None,...,None]-np.rad2deg(model.x0))*60*60*np.cos(model.y0),
        gridwz[None,None,...]]

model.xyz = xyz

# Deal with bowling and simming in TODs and setup noise
noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
bowl_str = process_tods(cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
)

tods = make_tod_stuff(todvec, skymap, x0=model.x0, y0=model.y0)

## for ti, tod in enumerate(tods):
##     x, y, rhs, v, weight, norm, data = tod  # unravel tod
##     x    = np.median(x,axis=0)
##     y    = np.median(y,axis=0)
##     data = np.max(data,axis=0)
##     plt.subplot(121); plt.plot(x[np.argsort(x)],data[np.argsort(x)])
##     plt.subplot(122); plt.plot(y[np.argsort(y)],data[np.argsort(y)])
## plt.show(); plt.close()
## sys.exit(1)

fixed_pars_ids = []
fixed_params = params[fixed_pars_ids]
params = params[[0,1,2,3]]
params2 = params*(1.00 + 1e-1*np.random.randn(5*len(params), len(params)))
params2[:,2] = np.abs(params2[:,2])

#jit partial-d sample function
cur_sample = functools.partial(sample, model)
jsample = jax.jit(cur_sample)

nwalkers, ndim = params2.shape

jsample(params, tods)
#my_sampler = construct_sampler(model_params, xyz, beam)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (tods, jsample, fixed_params, fixed_pars_ids),pool=None)

sampler.run_mcmc(params2, 10000, skip_initial_state_check = True, progress=True)

flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

for p in range(flat_samples.shape[1]):
    print(corner.quantile(flat_samples[:,p],[0.16, 0.50, 0.84]))

truths = params

edges = [corner.quantile(flat_samples[:,i],[0.16,0.50,0.84]) for i in range(flat_samples.shape[1])]
edges = [[edges[i][1]-5.00*(edges[i][1]-edges[i][0]),
          edges[i][1]+5.00*(edges[i][2]-edges[i][1])] for i in range(flat_samples.shape[1])]

fig = corner.corner(flat_samples, labels=model.par_names, truths=truths, range=edges, quantiles=[0.16, 0.50, 0.84], show_titles=True, title_fmt='.2e')

plt.savefig('1gauss_chi_25_off_corner.pdf')

with open('1gauss_chi_25_off_samples.pkl', 'wb') as f:
    dill.dump(flat_samples, f,dill.HIGHEST_PROTOCOL)