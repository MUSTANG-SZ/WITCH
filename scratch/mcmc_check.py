import argparse as argp
import functools
import glob
import os
import pickle as pk
import shutil
import sys
import time
from functools import partial

import emcee
import minkasi.parallel as parallel
import minkasi.tods.core as mtods
import minkasi.tods.io as io
import minkasi.tods.processing as tod_processing
import minkasi_jax.presets_by_source as pbs
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import Angle
from matplotlib import pyplot as plt
from minkasi.fitting import models
from minkasi.mapmaking import noise
import minkasi

from minkasi.maps.skymap import SkyMap
from minkasi_jax.core import helper
from minkasi_jax.core import model as mink_model
from minkasi_jax.forward_modeling import make_tod_stuff, sample
from minkasi_jax.forward_modeling import sampler as my_sampler
from minkasi_jax.utils import *

import minkasi

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
import functools


# %load_ext autoreload
# %autoreload 2

"""
def log_prior(theta):
    dx, dy, dz, r1, r2, r3, theta_1, beta_1, amp_1 = theta
    if np.abs(dx) < 20 and np.abs(dy) < 20 and np.abs(dz) <20 and 0 < r1 < 1 and 0 < r2 < 1 and 0 < r3 < 1 and 0 < theta_1 < 2*np.pi and 0 < beta_1 < 2 and 0 < amp_1 < 1e6:
        return 0.0
    return -np.inf

def log_probability(theta, tods):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + my_sampler(theta, tods)
"""



<<<<<<< HEAD
def log_prior(theta, da):
    dx, dy, sigma, amp_1 = theta
    if (
        np.abs(dx) < 20
        and np.abs(dy) < 20
        and 1e-2 * da < sigma < 30 * da
        and -10 < amp_1 < 10
    ):
        return 0.0
    return -np.inf


def log_probability(theta, tods, jsample, fixed_params, fixed_pars_ids, da):
    lp = log_prior(theta, da)


def log_prior(theta):
    dx, dy, sigma, amp_1 = theta
    if np.abs(dx) < 20 and np.abs(dy) < 20 and 1 < sigma < 8 and -10 < amp_1 < 10: 
        return 0.0
    return -np.inf

def log_probability(theta, tods, jsample, fixed_params, fixed_pars_ids):
    lp = log_prior(theta)    

=======
'''
def log_prior(theta):
    dx, dy, sigma, amp_1 = theta
    if np.abs(dx) < 20 and np.abs(dy) < 20 and 1 < sigma < 8 and -10 < amp_1 < 10: 
        return 0.0
    return -np.inf

def log_probability(theta, tods, jsample, fixed_params, fixed_pars_ids):
    lp = log_prior(theta)    
>>>>>>> 43883e3bbf287cc28dff7707784b612c8dfb3b86
    if not np.isfinite(lp):
        return -np.inf
    return lp + my_sampler(theta, tods, jsample, fixed_params, fixed_pars_ids)

<<<<<<< HEAD


# with open('/home/r/rbond/jorlo/dev/minkasi_jax/configs/sampler_sims/1gauss.yaml', "r") as file:
#    cfg = yaml.safe_load(file)
# with open('/home/jack/dev/minkasi_jax/configs/ms0735/ms0735.yaml', "r") as file:
#    cfg = yaml.safe_load(file)


=======
>>>>>>> 43883e3bbf287cc28dff7707784b612c8dfb3b86
path = "/home/jorlo/dev/minkasi_jax/configs/sampler_sims/"
with open(path + '/1gauss.yaml', "r") as file:
    cfg = yaml.safe_load(file)
# fit = True

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
<<<<<<< HEAD
=======

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
priors = [None if prior is None else "flat" for prior in prior_vals]

# Deal with bowling and simming in TODs and setup noise
noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
bowl_str = process_tods(
    args, cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
)
>>>>>>> 43883e3bbf287cc28dff7707784b612c8dfb3b86

lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = SkyMap(lims, pixsize, square=True, multiple = 2)
xyz = make_grid_from_skymap(skymap, cfg["coords"]["r_map"], cfg["coords"]["dr"])
model.xyz = xyz
print("skymap, xyz: ", skymap.map.shape, model.xyz[0].shape)
<<<<<<< HEAD

=======
>>>>>>> 43883e3bbf287cc28dff7707784b612c8dfb3b86

tods = make_tod_stuff(todvec, skymap)

# test_params = params[:13] #for speed only considering single isobeta model

truths = params

model_params = [0, 0, 1, 0, 0, 0, 0, 0]

fixed_pars_ids = []
fixed_params = params[fixed_pars_ids]
params = params[[0, 1, 2, 3]]
params2 = params + 1e-4 * np.random.randn(2 * len(params), len(params))
params2[:, 2] = np.abs(params2[:, 2])  # Force sigma positive

print(params2[:,2])
#jit partial-d sample function
cur_sample = functools.partial(sample, model)
<<<<<<< HEAD

=======
>>>>>>> 43883e3bbf287cc28dff7707784b612c8dfb3b86
jsample = jax.jit(cur_sample)

nwalkers, ndim = params2.shape

jsample(params, tods)

# my_sampler = construct_sampler(model_params, xyz, beam)



sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args = (tods, jsample, fixed_params, fixed_pars_ids) #comma needed to not unroll tods
)

sampler.run_mcmc(params2, 1000, skip_initial_state_check = True, progress=True)
<<<<<<< HEAD

=======
>>>>>>> 43883e3bbf287cc28dff7707784b612c8dfb3b86


flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import pickle as pk

<<<<<<< HEAD
dir = "./"#'/scratch/r/rbond/jorlo/forward-modeling/'
=======
odir = "./"#'/scratch/r/rbond/jorlo/forward-modeling/'
>>>>>>> 43883e3bbf287cc28dff7707784b612c8dfb3b86

import corner
truths = params
fig = corner.corner(
    flat_samples, labels=model.par_names, truths=truths
);

plt.savefig(odir+'1gauss_corner.pdf')
plt.savefig(odir+'1gauss_corner.png')

with open(odir+'mcmc_samples.pk', 'wb') as f:
    pk.dump(flat_samples, f)
<<<<<<< HEAD
=======


>>>>>>> 43883e3bbf287cc28dff7707784b612c8dfb3b86


plt.savefig(odir + "1gauss_corner.pdf")
plt.savefig(odir + "1gauss_corner.png")
