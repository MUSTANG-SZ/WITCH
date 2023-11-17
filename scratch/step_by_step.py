import os
import sys
import time
import glob
import shutil
import argparse as argp
from functools import partial
import yaml
import numpy as np

import minkasi.tods.io as io
import minkasi.parallel as parallel
import minkasi.tods.core as mtods
import minkasi.tods.processing as tod_processing
from minkasi.maps.skymap import SkyMap
from minkasi.fitting import models
from minkasi.mapmaking import noise

from astropy.coordinates import Angle
from astropy import units as u

import minkasi_jax.presets_by_source as pbs
from minkasi_jax.utils import *
from minkasi_jax.core import helper
from minkasi_jax.core import model
from minkasi_jax.forward_modeling import make_tod_stuff, sample
from minkasi_jax.forward_modeling import sampler as my_sampler
import emcee
import functools

import pickle as pk

from matplotlib import pyplot as plt

#%load_ext autoreload
#%autoreload 2

with open('/home/jack/dev/minkasi_jax/configs/sampler_sims/1gauss_home.yaml', "r") as file:
    cfg = yaml.safe_load(file)
#fit = True

# Setup coordindate stuff
z = eval(str(cfg["coords"]["z"]))
da = get_da(z)
r_map = eval(str(cfg["coords"]["r_map"]))
dr = eval(str(cfg["coords"]["dr"]))
xyz = make_grid(r_map, dr)
coord_conv = eval(str(cfg["coords"]["conv_factor"]))
x0 = eval(str(cfg["coords"]["x0"]))
y0 = eval(str(cfg["coords"]["y0"]))

# Load TODs
tod_names = glob.glob(os.path.join(cfg["paths"]["tods"], cfg["paths"]["glob"]))
bad_tod, addtag = pbs.get_bad_tods(
    cfg["cluster"]["name"], ndo=cfg["paths"]["ndo"], odo=cfg["paths"]["odo"]
)
tod_names = io.cut_blacklist(tod_names, bad_tod)
tod_names.sort()
tod_names = tod_names[parallel.myrank :: parallel.nproc]
print('tod #: ', len(tod_names))
parallel.barrier()  # Is this needed?

todvec = mtods.TodVec()
n_tod = 10
for i, fname in enumerate(tod_names):
    if fname == "/scratch/r/rbond/jorlo/MS0735/TS_EaCMS0f0_51_5_Oct_2021/Signal_TOD-AGBT21A_123_03-s20.fits": continue
    if i >= n_tod: break
    dat = io.read_tod_from_fits(fname)

    tod_processing.truncate_tod(dat)
    tod_processing.downsample_tod(dat)   #sometimes we have faster sampled data than we need.
                                  #this fixes that.  You don't need to, though.
    tod_processing.truncate_tod(dat)  
    
    # figure out a guess at common mode and (assumed) linear detector drifts/offset
    # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd, pred2, cm = tod_processing.fit_cm_plus_poly(dat["dat_calib"], cm_ord=3, full_out=True)
    dat["dat_calib"] = dd
    dat["pred2"] = pred2
    dat["cm"] = cm

    # Make pixelized RA/Dec TODs
    idx, idy = tod_to_index(dat["dx"], dat["dy"], x0, y0, xyz, coord_conv)
    idu, id_inv = np.unique(
        np.vstack((idx.ravel(), idy.ravel())), axis=1, return_inverse=True
    )
    dat["idx"] = idu[0]
    dat["idy"] = idu[1]
    dat["id_inv"] = id_inv
    dat["model_idx"] = idx
    dat["model_idy"] = idy

    tod = mtods.Tod(dat)
    todvec.add_tod(tod)

Te = eval(str(cfg["cluster"]["Te"]))
freq = eval(str(cfg["cluster"]["freq"]))
beam = beam_double_gauss(
    dr,
    eval(str(cfg["beam"]["fwhm1"])),
    eval(str(cfg["beam"]["amp1"])),
    eval(str(cfg["beam"]["fwhm2"])),
    eval(str(cfg["beam"]["amp2"])),
)

funs = []
npars = []
labels = []
params = []
to_fit = []
priors = []
prior_vals = []
re_eval = []
par_idx = {}
subtract = []

for mname, model in cfg["models"].items():
    npars.append(len(model["parameters"]))
    _to_fit = []
    _re_eval = []
    _par_idx = {}
    for name, par in model["parameters"].items():
        labels.append(name)
        par_idx[mname + "-" + name] = len(params)
        _par_idx[mname + "-" + name] = len(_to_fit)
        params.append(eval(str(par["value"])))
        _to_fit.append(eval(str(par["to_fit"])))
        if "priors" in par:
            priors.append(par["priors"]["type"])
            prior_vals.append(eval(str(par["priors"]["value"])))
        else:
            priors.append(None)
            prior_vals.append(None)
        if "re_eval" in par and par["re_eval"]:
            _re_eval.append(str(par["value"]))
        else:
            _re_eval.append(False)
    to_fit = to_fit + _to_fit
    re_eval = re_eval + _re_eval
    # Special case where function is helper
    if model["func"][:15] == "partial(helper,":
        func_str = model["func"][:-1]
        if "xyz" not in func_str:
            func_str += ", xyz=xyz"
        if "beam" not in func_str:
            func_str += ", beam=beam"
        if "argnums" not in func_str:
            func_str += ", argnums=np.where(_to_fit)[0]"
        if "re_eval" not in func_str:
            func_str += ", re_eval=_re_eval"
        if "par_idx" not in func_str:
            func_str += ", par_idx=_par_idx"
        func_str += ")"
        model["func"] = func_str

    funs.append(eval(str(model["func"])))
    if "sub" in model:
        subtract.append(model["sub"])
    else:
        subtract.append(True)

npars = np.array(npars)
labels = np.array(labels)
params = np.array(params)
to_fit = np.array(to_fit, dtype=bool)
priors = np.array(priors)

noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))

############################################################################
# Check if I can simply fit mapspace models                                #
############################################################################

from minkasi_jax.core import model
import numpy as np

def log_likelihood(theta, data):
     cur_model =  model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, theta)
     return -0.5 * np.sum(((data-cur_model)/1e-6)**2)

def log_prior(theta):
     dx, dy, sigma, amp_1 = theta
     if np.abs(dx) < 20 and np.abs(dy) < 20 and 1e-2*0.0036 < sigma < 30*0.0036 and -10 < amp_1 < 10:
         return 0.0
     return -np.inf

def log_probability(theta, data):
     lp = log_prior(theta)
     if not np.isfinite(lp):
         return -np.inf
     return lp + log_likelihood(theta, data)

dx = float(y2K_RJ(freq, Te)*dr*XMpc/me)
params[3] = 1e-3
vis_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, params)
noise = np.random.rand(330, 330)*0.1*params[3]

data = vis_model+noise

truths = params

params2 = params + 1e-4*np.random.randn(2*len(params), len(params))
params2[:,2] = np.abs(params2[:,2]) #Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args = (vis_model,)
)

sampler.run_mcmc(params2, 10000, skip_initial_state_check = True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=truths
)
############################################################################
# Now fit models in TOD space. Start with a square TOD that matches map.   #
############################################################################

def log_likelihood_tod(theta, idx, idy, data):
     cur_model =  model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, theta)
     cur_model = cur_model.at[idy.astype(int), idx.astype(int)].get(mode = "fill", fill_value = 0)

     return -1/2 * np.sum(((data-cur_model)/1e-6)**2)

def log_probability_tod(theta, idx, idy, data):
     lp = log_prior(theta)
     if not np.isfinite(lp):
         return -np.inf
     return lp + log_likelihood_tod(theta, idx, idy, data)

x = np.arange(0, len(xyz[0][0]), dtype=int)
y = np.arange(0, len(xyz[0][0]), dtype=int)
X, Y = np.meshgrid(x, y)
params[3] = 1e-3

data_tod = vis_model.at[Y, X].get(mode="fill", fill_value = 0)

params2 = params + 1e-4*np.random.randn(2*len(params), len(params))
params2[:,2] = np.abs(params2[:,2]) #Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_tod, args = (X, Y, data_tod)
)

sampler.run_mcmc(params2, 10000, skip_initial_state_check = True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=truths
)

############################################################################
# TOD shape TOD                                                            #
############################################################################

idx = todvec.tods[0].info["model_idx"]
idy = todvec.tods[0].info["model_idy"]
data_tod = vis_model.at[idy, idx].get(mode = "fill", fill_value = 0)

params2 = params + 1e-4*np.random.randn(2*len(params), len(params))
params2[:,2] = np.abs(params2[:,2]) #Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_tod, args = (idx, idy, data_tod)
)

sampler.run_mcmc(params2, 10000, skip_initial_state_check = True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=truths
)


############################################################################
# Real TOD Model                                                           #
############################################################################
from minkasi_jax.core import model
import numpy as np

def log_likelihood_tod(theta, idx, idy, data):
     cur_model =  model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, theta)
     cur_model = cur_model.at[idy.astype(int), idx.astype(int)].get(mode = "fill", fill_value = 0)

     return -1/2 * np.sum(((data-cur_model)/1e-6)**2)

def log_probability_tod(theta, idx, idy, data):
     lp = log_prior(theta)
     if not np.isfinite(lp):
         return -np.inf
     return lp + log_likelihood_tod(theta, idx, idy, data)

lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = SkyMap(lims, pixsize, square=True, multiple = 2)

sim = True #This script is for simming, the option to turn off is here only for debugging
#TODO: Write this to use minkasi_jax.core.model
for i, tod in enumerate(todvec.tods):
    print(tod.info["fname"])
    ipix = skymap.get_pix(tod)
    tod.info["ipix"] = ipix
    if sim:
        tod.info["dat_calib"] *= 0
        start = 0
        sim_model = 0 
        for n, fun in zip(npars, funs):
            sim_model += fun(params[start : (start + n)], tod)[1]
            start += n
        tod.info["dat_calib"] += np.array(sim_model)


    tod.set_noise(noise_class, *noise_args, **noise_kwargs)

idx = todvec.tods[0].info["model_idx"]
idy = todvec.tods[0].info["model_idy"]

data_tod = todvec.tods[0].info["dat_calib"]
truths = params

params2 = params + 1e-4*np.random.randn(2*len(params), len(params))
params2[:,2] = np.abs(params2[:,2]) #Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_tod, args = (idx, idy, data_tod)
)

sampler.run_mcmc(params2, 10000, skip_initial_state_check = True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=truths
)
