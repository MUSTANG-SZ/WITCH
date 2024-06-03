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

from minkasi.maps.skymap import SkyMap
from minkasi_jax.core import helper, model
from minkasi_jax.forward_modeling import make_tod_stuff, sample
from minkasi_jax.forward_modeling import sampler as my_sampler
from minkasi_jax.utils import *

import minkasi


from astropy.coordinates import Angle
from astropy import units as u

import minkasi_jax.presets_by_source as pbs
from minkasi_jax.utils import *
import minkasi_jax.core as core
from minkasi_jax.forward_modeling import make_tod_stuff, sample
from minkasi_jax.forward_modeling import sampler as my_sampler
from minkasi_jax.fitter import get_outdir, make_parser, load_tods, process_tods, load_config
from minkasi_jax.containers import Model

import emcee
import functools













path = "/home/jorlo/dev/minkasi_jax/configs/sampler_sims/"
with open(path + '/1gauss.yaml', "r") as file:
    cfg = yaml.safe_load(file)
# fit = True

parser = make_parser()
args = parser.parse_args()


# Load TODs
tod_names = glob.glob(os.path.join(cfg["paths"]["tods"], cfg["paths"]["glob"]))
bad_tod, addtag = pbs.get_bad_tods(
    cfg["cluster"]["name"], ndo=cfg["paths"]["ndo"], odo=cfg["paths"]["odo"]
)
tod_names = io.cut_blacklist(tod_names, bad_tod)
tod_names.sort()
tod_names = tod_names[parallel.myrank :: parallel.nproc]
print("tod #: ", len(tod_names))
parallel.barrier()  # Is this needed?

todvec = mtods.TodVec()
n_tod = 10
for i, fname in enumerate(tod_names):
    if (
        fname
        == "/scratch/r/rbond/jorlo/MS0735/TS_EaCMS0f0_51_5_Oct_2021/Signal_TOD-AGBT21A_123_03-s20.fits"
    ):
        continue
    if i >= n_tod:
        break
    dat = io.read_tod_from_fits(fname)

    tod_processing.truncate_tod(dat)
    tod_processing.downsample_tod(
        dat
    )  # sometimes we have faster sampled data than we need.
    # this fixes that.  You don't need to, though.
    tod_processing.truncate_tod(dat)

    # figure out a guess at common mode and (assumed) linear detector drifts/offset
    # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd, pred2, cm = tod_processing.fit_cm_plus_poly(
        dat["dat_calib"], cm_ord=3, full_out=True
    )
    dat["dat_calib"] = dd
    dat["pred2"] = pred2
    dat["cm"] = cm

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
priors = [None if prior is None else "flat" for prior in prior_vals]

# Deal with bowling and simming in TODs and setup noise
noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
bowl_str = process_tods(
    args, cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
)

# Get output
outdir = get_outdir(cfg, bowl_str, model)

# Now we fit
pars_fit = params.copy()

############################################################################
# Check if I can simply fit mapspace models                                #
############################################################################



import numpy as np
from minkasi_jax.core import model


def log_likelihood(theta, data):
    cur_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, theta)
    return -0.5 * np.sum(((data - cur_model) / 1e-6) ** 2)


vis_model = core.model(
    model.xyz,
    *model.n_struct,
    model.dz,
    model.beam,
    *params,
)

noise = scipy.stats.norm.rvs(size=vis_model.shape,loc=0.00,scale=1.00E-05) 
dat_model  = vis_model+noise

def log_likelihood(theta):
    cur_model = core.model(model.xyz, *model.n_struct, model.dz, model.beam, *theta) 
    return -0.5 * jp.sum(((dat_model-cur_model)/1e-5)**2)

def log_prior(theta):
     dx, dy, sigma, amp_1 = theta
    #if np.abs(dx) < 20 and np.abs(dy) < 20 and 1e-2*0.0036 < sigma < 30*0.0036 and -10 < amp_1 < 10:
     if np.abs(dx) < 20 and np.abs(dy) < 20 and 1 < sigma < 10 and -10 < amp_1 < 10:
         return 0.0
     return -np.inf

def log_probability(theta):
     lp = log_prior(theta)
     if not np.isfinite(lp):
         return -np.inf
     else:
        ll = log_likelihood(theta)
        return lp + ll

truths = params

params2 = params + 1e-1*np.random.randn(2*len(params), len(params))
params2[:,2] = np.abs(params2[:,2]) #Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability
)

sampler.run_mcmc(params2, 10000, skip_initial_state_check = True, progress=True)
flat_samples = sampler.get_chain(discard=5000, thin=15, flat=True)

import corner

fig = corner.corner(
    flat_samples, labels=model.par_names, truths=truths
)

plt.show(); plt.close()

out_model = np.array([core.model(model.xyz,*model.n_struct,model.dz,model.beam,*samp) for samp in flat_samples])
out_model = np.array([corner.quantile(out,0.50)[0] for out in out_model.reshape(out_model.shape[0],-1).T]).reshape(out_model.shape[1:])

plt.subplot(131); plt.imshow(dat_model); plt.colorbar()
plt.subplot(132); plt.imshow(out_model); plt.colorbar()
plt.subplot(133); plt.imshow(dat_model-out_model); plt.colorbar()
plt.show(); plt.close()

############################################################################
# Check if I can simply fit mapspace models                                #
############################################################################
>>>>>>> c6ac539 (map-space fit + no-data likelihood)

def log_likelihood(theta, data):
     cur_model = core.model(model.xyz, *model.n_struct, model.dz, model.beam, *params) 
     return -0.5 * np.sum(((data-cur_model)/1e-6)**2)


def log_prior(theta):
    dx, dy, sigma, amp_1 = theta
    if (
        np.abs(dx) < 20
        and np.abs(dy) < 20
        and 1e-2 * 0.0036 < sigma < 30 * 0.0036
        and -10 < amp_1 < 10
    ):
        return 0.0
    return -np.inf


def log_probability(theta, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data)



dx = float(y2K_RJ(freq, Te) * dr * XMpc / me)
params[3] = 1e-3
vis_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, params)
noise = np.random.rand(330, 330) * 0.1 * params[3]

vis_model = core.model(
    model.xyz,
    *model.n_struct,
    model.dz,
    model.beam,
    *params,
)

noise = np.random.rand(model.xyz[0].shape[0], model.xyz[1].shape[1])*0.1*params[3]


data = vis_model + noise

truths = params

params2 = params + 1e-4 * np.random.randn(2 * len(params), len(params))
params2[:, 2] = np.abs(params2[:, 2])  # Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(vis_model,))

sampler.run_mcmc(params2, 10000, skip_initial_state_check=True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner


fig = corner.corner(flat_samples, labels=labels, truths=truths)

fig = corner.corner(
    flat_samples, labels=model.par_names, truths=truths
)

plt.show()
plt.close()

############################################################################
# Now fit models in TOD space. Start with a square TOD that matches map.   #
############################################################################

import numpy as np
from minkasi_jax.core import model


def log_likelihood_tod(theta, idx, idy, data):
    cur_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, theta)
    cur_model = cur_model.at[idy.astype(int), idx.astype(int)].get(
        mode="fill", fill_value=0
    )

    return -1 / 2 * np.sum(((data - cur_model) / 1e-6) ** 2)


def log_prior(theta):
    dx, dy, sigma, amp_1 = theta
    if (
        np.abs(dx) < 20
        and np.abs(dy) < 20
        and 1e-2 * 0.0036 < sigma < 30 * 0.0036
        and -10 < amp_1 < 10
    ):
        return 0.0
    return -np.inf


def log_probability_tod(theta, idx, idy, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_tod(theta, idx, idy, data)


dx = float(y2K_RJ(freq, Te) * dr * XMpc / me)
params[3] = 1e-3
vis_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, params)


x = np.arange(0, len(xyz[0][0]), dtype=int)
y = np.arange(0, len(xyz[0][0]), dtype=int)
X, Y = np.meshgrid(x, y)
params[3] = 1e-3

data_tod = vis_model.at[Y, X].get(mode="fill", fill_value=0)

params2 = params + 1e-4 * np.random.randn(2 * len(params), len(params))
params2[:, 2] = np.abs(params2[:, 2])  # Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_tod, args=(X, Y, data_tod)
)

sampler.run_mcmc(params2, 5000, skip_initial_state_check=True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner

fig = corner.corner(flat_samples, labels=labels, truths=truths)

############################################################################
# TOD shape TOD                                                            #
############################################################################


def log_likelihood_tod(theta, idx, idy, data):
    cur_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, theta)
    cur_model = cur_model.at[idy.astype(int), idx.astype(int)].get(
        mode="fill", fill_value=0
    )

    return -1 / 2 * np.sum(((data - cur_model) / 1e-6) ** 2)


def log_prior(theta):
    dx, dy, sigma, amp_1 = theta
    if (
        np.abs(dx) < 20
        and np.abs(dy) < 20
        and 1e-2 * 0.0036 < sigma < 30 * 0.0036
        and -10 < amp_1 < 10
    ):
        return 0.0
    return -np.inf


def log_probability_tod(theta, idx, idy, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_tod(theta, idx, idy, data)


import numpy as np
from minkasi_jax.core import model

dx = float(y2K_RJ(freq, Te) * dr * XMpc / me)
params[3] = 1e-3
vis_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, params)

idx = todvec.tods[0].info["model_idx"]
idy = todvec.tods[0].info["model_idy"]
data_tod = vis_model.at[idy, idx].get(mode="fill", fill_value=0)

params2 = params + 1e-4 * np.random.randn(2 * len(params), len(params))
params2[:, 2] = np.abs(params2[:, 2])  # Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_tod, args=(idx, idy, data_tod)
)

sampler.run_mcmc(params2, 5000, skip_initial_state_check=True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner

fig = corner.corner(flat_samples, labels=labels, truths=params)


import numpy as np

############################################################################
# Real TOD Model                                                           #
############################################################################
from minkasi_jax.core import model


def log_likelihood_tod(theta, idx, idy, data):
    cur_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, theta)
    cur_model = cur_model.at[idy.astype(int), idx.astype(int)].get(
        mode="fill", fill_value=0
    )

    return -1 / 2 * np.sum(((data - cur_model) / 1e-6) ** 2)


def log_probability_tod(theta, idx, idy, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_tod(theta, idx, idy, data)


def log_prior(theta):
    dx, dy, sigma, amp_1 = theta
    if (
        np.abs(dx) < 20
        and np.abs(dy) < 20
        and 1e-2 * 0.0036 < sigma < 30 * 0.0036
        and -10 < amp_1 < 10
    ):
        return 0.0
    return -np.inf


lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = SkyMap(lims, pixsize, square=True, multiple=2)
dx = float(y2K_RJ(freq, Te) * dr * XMpc / me)
params[3] = 1e-3


sim = True  # This script is for simming, the option to turn off is here only for debugging
# TODO: Write this to use minkasi_jax.core.model
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

params2 = params + 1e-4 * np.random.randn(2 * len(params), len(params))
params2[:, 2] = np.abs(params2[:, 2])  # Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_tod, args=(idx, idy, data_tod)
)

sampler.run_mcmc(params2, 5000, skip_initial_state_check=True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner

fig = corner.corner(flat_samples, labels=labels, truths=truths)

import numpy as np

############################################################################
# Real TOD Model, func call                                                #
############################################################################
from minkasi_jax.core import model


def log_likelihood_tod(theta, idx, idy, data):
    cur_model = model(xyz, 0, 0, 1, 0, 0, 0, 0, 0, dx, beam, theta)
    cur_model = cur_model.at[idy.astype(int), idx.astype(int)].get(
        mode="fill", fill_value=0
    )

    return -1 / 2 * np.sum(((data - cur_model) / 1e-6) ** 2)


def log_probability_tod(theta, idx, idy, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_tod(theta, idx, idy, data)


def log_prior(theta):
    dx, dy, sigma, amp_1 = theta
    if (
        np.abs(dx) < 20
        and np.abs(dy) < 20
        and 1e-2 * 0.0036 < sigma < 30 * 0.0036
        and -10 < amp_1 < 10
    ):
        return 0.0
    return -np.inf


lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = SkyMap(lims, pixsize, square=True, multiple=2)

for i, tod in enumerate(todvec.tods):
    print(tod.info["fname"])
    ipix = skymap.get_pix(tod)
    tod.info["ipix"] = ipix

    tod.set_noise(noise_class, *noise_args, **noise_kwargs)

    if sim:
        # tod.info["dat_calib"] *= (-1) ** ((parallel.myrank + parallel.nproc * i) % 2)
        tod.info["dat_calib"] = 0
        start = 0
        cur_model = 0
        for n, fun in zip(npars, funs):
            cur_model += fun(params[start : (start + n)], tod)[1]
            start += n
        tod.info["dat_calib"] += np.array(cur_model)

idx = todvec.tods[0].info["model_idx"]
idy = todvec.tods[0].info["model_idy"]

data_tod = todvec.tods[0].info["dat_calib"]
truths = params

params2 = params + 1e-4 * np.random.randn(2 * len(params), len(params))
params2[:, 2] = np.abs(params2[:, 2])  # Force sigma positive

nwalkers, ndim = params2.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_tod, args=(idx, idy, data_tod)
)

sampler.run_mcmc(params2, 5000, skip_initial_state_check=True, progress=True)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import corner


fig = corner.corner(flat_samples, labels=labels, truths=truths)

fig = corner.corner(
    flat_samples, labels=labels, truths=truths
)

