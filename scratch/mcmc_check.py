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
import minkasi.maps.skymap as skymap
from minkasi.fitting import models
from minkasi.mapmaking import noise

from astropy.coordinates import Angle
from astropy import units as u

import minkasi_jax.presets_by_source as pbs
from minkasi_jax.utils import *
from minkasi_jax.core import helper
from minkasi_jax.core import model as mink_model
from minkasi_jax.forward_modeling import make_tod_stuff, sample
from minkasi_jax.forward_modeling import sampler as my_sampler
import emcee
import functools

import pickle as pk

from matplotlib import pyplot as plt

#%load_ext autoreload
#%autoreload 2

'''
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

'''
def log_prior(theta, da):
    dx, dy, sigma, amp_1 = theta
    if np.abs(dx) < 20 and np.abs(dy) < 20 and 1e-2*da < sigma < 30*da and -10 < amp_1 < 10: 
        return 0.0
    return -np.inf

def log_probability(theta, tods, jsample, fixed_params, fixed_pars_ids, da):
    lp = log_prior(theta, da)    
    if not np.isfinite(lp):
        return -np.inf
    return lp + my_sampler(theta, tods, jsample, fixed_params, fixed_pars_ids)

#with open('/home/r/rbond/jorlo/dev/minkasi_jax/configs/sampler_sims/1gauss.yaml', "r") as file:
#    cfg = yaml.safe_load(file)
#with open('/home/jack/dev/minkasi_jax/configs/ms0735/ms0735.yaml', "r") as file:
#    cfg = yaml.safe_load(file)
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

lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = SkyMap(lims, pixsize, square=True, multiple = 2)
print("skymap, xyz: ", skymap.map.shape, xyz[0].shape)
#dr = pixsize*da * (3600*180/np.pi)
#r_map = skymap.map.shape[1]*dr/2
#xyz = make_grid(r_map, dr)

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
for cur_model in cfg["models"].values():
    npars.append(len(cur_model["parameters"]))
    for name, par in cur_model["parameters"].items():
        labels.append(name)
        par_idx[name] = len(params)
        params.append(eval(str(par["value"])))
        to_fit.append(eval(str(par["to_fit"])))
        if "priors" in par:
            priors.append(par["priors"]["type"])
            prior_vals.append(eval(str(par["priors"]["value"])))
        else:
            priors.append(None)
            prior_vals.append(None)
        if "re_eval" in par and par["re_eval"]:
            re_eval.append(str(par["value"]))
        else:
            re_eval.append(False)
    2.627 * da, funs.append(eval(str(cur_model["func"])))

npars = np.array(npars)
labels = np.array(labels)
params = np.array(params)
to_fit = np.array(to_fit, dtype=bool)
priors = np.array(priors)

noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))

sim = True #This script is for simming, the option to turn off is here only for debugging
dx = float(y2K_RJ(freq, Te)*dr*XMpc/me)

#TODO: Write this to use minkasi_jax.core.model
for i, tod in enumerate(todvec.tods):
    print(tod.info["fname"])
    ipix = skymap.get_pix(tod)
    tod.info["ipix"] = ipix

    if sim:
        tod.info["dat_calib"] *= (-1) ** ((parallel.myrank + parallel.nproc * i) % 2)
        #tod.info["dat_calib"] = 0
        start = 0
        model = 0 
        for n, fun in zip(npars, funs):
            model += fun(params[start : (start + n)], tod)[1]
            start += n

        tod.info["dat_calib"] += np.array(model)

    tod.set_noise(noise_class, *noise_args, **noise_kwargs)



    

tods = make_tod_stuff(todvec, skymap)

#test_params = params[:13] #for speed only considering single isobeta model

truths = params

model_params = [0,0,1,0,0,0,0,0]

fixed_pars_ids = []
fixed_params = params[fixed_pars_ids]
params = params[[0,1,2,3]]
params2 = params + 1e-4*np.random.randn(2*len(params), len(params))
params2[:,2] = np.abs(params2[:,2]) #Force sigma positive

print(params2[:,2])
#jit partial-d sample function
cur_sample = functools.partial(sample, model_params, xyz, beam)
jsample = jax.jit(cur_sample)

nwalkers, ndim = params2.shape

jsample(params, tods)
#my_sampler = construct_sampler(model_params, xyz, beam)




sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args = (tods, jsample, fixed_params, fixed_pars_ids, da) #comma needed to not unroll tods
)

sampler.run_mcmc(params2, 10000, skip_initial_state_check = True, progress=True)


flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

import pickle as pk

odir = '/scratch/r/rbond/jorlo/forward-modeling/'

with open(odir+'mcmc_samples.pk', 'wb') as f:
    pk.dump(flat_samples, f)

truths = params

import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=truths
);

plt.savefig(odir+'1gauss_corner.pdf')
plt.savefig(odir+'1gauss_corner.png')
