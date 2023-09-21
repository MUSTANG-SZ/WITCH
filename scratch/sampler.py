import os
import sys
import time
import glob
import shutil
import argparse as argp
from functools import partial
import yaml
import numpy as np
import minkasi
#from jack_minkasi import minkasi
from astropy.coordinates import Angle
from astropy import units as u
import minkasi_jax.presets_by_source as pbs
from minkasi_jax.utils import *
from minkasi_jax import helper
from minkasi_jax.core import model
from minkasi_jax.forward_modeling import construct_sampler, make_tod_stuff
import emcee

import pickle as pk

from matplotlib import pyplot as plt

with open('/home/jorlo/dev/minkasi_jax/configs/ms0735_noSub.yaml', "r") as file:
    cfg = yaml.safe_load(file)
#with open('/home/r/rbond/jorlo/dev/minkasi_jax/configs/ms0735/ms0735.yaml', "r") as file:
#    cfg = yaml.safe_load(file)
fit = True

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
tod_names = minkasi.cut_blacklist(tod_names, bad_tod)
tod_names.sort()
tod_names = tod_names[minkasi.myrank :: minkasi.nproc]
print('tod #: ', len(tod_names))
minkasi.barrier()  # Is this needed?

todvec = minkasi.TodVec()
n_tod = 2
for i, fname in enumerate(tod_names):
    if i >= n_tod: break
    dat = minkasi.read_tod_from_fits(fname)
    minkasi.truncate_tod(dat)

    
    # figure out a guess at common mode and (assumed) linear detector drifts/offset
    # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd, pred2, cm = minkasi.fit_cm_plus_poly(dat["dat_calib"], cm_ord=3, full_out=True)
    dat["dat_calib"] = dd
    dat["pred2"] = pred2
    dat["cm"] = cm

    # Make pixelized RA/Dec TODs
    idx, idy = tod_to_index(dat["dx"], dat["dy"], x0, y0, r_map, dr, coord_conv)
    idu, id_inv = np.unique(
        np.vstack((idx.ravel(), idy.ravel())), axis=1, return_inverse=True
    )
    dat["idx"] = idu[0]
    dat["idy"] = idu[1]
    dat["id_inv"] = id_inv
    dat["model_idx"] = idx
    dat["model_idy"] = idy

    tod = minkasi.Tod(dat)
    todvec.add_tod(tod)

lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = minkasi.SkyMap(lims, pixsize)

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

#TODO: Implement tsBowl here
if "bowling" in cfg:
    sub_poly = cfg["bowling"]["sub_poly"]

sim = False #This script is for simming, the option to turn off is here only for debugging
#TODO: Write this to use minkasi_jax.core.model
for i, tod in enumerate(todvec.tods):

    ipix = skymap.get_pix(tod)
    tod.info["ipix"] = ipix
    tod.set_noise(noise_class, *noise_args, **noise_kwargs)

tods = make_tod_stuff(todvec)

test_params = params[:9] #for speed only considering single isobeta model
test_params2 = test_params + 1e-4 * np.random.randn(20, len(test_params))

nwalkers, ndim = test_params2.shape

model_params = [1,0,0,0,0,0,0]

#my_sampler = construct_sampler(model_params, xyz, beam)

from minkasi_jax.forward_modeling import jget_chis

@partial(
    jax.jit,
    static_argnums=(1, 2, 3, 4, 5, 6, 7),
)
def my_sampler(params, n_iso, n_gnfw, n_gauss, n_uni, n_expo, n_power, n_power_cos, xyz, beam, tods):#, model_params, xyz, beam):
    """
    Generate a model realization and compute the chis of that model to data.
    TODO: model components currently hard coded.

    Arguements:

        tods: Array of tod parameters. See prep tods

        params: model parameters

        model_params: number of each model componant

        xyz: grid to evaluate model at

        beam: Beam to smooth by

    Returns:

        chi2: the chi2 difference of the model to the tods

    """
    chi2 = 0
    for i, tod in enumerate(tods):
        idx_tod, idy_tod, dat, v, weight, id_inv = tod #unravel tod

        pred = model(xyz, n_iso, n_gnfw, n_gauss, n_uni, n_expo, n_power, n_power_cos,
                     -2.5e-05, beam, idx_tod, idy_tod, params)

        pred = pred[id_inv].reshape(dat.shape)
        chi2 += jget_chis(dat, pred, v, weight)

    return chi2

my_sampler(params, 1,0,0,0,0,0,0, xyz, beam, tods)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, my_sampler, args = (1,0,0,0,0,0,0, xyz, beam, tods)
)

sampler.run_mcmc(test_params2, 50, skip_initial_state_check = True, progress=True)

