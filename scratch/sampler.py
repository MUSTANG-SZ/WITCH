import argparse as argp
import glob
import os
import pickle as pk
import shutil
import sys
import time
from functools import partial

import emcee
import minkasi.maps.skymap as skymap
import minkasi.parallel as parallel
import minkasi.tods.core as tods
import minkasi.tods.io as io
import minkasi.tods.processing as tod_processing
import minkasi_jax.presets_by_source as pbs
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import Angle
from matplotlib import pyplot as plt
from minkasi_jax import helper
from minkasi_jax.core import model
from minkasi_jax.forward_modeling import construct_sampler, make_tod_stuff
from minkasi_jax.utils import *

with open("/home/r/rbond/jorlo/dev/minkasi_jax/configs/ms0735_noSub.yaml") as file:
    cfg = yaml.safe_load(file)
# with open('/home/r/rbond/jorlo/dev/minkasi_jax/configs/ms0735/ms0735.yaml', "r") as file:
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
tod_names = io.cut_blacklist(tod_names, bad_tod)
tod_names.sort()
tod_names = tod_names[parallel.myrank :: parallel.nproc]
print("tod #: ", len(tod_names))
parallel.barrier()  # Is this needed?

todvec = tods.TodVec()
n_tod = 2
for i, fname in enumerate(tod_names):
    if i >= n_tod:
        break
    dat = io.read_tod_from_fits(fname)
    tod_processing.truncate_tod(dat)

    # figure out a guess at common mode and (assumed) linear detector drifts/offset
    # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd, pred2, cm = tod_processing.fit_cm_plus_poly(
        dat["dat_calib"], cm_ord=3, full_out=True
    )
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

    tod = tods.Tod(dat)
    todvec.add_tod(tod)

lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = skymap.SkyMap(lims, pixsize)

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

# TODO: Implement tsBowl here
if "bowling" in cfg:
    sub_poly = cfg["bowling"]["sub_poly"]

sim = False  # This script is for simming, the option to turn off is here only for debugging
# TODO: Write this to use minkasi_jax.core.model
for i, tod in enumerate(todvec.tods):
    ipix = skymap.get_pix(tod)
    tod.info["ipix"] = ipix
    tod.set_noise(noise_class, *noise_args, **noise_kwargs)

tods = make_tod_stuff(todvec)

test_params = params[:9]  # for speed only considering single isobeta model
test_params2 = test_params + 1e-4 * np.random.randn(20, len(test_params))

nwalkers, ndim = test_params2.shape

model_params = [1, 0, 0, 0, 0, 0, 0]

my_sampler = construct_sampler(model_params, xyz, beam)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, my_sampler, args=(tods,)  # comma needed to not unroll tods
)

sampler.run_mcmc(test_params2, 50, skip_initial_state_check=True, progress=True)
