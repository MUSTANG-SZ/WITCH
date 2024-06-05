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

import jax; jax.config.update("jax_enable_x64", True)

np.random.seed(435892)

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
from minkasi_jax.forward_modeling import make_tod_stuff, sample, Mapset, todcore
from minkasi_jax.forward_modeling import sampler as my_sampler
from minkasi_jax.fitter import get_outdir, make_parser, load_tods, process_tods, load_config
from minkasi_jax.containers import Model

import emcee
import corner

import functools
import dill

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
pixsize = 1.00 / 3600 * np.pi / 180
skymap = minkasi.maps.SkyMap(lims, pixsize)

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

noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
bowl_str = process_tods(cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model)

tods = make_tod_stuff(todvec,skymap,x0=model.x0,y0=model.y0)

# ------------------------------------------------------------

dx, dy, rhs, v, weight, norm, data = tods[0]

params1 = params.copy(); params1[-1] = 1.00
params2 = params.copy(); params2[-1] = 0.10

# ------------------------------------------------------------

if False:
    factor = []
    for pars in [params1,params2]:
        model_map = mink_model(model.xyz,*model.n_struct,model.dz,model.beam,*pars)

        model_tod = bilinear_interp(dx,dy,xyz[0].ravel(),xyz[1].ravel(),model_map)

        model_rot = jnp.dot(v,model_tod)
        model_tmp = jnp.hstack([model_rot, jnp.fliplr(model_rot[:, 1:-1])])
        model_rft = jnp.real(jnp.fft.rfft(model_tmp,axis=1))

        model_filt = jnp.fft.irfft(weight*model_rft,axis=1)[:,:model_rft.shape[1]]
        model_filt_rot = jnp.dot(v.T,model_filt)

        factor.append(-2.00*jnp.sum(model_filt_rot*data)+jnp.sum(model_filt_rot*model_tod))

    print('test 1:',factor[0]-factor[1],factor[0],factor[1])

# ------------------------------------------------------------

factor1 = []
for pars in [params1,params2]:
    factor1.append(sample(model,pars,[tods[0]]))

# ------------------------------------------------------------

factor2 = []
for pars in [params1,params2]:
    model_map = mink_model(model.xyz,*model.n_struct,model.dz,model.beam,*pars)

    model_tod = bilinear_interp(dx,dy,xyz[0].ravel(),xyz[1].ravel(),model_map)

    tods = todvec.tods[0]
    data = tods.info['dat_calib'].copy()
    model_filt_rot = tods.noise.apply_noise(model_tod)

    factor2.append(jnp.sum(model_filt_rot*data)-0.50*jnp.sum(model_filt_rot*model_tod))

#print((factor2[0]-factor2[1])-(factor1[0]-factor1[1]))

# ------------------------------------------------------------

factor3 = []
for pars in [params1,params2]:
    model_map = mink_model(model.xyz,*model.n_struct,model.dz,model.beam,*pars)
    model_tod = bilinear_interp(dx,dy,xyz[0].ravel(),xyz[1].ravel(),model_map)
    model_filt_rot = todvec.tods[0].noise.apply_noise(model_tod)

    refmap = skymap.copy()
    refmap.clear()

    mapset = Mapset()
    mapset.add_map(refmap)
    todtmp = todcore.TodVec()
    todtmp.add_tod(todvec.tods[0])
    todtmp.make_rhs(mapset,False)

    data_rhs = mapset.maps[0].map.copy()

    #print(data_rhs.shape,model_map.shape)
    factor3.append(jnp.sum(data_rhs*model_map.T)-0.50*jnp.sum(model_filt_rot*model_tod))

print('test with sample foo  :',factor1[0]-factor1[1],factor1[0],factor1[1])
print('test with apply_noise :',factor2[0]-factor2[1],factor2[0],factor2[1])
print('test with rhs         :',factor3[0]-factor3[1],factor3[0],factor3[1])

# ------------------------------------------------------------
for pars in [params1]:
    model_map = mink_model(model.xyz,*model.n_struct,model.dz,model.beam,*pars)
    
    refmap = skymap.copy()
    refmap.clear()

    mapset = Mapset()
    mapset.add_map(refmap)
    todtmp = todcore.TodVec()
    todtmp.add_tod(todvec.tods[0])
    todtmp.make_rhs(mapset,False)
    data_rhs = mapset.maps[0].map.copy()

    modtod = 0*todvec.tods[0].info["dat_calib"]
    mapset.maps[0].map[:] = model_map.T
    mapset.maps[0].map2tod(todvec.tods[0],modtod)

    r = todvec.tods[0].info["dat_calib"]-modtod
    r_filt = todvec.tods[0].noise.apply_noise(r)
    factor4 = jnp.sum(r_filt*r)

    print('test minkasi-only (1):',factor4)

    dfilt = todvec.tods[0].noise.apply_noise(todvec.tods[0].info["dat_calib"])
    mfilt = todvec.tods[0].noise.apply_noise(modtod)
    mrhs = jnp.sum(model_map.T*data_rhs)
    dmfilt = jnp.sum(mfilt*todvec.tods[0].info["dat_calib"])
    print('test minkasi-only (2):',mrhs,dmfilt,mrhs-dmfilt)

    dNd = jnp.sum(dfilt*todvec.tods[0].info["dat_calib"])
    mNm = jnp.sum(mfilt*modtod)
    chisq = dNd-2.00*dmfilt+mNm
    print('test minkasi-only (3):',chisq,factor4,chisq-factor4)

