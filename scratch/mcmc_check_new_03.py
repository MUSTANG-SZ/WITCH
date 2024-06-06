import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

import emcee
import corner

import minkasi

from minkasi_jax.fitter import make_parser, load_tods, process_tods, load_config
from minkasi_jax.containers import Model

from minkasi_jax.sampler import sample

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

xyz = [(gridwx[0,:][...,None,None]-np.rad2deg(model.x0))*60*60*np.cos(model.y0),
       (gridwy[:,0][None,...,None]-np.rad2deg(model.y0))*60*60,
        gridwz[None,None,...]]

model.xyz = xyz

# Deal with bowling and simming in TODs and setup noise
noise_class  = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args   = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
bowl_str = process_tods(cfg,todvec,skymap,noise_class,noise_args,noise_kwargs,model)

steps = 2
for step in range(steps):
    sampler = sample(model,todvec,skymap,nwalk=3*len(params),nburn=200,nstep=400,pinit=params)
    samples = sampler.get_chain(thin=10,flat=True)

    for p in range(samples.shape[1]):
        print(corner.quantile(samples[:,p],[0.16,0.50,0.84]))

    if step==steps-1 or steps==1:
        edges = [corner.quantile(samples[:,i],[0.16,0.50,0.84]) for i in range(samples.shape[1])]
        edges = [[edges[i][1]-5.00*(edges[i][1]-edges[i][0]),
                edges[i][1]+5.00*(edges[i][2]-edges[i][1])] for i in range(samples.shape[1])]

        for i in range(samples.shape[1]):
            if edges[i][0]<model.priors[i].support()[0]: edges[i][0] = model.priors[i].support()[0]
            if edges[i][1]>model.priors[i].support()[1]: edges[i][1] = model.priors[i].support()[1]

            if edges[i][0]==edges[i][1]: 
                edges[i][0] = edges[i][0]-0.10*edges[i][0]
                edges[i][1] = edges[i][1]+0.10*edges[i][1]
            
        fig = corner.corner(samples,labels=model.par_names,truths=params,quantiles=[0.16,0.50,0.84],show_titles=True,title_fmt='.2e')
        plt.show(); plt.close()
        