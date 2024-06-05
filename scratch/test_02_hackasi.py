import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

import emcee
import corner

import minkasi

from minkasi_jax.fitter import make_parser, load_tods, process_tods, load_config
from minkasi_jax.forward_modeling import *
from minkasi_jax.containers import Model
from minkasi_jax import core as mjcore

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
pixsize = np.deg2rad(1.00/3600)
skymap = minkasi.maps.SkyMap(lims, pixsize)

# Define the model and get stuff setup for minkasi
model = Model.from_cfg(cfg)
params = np.array(model.pars)

gridmx, gridmy = np.meshgrid(np.arange(skymap.nx),np.arange(skymap.ny))
gridwx, gridwy = skymap.wcs.all_pix2world(gridmx,gridmy,0)
gridwx = np.fliplr(gridwx)
gridwz = np.linspace(-1 * cfg["coords"]["r_map"], cfg["coords"]["r_map"], 2 * int(cfg["coords"]["r_map"] / cfg["coords"]["dr"]), dtype=float)

# xyz = [(gridwy[:,0][...,None,None]-np.rad2deg(model.y0))*60*60,
#        (gridwx[0,:][None,...,None]-np.rad2deg(model.x0))*60*60*np.cos(model.y0),
#         gridwz[None,None,...]]

xyz = [(gridwx[0,:][...,None,None]-np.rad2deg(model.x0))*60*60*np.cos(model.y0),
       (gridwy[:,0][None,...,None]-np.rad2deg(model.y0))*60*60,
        gridwz[None,None,...]]

model.xyz = xyz

# Deal with bowling and simming in TODs and setup noise
noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))
bowl_str = process_tods(cfg,todvec,skymap,noise_class,noise_args,noise_kwargs,model)

# ------------------------------------------------------------

tods = make_tod_stuff(todvec,skymap,x0=model.x0,y0=model.y0)
dx, dy, rhs, v, weight, norm, data = tods[0]

r = mjcore.model(model.xyz,*model.n_struct,model.dz,model.beam,*params)
r_tod = bilinear_interp(dx,dy,model.xyz[0].ravel(),model.xyz[1].ravel(),r)
r_rot = jnp.dot(v,r_tod)

r_tmp = jnp.hstack([r_rot,jnp.fliplr(r_rot[:,1:-1])])
r_rft = jnp.real(jnp.fft.rfft(r_tmp,axis=1))

r_ift = jnp.fft.irfft(weight*r_rft,axis=1,norm='forward')[:,:r_rot.shape[1]]
r_irt = jnp.dot(v.T,r_ift)
r_irt = r_irt.at[:, 0].multiply(0.50)
r_irt = r_irt.at[:,-1].multiply(0.50)
print(r_irt.shape)

# ------------------------------------------------------------

m = mjcore.model(model.xyz,*model.n_struct,model.dz,model.beam,*params)
m_tod = bilinear_interp(dx,dy,model.xyz[0].ravel(),model.xyz[1].ravel(),m)
m_fil = todvec.tods[0].noise.apply_noise(m_tod)

# ------------------------------------------------------------

plt.plot(np.median(r_irt,axis=0)/np.median(m_fil,axis=0),'.')
plt.show(); plt.close()