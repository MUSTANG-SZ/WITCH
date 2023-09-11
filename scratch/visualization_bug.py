from astropy.coordinates import Angle
import astropy.units as u
import numpy as np
import yaml

from minkasi_jax.utils import *
from minkasi_jax.core import model

import matplotlib.pyplot as plt


with open('../configs/ms0735_noSub.yaml', "r") as file:
    cfg = yaml.safe_load(file)

# Setup coordindate stuff
z = eval(str(cfg["coords"]["z"]))
da = get_da(z)
r_map = eval(str(cfg["coords"]["r_map"]))
dr = eval(str(cfg["coords"]["dr"]))
xyz = make_grid(r_map, dr)
coord_conv = eval(str(cfg["coords"]["conv_factor"]))
x0 = eval(str(cfg["coords"]["x0"]))
y0 = eval(str(cfg["coords"]["y0"]))


Te = eval(str(cfg["cluster"]["Te"]))
freq = eval(str(cfg["cluster"]["freq"]))
beam = beam_double_gauss(
    dr,
    eval(str(cfg["beam"]["fwhm1"])),
    eval(str(cfg["beam"]["amp1"])),
    eval(str(cfg["beam"]["fwhm2"])),
    eval(str(cfg["beam"]["amp2"])),
)

npars = []
labels = []
params = []
par_idx = {}
for cur_model in cfg["models"].values():
    npars.append(len(cur_model["parameters"]))
    for name, par in cur_model["parameters"].items():
        labels.append(name)
        par_idx[name] = len(params)
        params.append(eval(str(par["value"])))

npars = np.array(npars)
labels = np.array(labels)
params = np.array(params)

vis_pars = params[:42]
vis_pars[8], vis_pars[17] = -1e-5, -1e-5


x = np.arange(0, 2*len(xyz[1]), dtype = int)
y = np.arange(0, 2*len(xyz[1]), dtype = int)
X, Y = np.meshgrid(x, y)

# dr = eval(str(cfg["coords"]["dr"]))*2 #Things look wrong even without this but they look really wrong with it
# xyz = make_grid(r_map, dr)

dx = float(y2K_RJ(freq, Te)*dr*XMpc/me)
vis_model = model(xyz, 2, 0, 0, 3, 0, 0, 0, dx, beam, X, Y, vis_pars)

plt.imshow(vis_model, extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)], origin = 'lower')
plt.colorbar()
plt.show()
