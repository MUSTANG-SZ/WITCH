from witch.core import model
from witch.utils import *
from witch.fitter import *
from witch.containers import Model
import witch.mapmaking as mm
from witch.nonparametric import broken_power, nonpara_power

import minkasi.tools.presets_by_source as pbs
import minkasi

from astropy.coordinates import Angle
import astropy.units as units

import yaml
import numpy as np
import os
import glob
import shutil

import matplotlib.pyplot as plt


rs = np.linspace(1, 10, 1000)
pows = np.array([-1. , -1.5, -2. , -2.5, -3., -4 ])
amps = np.array([-2, -3, -4, -5, -6, 0])
rbins = np.array([0, 1, 2, 3, 5, 7, 999999])

path = "/home/jorlo/dev/minkasi_jax/unit_tests/cyl_unit.yaml"
with open(path, "r") as file:
    cfg = yaml.safe_load(file)
if "models" not in cfg:
    cfg["models"] = {}

cfg = load_config({}, path)
cfg["fit"] = cfg.get("fit", "model" in cfg)
cfg["sim"] = cfg.get("sim", False)
cfg["wnoise"] = cfg.get("wnoise", False)
cfg["map"] = cfg.get("map", True)
cfg["sub"] = cfg.get("sub", True)

cfg["sim"] = False

todroot = cfg["paths"]["tods"]
if not os.path.isabs(todroot):
    todroot = os.path.join(
        os.environ.get("MJ_TODROOT", os.environ["HOME"]), todroot
    )
tod_names = glob.glob(os.path.join(todroot, cfg["paths"]["glob"]))
bad_tod, _ = pbs.get_bad_tods(
    cfg["name"], ndo=cfg["paths"]["ndo"], odo=cfg["paths"]["odo"]
)
if "cut" in cfg["paths"]:
    bad_tod += cfg["paths"]["cut"]
tod_names = minkasi.tods.io.cut_blacklist(tod_names, bad_tod)
tod_names.sort()
ntods = cfg["minkasi"].get("ntods", None)
tod_names = tod_names[:ntods]
tod_names = tod_names[minkasi.myrank :: minkasi.nproc]
minkasi.barrier()  # Is this needed?

todvec = minkasi.tods.TodVec()
ntods = 2
for i, fname in enumerate(tod_names):
    if i > ntods: break
    dat = minkasi.tods.io.read_tod_from_fits(fname)
    minkasi.tods.processing.truncate_tod(dat)
    minkasi.tods.processing.downsample_tod(dat)
    minkasi.tods.processing.truncate_tod(dat)
    # figure out a guess at common mode and (assumed) linear detector drifts/offset
    # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd, pred2, cm = minkasi.tods.processing.fit_cm_plus_poly(
        dat["dat_calib"], cm_ord=3, full_out=True
    )
    dat["dat_calib"] = dd
    dat["pred2"] = pred2
    dat["cm"] = cm

    tod = minkasi.tods.Tod(dat)
    todvec.add_tod(tod)

lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = minkasi.maps.SkyMap(lims, pixsize)


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
    cfg, todvec, skymap, noise_class, noise_args, noise_kwargs, model
)

# Get output
outdir = get_outdir(cfg, model)

mm.make_maps(
    todvec,
    skymap,
    noise_class,
    noise_args,
    noise_kwargs,
    os.path.join(outdir, "signal"),
    cfg["minkasi"]["npass"],
    cfg["minkasi"]["dograd"],
)

power_law = broken_power(rs, rbins, amps, pows, 0)

pressure = nonpara_power(0,0,0, rbins, amps, pows, 0, 0, model.xyz)

asdf



