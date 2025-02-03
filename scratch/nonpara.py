import glob
import os
import shutil

import astropy.units as units
import matplotlib.pyplot as plt
import minkasi
import minkasi.tools.presets_by_source as pbs
import numpy as np
import yaml
from astropy.coordinates import Angle

import witch.external.minkasi.mapmaking as mm
from witch.containers import Model
from witch.core import model
from witch.fitter import *
from witch.nonparametric import broken_power
from witch.structure import nonpara_power
from witch.utils import *


def array_to_tuple(arr):
    if isinstance(arr, list) or type(arr) is np.ndarray:
        return tuple(array_to_tuple(item) for item in arr)
    else:
        return arr


rs = np.linspace(1, 10, 1000)
pows = np.array([-1.0, -1.5, -2.0, -2.5, -3.0, -4])
amps = np.array([-2, -3, -4, -5, -6, 0])
rbins = (0, 1, 2, 3, 5, 7, 999999)

path = "/home/jorlo/dev/minkasi_jax/unit_tests/cyl_unit.yaml"
with open(path) as file:
    cfg = yaml.safe_load(file)
if "models" not in cfg:
    cfg["models"] = {}

# TODO: Serialize cfg to a data class (pydantic?)
cfg = load_config({}, path)
cfg["fit"] = cfg.get("fit", "model" in cfg)
cfg["sim"] = cfg.get("sim", False)
cfg["wnoise"] = cfg.get("wnoise", False)
cfg["map"] = cfg.get("map", True)
cfg["sub"] = cfg.get("sub", True)
cfg["mem_starved"] = cfg.get("mem_starved", False)


# Do imports
for module, name in cfg.get("imports", {}).items():
    mod = import_module(module)
    if isinstance(name, str):
        locals()[name] = mod
    elif isinstance(name, list):
        for n in name:
            locals()[n] = getattr(mod, n)
    else:
        raise TypeError("Expect import name to be a string or a list")

# Get the functions needed to work with out dataset
# TODO: make protocols for these and check them
dset_name = list(cfg["datasets"].keys())[0]
load_tods = eval(cfg["datasets"][dset_name]["funcs"]["load_tods"])
get_info = eval(cfg["datasets"][dset_name]["funcs"]["get_info"])
make_beam = eval(cfg["datasets"][dset_name]["funcs"]["make_beam"])
preproc = eval(cfg["datasets"][dset_name]["funcs"]["preproc"])
postproc = eval(cfg["datasets"][dset_name]["funcs"]["postproc"])
postfit = eval(cfg["datasets"][dset_name]["funcs"]["postfit"])

# Get TODs
todvec = load_tods(dset_name, cfg, comm)

# Get any info we need specific to an expiriment
info = get_info(dset_name, cfg, todvec)

# Get the beam
beam = make_beam(dset_name, cfg, info)

# Define the model and get stuff setup fitting
if "model" in cfg:
    model = Model.from_cfg(cfg, beam)
else:
    model = None
    print_once("No model defined, setting fit, sim, and sub to False")
    cfg["fit"] = False
    cfg["sim"] = False
    cfg["sub"] = False

# Setup noise
noise_class = eval(str(cfg["datasets"][dset_name]["noise"]["class"]))
noise_args = tuple(eval(str(cfg["datasets"][dset_name]["noise"]["args"])))
noise_kwargs = eval(str(cfg["datasets"][dset_name]["noise"]["kwargs"]))
info["noise_class"] = noise_class
info["noise_args"] = noise_args
info["noise_kwargs"] = noise_kwargs

# Get output
if "base" in cfg.keys():
    del cfg["base"]  # We've collated to the cfg files so no need to keep the base
outdir = get_outdir(cfg, model)
info["outdir"] = outdir

# Process the TODs
preproc(dset_name, cfg, todvec, model, info)
todvec = process_tods(cfg, todvec, noise_class, noise_args, noise_kwargs, model)
todvec = jax.block_until_ready(todvec)
postproc(dset_name, cfg, todvec, model, info)


condlist = [
    (rbins[i] <= rs) & (rs < rbins[i + 1]) for i in range(len(pows) - 1, -1, -1)
]
condlist = array_to_tuple(condlist)
power_law = broken_power(rs, condlist, rbins, amps, pows, 0)

pressure = nonpara_power(0, 0, 0, rbins, amps, pows, 0, 0, model.xyz)

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    pressure = nonpara_power(
        0, 0, 0, rbins, amps, pows, 0, 0, model.xyz
    ).block_until_ready()

asdf
