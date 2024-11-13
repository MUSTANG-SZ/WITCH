"""
Update an older config to one that supports flexable dataset backends.
This is assuming you are working with mustang2 data.

This script is not that smart so use with caution.
You probably only want to run this on your base config.
"""

import argparse

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("path", help="The path to the config")
parser.add_argument(
    "--overwrite",
    "-o",
    action="store_true",
    help="If passed overwrite the input config instead of making a new one",
)
parser.add_argument(
    "--wrap_noise",
    "-w",
    action="store_true",
    help="If passed wrap the minkasi noise rather than use the JITted version of NoiseSmoothedSVD",
)
args = parser.parse_args()


with open(args.path) as file:
    cfg = yaml.safe_load(file)

if "minkasi" not in cfg:
    raise ValueError(
        "No minkasi block found! This config probably doesn't need to be converted. If it does please contact a WITCH maintainer for help doing a manual conversion"
    )

# Add imports
imports = cfg.get("imports", {})
imports.update({"witch.external.minkasi.funcs": "mf", "jitkasi.noise": "jn"})
cfg["imports"] = imports

# Add fitting section
maxiter = cfg["minkasi"].get("maxiter", 10)
cfg["minkasi"].pop("maxiter", None)
chitol = cfg["minkasi"].get("chitol", 1e-5)
cfg["minkasi"].pop("chitol", None)
fitting = {"maxiter": maxiter, "chitol": chitol}
cfg["fitting"] = fitting

# Get the beam
if "beam" in cfg:
    beam = cfg["beam"].copy()
    cfg.pop("beam", None)
else:
    print(
        "No beam found, using default M2 beam. If you define the beam elsewhere you will want to manually edit the new config file"
    )
    beam = {"fwhm1": 9.735, "amp1": 0.9808, "fwhm2": 32.627, "amp2": 0.0192}

# Now for the main section
mustang2 = cfg["minkasi"].copy()
cfg.pop("minkasi", None)
mustang2["minkasi_noise"] = mustang2["noise"].copy()
mustang2["beam"] = beam
mustang2["funcs"] = {
    "load_tods": "mf.load_tods",
    "get_info": "mf.get_info",
    "make_beam": "mf.make_beam",
    "preproc": "mf.preproc",
    "postproc": "mf.postproc",
    "postfit": "mf.postfit",
}
if args.wrap_noise:
    print("Wrapping minkasi noise class")
    mustang2["copy_noise"] = True
    nargs = f"[{mustang2['noise']['class']}, '__call__', 'apply_noise', False, {mustang2['noise']['args'][1:]}"
    mustang2["noise"]["class"] = "jn.NoiseWrapper"
    mustang2["noise"]["args"] = nargs
else:
    print("Using default JITted noise, please check and tweak this as needed")
    mustang2["noise"]["class"] = "jn.NoiseSmoothedSVD"
    mustang2["noise"]["args"] = "[]"
    mustang2["noise"]["kwargs"] = "{'fwhm':10}"
cfg["datasets"] = {"mustang2": mustang2}

# Now lets save
if args.overwrite:
    outpath = args.path
    print("Overwriting existing file")
else:
    outpath = args.path + ".new"
    print(f"Saving to {outpath}")

with open(outpath, "w") as file:
    yaml.dump(cfg, file, sort_keys=False)
