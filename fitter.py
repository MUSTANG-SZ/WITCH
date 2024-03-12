"""
Master fitting and map making script.
Docs will exist someday...
"""

import argparse as argp
import glob
import os
import shutil
import sys
import time
from functools import partial

import jax
import minkasi.minkasi_all as minkasi
import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import Angle

import minkasi_jax.presets_by_source as pbs
from minkasi_jax import helper
from minkasi_jax.utils import *
import minkasi.minkasi_all as minkasi

def print_once(*args):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Arguments:

        *args: Arguments to pass to print.
    """
    if minkasi.myrank == 0:
        print(*args)
        sys.stdout.flush()


# Parse arguments
parser = argp.ArgumentParser(
    description="Fit cluster profiles using mikasi and minkasi_jax"
)
parser.add_argument("config", help="Path to config file")
parser.add_argument(
    "--nofit",
    "-nf",
    action="store_true",
    help="Don't actually fit, just use values from config",
)
parser.add_argument(
    "--nosub",
    "-ns",
    action="store_true",
    help="Don't subtract the sim'd or fit model. Useful for mapmaking a sim'd cluster"
)
parser.add_argument(
    "--wnoise",
    "-wn",
    action="store_true",
    help="Use whitenoise instead of map noise. Only for use with sim"
)
args = parser.parse_args()

with open(args.config, "r") as file:
    cfg = yaml.safe_load(file)
if "models" not in cfg:
    cfg["models"] = {}
fit = (not args.nofit) & bool(len(cfg["models"]))

# Get device
# TODO: multi device setups
dev_id = cfg.get("jax_device", 0)
device = jax.devices()[dev_id]

# Setup coordindate stuff
z = eval(str(cfg["coords"]["z"]))
da = get_da(z)
r_map = eval(str(cfg["coords"]["r_map"]))
dr = eval(str(cfg["coords"]["dr"]))
dr = (lambda x: x if type(x) is tuple else (x,))(dr)
coord_conv = eval(str(cfg["coords"]["conv_factor"]))
x0 = eval(str(cfg["coords"]["x0"]))
y0 = eval(str(cfg["coords"]["y0"]))

xyz_host = make_grid(r_map, *dr)
xyz = jax.device_put(xyz_host, device)
xyz[0].block_until_ready()
xyz[1].block_until_ready()
xyz[2].block_until_ready()
dr = eval(str(cfg["coords"]["dr"]))
# Load TODs
tod_names = glob.glob(os.path.join(cfg["paths"]["tods"], cfg["paths"]["glob"]))
bad_tod, addtag = pbs.get_bad_tods(
    cfg["cluster"]["name"], ndo=cfg["paths"]["ndo"], odo=cfg["paths"]["odo"]
)
if "cut" in cfg["paths"]:
    bad_tod += cfg["paths"]["cut"]
#tod_names = minkasi.cut_blacklist(tod_names, bad_tod)
tod_names.sort()
ntods = cfg["minkasi"].get("ntods", None)
tod_names = tod_names[:ntods]
tod_names = tod_names[minkasi.myrank :: minkasi.nproc]
minkasi.barrier()  # Is this needed?

n_tods = 999999
todvec = minkasi.TodVec()
for i, fname in enumerate(tod_names):
    if i >= n_tods: continue
    dat = minkasi.read_tod_from_fits(fname)
    minkasi.truncate_tod(dat)
    minkasi.downsample_tod(dat)
    minkasi.truncate_tod(dat)
    # figure out a guess at common mode and (assumed) linear detector drifts/offset
    # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd, pred2, cm = minkasi.fit_cm_plus_poly(dat["dat_calib"], cm_ord=3, full_out=True)
    dat["dat_calib"] = dd
    dat["pred2"] = pred2
    dat["cm"] = cm

    # Make pixelized RA/Dec TODs
    idx, idy = tod_to_index(dat["dx"], dat["dy"], x0, y0, xyz_host, coord_conv)
    idu, id_inv = np.unique(
        np.vstack((idx.ravel(), idy.ravel())), axis=1, return_inverse=True
    )
    dat["idx"] = jax.device_put(idu[0], device)
    dat["idy"] = jax.device_put(idu[1], device)
    dat["id_inv"] = id_inv

    tod = minkasi.Tod(dat)
    todvec.add_tod(tod)

# make a template map with desired pixel size an limits that cover the data
# todvec.lims() is MPI-aware and will return global limits, not just
# the ones from private TODs
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
beam = jax.device_put(beam, device)

# Setup fit parameters
funs = []
npars = []
labels = []
params = []
to_fit = []
priors = []
prior_vals = []
re_eval = []
par_idx = {}
subtract = []
for mname, model in cfg["models"].items():
    npars.append(len(model["parameters"]))
    _to_fit = []
    _re_eval = []
    _par_idx = {}
    for name, par in model["parameters"].items():
        labels.append(name)
        par_idx[mname + "-" + name] = len(params)
        _par_idx[mname + "-" + name] = len(_to_fit)
        params.append(eval(str(par["value"])))
        _to_fit.append(eval(str(par["to_fit"])))
        if "priors" in par:
            priors.append(par["priors"]["type"])
            prior_vals.append(eval(str(par["priors"]["value"])))
        else:
            priors.append(None)
            prior_vals.append(None)
        if "re_eval" in par and par["re_eval"]:
            _re_eval.append(str(par["value"]))
        else:
            _re_eval.append(False)
    to_fit = to_fit + _to_fit
    re_eval = re_eval + _re_eval
    # Special case where function is helper
    if model["func"][:15] == "partial(helper,":
        func_str = model["func"][:-1]
        if "xyz" not in func_str:
            func_str += ", xyz=xyz"
        if "beam" not in func_str:
            func_str += ", beam=beam"
        if "argnums" not in func_str:
            func_str += ", argnums=np.where(_to_fit)[0]"
        if "re_eval" not in func_str:
            func_str += ", re_eval=_re_eval"
        if "par_idx" not in func_str:
            func_str += ", par_idx=_par_idx"
        func_str += ")"
        model["func"] = func_str

    funs.append(eval(str(model["func"])))
    if "sub" in model:
        subtract.append(model["sub"])
    else:
        subtract.append(True)
npars = np.array(npars)
labels = np.array(labels)
params = np.array(params)
to_fit = np.array(to_fit, dtype=bool)
priors = np.array(priors)

noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))

sub_poly = False
if "bowling" in cfg:
    sub_poly = cfg["bowling"]["sub_poly"]
if sub_poly:
    method = cfg["bowling"]["method"]
    degree = cfg["bowling"]["degree"]
sim = False
if "sim" in cfg:
    sim = cfg["sim"]
for i, tod in enumerate(todvec.tods):
    ipix = skymap.get_pix(tod)
    tod.info["ipix"] = ipix

    if sub_poly:
        tod.set_apix()
        for j in range(tod.info["dat_calib"].shape[0]):
            x, y = tod.info["apix"][j], tod.info["dat_calib"][j] - tod.info[method][j]
            res = np.polynomial.polynomial.polyfit(x, y, cfg["bowling"]["degree"])
            tod.info["dat_calib"][j] -= np.polynomial.polynomial.polyval(x, res)

    if sim:
        if args.wnoise:
            temp = np.percentile(np.diff(tod.info["dat_calib"]), [33,68])
            scale = (temp[1]-temp[0])/np.sqrt(8) 
            tod.info["dat_calib"] = np.random.normal(0, scale, size=tod.info["dat_calib"].shape) 
        else: 
            tod.info["dat_calib"] *= (-1) ** ((minkasi.myrank + minkasi.nproc * i) % 2)
        start = 0
        model = 0
        for n, fun in zip(npars, funs):
            model += fun(params[start : (start + n)], tod)[1]
            start += n
            print(np.amax(np.abs(model)))
        tod.info["dat_calib"] += np.array(model)

    tod.set_noise(noise_class, *noise_args, **noise_kwargs)

# Figure out output
models = [
    mn + ("_ns" * (not ns)) for mn, ns in zip(list(cfg["models"].keys()), subtract)
]
outdir = os.path.join(
    cfg["paths"]["outroot"],
    cfg["cluster"]["name"],
    "-".join(mn for mn in models),
)
if "subdir" in cfg["paths"]:
    outdir = os.path.join(outdir, cfg["paths"]["subdir"])
if fit:
    outdir = os.path.join(outdir, "-".join(l for l in labels[to_fit]))
else:
    outdir = os.path.join(outdir, "not_fit")
if sub_poly:
    outdir += "-" + method + "_" + str(degree)
if sim:
    outdir += "-sim"
if args.nosub:
    outdir += "-no_sub"

print_once("Outputs can be found in", outdir)
if minkasi.myrank == 0:
    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(outdir, "config.yaml"))

# Fit TODs
pars_fit = params

if sim:
    print_once("Starting pars: \n")
    for i, label in enumerate(labels):
        print_once(label, ": {:.2e}".format(params[i]))
    params[to_fit] *= 1.1 #Don't start at exactly the right value

if fit:
    t1 = time.time()
    print_once("Started actual fitting")
    pars_fit, chisq, curve, errs = minkasi.fit_timestreams_with_derivs_manyfun(
        funs,
        params,
        npars,
        todvec,
        to_fit,
        maxiter=cfg["minkasi"]["maxiter"],
        priors=priors,
        prior_vals=prior_vals,
    )
    minkasi.comm.barrier()
    t2 = time.time()
    print_once("Took", t2 - t1, "seconds to fit")

    params = pars_fit
    for i, re in enumerate(re_eval):
        if not re:
            continue
        pars_fit[i] = eval(re)

    print_once("Fit parameters:")
    for l, pf, err in zip(labels, pars_fit, errs):
        print_once("\t", l, "= {:.2e} +/- {:.2e}".format(pf, err))
    print_once("chisq =", chisq)

    if minkasi.myrank == 0:
        res_path = os.path.join(outdir, "results")
        print_once("Saving results to", res_path + ".npz")
        np.savez_compressed(
            res_path, pars_fit=pars_fit, chisq=chisq, errs=errs, curve=curve
        )

# Subtract model from TODs
if not args.nosub:
    for tod in todvec.tods:
        start = 0
        model = 0
        for n, fun, sub in zip(npars, funs, subtract):
            if not sub:
                continue
            model += fun(pars_fit[start : (start + n)], tod)[1]
            start += n
        tod.info["dat_calib"] -= np.array(model)
        tod.set_noise(noise_class, *noise_args, **noise_kwargs)

# Make maps
npass = cfg["minkasi"]["npass"]
dograd = cfg["minkasi"]["dograd"]
# get the hit count map.  We use this as a preconditioner
# which helps small-scale convergence quite a bit.
print_once("starting hits")
hits = minkasi.make_hits(todvec, skymap)
print_once("finished hits.")
naive = skymap.copy()
naive.clear()
for tod in todvec.tods:
    tmp = tod.info["dat_calib"].copy()
    u, s, v = np.linalg.svd(tmp, 0)
    pred = np.outer(u[:, 0], s[0] * v[0, :])
    tmp = tmp - pred
    naive.tod2map(tod, tmp)
naive.mpi_reduce()
naive.map[hits.map > 0] = naive.map[hits.map > 0] / hits.map[hits.map > 0]
if minkasi.myrank == 0:
    naive.write(os.path.join(outdir, "naive.fits"))
    hits.write(os.path.join(outdir, "hits.fits"))
hits_org = hits.copy()
hits.invert()

# setup the mapset.  In general this can have many things
# in addition to map(s) of the sky, but for now we'll just
# use a single skymap.
weightmap = minkasi.make_hits(todvec, skymap, do_weights=True)
mask = weightmap.map > 0
tmp = weightmap.map.copy()
tmp[mask] = 1.0 / np.sqrt(tmp[mask])
noisemap = weightmap.copy()
noisemap.map[:] = tmp
if minkasi.myrank == 0:
    noisemap.write(os.path.join(outdir, "noise.fits"))
    weightmap.write(os.path.join(outdir, "weights.fits"))

mapset = minkasi.Mapset()
mapset.add_map(skymap)

# make A^T N^1 d.  TODs need to understand what to do with maps
# but maps don't necessarily need to understand what to do with TODs,
# hence putting make_rhs in the vector of TODs.
# Again, make_rhs is MPI-aware, so this should do the right thing
# if you run with many processes.
rhs = mapset.copy()
todvec.make_rhs(rhs)

# this is our starting guess.  Default to starting at 0,
# but you could start with a better guess if you have one.
x0 = rhs.copy()
x0.clear()

# preconditioner is 1/ hit count map.  helps a lot for
# convergence.
precon = mapset.copy()
precon.maps[0].map[:] = hits.map[:]

# run PCG
iters = [5, 25, 100]
mapset_out = minkasi.run_pcg_wprior(
    rhs,
    x0,
    todvec,
    None,
    precon,
    maxiter=26,
    outroot=os.path.join(outdir, "noprior"),
    save_iters=iters,
)
if minkasi.myrank == 0:
    mapset_out.maps[0].write(
        os.path.join(outdir, "initial.fits")
    )  # and write out the map as a FITS file

for niter in range(npass):
    maxiter = 26 + 25 * (niter + 1)
    # first, re-do the noise with the current best-guess map
    for tod in todvec.tods:
        mat = 0 * tod.info["dat_calib"]
        for mm in mapset_out.maps:
            mm.map2tod(tod, mat)
        tod.set_noise(
            noise_class, dat=tod.info["dat_calib"] - mat, *noise_args, **noise_kwargs
        )

    gradmap = hits.copy()

    if dograd:
        gradmap.map[:] = minkasi.get_grad_mask_2d(
            mapset_out.maps[0], todvec, thresh=1.8
        )
        prior = minkasi.tsModel(todvec, minkasi.CutsCompact)
        for tod in todvec.tods:
            prior.data[tod.info["fname"]] = tod.prior_from_skymap(gradmap)
            print(
                "prior on tod "
                + tod.info["fname"]
                + " length is "
                + repr(prior.data[tod.info["fname"]].map.size)
            )

        mapset = minkasi.Mapset()
        mapset.add_map(mapset_out.maps[0])
        pp = prior.copy()
        pp.clear()
        mapset.add_map(pp)

        priorset = minkasi.Mapset()
        priorset.add_map(skymap)
        priorset.add_map(prior)
        priorset.maps[0] = None
    else:
        priorset = None

    rhs = mapset.copy()
    todvec.make_rhs(rhs)

    precon = mapset.copy()
    precon.maps[0].map[:] = hits.map[:]
    mapset_out = minkasi.run_pcg_wprior(
        rhs,
        mapset,
        todvec,
        priorset,
        precon,
        maxiter=maxiter,
        outroot=os.path.join(outdir, "niter_" + repr(niter + 1)),
        save_iters=iters,
    )
    if minkasi.myrank == 0:
        mapset_out.maps[0].write(
            os.path.join(outdir, "niter_" + repr(niter + 1) + ".fits")
        )

minkasi.barrier()
