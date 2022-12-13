"""
Master fitting and map making script.
Docs will exist someday...
"""

import os
import sys
import time
import glob
import shutil
import argparse as ap
from functools import partial
import yaml
import numpy as np
import mikasi
from astropy.coordinates import Angle
from astropy import units as u
import minkasi_jax.presets_by_source as pbs
from minkasi_jax.utils import *
from minkasi_jax.isobeta import isobeta_helper


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
parser = ap.ArgumentParser(
    description="Fit cluster profiles using mikasi and minkasi_jax"
)
parser.add_argument("config", help="Path to config file")
parser.add_argument(
    "--nofit",
    "-nf",
    action="store_true",
    help="Don't actually fit, just use values from config",
)
args = parser.parse_args()

with open(args.config, "r") as file:
    cfg = yaml.safe_load(file)
fit = not args.nofit

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
minkasi.barrier()  # Is this needed?

todvec = minkasi.TodVec()
for fname in tod_names:
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
    dat["idx"] = idx
    dat["idy"] = idy

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

# Setup fit parameters
funs = []
npars = []
labels = []
pars = []
to_fit = []
priors = []
prior_vals = []
for model in cfg["models"].values():
    funs.append(eval(str(model["func"])))
    npars.append(len(model["parameters"]))
    for name, par in model["parameters"].items():
        labels.append(name)
        pars.append(eval(str(par["value"])))
        to_fit.append(eval(str(par["to_fit"])))
        if "priors" in model:
            priors.append(model["priors"]["type"])
            prior_vals.append(eval(str(model["priors"]["value"])))
        else:
            priors.append(None)
            prior_vals.append(None)
npars = np.array(npars)
labels = np.array(labels)
pars = np.array(pars)
to_fit = np.array(to_fit, dtype=bool)


noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))

# TODO: Add sim option
sub_poly = False
if "bowling" in cfg:
    sub_poly = cfg["bowling"]["sub_poly"]
if sub_poly:
    method = cfg["bowling"]["method"]
    degree = cfg["bowling"]["degree"]
for i, tod in enumerate(todvec.tods):
    ipix = skymap.get_pix(tod)
    tod.info["ipix"] = ipix

    if sub_poly:
        tod.set_apix()
        for j in range(tod.info["dat_calib"].shape[0]):
            x, y = tod.info["apix"][j], tod.info["dat_calib"][j] - tod.info[method][j]
            res, stats = np.polynomial.polynomial.polyfit(
                x, y, cfg["bowling"]["degree"]
            )
            tod.info["dat_calib"][j] -= np.polynomial.polynomial.polyval(x, res)

    tod.set_noise(noise_class, *noise_args, **noise_kwargs)

# Figure out output
outdir = os.path.join(
    cfg["paths"]["outroot"],
    cfg["cluster"]["name"],
    "-".join(mn for mn in cfg["models"].keys()),
)
if cfg["paths"]["subdir"]:
    outdir = os.path.join(outdir, cfg["paths"]["subdir"])
if fit:
    outdir = os.path.join(outdir, "-".join(l for l in labels[to_fit]))
else:
    outdir = os.path.join(outdir, "not_fit")
print_once("Outputs can be found in ", outdir)
if minkasi.myrank == 0:
    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(outdir, "config.yaml"))

# Fit TODs
pars_fit = pars
if fit:
    t1 = time.time()
    print_once("Started actual fitting")
    pars_fit, chisq, curve, errs = minkasi.fit_timestreams_with_derivs_manyfun(
        funs,
        pars,
        npars,
        todvec,
        to_fit,
        maxiter=cfg["minkasi"]["maxiter"],
        priors=priors,
        prior_vals=prior_vals,
    )
    minkasi.comm.barrier()
    t2 = time.time()
    print_once("Took ", t2 - t1, " seconds to fit")

    print_once("Fit parameters:")
    for l, pf, err in zip(labels, pars_fit, errs):
        print_once("\t", l, " = ", pf, " ± ", err)
    print_once("chisq = ", chisq)

    if minkasi.myrank == 0:
        res_path = os.path.join(outdir, "results")
        print_once("Saving results to ", res_path)
        np.savez_compressed(
            res_path, pars_fit=pars_fit, chisq=chisq, errs=errs, curve=curve
        )

# Subtract model from TODs
for tod in todvec.tods:
    start = 0
    model = 0
    for n, fun in zip(npars, funs):
        model += fun(pars_fit[start : (start + n)], tod)[1]
        start += n
    tod.info["dat_calib"] -= model
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
        outroot=os.path.join(outroot, "niter_" + repr(niter + 1)),
        save_iters=iters,
    )
    if minkasi.myrank == 0:
        mapset_out.maps[0].write(
            os.path.join(outdir, "niter_" + repr(niter + 1) + ".fits")
        )

minkasi.barrier()
