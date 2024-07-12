"""
Functions that wrap useful minkasi recipes
"""

import contextlib
import os
from typing import Optional

import minkasi
import numpy as np
from typing_extensions import Unpack


def make_naive(
    todvec: minkasi.tods.TodVec, skymap: minkasi.maps.MapType, outdir: str
) -> tuple[minkasi.maps.MapType, minkasi.maps.MapType]:
    """
    Make a naive map where we just bin common mode subtracted TODs.

    Parameters
    ----------
    todvec : minkasi.tods.TodVec
        The TODs to mapmake.
    skymap : minkasi.maps.MapType
        Map to use as footprint for outputs.

    Returns
    -------
    naive : minkasi.maps.MapType
        The naive map.
    hits : minkasi.maps.MapType
        The hit count map.
        We use this as a preconditioner which helps small-scale convergence quite a bit.
    """
    hits = minkasi.mapmaking.make_hits(todvec, skymap)

    # Make a naive map where we just bin the CM subbed tods
    naive = skymap.copy()
    naive.clear()
    for tod in todvec.tods:
        tmp = tod.info["dat_calib"].copy()
        u, s, v = np.linalg.svd(tmp, False)
        tmp -= np.outer(u[:, 0], s[0] * v[0, :])
        naive.tod2map(tod, tmp)
    naive.mpi_reduce()
    naive.map[hits.map > 0] = naive.map[hits.map > 0] / hits.map[hits.map > 0]
    if minkasi.myrank == 0:
        naive.write(os.path.join(outdir, "naive.fits"))
        hits.write(os.path.join(outdir, "hits.fits"))
    naive.clear()
    return naive, hits


def make_weights(
    todvec: minkasi.tods.TodVec, skymap: minkasi.maps.MapType, outdir: str
) -> tuple[minkasi.maps.MapType, minkasi.maps.MapType]:
    """
    Make weights and noise map.

    Parameters
    ----------
    todvec : minkasi.tods.TodVec
        The TODs to mapmake.
    skymap : minkasi.maps.MapType
        Map to use as footprint for outputs.

    Returns
    -------
    weightmap : minkasi.maps.MapType
        The weights map.
    noisemap : minkasi.maps.MapType
        The noise map.
        This is just 1/sqrt(weights).
    """
    weightmap = minkasi.mapmaking.make_hits(todvec, skymap, do_weights=True)
    mask = weightmap.map > 0
    tmp = weightmap.map.copy()
    tmp[mask] = 1.0 / np.sqrt(tmp[mask])
    noisemap = weightmap.copy()
    noisemap.map[:] = tmp
    if minkasi.myrank == 0:
        noisemap.write(os.path.join(outdir, "noise.fits"))
        weightmap.write(os.path.join(outdir, "weights.fits"))

    return weightmap, noisemap


def reestimate_noise_from_map(
    todvec: minkasi.tods.TodVec,
    mapset: minkasi.maps.Mapset,
    noise_class: minkasi.mapmaking.NoiseModelType,
    noise_args: tuple,
    noise_kwargs: dict,
):
    """
    Use the current guess at the map to reestimate the noise:

    Parameters
    ----------
    todvec : minkasi.tods.TodVec
        The TODs to reestimate noise for.
    mapset : minkasi.maps.Mapset
        Mapset containing the current map solution.
    noise_class : minkasi.mapmaking.NoiseModelType
        Which noise model to use.
    noise_args : tuple
        Additional arguments to pass to `minkasi.tods.Tod.set_noise`.
    noise_kwargs : dict
        Additional keyword arguments to pass to `minkasi.tods.Tod.set_noise`.
    """
    for tod in todvec.tods:
        mat = 0 * tod.info["dat_calib"]
        for mm in mapset.maps:
            mm.map2tod(tod, mat)
        tod.set_noise(
            noise_class,
            dat=tod.info["dat_calib"] - mat,
            *noise_args,
            **noise_kwargs,
        )


def get_grad_prior(
    todvec: minkasi.tods.TodVec,
    mapset: minkasi.maps.Mapset,
    gradmap: minkasi.maps.MapType,
    *args: Unpack[tuple],
    **kwargs,
) -> minkasi.maps.Mapset:
    """
    Make a gradient based prior from a map.
    This helps avoid errors due to sharp features.

    Parameters
    ----------
    todvec : minkasi.tods.TodVec
        The TODs what we are mapmaking.
    mapset : minkasi.maps.Mapset
        The mapset to compute priors with.
        We assume that the first element is the map we care about.
    gradmap : minkasi.maps.MapType
        Containter to use as the gradient map.
    *args : Unpack[tuple]
        Additional arguments to pass to get_grad_mask_2d.
    **kwargs
        Keyword arguments to pass to get_grad_mask_2d.
    Returns
    -------
    new_mapset : minkasi.maps.Mapset
        A mapset with the original map and a cleared prior map.
    """
    gradmap.map[:] = minkasi.mapmaking.noise.get_grad_mask_2d(
        mapset.maps[0], todvec, *args, **kwargs
    )
    prior = minkasi.mapmaking.timestream.tsModel(todvec, minkasi.tods.cuts.CutsCompact)
    for tod in todvec.tods:
        prior.data[tod.info["fname"]] = tod.prior_from_skymap(gradmap)
        print(
            "prior on tod "
            + tod.info["fname"]
            + " length is "
            + repr(prior.data[tod.info["fname"]].map.size)
        )

    new_mapset = minkasi.maps.Mapset()
    new_mapset.add_map(mapset.maps[0])
    pp = prior.copy()
    pp.clear()
    new_mapset.add_map(pp)

    return new_mapset


def solve_map(
    todvec: minkasi.tods.TodVec,
    x0: minkasi.maps.Mapset,
    ihits: minkasi.maps.MapType,
    prior: Optional[minkasi.mapmaking.pcg.HasPrior],
    maxiters: int,
    save_iters: list[int],
    outdir: str,
    desc_str: str,
) -> minkasi.maps.Mapset:
    """
    Solve for map with PCG.

    Parameters
    ----------
    todvec : minkasi.tods.TodVec
        The TODs what we are mapmaking.
    x0 : minkasi.maps.Mapset
        The initial guess mapset.
    ihits : minkasi.maps.MapType
        The inverse hits map.
    prior : Optional[minkasi.mapmaking.pgc.HasPrior]
        Prior to use when mapmaking, set to None to not use.
    maxiters : int
        Maximum PCG iters to use.
    save_iters : list[int]
        Which iterations to save the map at.
    outdir : str
        The output directory
    desc_str : str
        String used to determine outroot.

    Returns
    -------
    mapset : minkasi.maps.Mapset
        The mapset with the solved map.
    """
    # make A^T N^1 d.  TODs need to understand what to do with maps
    # but maps don't necessarily need to understand what to do with TODs,
    # hence putting make_rhs in the vector of TODs.
    # Again, make_rhs is MPI-aware, so this should do the right thing
    # if you run with many processes.
    rhs = x0.copy()
    todvec.make_rhs(rhs)

    # Preconditioner is 1/ hit count map.
    # Helps a lot for convergence.
    precon = x0.copy()
    precon.maps[0].map[:] = ihits.map[:]

    # run PCG to solve
    # Supressing print here, probably want a verbosity setting on the minkasi side...
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        mapset = minkasi.mapmaking.run_pcg_wprior(
            rhs,
            x0,
            todvec,
            prior,
            precon,
            maxiter=maxiters,
            outroot=os.path.join(outdir, desc_str),
            save_iters=save_iters,
        )

    if minkasi.myrank == 0:
        mapset.maps[0].write(
            os.path.join(outdir, f"{desc_str}.fits")
        )  # and write out the map as a FITS file

    return mapset


def make_maps(
    todvec: minkasi.tods.TodVec,
    skymap: minkasi.maps.MapType,
    noise_class: minkasi.mapmaking.noise.NoiseModelType,
    noise_args: tuple,
    noise_kwargs: dict,
    outdir: str,
    npass: int,
    dograd: bool,
):
    """
    Make a minkasi map with multple passes and noise reestimation.
    Unless you are an expert this will usually be all you need.

    Parameters
    ----------
    todvec : minkasi.tods.TodVec
        The tods to mapmake.
    skymap : minkasi.maps.MapType
        Map to use as a template.
        The contents don't matter only the shape and WCS info.
    noise_class : minkasi.mapmaking.noise.NoiseModelType
        The noise model to use on the TODs.
    noise_args : tuple
        Arguments to pass to `minkasi.tods.Tod.set_noise`.
    noise_kwargs : dict
        Keyword arguments to pass to `minkasi.tods.Tod.set_noise`.
    outdir : str
        The output directory.
    npass : int
        The number of times to mapmake and then reestimate the noise.
    dograd : bool
        If True make a map based prior to avoid biases from sharp features.
    """
    naive, hits = make_naive(todvec, skymap, outdir)

    # Take 1 over hits map
    ihits = hits.copy()
    ihits.invert()

    # Save weights and noise maps
    _ = make_weights(todvec, skymap, outdir)

    # Setup the mapset
    # For now just include the naive map so we can use it as the initial guess.
    mapset = minkasi.maps.Mapset()
    mapset.add_map(naive)

    # run PCG to solve for a first guess
    iters = [5, 25, 100]
    mapset = solve_map(todvec, mapset, ihits, None, 26, iters, outdir, "initial")

    # Now we iteratively solve and reestimate the noise
    for niter in range(npass):
        maxiter = 26 + 25 * (niter + 1)
        reestimate_noise_from_map(todvec, mapset, noise_class, noise_args, noise_kwargs)

        # Make a gradient based prior
        if dograd:
            mapset = get_grad_prior(todvec, mapset, hits.copy(), thresh=1.8)
        # Solve
        mapset = solve_map(
            todvec, mapset, ihits, None, maxiter, iters, outdir, f"niter_{niter+1}"
        )

    minkasi.barrier()
