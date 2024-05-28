"""
Functions that wrap useful minkasi recipes
"""
import os
from typing import Optional

import minkasi
import numpy as np


def make_naive(
    todvec: minkasi.tods.TodVec, skymap: minkasi.maps.MapType, outdir: str
) -> tuple[minkasi.maps.MapType, minkasi.maps.MapType]:
    """
    Make a naive map where we just bin common mode subtracted TODs.

    Arguments:

        todvec: The TODs to mapmake.

        skymap: Map to use as footprint for outputs.

    Returns:

        naive: The navie map.

        hits: The hit count map.
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

    return naive, hits


def make_weights(
    todvec: minkasi.tods.TodVec, skymap: minkasi.maps.MapType, outdir: str
) -> tuple[minkasi.maps.MapType, minkasi.maps.MapType]:
    """
    Make weights and noise map.

    Arguments:

        todvec: The TODs to mapmake.

        skymap: Map to use as footprint for outputs.

    Returns:

        weightmap: The weights map.

        noisemap: The noise map.
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
    noise_args: list,
    noise_kwargs: dict,
):
    """
    Use the current guess at the map to reestimate the noise:

    Arguments:

        todvec: The TODs to reestimate noise for.

        mapset: Mapset containing the current map solution.

        noise_class: Which noise model to use.

        noise_args: Additional arguments to pass to set_noise.

        noise_kwargs: Additional keyword argmuents to pass to set_noise.
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
    gradmap: minkasi.maps.Maptype,
    *args,
    **kwargs,
) -> tuple[minkasi.mapmaking.HasPrior, minkasi.maps.Mapset]:
    """
    Make a gradient based prior. This helps avoid errors due to sharp features.

    Arguments:

        todvec: The TODs what we are mapmaking.

        mapset: The mapset to compute priors with.
                We assume that the first element is the map we care about.

        gradmap: Containter to use as the gradient map.

        *args: Additional arguments to pass to get_grad_mask_2d.

        **kwargs: Kewword arguments to pass to get_grad_mask_2d.

    Returns:

        prior: A prior to pass to run_pcg_wprior.

        new_mapset: A mapset with the original map and a cleared prior map.
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

    return prior, new_mapset


def solve_map(
    todvec: minkasi.tods.TodVec,
    x0: minkasi.maps.Mapset,
    ihits: minkasi.maps.MapType,
    prior: Optional[minkasi.mapmaking.HasPrior],
    maxiters: int,
    save_iters: list[int],
    outdir: str,
    desc_str: str,
) -> minkasi.maps.Mapset:
    """
    Solve for map with PCG.

    Arguments:

        todvec: The TODs what we are mapmaking.

        x0: The initial guess mapset.

        ihits: The inverse hits map.

        prior: Prior to use when mapmaking, set to None to not use.

        maxiters: Maximum PCG iters to use.

        save_iters: Which iterations to save the map at.

        outdir: The output directory

        desc_str: String used to deterime outroot.

    Returns:

        mapset: The mapset with the solved map.
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
