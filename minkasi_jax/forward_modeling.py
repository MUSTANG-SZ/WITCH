"""
Functions for performing forward modeling
"""

import functools

import jax
import jax.numpy as jnp
import minkasi.maps.skymap as skymap
import minkasi.tods.core as todcore
import numpy as np
from minkasi.maps.mapset import Mapset

from minkasi_jax.core import model
from minkasi_jax.utils import make_grid

from .utils import bilinear_interp, rad_to_arcsec
    
@jax.jit
def get_mfilt(m_tod,v,weight):
    m_rot = jnp.dot(v,m_tod)

    m_tmp = jnp.hstack([m_rot,jnp.fliplr(m_rot[:,1:-1])])
    m_rft = jnp.real(jnp.fft.rfft(m_tmp,axis=1))

    m_ift = jnp.fft.irfft(weight*m_rft,axis=1,norm='forward')[:,:m_tod.shape[1]]
    m_irt = jnp.dot(v.T,m_ift)
    m_irt = m_irt.at[:, 0].multiply(0.50)
    m_irt = m_irt.at[:,-1].multiply(0.50)
    return m_irt

@jax.jit
def get_chis(m, dx, dy, xyz, rhs, v, weight, dd=None):
    """
    A faster, but more importantly much less memory intensive, way to get chis.
    The idea is chi^2 = (d-Am)^T N^-1 (d-Am). Previously we would calculate the residuals d-Am
    and calculate directly. However d-Am has shape [ndet, nsamp], which is very big. m has shape
    [nx, ny], much smaller. A, the pointing reconstruction, can be encapsulated into a few pars.
    Therefore if we can do everthing in map shape, we save a lot on the ammount of stuff we
    need to put on the GPU. We can expand chi^2 as

    d^T N^-1 d -2d^T N^-1 A m + m^T A^T N^-1 A m = dd - 2*dot(m, rhs) + mm

    the first term only has to do with the data. If we care about the absolute value of chi2,
    which we do at the end, then we can include it in calculation. For MCMC however, we only
    care about the relative delta chi2 between models. So we can drop that term. For the other
    terms

    mm = m^T A^T N^-1 A m

    This term is essentially what we've been doing before, except that m is now in map shape,
    whereas before m was in tod shape so we essentially had Am. So we need to do Am, but this is
    in general fast and Jon has a very fast way of doing it. Finally rhs need only be computed
    once, and while it is an additional thing to put on the gpu it is small, map shape.

    Parameters
    ----------
    m : NDArray[np.floating]
        The model evaluated at all the map pixels
    dx : NDArray[np.floating]
        RA TOD in the same units as the grid.
    dy : NDArray[np.floating]
        Dec TOD in the same units as the grid.
    xyz : tuple[jax.Array]
        Grid that model was evaluated on, used for interpolation.
    rhs : NDArray[np.floating]
        The map output of todvec.make_rhs. Note this is how the data enters into the chi2 calc.
    v : NDArray[np.floating]
        The right singular vectors for the noise SVD. These rotate the data into the basis of
        the SVD.
    weight : NDArray[np.floating]
        The noise weights, in fourier space, SVD decomposed.
    dd : None | np.floating
        Optional chi2 from dd. Not necessary for MCMC but is important for evaluating goodness
        of fit.

    Outputs
    -------
    chi2 : np.floating
        The chi2 of the model m to the data.
    """

    m_tod = bilinear_interp(dx,dy,xyz[0].ravel(),xyz[1].ravel(),m)
    m_irt = get_mfilt(m_tod,v,weight)
    
    chisq =  -2.00*jnp.sum(rhs*m)+jnp.sum(m_irt*m_tod)
    if dd is not None: chisq += dd
    return chisq

def sampler(theta, tods, jsample, fixed_pars, fix_pars_ids):
    _theta = np.zeros(len(theta) + len(fixed_pars))

    par_idx = 0
    fix_idx = 0
    for i in range(len(_theta)):
        if i in fix_pars_ids:
            _theta[i] = fixed_pars[fix_idx]
            fix_idx += 1
        else:
            _theta[i] = theta[par_idx]
            par_idx += 1

    return jsample(_theta, tods)


def sample(cur_model, theta, tods):  # , model_params, xyz, beam):
    """
    Generate a model realization and compute the chis of that model to data.

    Arguements:

        tods: Array of tod parameters. See prep tods

        params: model parameters

        model_params: number of each model componant

        xyz: grid to evaluate model at

        beam: Beam to smooth by

    Returns:

        chi2: the chi2 difference of the model to the tods

    """
    log_like = 0

    m = model(
        cur_model.xyz,
        *cur_model.n_struct,
        cur_model.dz, 
        cur_model.beam,
        *theta,
    )

    for i, tod in enumerate(tods):
        x, y, rhs, v, weight, norm, dd = tod  # unravel tod

        log_like += -0.50 * (jget_chis(m, x, y, cur_model.xyz, rhs, v, weight, dd) + norm)

    #   model_tod = data-bilinear_interp(x, y, cur_model.xyz[0].ravel(), cur_model.xyz[1].ravel(),m)
    #   model_rot = jnp.dot(v, model_tod)
    #   model_tmp = jnp.hstack([model_rot, jnp.fliplr(model_rot[:, 1:-1])])
    #   model_rft = jnp.real(jnp.fft.rfft(model_tmp, axis=1))

    #   log_like += -0.50*jnp.sum(weight*model_rft**2)
    return log_like


jget_chis = jax.jit(get_chis)


def make_tod_stuff(
    todvec, skymap, lims=None, pixsize=2.0 / 3600 * np.pi / 180, x0=0.0, y0=0.0
):
    tods = []
    if lims == None:
        lims = todvec.lims()
    refmap = skymap.copy()

    for i, tod in enumerate(todvec.tods):
        temp_todvec = todcore.TodVec()
        temp_todvec.add_tod(tod)
        mapset = Mapset()
        refmap.clear()
        mapset.add_map(refmap)
        temp_todvec.make_rhs(mapset)

        # todgrid = refmap.wcs.wcs_world2pix(np.array([np.rad2deg(tod.info['dx'].flatten()),
        #                                            np.rad2deg(tod.info['dy'].flatten())]).T,1)
        # di = todgrid[:,0].reshape(tod.get_data_dims()).flatten()
        # dj = todgrid[:,1].reshape(tod.get_data_dims()).flatten()
        # un wrap stuff cause jit doesn't like having tod objects
        norm = -np.sum(
            np.log(tod.noise.mywt[tod.noise.mywt != 0.00]) - np.log(2.00 * np.pi)
        )
        
        v  = jnp.array(tod.noise.v)
        wt = jnp.array(tod.noise.mywt)
        
        dd = jnp.sum(get_mfilt(tod.info["dat_calib"],v,wt)*tod.info["dat_calib"])

        tods.append(
            [  # jnp.array(di),
                # jnp.array(dj),
                (jnp.array(tod.info["dx"]) - x0) * rad_to_arcsec * jnp.cos(y0),
                (jnp.array(tod.info["dy"]) - y0) * rad_to_arcsec,
                jnp.array(jnp.flipud(mapset.maps[0].map.copy())),
                v,
                wt,
                norm,
                dd
            ]
        )
    return tods
