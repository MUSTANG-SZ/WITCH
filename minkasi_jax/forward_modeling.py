"""
Functions for performing forward modeling
"""

import jax
import jax.numpy as jnp
import numpy as np

from minkasi_jax.core import model
from minkasi_jax.utils import make_grid

import minkasi.maps.skymap as skymap
import minkasi.tods.core as todcore
from minkasi.maps.mapset import Mapset

import functools

@jax.jit
def get_chis(m, x, y, rhs, v, weight, dd = None):
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
    x : NDArray[np.floating]
        2D array of tod x indicies
    y : NDArray[np.floating]
        2D array of tod y indicies
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

    model = jax.scipy.ndimage.map_coordinates(m, (x,y),0).reshape((v.shape[1], -1))

    model = model.at[:,0].set((jnp.sqrt(0.5)*model)[:,0])
    model = model.at[:,-1].set((jnp.sqrt(0.5)*model)[:,-1])
    model_rot = jnp.dot(v,model)
    tmp=jnp.hstack([model_rot,jnp.fliplr(model_rot[:,1:-1])]) #mirror pred so we can do dct of first kind
    predft=jnp.real(jnp.fft.rfft(tmp,axis=1))
    nn=predft.shape[1]

    chisq = jnp.sum(weight[:,:nn]*predft**2) - 2*jnp.dot(rhs.ravel(), m.ravel())
    
    return chisq
    
def sampler(params, tods, jsample, fixed_pars, fix_pars_ids):
    _params = np.zeros(len(params) + len(fixed_pars))
 
    par_idx = 0
    fix_idx = 0
    for i in range(len(_params)):
        if i in fix_pars_ids:
            _params[i] = fixed_pars[fix_idx]
            fix_idx += 1
        else:
            _params[i] = params[par_idx]
            par_idx += 1

    return jsample(_params, tods)

def sample(model_params, xyz, beam, x_map, y_map, params, tods):#, model_params, xyz, beam):
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
    n_iso, n_gnfw, n_gauss, n_egauss, n_uni, n_expo, n_power, n_power_cos = model_params

    m = model(xyz, n_iso, n_gnfw, n_gauss, n_egauss, n_uni, n_expo, n_power, n_power_cos,
                     -2.5e-05, beam, x_map, y_map, params)

    for i, tod in enumerate(tods):
        x, y, rhs, v, weight, norm = tod #unravel tod
       
        log_like += jget_chis(m, x, y, rhs, v, weight) / norm 

    return log_like

jget_chis = jax.jit(get_chis)

def make_tod_stuff(todvec, skymap, lims = None, pixsize =  2.0 / 3600 * np.pi / 180):
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
        
        todgrid = refmap.wcs.wcs_world2pix(np.array([np.rad2deg(tod.info['dx'].flatten()),
                                                    np.rad2deg(tod.info['dy'].flatten())]).T,1)
        di = todgrid[:,0].reshape(tod.get_data_dims()).flatten()
        dj = todgrid[:,1].reshape(tod.get_data_dims()).flatten()
        #un wrap stuff cause jit doesn't like having tod objects
        norm = -np.sum(np.log(tod.noise.mywt[tod.noise.mywt!=0.00])-np.log(2.00*np.pi))

        tods.append([jnp.array(di),
                     jnp.array(dj),
                     jnp.array(mapset.maps[0].map),
                     jnp.array(tod.noise.v),
                     jnp.array(tod.noise.mywt),
                     norm]) 
    return tods


