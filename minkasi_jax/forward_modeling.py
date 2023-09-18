"""
Functions for performing forward modeling
"""

import jax
import jax.numpy as jnp
import numpy as np

from minkasi_jax.core import model
from minkasi_jax.utils import make_grid

import functools

def construct_sampler(model_params, xyz, beam):
    cur_sample = functools.partial(sample, model_params, xyz, beam)
    jsample = jax.jit(cur_sample)

    return jsample

def sample(model_params, xyz, beam, params, tods):#, model_params, xyz, beam):
    """
    Generate a model realization and compute the chis of that model to data.
    TODO: model components currently hard coded.

    Arguements:

        tods: Array of tod parameters. See prep tods

        params: model parameters

        model_params: number of each model componant 

        xyz: grid to evaluate model at

        beam: Beam to smooth by

    Returns:

        chi2: the chi2 difference of the model to the tods

    """
    chi2 = 0
    n_iso, n_gnfw, n_gauss, n_uni, n_expo, n_power, n_power_cos = model_params
    for i, tod in enumerate(tods):
        idx_tod, idy_tod, dat, v, weight, id_inv = tod #unravel tod
   
        pred = model(xyz, n_iso, n_gnfw, n_gauss, n_uni, n_expo, n_power, n_power_cos,
                     -2.5e-05, beam, idx_tod, idy_tod, params)
    
        pred = pred[id_inv].reshape(dat.shape)
        chi2 += jget_chis(dat, pred, v, weight)

    return chi2

def get_chis(dat, pred, v, weight):
    """
    Compute the chi2 difference of data and model given some weights.

    Arguments:

        data: the data

        pred: the model prediction

        v: tod.noise.v ?

        weights: data weights

    Returns:

        chis2: the chi2 difference of the model to the data given the weights

    """


    resid = dat - pred #TODO: Due to time to put arrays on gpu it's faster to compute resid
                       #in numpy and then pass to get_chis. Most time this will itself be inside
                       #a jitted function so it won't matter but for the times it's not it will
                       #be a good speedup
    resid = resid.at[:,0].set((jnp.sqrt(0.5)*resid)[:,0])
    resid = resid.at[:,-1].set((jnp.sqrt(0.5)*resid)[:,-1])
    resid_rot = jnp.dot(v,resid)
    tmp=jnp.hstack([resid_rot,jnp.fliplr(resid_rot[:,1:-1])]) #mirror pred so we can do dct of first kind
    predft=jnp.real(jnp.fft.rfft(tmp,axis=1))
    nn=predft.shape[1]
    chisq = jnp.sum(weight[:,:nn]*predft**2)
    return chisq

jget_chis = jax.jit(get_chis)


def make_tod_stuff(todvec):
    """
    Generate variables needed for sample from a todvec. TODO: since we're moving away from
    indexing in sample, this function can be greatly simplified. Note it should generally only be    run once so there's no point in jitting it.
    

    Arguments:
        
        todvec: a minkasi todvec object

    Returns:

        tods: a list of arrays with tod parameters that are required by sample    
    """
    tods = []

    for i,tod in enumerate(todvec.tods):

        #un wrap stuff cause jit doesn't like having tod objects
        tods.append([jnp.array(tod.info["idx"]), jnp.array(tod.info["idy"]),
                     jnp.array(tod.info["dat_calib"]), jnp.array(tod.noise.v),
                     jnp.array(tod.noise.mywt), jnp.array(tod.info["id_inv"])])

    return tods
