"""
Core module for generating models aed their gradients.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .structure import (
    add_exponential,
    add_powerlaw,
    add_powerlaw_cos,
    add_uniform,
    gaussian,
    gnfw,
    isobeta,
)
from .utils import fft_conv

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")

N_PAR_ISOBETA = 9
N_PAR_GNFW = 15
N_PAR_GAUSSIAN = 9
N_PAR_UNIFORM = 8
N_PAR_EXPONENTIAL = 14
N_PAR_POWERLAW = 11

ARGNUM_SHIFT = 12


def helper(
    params,
    tod,
    xyz,
    dx,
    beam,
    argnums,
    re_eval,
    par_idx,
    n_isobeta=0,
    n_gnfw=0,
    n_gaussian=0,
    n_uniform=0,
    n_exponential=0,
    n_powerlaw=0,
    n_powerlaw_cos=0,
):
    """
    Helper function to be used when fitting with Minkasi.
    Use functools.partial to set all parameters but params and tod before passing to Minkasi.

    Arguments:

        params: 1D array of model parameters.

        tod: The TOD, assumed that idx and idy are in tod.info.
             Optionally also include id_inv. In this case it is assumed that
             idx and idy are flattened and passed through np.unique such that
             each (idx[i], idy[i]) is unique and id_inv is the inverse index
             mapping obtained by setting return_inverse=True.

        xyz: Coordinate grid to compute profile on.

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        argnums: Arguments to evaluate the gradient at.

        re_eval: Array where each element is eather False or a string.
                 If element is a string it will be evaluated and used to
                 set the value of the corresponsinding parameter.

        par_idx: Dictionairy that maps parameter names to indices.

        n_isobeta: Number of isobeta profiles to add.

        n_gnfw: Number of gnfw profiles to add.

        n_gaussian: Number of gaussians to add.

        n_uniform: Number of uniform ellipsoids to add.

        n_exponential: Number of exponential ellipsoids to add.

        n_powerlaw: Number of power law ellipsoids to add.

        n_powerlaw_cos: Number of radial power law ellipsoids with angulas cos term to add.

    Returns:

        grad: The gradient of the model with respect to the model parameters.
                Reshaped to be the correct format for Minkasi.

        pred: The isobeta model with the specified substructure.
    """
    idx = tod.info["idx"]
    idy = tod.info["idy"]

    for i, re in enumerate(re_eval):
        if re is False:
            continue
        params[i] = eval(re)

    pred, grad = model_grad(
        xyz,
        n_isobeta,
        n_gnfw,
        n_gaussian,
        n_uniform,
        n_exponential,
        n_powerlaw,
        n_powerlaw_cos,
        dx,
        beam,
        idx,
        idy,
        tuple(argnums + ARGNUM_SHIFT),
        *params,
    )
    pred = jax.device_get(pred)
    grad = jax.device_get(grad)

    if "id_inv" in tod.info:
        id_inv = tod.info["id_inv"]
        shape = tod.info["dx"].shape
        pred = pred[id_inv].reshape(shape)
        grad = grad[:, id_inv].reshape((len(grad),) + shape)

    return grad, pred


@partial(
    jax.jit,
    static_argnums=(1, 2, 3, 4, 5, 6, 7, 8),
)
def model(
    xyz,
    n_isobeta,
    n_gnfw,
    n_gaussian,
    n_uniform,
    n_exponential,
    n_powerlaw,
    n_powerlaw_cos,
    dx,
    beam,
    idx,
    idy,
    *params,
):
    """
    Generically create models with substructure.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_isobeta: Number of isobeta profiles to add.

        n_gnfw: Number of gnfw profiles to add.

        n_gaussian: Number of gaussians to add.

        n_uniform: Number of uniform ellipsoids to add.

        n_exponential: Number of exponential ellipsoids to add.

        n_powerlaw: Number of power law ellipsoids to add.

        n_powerlaw_cos: Number of radial power law ellipsoids with angulas cos term to add.

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        idx: RA TOD in units of pixels.
             Should have Dec stretch applied.

        idy: Dec TOD in units of pixels.

        params: 1D array of model parameters.

    Returns:

        model: The model with the specified substructure.
    """
    params = jnp.array(params)
    params = jnp.ravel(params)  # Fixes strange bug with params having dim (1,n)
    isobetas = jnp.zeros((1, 1), dtype=float)
    gnfws = jnp.zeros((1, 1), dtype=float)
    gaussians = jnp.zeros((1, 1), dtype=float)
    uniforms = jnp.zeros((1, 1), dtype=float)
    exponentials = jnp.zeros((1, 1), dtype=float)
    powerlaws = jnp.zeros((1, 1), dtype=float)
    powerlaw_coses = jnp.zeros((1, 1), dtype=float)

    start = 0
    if n_isobeta:
        delta = n_isobeta * N_PAR_ISOBETA
        isobetas = params[start : start + delta].reshape((n_isobeta, N_PAR_ISOBETA))
        start += delta
    if n_gnfw:
        delta = n_gnfw * N_PAR_GNFW
        gnfws = params[start : start + delta].reshape((n_gnfw, N_PAR_GNFW))
        start += delta
    if n_gaussian:
        delta = n_gaussian * N_PAR_GAUSSIAN
        gaussians = params[start : start + delta].reshape((n_gaussian, N_PAR_GAUSSIAN))
        start += delta
    if n_uniform:
        delta = n_uniform * N_PAR_UNIFORM
        uniforms = params[start : start + delta].reshape((n_uniform, N_PAR_UNIFORM))
        start += delta
    if n_exponential:
        delta = n_exponential * N_PAR_EXPONENTIAL
        exponentials = params[start : start + delta].reshape(
            (n_exponential, N_PAR_EXPONENTIAL)
        )
        start += delta
    if n_powerlaw:
        delta = n_powerlaw * N_PAR_POWERLAW
        powerlaws = params[start : start + delta].reshape((n_powerlaw, N_PAR_POWERLAW))
        start += delta
    if n_powerlaw_cos:
        delta = n_powerlaw_cos * N_PAR_POWERLAW
        powerlaw_coses = params[start : start + delta].reshape(
            (n_powerlaw_cos, N_PAR_POWERLAW)
        )
        start += delta

    pressure = jnp.zeros((xyz[0].shape[1], xyz[1].shape[0], xyz[2].shape[2]))
    for i in range(n_isobeta):
        pressure = jnp.add(pressure, isobeta(*isobetas[i], xyz))

    for i in range(n_gnfw):
        pressure = jnp.add(pressure, gnfw(*gnfws[i], xyz))

    for i in range(n_gaussian):
        pressure = jnp.add(pressure, gaussian(*gaussians[i], xyz))

    for i in range(n_uniform):
        pressure = add_uniform(pressure, xyz, *uniforms[i])

    for i in range(n_exponential):
        pressure = add_exponential(pressure, xyz, *exponentials[i])

    for i in range(n_powerlaw):
        pressure = add_powerlaw(pressure, xyz, *powerlaws[i])

    for i in range(n_powerlaw_cos):
        pressure = add_powerlaw_cos(pressure, xyz, *powerlaw_coses[i])

    # Integrate along line of site
    ip = jnp.trapz(pressure, dx=dx, axis=-1)

    bound0, bound1 = int((ip.shape[0] - beam.shape[0]) / 2), int(
        (ip.shape[1] - beam.shape[1]) / 2
    )
    beam = jnp.pad(
        beam,
        (
            (bound0, ip.shape[0] - beam.shape[0] - bound0),
            (bound1, ip.shape[1] - beam.shape[1] - bound1),
        ),
    )

    ip = fft_conv(ip, beam)

    model_out = ip.at[idy.ravel(), idx.ravel()].get(mode="fill", fill_value=0)
    return model_out.reshape(idx.shape)


@partial(
    jax.jit,
    static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 12),
)
def model_grad(
    xyz,
    n_isobeta,
    n_gnfw,
    n_gaussian,
    n_uniform,
    n_exponential,
    n_powerlaw,
    n_powerlaw_cos,
    dx,
    beam,
    idx,
    idy,
    argnums,
    *params,
):
    """
    Generically create models with substructure and get their gradients.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_isobeta: Number of isobeta profiles to add.

        n_gnfw: Number of gnfw profiles to add.

        n_gaussian: Number of gaussians to add.

        n_uniform: Number of uniform ellipsoids to add.

        n_exponential: Number of exponential ellipsoids to add.

        n_powerlaw: Number of power law ellipsoids to add.

        n_powerlaw_cos: Number of radial power law ellipsoids with angulas cos term to add.

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        idx: RA TOD in units of pixels.
             Should have Dec stretch applied.

        idy: Dec TOD in units of pixels.

        argnums: The arguments to evaluate the gradient at

        params: 1D array of model parameters.

    Returns:

        model: The model with the specified substructure.

        grad: The gradient of the model with respect to the model parameters.
    """
    pred = model(
        xyz,
        n_isobeta,
        n_gnfw,
        n_gaussian,
        n_uniform,
        n_exponential,
        n_powerlaw,
        n_powerlaw_cos,
        dx,
        beam,
        idx,
        idy,
        *params,
    )

    grad = jax.jacfwd(model, argnums=argnums)(
        xyz,
        n_isobeta,
        n_gnfw,
        n_gaussian,
        n_uniform,
        n_exponential,
        n_powerlaw,
        n_powerlaw_cos,
        dx,
        beam,
        idx,
        idy,
        *params,
    )
    grad_padded = jnp.zeros((len(params),) + idx.shape)
    grad_padded = grad_padded.at[jnp.array(argnums) - ARGNUM_SHIFT].set(jnp.array(grad))

    return pred, grad_padded
