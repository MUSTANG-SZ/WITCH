"""
Core module for generating models and their gradients.
"""

import inspect

import jax
import jax.numpy as jnp
from typing_extensions import Unpack

if hasattr(jnp, "trapz"):
    trapz = jnp.trapz
else:
    from jax.scipy.integrate import trapezoid as trapz

import numpy as np

from .structure import STRUCT_FUNCS, STRUCT_N_PAR, STRUCT_STAGE
from .utils import fft_conv

ORDER = (
    "isobeta",
    "gnfw",
    "a10",
    "ea10",
    "cylindrical_beta",
    "egaussian",
    "uniform",
    "exponential",
    "powerlaw",
    "powerlaw_cos",
    "gaussian",
)

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")


def _get_static(signature, prefix_list=["n_", "argnums"]):
    par_names = np.array(list(signature.parameters.keys()), dtype=str)
    static_msk = np.zeros_like(par_names, dtype=bool)
    for prefix in prefix_list:
        static_msk += np.char.startswith(par_names, prefix)
    return tuple(np.where(static_msk)[0])


def _check_order():
    or_uniq, or_cts = np.unique(ORDER, return_counts=True)
    if len(or_uniq) != len(ORDER):
        raise ValueError(f"Non-unique entries found in ORDER: {or_uniq[or_cts > 1]}")
    for name, struct_dict in zip(
        ["STRUCT_FUNCS", "STRUCT_N_PAR", "STRUCT_STAGE"],
        [STRUCT_FUNCS, STRUCT_N_PAR, STRUCT_STAGE],
    ):
        keys = list(struct_dict.keys())
        or_missing = np.setdiff1d(keys, ORDER, True)
        if len(or_missing):
            raise ValueError(f"ORDER missing entries: {or_missing}")
        sd_missing = np.setdiff1d(ORDER, keys, True)
        if len(sd_missing):
            raise ValueError(f"{name} missing entries: {sd_missing}")
    stages = [STRUCT_STAGE[struct] for struct in ORDER]
    if not np.array_equal(stages, np.sort(stages)):
        raise ValueError("ORDER seems to have elements with stages out of order")


def stage2_model(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    dz: float,
    beam: jax.Array,
    *pars: Unpack[tuple[float, ...]],
):
    """
    Only returns the second stage of the model. Used for visualizing shocks, etc.
    that can otherwise be hard to see in a model plot

    Parameters
    ----------
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Grid to compute model on.
        See `containers.Model.xyz` for details.
    n_structs : tuple[int, ...]
        Number of each structure to use.
        Should be in the same order as `order`.
    dz : float
        Factor to scale by while integrating.
        Should at least include the pixel size along the LOS.
    beam : jax.Array
        Beam to convolve by, should be a 2d array.
    *pars : Unpack[tuple[float,...]]
        1D container of model parameters.

    Returns
    -------
    model : jax.Array
        The model with the specified substructure evaluated on the grid.
        No stage 3 structures are included.
    """
    params = jnp.array(pars)
    params = jnp.ravel(params)  # Fixes strange bug with params having dim (1,n)

    pressure = jnp.ones((xyz[0].shape[0], xyz[1].shape[1], xyz[2].shape[2]))
    start = 0

    # Stage 0, track delta but don't add anything
    for n_struct, struct in zip(n_structs, ORDER):
        if STRUCT_STAGE[struct] != 0:
            continue
        if not n_struct:
            continue
        delta = n_struct * STRUCT_N_PAR[struct]
        struct_pars = params[start : start + delta].reshape(
            (n_struct, STRUCT_N_PAR[struct])
        )
        start += delta

    # Stage 1, modify the 3d grid
    for n_struct, struct in zip(n_structs, ORDER):
        if STRUCT_STAGE[struct] != 1:
            continue
        if not n_struct:
            continue

        delta = n_struct * STRUCT_N_PAR[struct]
        struct_pars = params[start : start + delta].reshape(
            (n_struct, STRUCT_N_PAR[struct])
        )

        start += delta
        for i in range(n_struct):
            pressure = STRUCT_FUNCS[struct](pressure, xyz, *struct_pars[i])

    # Integrate along line of site
    ip = trapz(pressure, dx=dz, axis=-1)

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

    return ip


def model(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    dz: float,
    beam: jax.Array,
    *pars: Unpack[tuple[float, ...]],
):
    """
    Generically create models with substructure.

    Parameters
    ----------
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Grid to compute model on.
        See `containers.Model.xyz` for details.
    n_structs : tuple[int, ...]
        Number of each structure to use.
        Should be in the same order as `order`.
    dz : float
        Factor to scale by while integrating.
        Should at least include the pixel size along the LOS.
    beam : jax.Array
        Beam to convolve by, should be a 2d array.
    *pars : Unpack[tuple[float,...]]
        1D container of model parameters.

    Returns
    -------
    model : jax.Array
        The model with the specified substructure evaluated on the grid.
    """
    params = jnp.array(pars)
    params = jnp.ravel(params)  # Fixes strange bug with params having dim (1,n)

    pressure = jnp.zeros((xyz[0].shape[0], xyz[1].shape[1], xyz[2].shape[2]))
    start = 0

    # Stage 0, add to the 3d grid
    for n_struct, struct in zip(n_structs, ORDER):
        if STRUCT_STAGE[struct] != 0:
            continue
        if not n_struct:
            continue
        delta = n_struct * STRUCT_N_PAR[struct]
        struct_pars = params[start : start + delta].reshape(
            (n_struct, STRUCT_N_PAR[struct])
        )
        start += delta
        for i in range(n_struct):
            pressure = jnp.add(pressure, STRUCT_FUNCS[struct](*struct_pars[i], xyz))

    # Stage 1, modify the 3d grid
    for n_struct, struct in zip(n_structs, ORDER):
        if STRUCT_STAGE[struct] != 1:
            continue
        if not n_struct:
            continue
        delta = n_struct * STRUCT_N_PAR[struct]
        struct_pars = params[start : start + delta].reshape(
            (n_struct, STRUCT_N_PAR[struct])
        )
        start += delta
        for i in range(n_struct):
            pressure = STRUCT_FUNCS[struct](pressure, xyz, *struct_pars[i])

    # Integrate along line of site
    ip = trapz(pressure, dx=dz, axis=-1)

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

    # Stage 2, add to the integrated profile
    for n_struct, struct in zip(n_structs, ORDER):
        if STRUCT_STAGE[struct] != 2:
            continue
        if not n_struct:
            continue
        delta = n_struct * STRUCT_N_PAR[struct]
        struct_pars = params[start : start + delta].reshape(
            (n_struct, STRUCT_N_PAR[struct])
        )
        start += delta
        for i in range(n_struct):
            ip = jnp.add(ip, STRUCT_FUNCS[struct](*struct_pars[i], xyz))

    return ip


def model_grad(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    dz: float,
    beam: jax.Array,
    argnums: tuple[int, ...],
    *pars: Unpack[tuple[float, ...]],
):
    """
    A wrapper around model that also returns the gradients of the model.
    Only the additional arguments are described here, see `model` for the others.
    Note that the additional arguments are passed **before** the *params argument.

    Parameters
    ----------
    argnums : tuple[int,...]
        The indices of the arguments to evaluate the gradient at.

    Returns
    -------
    model : jax.Array
        The model with the specified substructure evaluated on the grid.
    grad : jax.Array
        The gradient of the model with respect to the model parameters.
        Has shape `(len(pars),) + model.shape)`.
    """
    pred = model(
        xyz,
        n_structs,
        dz,
        beam,
        *pars,
    )

    grad = jax.jacfwd(model, argnums=argnums)(
        xyz,
        n_structs,
        dz,
        beam,
        *pars,
    )
    grad_padded = jnp.zeros((len(pars),) + pred.shape)
    grad_padded = grad_padded.at[jnp.array(argnums) - ARGNUM_SHIFT].set(jnp.array(grad))

    return pred, grad_padded


# Check that ORDER is ok...
_check_order()

# Do some signature inspection to avoid hard coding
model_sig = inspect.signature(model)
model_grad_sig = inspect.signature(model_grad)

# Get argnum shifts, -1 is for param
ARGNUM_SHIFT = len(model_sig.parameters) - 1

# Figure out static argnums
model_static = _get_static(model_sig)
model_grad_static = _get_static(model_grad_sig)

# Now JIT
model = jax.jit(model, static_argnums=model_static)
model_grad = jax.jit(model_grad, static_argnums=model_grad_static)
