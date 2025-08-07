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

from .structure import STRUCT_FUNCS, STRUCT_N_NONPARA, STRUCT_N_PAR, STRUCT_STAGE
from .utils import fft_conv

ORDER = (
    "nonpara_power",
    "isobeta",
    "sph_isobeta",
    "gnfw",
    "gnfw_rs",
    "egnfw",
    "a10",
    "ea10",
    "cylindrical_beta",
    "egaussian",
    "uniform",
    "exponential",
    "powerlaw",
    "powerlaw_cos",
    "cylindrical_beta_2d",
    "gaussian",
)

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")


def _get_static(signature, prefix_list=["n_", "argnums", "to_run"]):
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


def _stage_m1(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    n_rbins: tuple[int, ...],
    params: jax.Array,
    start: int,
    pressure: jax.Array,
    run: bool = True,
) -> tuple[jax.Array, int]:
    for i, (n_struct, struct) in enumerate(zip(n_structs, ORDER)):
        if STRUCT_STAGE[struct] != -1:
            continue
        if not n_struct:
            continue
        delta = n_struct * (
            n_rbins[i] * STRUCT_N_NONPARA[struct]
            + STRUCT_N_PAR[struct]
            - STRUCT_N_NONPARA[struct]
        )
        struct_pars = params[start : start + delta].reshape(
            (n_struct, int(delta / n_struct))
        )
        start += delta
        if not run:
            continue
        for j in range(n_struct):
            cur_struct_pars = struct_pars[j]
            nonpara_struct_pars = cur_struct_pars[
                : n_rbins[i] * STRUCT_N_NONPARA[struct]
            ].reshape((STRUCT_N_NONPARA[struct], n_rbins[i]))
            cur_struct_pars = cur_struct_pars[n_rbins[i] * STRUCT_N_NONPARA[struct] :]
            cur_pars = [
                nonpara_struct_pars[k] for k in range(STRUCT_N_NONPARA[struct])
            ] + [
                cur_struct_pars[k]
                for k in range(STRUCT_N_PAR[struct] - STRUCT_N_NONPARA[struct])
            ]
            pressure = jnp.add(pressure, STRUCT_FUNCS[struct](*cur_pars, xyz))
    return pressure, int(start)


def _stage_0(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    params: jax.Array,
    start: int,
    pressure: jax.Array,
    run: bool = True,
) -> tuple[jax.Array, int]:
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
        if not run:
            continue
        for i in range(n_struct):
            pressure = jnp.add(pressure, STRUCT_FUNCS[struct](*struct_pars[i], xyz))
    return pressure, int(start)


def _stage_1(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    params: jax.Array,
    start: int,
    pressure: jax.Array,
    run: bool = True,
) -> tuple[jax.Array, int]:
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
        if not run:
            continue
        for i in range(n_struct):
            pressure = STRUCT_FUNCS[struct](pressure, xyz, *struct_pars[i])
    return pressure, int(start)


def _stage_2(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    params: jax.Array,
    start: int,
    ip: jax.Array,
    run: bool = True,
) -> tuple[jax.Array, int]:
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
        if not run:
            continue
        for i in range(n_struct):
            ip = jnp.add(ip, STRUCT_FUNCS[struct](*struct_pars[i], xyz))
    return ip, int(start)


def _beam_conv(
    ip: jax.Array,
    beam: jax.Array,
    run: bool = True,
) -> jax.Array:
    if not run:
        return ip
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


def _stage_3(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    params: jax.Array,
    start: int,
    ip: jax.Array,
    run: bool = True,
) -> tuple[jax.Array, int]:
    for n_struct, struct in zip(n_structs, ORDER):
        if STRUCT_STAGE[struct] != 3:
            continue
        if not n_struct:
            continue
        delta = n_struct * STRUCT_N_PAR[struct]
        struct_pars = params[start : start + delta].reshape(
            (n_struct, STRUCT_N_PAR[struct])
        )
        start += delta
        if not run:
            continue
        for i in range(n_struct):
            ip = jnp.add(ip, STRUCT_FUNCS[struct](*struct_pars[i], xyz))
    return ip, int(start)


def model3D(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    n_rbins: tuple[int, ...],
    params: jax.Array,
) -> jax.Array:
    """
    Generate a 3D profile from params on xyz.

    Parameters
    ----------
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Grid to compute model on.
        See `containers.Model.xyz` for details.
    n_structs : tuple[int, ...]
        Number of each structure to use.
        Should be in the same order as `order`.
    n_rbins : tuple[int]
        Number of rbins for each non-parametric model
    params : jax.Array
        1D container of model parameters.

    Returns
    -------
    pressure : jax.Array
        The 3D model with the specified substructure evaluated on the grid.
    """
    return model(
        xyz,
        n_structs,
        n_rbins,
        0,
        jnp.array([1]),
        (True, True, True, False, False, False),
        *params,
    )


def make_to_run(
    stage_m1: bool = True,
    stage_0: bool = True,
    stage_1: bool = True,
    stage_2: bool = True,
    beam_conv: bool = True,
    stage_3: bool = True,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """
    Constructs the `to_run` tuple needed for calls to `model` and `model_grad` in the correct order.
    This is genrally prefered over manual construction, if you need to contsruct manualy the order of parameters in
    this function is the order of the `to_run` tuple.
    Any stages that are set to `False` here will not be run when computing the model.

    Parameters
    ----------
    stage_m1 : bool, default: True
        If True run stage -1.
        This is non-parametric models.
    stage_0 : bool, default: True
        If True run stage 0.
        This is 3D parametric models.
    stage_1 : bool, default: True
        If True run stage 1.
        This is 3D sub-structure.
    stage_2 : bool, default: True
        If True run stage 2.
        This is 2D structure added prior to beam convolution.
    beam_conv : bool, default: True
        If True apply beam convolution after stage 2.
    stage_2 : bool, default: True
        If True run stage 3.
        This is 2D structure added after beam convolution.

    Returns
    -------
    to_run : tuple[bool, bool, bool, bool, bool, bool]
        Which stages to run in the order needed for `model` and `model_grad`.
    """
    return (stage_m1, stage_0, stage_1, stage_2, beam_conv, stage_3)


def model(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    n_rbins: tuple[int, ...],
    dz: float,
    beam: jax.Array,
    to_run: tuple[bool, bool, bool, bool, bool, bool],
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
    n_rbins : tuple[int]
        Number of rbins for each non-parametric model
    dz : float
        Factor to scale by while integrating.
        Should at least include the pixel size along the LOS.
    beam : jax.Array
        Beam to convolve by, should be a 2d array.
    to_run : tuple[bool, bool, bool, bool, bool, bool]
        Which stages to run.
        Should be a 6 element tuple of bools, see `make_to_run` for details.
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

    # Stage -1, non para
    pressure, start = _stage_m1(
        xyz, n_structs, n_rbins, params, start, pressure, to_run[0]
    )

    # Stage 0, add to the 3d grid
    pressure, start = _stage_0(xyz, n_structs, params, start, pressure, to_run[1])

    # Stage 1, modify the 3d grid
    pressure, start = _stage_1(xyz, n_structs, params, start, pressure, to_run[2])

    # Integrate along line of site
    ip = trapz(pressure, dx=dz, axis=-1)

    # Stage 2, add non-beam convolved to integrated profile
    ip, start = _stage_2(xyz, n_structs, params, start, ip, to_run[3])

    # Beam conv
    ip = _beam_conv(ip, beam, to_run[4])

    # Stage 3, add beam convolved to the integrated profile
    ip, start = _stage_3(xyz, n_structs, params, start, ip, to_run[5])

    return ip


def model_grad(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    n_rbins: tuple[int, ...],
    dz: float,
    beam: jax.Array,
    to_run: tuple[bool, bool, bool, bool, bool, bool],
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
        n_rbins,
        dz,
        beam,
        to_run,
        *pars,
    )

    grad = jax.jacfwd(model, argnums=argnums)(
        xyz,
        n_structs,
        n_rbins,
        dz,
        beam,
        to_run,
        *pars,
    )
    grad_padded = jnp.zeros((len(pars),) + pred.shape)
    grad_padded = grad_padded.at[jnp.array(argnums) - ARGNUM_SHIFT].set(jnp.array(grad))

    return pred, grad_padded


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

    # Stage 0, but just track param start
    pressure, start = _stage_0(xyz, n_structs, params, start, pressure, False)

    # Stage 1, modify the 3d grid
    pressure, start = _stage_1(xyz, n_structs, params, start, pressure)

    # Integrate along line of site
    ip = trapz(pressure, dx=dz, axis=-1)
    ip = _beam_conv(ip, beam)

    return ip


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
