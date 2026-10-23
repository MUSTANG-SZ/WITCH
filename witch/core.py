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

from .grid import transform_grid
from .structure import STRUCT_FUNCS, STRUCT_N_NONPARA, STRUCT_N_PAR, STRUCT_STAGE

ORDER = (
    "nonpara_power",
    "isobeta",
    "qso",
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

ORDER_DICT = {name: i for i, name in enumerate(ORDER)}

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


# Spherical stage 0 structures. All take (dx, dy, dz, ...) as their first parameters.
# When there is no stage 1 (which needs the full 3D pressure), the LOS integral of
# these only depends on the projected radius so it can be tabulated in 1D.
SPHERICAL = ("sph_isobeta", "gnfw", "gnfw_rs", "a10")
FAST_LOS = True  # Set False to always integrate on the full 3D grid
N_LOS_TABLE = 4096  # Number of radial nodes in the 1D LOS integral table


def _catmull_rom(table: jax.Array, t: jax.Array) -> jax.Array:
    """
    Cubic (Catmull-Rom) interpolation on a uniformly spaced table.

    Parameters
    ----------
    table : jax.Array
        1D table of values at nodes `0, 1, ..., len(table) - 1`.
    t : jax.Array
        Positions to interpolate at in units of node spacing, must be in `[0, len(table) - 1]`.

    Returns
    -------
    vals : jax.Array
        Interpolated values with the shape of `t`.
    """
    n = table.shape[0]
    # Reflect about R=0 (profiles are even in R) and extrapolate linearly past the end
    padded = jnp.concatenate([table[1:2], table, 2 * table[-1:] - table[-2:-1]])
    i = jnp.clip(jnp.floor(t).astype(int), 0, n - 2)
    f = t - i
    p0, p1, p2, p3 = (padded[i + k] for k in range(4))
    return 0.5 * (
        2 * p1
        + (p2 - p0) * f
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f**2
        + (3 * (p1 - p2) + p3 - p0) * f**3
    )


def _los_spherical(
    struct: str,
    pars: jax.Array,
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    dz: float,
) -> jax.Array:
    """
    Line of sight integral of a spherical structure using a 1D radial table.
    The table is built with the structure function itself on a synthetic grid
    (same LOS samples and trapezoid weights as the 3D grid), then interpolated
    to the projected radius of every pixel.

    Parameters
    ----------
    struct : str
        Name of the spherical structure, must be in `SPHERICAL`.
    pars : jax.Array
        Parameters of the structure, in the order of the structure function.
    xyz : tuple[jax.Array, jax.Array, jax.Array, float, float]
        Grid to compute model on.
        See `containers.Model.xyz` for details.
    dz : float
        Factor to scale by while integrating.

    Returns
    -------
    ip : jax.Array
        Integrated profile with shape `(xyz[0].shape[0], xyz[1].shape[1])`.
    """
    x, y, *_ = transform_grid(pars[0], pars[1], 0.0, 1, 1, 1, 0, xyz)
    rad = jnp.sqrt(x**2 + y**2)[..., 0]
    rmax = jax.lax.stop_gradient(jnp.max(rad)) * (1 + 1e-6)
    radii = jnp.linspace(0.0, rmax, N_LOS_TABLE)
    xyz_t = (radii[:, None, None], jnp.zeros((1, 1, 1)), xyz[2], 0.0, 0.0)
    pars_t = [0.0, 0.0] + [pars[k] for k in range(2, len(pars))]
    table = trapz(STRUCT_FUNCS[struct](*pars_t, xyz_t), dx=dz, axis=-1)[:, 0]
    return _catmull_rom(table, rad / rmax * (N_LOS_TABLE - 1))


def _stage_0(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    params: jax.Array,
    start: int,
    pressure: jax.Array,
    run: bool = True,
    los: tuple[float, jax.Array] | None = None,
) -> tuple[jax.Array, int, jax.Array | None]:
    # If los is given as (dz, ip) spherical structures are integrated with
    # `_los_spherical` and added to ip instead of the 3D pressure
    ip = None if los is None else los[1]
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
            if los is not None and struct in SPHERICAL:
                ip = jnp.add(ip, _los_spherical(struct, struct_pars[i], xyz, los[0]))
            else:
                pressure = jnp.add(pressure, STRUCT_FUNCS[struct](*struct_pars[i], xyz))
    return pressure, int(start), ip


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
    params = jnp.array(params)
    params = jnp.ravel(params)  # Fixes strange bug with params having dim (1,n)

    pressure = jnp.zeros((xyz[0].shape[0], xyz[1].shape[1], xyz[2].shape[2]))
    start = 0

    # Stage -1, non para
    pressure, start = _stage_m1(xyz, n_structs, n_rbins, params, start, pressure, True)

    # Stage 0, add to the 3d grid
    pressure, start, _ = _stage_0(xyz, n_structs, params, start, pressure, True)

    # Stage 1, modify the 3d grid
    pressure, start = _stage_1(xyz, n_structs, params, start, pressure, True)

    return pressure


def make_to_run(
    stage_m1: bool = True,
    stage_0: bool = True,
    stage_1: bool = True,
    stage_2: bool = True,
) -> tuple[bool, bool, bool, bool]:
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
        This is 2D structure.

    Returns
    -------
    to_run : tuple[bool, bool, bool, bool, bool, bool]
        Which stages to run in the order needed for `model` and `model_grad`.
    """
    return (stage_m1, stage_0, stage_1, stage_2)


def model(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    n_rbins: tuple[int, ...],
    dz: float,
    to_run: tuple[bool, bool, bool, bool],
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
    to_run : tuple[bool, bool, bool, bool]
        Which stages to run.
        Should be a 5 element tuple of bools, see `make_to_run` for details.
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

    # Stage 1 needs the full 3D pressure, if there is none spherical stage 0
    # structures can use the 1D LOS table instead
    has_stage_1 = any(
        n and STRUCT_STAGE[struct] == 1 for n, struct in zip(n_structs, ORDER)
    )
    fast_los = FAST_LOS and not has_stage_1
    ip_los = jnp.zeros(pressure.shape[:2]) if fast_los else None

    # Stage 0, add to the 3d grid
    pressure, start, ip_los = _stage_0(
        xyz,
        n_structs,
        params,
        start,
        pressure,
        to_run[1],
        (dz, ip_los) if fast_los else None,
    )

    # Stage 1, modify the 3d grid
    pressure, start = _stage_1(xyz, n_structs, params, start, pressure, to_run[2])

    # Integrate along line of site
    ip = trapz(pressure, dx=dz, axis=-1)
    if fast_los:
        ip = ip + ip_los

    # Stage 2, add to integrated profile
    ip, start = _stage_2(xyz, n_structs, params, start, ip, to_run[3])

    return ip


def model_grad(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    n_rbins: tuple[int, ...],
    dz: float,
    to_run: tuple[bool, bool, bool, bool],
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
        to_run,
        *pars,
    )

    grad_padded = jnp.zeros((len(pars),) + pred.shape)
    if len(argnums) > 0:
        grad = jax.jacfwd(model, argnums=argnums)(
            xyz,
            n_structs,
            n_rbins,
            dz,
            to_run,
            *pars,
        )
        grad_padded = grad_padded.at[jnp.array(argnums) - ARGNUM_SHIFT].set(
            jnp.array(grad)
        )

    return pred, grad_padded


def stage2_model(
    xyz: tuple[jax.Array, jax.Array, jax.Array, float, float],
    n_structs: tuple[int, ...],
    dz: float,
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
    pressure, start, _ = _stage_0(xyz, n_structs, params, start, pressure, False)

    # Stage 1, modify the 3d grid
    pressure, start = _stage_1(xyz, n_structs, params, start, pressure)

    # Integrate along line of site
    ip = trapz(pressure, dx=dz, axis=-1)

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
