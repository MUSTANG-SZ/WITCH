import warnings
from copy import deepcopy
from typing import Optional

import jax.numpy as jnp
import numpy as np

from . import core
from . import utils as wu
from .containers import Model, Parameter, Structure
from .powerlaw import profile_to_broken_power


def array_to_tuple(arr):
    if isinstance(arr, list) or type(arr) is np.ndarray:
        return tuple(array_to_tuple(item) for item in arr)
    else:
        return arr


def bin_map(hdu, rbins, x0=None, y0=None, cunit=None):
    """
    Radially bin a map into rbins. Code adapted from CLASS

    Parameters
    ----------
    hdu : fits.HDUList
        hdu containing map to bin
    rbins : NDArray[np.floating]
        Bin edges in radians
    cunit : Union[None, np.floating], Default: None
        Pixel units. If None, will atempt to infer from imap

    Returns
    -------
    bin1d : NDArray[np.floating]
        Bin center values
    var1d : NDArray[np.floating]
        Bin variance estimate
    """

    if cunit is None:
        try:
            cunit = hdu[0].header["CUNIT1"].lower()
        except KeyError as e:
            raise e

    if (
        cunit.lower() == "rad"
        or cunit.lower() == "radian"
        or cunit.lower() == "radians"
    ):
        pixunits = 1
    elif (
        cunit.lower() == "deg"
        or cunit.lower() == "degree"
        or cunit.lower() == "degrees"
    ):
        pixunits = wu.rad_to_deg
    elif (
        cunit.lower() == "arcmin"
        or cunit.lower() == "arcminute"
        or cunit.lower() == "arcminutes"
    ):
        pixunits = wu.rad_to_arcmin
    elif (
        cunit.lower() == "arcsec"
        or cunit.lower() == "arcsecond"
        or cunit.lower() == "arcseconds"
    ):
        pixunits = wu.rad_to_arcsec
    else:
        raise ValueError("Error: cunit {} is not a valid pixel unit".format(cunit))

    pixsize = np.abs(hdu[0].header["CDELT1"]) / pixunits
    x0 = hdu[0].header["CRVAL1"] / pixunits
    y0 = hdu[0].header["CRVAL2"] / pixunits

    if np.abs(hdu[0].header["CDELT1"]) != np.abs(hdu[0].header["CDELT2"]):
        warnings.warn(
            "Warning: non-square pixels: RA: {} Dec{}".format(
                np.abs(hdu[0].header["CDELT1"]), np.abs(hdu[0].header["CDELT2"])
            )
        )

    # The offset is redundent if the binning center is taken to be the map center but frequently it is not
    x = np.linspace(
        -hdu[0].data.shape[1] / 2 * pixsize + hdu[0].header["CRVAL1"] / pixunits,
        hdu[0].data.shape[1] / 2 * pixsize + hdu[0].header["CRVAL1"] / pixunits,
        hdu[0].data.shape[1],
    )
    y = np.linspace(
        -hdu[0].data.shape[0] / 2 * pixsize + hdu[0].header["CRVAL2"] / pixunits,
        hdu[0].data.shape[0] / 2 * pixsize + hdu[0].header["CRVAL2"] / pixunits,
        hdu[0].data.shape[0],
    )

    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    rbins = rbins.append(999999)
    bin1d = np.zeros(len(rbins) - 1)
    var1d = np.zeros(len(rbins) - 1)

    for k in range(len(rbins) - 1):
        pixels = [
            hdu[0].data[i, j]
            for i in range(len(y))
            for j in range(len(x))
            if rbins[k] < R[i, j] <= rbins[k + 1]
        ]
        bin1d[k] = np.mean(pixels)
        var1d[k] = np.var(pixels)

    return bin1d, var1d


def get_rbins(
    model,
    rmax: float = 3.0 * 60.0,
    struct_num: int = 0,
    sig_params: list[str] = ["amp", "P0"],
    default: tuple[int, ...] = (0, 10, 20, 30, 50, 80, 120, 180),
) -> tuple[int, ...]:
    """
    Function which returns a good set of rbins for a non-parametric fit given the significance of the underlying parametric model.

    Parameters
    ----------
    model : container.Model
        Parametric model to calculate rbins on
    rmax : float, default: 180
        Maximum radius of the rbins
    struct_num : int, defualt: 0
        Structure within model to calculate rbins on
    sig_params: list[str], default: ["amp", "P0"]
        Parameters to consider for computing significance.
        Only first match will be used.
    default: tuple[int, ...], default: (0, 10, 20, 30, 50, 80, 120, 180)
        Default rbins to be returned if generation fails.

    Returns
    -------
    rbins: tuple[int, ...]
        rbins for nonparametric fit
    """
    sig = 0
    for par in model.structures[struct_num].parameters:
        if par.name in sig_params:
            sig = par.val / par.err
            break
    if sig == 0:
        warnings.warn(
            "Warning: model does not contain any valid significance parameters {}. Returning default bins.".format(
                sig_params
            )
        )
        return default

    if sig < 10:
        warnings.warn(
            "Warning, significance {} too low to calculate bins. Returning default bins.".format(
                sig
            )
        )
        return default

    rbins = [0, 10, 20]
    rmin = 30
    nrbins = int(np.floor(sig / 5)[0] - 3)

    if nrbins == 1:
        rbins = np.array(rbins)
        rbins = np.append(rbins, rmax)

        return tuple(rbins.ravel())

    logrange = np.logspace(np.log10(rmin), np.log10(rmax), nrbins)
    step = logrange[1] - logrange[0]

    while step < 10:
        rbins.append(rmin)
        rmin += 10
        nrbins -= 1
        logrange = np.logspace(np.log10(rmin), np.log10(rmax), nrbins)
        step = logrange[1] - logrange[0]
        if rmin > rmax or nrbins < 1:
            return tuple(rbins)
    rbins = np.array(rbins)
    rbins = np.append(rbins, logrange)

    return tuple(rbins)


def para_to_non_para(
    model,
    n_rounds: Optional[int] = None,
    to_copy: list[str] = ["gnfw", "gnfw_rs", "a10", "isobeta", "uniform"],
) -> Model:
    """
    Function which approximately converts cluster profiles into a non-parametric form. Note this is
    only approximate and should be fit afterwords.

    Parameters
    ----------
    model : Model
        The parametric model to start from.
    n_rounds: Optional int | None, default: None
        Number of rounds to fit for output model. If none, copy from self
    to_copy : list[str], default: gnfw, gnfw_rs, a10, isobeta, uniform
        List of structures, by name, to copy.
    Returns
    -------
    Model : Model
        Model with a non-parametric representation of input model
    Raises
    ------
    ValueError
        If there are no models to copy
    """
    cur_model = deepcopy(
        model
    )  # Make a copy of model, we don't want to lose structures
    i = 0  # Make sure we keep at least one struct
    for structure in cur_model.structures:
        if structure.structure not in to_copy:
            cur_model.remove_struct(structure.name)
        else:
            i += 1
    if i == 0:
        raise ValueError("Error: no model structures in {}".format(to_copy))
    params = jnp.array(cur_model.pars)
    params = jnp.ravel(params)
    pressure, _ = core.model3D(
        cur_model.xyz, tuple(cur_model.n_struct), tuple(cur_model.n_rbins), params
    )
    pressure = pressure[
        ..., int(pressure.shape[2] / 2)
    ]  # Take middle slice. Close enough is good enough here, dont care about rounding

    pixsize = np.abs(cur_model.xyz[1][0][1] - cur_model.xyz[1][0][0]).item()

    rs, bin1d, _ = wu.bin_map(pressure, pixsize)

    rbins = get_rbins(cur_model)
    rbins = np.append(rbins, np.array([np.amax(rs)]))

    condlist = [
        jnp.array((rbins[i] <= rs) & (rs < rbins[i + 1]))
        for i in range(len(rbins) - 2, -1, -1)
    ]

    amps, pows, c = profile_to_broken_power(rs, bin1d, condlist, rbins)

    priors = (-1 * np.inf, np.inf)
    if n_rounds is None:
        n_rounds = model.n_rounds
    if not isinstance(n_rounds, int):
        raise ValueError("Non int n_rounds")
    parameters = [
        Parameter(
            "rbins",
            tuple([False] * n_rounds),
            jnp.atleast_1d(jnp.array(rbins[:-1], dtype=float)),  # Drop last bin
            jnp.zeros_like(jnp.atleast_1d(jnp.array(rbins[:-1])), dtype=float),
            jnp.array(priors, dtype=float),
        ),
        Parameter(
            "amps",
            tuple([True] * n_rounds),
            jnp.atleast_1d(jnp.array(amps, dtype=float)),
            jnp.zeros_like(jnp.atleast_1d(jnp.array(amps)), dtype=float),
            jnp.array(priors, dtype=float),
        ),
        Parameter(
            "pows",
            tuple([True] * n_rounds),
            jnp.atleast_1d(jnp.array(pows, dtype=float)),
            jnp.zeros_like(jnp.atleast_1d(jnp.array(pows)), dtype=float),
            jnp.array(priors, dtype=float),
        ),
        Parameter(
            "dx",  # TODO: miscentering
            tuple([False] * n_rounds),
            jnp.atleast_1d(jnp.array(0, dtype=float)),
            jnp.zeros_like(jnp.atleast_1d(jnp.array(0)), dtype=float),
            jnp.array(priors, dtype=float),
        ),
        Parameter(
            "dy",  # TODO: miscentering
            tuple([False] * n_rounds),
            jnp.atleast_1d(jnp.array(0, dtype=float)),
            jnp.zeros_like(jnp.atleast_1d(jnp.array(0)), dtype=float),
            jnp.array(priors, dtype=float),
        ),
        Parameter(
            "dz",  # TODO: miscentering
            tuple([False] * n_rounds),
            jnp.atleast_1d(jnp.array(0, dtype=float)),
            jnp.zeros_like(jnp.atleast_1d(jnp.array(0)), dtype=float),
            jnp.array(priors, dtype=float),
        ),
        Parameter(
            "c",
            tuple([True] * n_rounds),
            jnp.atleast_1d(jnp.array(c, dtype=float)),
            jnp.zeros_like(jnp.atleast_1d(jnp.array(0)), dtype=float),
            jnp.array(priors, dtype=float),
        ),
    ]

    structures = [
        Structure("nonpara_power", "nonpara_power", parameters, n_rbins=len(rbins) - 1)
    ]
    for structure in model.structures:
        if structure.name not in to_copy:
            structures.append(structure)

    return Model(
        name="test",
        structures=structures,
        xyz=model.xyz,
        dz=model.dz,
        beam=model.beam,
        n_rounds=n_rounds,
        cur_round=0,
    )
