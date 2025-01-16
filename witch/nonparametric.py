import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import utils as wu
from .grid import transform_grid


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


@jax.jit
def power(x, rbin, cur_amp, cur_pow, c):
    """
    Function which returns the powerlaw, given the bin-edge constraints. Exists to be partialed.

    Parameters:
    -----------
    x : float
        Dummy variable to be partialed over
    rbin : float
        Edge of bin for powerlaw
    cur_amp : float
        Amplitude of power law
    cur_pow : float
        Power of power law
    c : float
        Constant offset

    Returns
    -------
    tmp : float
        Powerlaw evaluated at x
    """
    tmp = cur_amp * (x**cur_pow - rbin**cur_pow) + c
    return tmp


@jax.jit
def broken_power(
    rs: jax.Array,
    condlist: tuple,
    rbins: jax.Array,
    amps: jax.Array,
    pows: jax.Array,
    c: float,
) -> jax.Array:
    """
    Function which returns a broken powerlaw evaluated at rs.

    Parameters:
    -----------
    rs : jax.Array
        Array of rs at which to compute pl.
    condlist : tuple
        tuple which enocdes which rs are evaluated by which parametric function
    rbins : jax.Array
        Array of bin edges for power laws
    amps : jax.Array
        Amplitudes of power laws
    pows : jax.Array                                                                                                                                                                                                                                                                            Exponents of power laws
    c : float
        Constant offset for powerlaws
    """
    cur_c = c  # TODO: necessary?
    funclist = []
    for i in range(len(condlist) - 1, -1, -1):
        funclist.append(
            partial(power, rbin=rbins[i + 1], cur_amp=amps[i], cur_pow=pows[i], c=cur_c)
        )
        cur_c += amps[i] * (rbins[i] ** pows[i] - rbins[i + 1] ** pows[i])
    return jnp.piecewise(rs, condlist, funclist)
