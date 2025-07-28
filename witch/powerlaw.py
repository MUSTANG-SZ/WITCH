from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from scipy.optimize import curve_fit


@jax.jit
def power(x: float, rbin: float, cur_amp: float, cur_pow: float, c: float):
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


def profile_to_broken_power(
    rs: ArrayLike, ys: ArrayLike, condlist: list[jax.Array], rbins: ArrayLike
) -> tuple[jax.Array, jax.Array, float]:
    """
    Estimates a non-parametric broken power profile from a generic profile.
    Note this is an estimation only; in partciular since we fit piece-wise
    the c's get messed up. This broken powerlaw should then be fit to the
    data.

    Parameters
    ----------
    rs : ArrayLike
        Array of radius values for the profile
    ys : ArrayLike
        Profile y values
    condlist : list[ArrayLike]
        List which defines which powerlaws map to which radii. See broken_power
    rbins : ArrayLike
        Array of bin edges defining the broken powerlaws

    Returns
    -------
    amps : jnp.array
        Best fit amps for the powerlaws
    pows : jnp.array
        Best fit powers for the powerlaws
    c : float
        Best fit c for only the outermost powerlaw
    """
    rs = jnp.array([x if x != 0 else 1e-1 for x in rs])  # Dont blow up

    rbins = jnp.array(
        [x if x != 0 else jnp.amin(rs) for x in rbins]
    )  # Dont blow up 2.0

    amps = jnp.zeros(len(condlist))
    pows = jnp.zeros(len(condlist))

    for i in range(len(condlist)):
        xdata = rs[condlist[i]]
        ydata = ys[condlist[i]]
        if i == len(condlist) - 1:
            popt, pcov = curve_fit(power, xdata, ydata, method="trf")
        else:
            popt, pcov = curve_fit(
                power,
                xdata,
                ydata,
                method="trf",
                p0=[rbins[::-1][i], np.amax(ydata) * 1e5, -2, 0.0],
            )
        if i == 0:
            c = popt[3]
        amps = amps.at[i].set(popt[1])
        pows = pows.at[i].set(popt[2])

    return amps[::-1], pows[::-1], c
