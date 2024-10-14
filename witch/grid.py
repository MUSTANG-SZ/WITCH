"""
Functions for building and working with the model grid.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from astropy.wcs import WCS
from numpy.typing import NDArray

from .utils import rad_to_arcsec

Grid = tuple[jax.Array, jax.Array, jax.Array, float, float]


def make_grid(
    r_map: float,
    dx: float,
    dy: Optional[float] = None,
    dz: Optional[float] = None,
    x0: float = 0,
    y0: float = 0,
) -> Grid:
    """
    Make coordinate grids to build models in.
    All grids are sparse and are `int(2*r_map / dr)` in each the non-sparse dimension.

    Parameters
    ----------
    r_map : float
        Size of grid radially.
    dx : float
        Grid resolution in x, should be in same units as r_map.
    dy : Optional[float], default: None
        Grid resolution in y, should be in same units as r_map.
        If None then dy is set to dx.
    dz : Optional[float], default: None
        Grid resolution in z, should be in same units as r_map.
        If None then dz is set to dx.
    x0 : float, default: 0
        Origin of grid in RA, assumed to be in same units as r_map.
    y0 : float, default: 0
        Origin of grid in Dec, assumed to be in same units as r_map.

    Returns
    -------
    x : jax.Array
        Grid of x coordinates in same units as r_map.
        Has shape (`int(2*r_map / dr), 1, 1).
    y : jax.Array
        Grid of y coordinates in same units as r_map.
        Has shape (1, `int(2*r_map / dr), 1).
    z : jax.Array
        Grid of z coordinates in same units as r_map.
        Has shape (1, 1, `int(2*r_map / dr)`).
    x0 : float
        Origin of grid in RA, in same units as r_map.
    y0 : float
        Origin of grid in Dec, in same units as r_map.
    """
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx

    # Make grid with resolution dr and size r_map
    x = (
        jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dx))
        / jnp.cos(y0 / rad_to_arcsec)
        + x0
    )
    y = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dy)) + y0
    z = jnp.linspace(-1 * r_map, r_map, 2 * int(r_map / dz))
    x, y, z = jnp.meshgrid(x, y, z, sparse=True, indexing="ij")

    return (x, y, z, x0, y0)


def make_grid_from_wcs(
    wcs: WCS,
    nx: int,
    ny: int,
    z_map: float,
    dz: float,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
) -> Grid:
    """
    Make coordinate grids to build models in from a minkasi skymap.
    All grids are sparse and match the input map and xy and have size `int(2*z_map/dz)` in z.
    Unlike `make_grid` here we assume things are radians.

    Parameters
    ----------
    wcs : WCS
        The WCS to base the grid off of.
    nx : int
        The number of pixels in x.
    ny : int
        The number of pixels in y.
    z_map : float
        Size of grid along LOS, in radians.
    dz : float
        Grid resolution along LOS, in radians.
    x0 : Optional[float], default: None
        Map x center in radians.
        If None, grid center is used.
    y0 : Optional[float], default: None
        Map y center in radians. If None, grid center is used.

    Returns
    -------
    x : jax.Array
        Grid of x coordinates in radians.
        Has shape (`skymap.nx`, 1, 1).
    y : jax.Array
        Grid of y coordinates in radians.
        Has shape (1, `skymap.ny`, 1).
    z : jax.Array
        Grid of z coordinates in same units as radians.
        Has shape (1, 1, `int(2*z_map / dz)`).
    x0 : float
        Origin of grid in RA, in radians.
    y0 : float
        Origin of grid in Dec, in radians.
    """
    # make grid
    _x = jnp.arange(nx, dtype=float)
    _y = jnp.arange(ny, dtype=float)
    _z = jnp.linspace(-1 * z_map, z_map, 2 * int(z_map / dz), dtype=float)
    x, y, z = jnp.meshgrid(_x, _y, _z, sparse=True, indexing="ij")

    # Pad so we don't need to broadcast
    x_flat = x.ravel()
    y_flat = y.ravel()
    len_diff = len(x_flat) - len(y_flat)
    if len_diff > 0:
        y_flat = jnp.pad(y_flat, (0, len_diff), "edge")
    elif len_diff < 0:
        x_flat = jnp.pad(x_flat, (0, abs(len_diff)), "edge")

    # Convert x and y to ra/dec
    ra_dec = wcs.wcs_pix2world(jnp.column_stack((x_flat, y_flat)), 0, ra_dec_order=True)
    ra_dec = np.deg2rad(ra_dec)
    ra = ra_dec[:, 0]
    dec = ra_dec[:, 1]

    # Remove padding
    if len_diff > 0:
        dec = dec[: (-1 * len_diff)]
    elif len_diff < 0:
        ra = ra[:len_diff]

    if not x0:
        x0 = (np.max(ra) + np.min(ra)) / 2
    if not y0:
        y0 = (np.max(dec) + np.min(dec)) / 2

    if x0 is None or y0 is None:
        raise TypeError("Origin still None")

    # Sparse indexing to save mem
    x = x.at[:, 0, 0].set(ra * rad_to_arcsec)
    y = y.at[0, :, 0].set(dec * rad_to_arcsec)
    z = z * rad_to_arcsec
    x0 *= rad_to_arcsec
    y0 *= rad_to_arcsec

    return x, y, z, float(x0), float(y0)


@jax.jit
def transform_grid(
    dx: float,
    dy: float,
    dz: float,
    r_1: float,
    r_2: float,
    r_3: float,
    theta: float,
    xyz: Grid,
):
    """
    Shift, rotate, and apply ellipticity to coordinate grid.
    Note that the `Grid` type is an alias for `tuple[jax.Array, jax.Array, jax.Array, float, float]`.

    Parameters
    ----------
    dx : float
        Amount to move grid origin in x
    dy : float
        Amount to move grid origin in y
    dz : float
        Amount to move grid origin in z
    r_1 : float
        Amount to scale along x-axis
    r_2 : float
        Amount to scale along y-axis
    r_3 : float
        Amount to scale along z-axis
    theta : float
        Angle to rotate in xy-plane in radians
    xyz : Grid
        Coordinte grid to transform

    Returns
    -------
    trasnformed : Grid
        Transformed coordinate grid.
    """
    # Get origin
    x0, y0 = xyz[3], xyz[4]
    # Shift origin
    x = (xyz[0] - (x0 + dx / jnp.cos(y0 / rad_to_arcsec))) * jnp.cos(
        (y0 + dy) / rad_to_arcsec
    )
    y = xyz[1] - (y0 + dy)
    z = xyz[2] - dz

    # Rotate
    xx = x * jnp.cos(theta) + y * jnp.sin(theta)
    yy = y * jnp.cos(theta) - x * jnp.sin(theta)

    # Apply ellipticity
    x = xx / r_1
    y = yy / r_2
    z = z / r_3

    return x, y, z, x0 - dx, y0 - dy


def tod_to_index(
    xi: NDArray[np.floating],
    yi: NDArray[np.floating],
    x0: float,
    y0: float,
    grid: Grid,
    conv_factor: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Convert RA/Dec TODs to index space.

    Parameters
    ----------
    xi : NDArray[np.floating]
        RA TOD, usually in radians
    yi : NDArray[np.floating]
        Dec TOD, usually in radians
    grid : Grid
        The grid to index on.
    conv_factor : float, default: 1.
        Conversion factor to put RA and Dec in same units as the grid.

    Returns
    -------
    idx : jax.Array
        The RA TOD in index space
    idy : jax.Array
        The Dec TOD in index space.
    """
    x0, y0 = grid[-2:]
    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    dx *= conv_factor
    dy *= conv_factor

    # Assuming sparse indexing here
    idx = np.digitize(dx, grid[0].ravel())
    idy = np.digitize(dy, grid[1].ravel())

    idx = np.rint(idx).astype(int)
    idy = np.rint(idy).astype(int)

    # Ensure out of bounds for stuff not in grid
    idx = jnp.where((idx < 0) + (idx >= grid[0].shape[0]), 2 * grid[0].shape[0], idx)
    idy = jnp.where((idy < 0) + (idy >= grid[1].shape[1]), 2 * grid[1].shape[1], idy)

    return idx, idy
