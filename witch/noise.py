"""
Noise models for maps where the CMB is a significant noise source.
"""

from functools import lru_cache
from typing import Self

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import register_pytree_node_class
from scipy.special import j0


@lru_cache(maxsize=1)
def _lensed_dl(lmax: int = 10000) -> np.ndarray:
    """
    Lensed LCDM TT spectrum from CAMB with Planck 2018 parameters.

    Parameters
    ----------
    lmax : int, default: 10000
        Maximum multipole.

    Returns
    -------
    dl : np.ndarray
        D_l = l(l+1)C_l/2pi in uK_CMB^2, indexed by l.
    """
    import camb

    pars = camb.set_params(
        H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544, As=2.1e-9, ns=0.9649,
        lmax=lmax, lens_potential_accuracy=1,
    )  # fmt: skip
    res = camb.get_results(pars)
    return res.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK", raw_cl=False)[:, 0]


def cmb_cl(
    ell: np.ndarray, fg_dl3000: float = 100.0, fwhm_arcmin: float = 2.0
) -> np.ndarray:
    """
    Beam smoothed CAMB CMB plus a crude white foreground power spectrum.

    Parameters
    ----------
    ell : np.ndarray
        Multipoles.
    fg_dl3000 : float, default: 100.0
        Foreground D_l at l=3000 in uK^2, scaled as l^2 (white in C_l).
    fwhm_arcmin : float, default: 2.0
        FWHM of the Gaussian beam the map was smoothed with.

    Returns
    -------
    cl : np.ndarray
        C_l in uK^2 sr, including the beam window squared.
    """
    dl = _lensed_dl()[ell.astype(int)] + fg_dl3000 * (ell / 3000.0) ** 2
    sigma = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8 * np.log(2))
    return dl * 2 * np.pi / (ell * (ell + 1)) * np.exp(-((ell * sigma) ** 2))


def cmb_covariance(
    shape: tuple[int, int], pix_arcmin: float, **kwargs: float
) -> np.ndarray:
    """
    Pixel-pixel covariance of the beam smoothed CMB on a flat patch.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the map.
    pix_arcmin : float
        Pixel size in arcmin.
    **kwargs
        Passed to `cmb_cl`.

    Returns
    -------
    cov : np.ndarray
        The `(npix, npix)` covariance in uK^2 with the map flattened in C order.
    """
    ell = np.arange(2, 10001, dtype=float)
    cl = cmb_cl(ell, **kwargs)
    ix, iy = np.indices(shape)
    xy = np.stack([ix.ravel(), iy.ravel()], axis=1) * np.deg2rad(pix_arcmin / 60.0)
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    # Flat sky: C(r) = int l dl / 2pi C_l J0(l r), evaluated on a radial grid and interpolated
    rgrid = np.linspace(0, dist.max(), 2000)
    crad = j0(rgrid[:, None] * ell) @ (ell * cl / (2 * np.pi))
    return np.interp(dist, rgrid, crad)


@register_pytree_node_class
class NoiseCMBMap:
    """
    Dense map noise model: white instrument noise from `ivar` plus a CMB covariance.
    The mean of the map is also marginalised over since cutouts have an arbitrary offset.
    This is a registered pytree and only suitable for small maps (the inverse is dense).

    Attributes
    ----------
    ninv : jax.Array
        The `(npix, npix)` inverse covariance. This is a child of the pytree.
    shape : tuple[int, int]
        Shape of the maps this is applied to. This is static pytree data.
    """

    def __init__(self, ninv, shape: tuple[int, int]):
        self.ninv = ninv
        self.shape = shape

    @jit
    def apply_noise(self, dat: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the inverse covariance to a map.

        Parameters
        ----------
        dat : jax.Array
            Map to filter, with shape `self.shape`.

        Returns
        -------
        dat_filt : jax.Array
            `N^-1 dat` with the same shape as `dat`.
        """
        return (self.ninv @ dat.ravel()).reshape(self.shape)

    @classmethod
    def compute(
        cls,
        dat: jnp.ndarray,
        ivar: jnp.ndarray,
        pix_deg: float,
        cmb_scale: float = 1.0,
        mean_sigma: float = 1e3,
        **kwargs: float,
    ) -> Self:
        """
        Build the noise model.

        Parameters
        ----------
        dat : jax.Array
            The map, only used for its shape.
        ivar : jax.Array
            Inverse variance of the white noise per pixel. Pixels with 0 get no weight.
        pix_deg : float
            Pixel size in degrees (sign is ignored).
        cmb_scale : float, default: 1.0
            Multiplier on the CMB covariance.
        mean_sigma : float, default: 1e3
            Prior width on the map offset in uK.
        **kwargs
            Passed to `cmb_covariance`, e.g. `fwhm_arcmin`.

        Returns
        -------
        noise_model : NoiseCMBMap
            The noise model.
        """
        ivar = np.asarray(ivar, dtype=float).ravel()
        white = np.where(ivar > 0, 1.0 / np.where(ivar > 0, ivar, 1.0), 1e12)
        cov = cmb_scale * cmb_covariance(
            dat.shape, abs(float(pix_deg)) * 60.0, **kwargs
        )
        cov += np.diag(white) + mean_sigma**2
        return cls(jnp.asarray(np.linalg.inv(cov)), tuple(dat.shape))

    def tree_flatten(self) -> tuple[tuple, tuple[int, int]]:
        return (self.ninv,), self.shape

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(children[0], aux_data)
