"""
Noise models for maps where the CMB is a significant noise source.

The CMB power spectrum is the lensed TT spectrum of the Planck 2018 cosmology, stored in
`witch/data/lensed_tt_planck2018.npz`, so CAMB is not needed by default. To use another
cosmology generate a spectrum with `generate_lensed_dl` (needs the optional `camb` extra) and
pass it as `dl` to `cmb_cl`, `cmb_covariance` or `NoiseCMBMap`, see `scripts/make_lensed_dl.py`
to store one. CAMB also covers the scales above the patch size that a covariance measured
from a map cannot.

Note
----
This is an interim model that is expected to be replaced by the multi-frequency
`NoiseMapCov` model, which takes cross-covariance files for every pair of datasets.
See the PR that added this module for what the replacement has to keep or change
(the `sample` hook in `fitter.process_maps`, `resolve_noise_args`, `sim_seed` and
`sim_noise_kwargs`). The stored CMB spectrum is still useful there for the scales
above the patch size.
"""

import warnings
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.tree_util import register_pytree_node_class
from scipy.special import j0

LENSED_TT_FILE = Path(__file__).parent / "data" / "lensed_tt_planck2018.npz"
# Planck 2018 TT,TE,EE+lowE+lensing parameters used for the stored spectrum
PLANCK2018 = {
    "H0": 67.36,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "tau": 0.0544,
    "As": 2.1e-9,
    "ns": 0.9649,
}


def generate_lensed_dl(lmax: int = 10000, **cosmology: float) -> np.ndarray:
    """
    Lensed LCDM TT spectrum from CAMB.

    Needs the optional `camb` dependency. The stored default spectrum was made with this
    function and the Planck 2018 parameters, see `scripts/make_lensed_dl.py`.

    Parameters
    ----------
    lmax : int, default: 10000
        Maximum multipole.
    **cosmology
        Parameters passed to `camb.set_params`, overriding the Planck 2018 values in
        `PLANCK2018`, e.g. `H0=70.0`.

    Returns
    -------
    dl : np.ndarray
        D_l = l(l+1)C_l/2pi in uK_CMB^2, indexed by l.

    Raises
    ------
    ImportError
        If `camb` is not installed.
    """
    try:
        import camb
    except ImportError as e:
        raise ImportError(
            "Generating a CMB spectrum needs the optional `camb` dependency, "
            "install it with `pip install camb`."
        ) from e

    pars = camb.set_params(
        **{**PLANCK2018, **cosmology}, lmax=lmax, lens_potential_accuracy=1
    )
    res = camb.get_results(pars)
    return res.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK", raw_cl=False)[:, 0]


@lru_cache(maxsize=1)
def _lensed_dl(lmax: int = 10000) -> np.ndarray:
    """
    Stored lensed LCDM TT spectrum for the Planck 2018 cosmology.

    Parameters
    ----------
    lmax : int, default: 10000
        Maximum multipole, at most the `lmax` of the stored spectrum.

    Returns
    -------
    dl : np.ndarray
        D_l = l(l+1)C_l/2pi in uK_CMB^2, indexed by l.

    Raises
    ------
    ValueError
        If `lmax` is above the stored range, use `generate_lensed_dl` instead.
    """
    with np.load(LENSED_TT_FILE) as f:
        dl = f["dl"]
    if lmax >= len(dl):
        raise ValueError(
            f"The stored CMB spectrum only goes to l={len(dl) - 1}, got lmax={lmax}. "
            "Use generate_lensed_dl for a larger lmax."
        )
    return dl[: lmax + 1]


def cmb_cl(
    ell: np.ndarray,
    fg_dl3000: float = 100.0,
    fwhm_arcmin: float = 2.0,
    dl: np.ndarray | None = None,
) -> np.ndarray:
    """
    Beam smoothed lensed CMB plus a crude white foreground power spectrum.

    Parameters
    ----------
    ell : np.ndarray
        Multipoles.
    fg_dl3000 : float, default: 100.0
        Foreground D_l at l=3000 in uK^2, scaled as l^2 (white in C_l).
    fwhm_arcmin : float, default: 2.0
        FWHM of the Gaussian beam the map was smoothed with.
    dl : np.ndarray, optional
        Lensed CMB D_l in uK_CMB^2 indexed by l, from `generate_lensed_dl`.
        Defaults to the stored Planck 2018 spectrum.

    Returns
    -------
    cl : np.ndarray
        C_l in uK^2 sr, including the beam window squared.
    """
    if dl is None:
        dl = _lensed_dl()
    dl = dl[ell.astype(int)] + fg_dl3000 * (ell / 3000.0) ** 2
    sigma = np.deg2rad(fwhm_arcmin / 60.0) / np.sqrt(8 * np.log(2))
    return dl * 2 * np.pi / (ell * (ell + 1)) * np.exp(-((ell * sigma) ** 2))


def cmb_covariance(
    shape: tuple[int, int], pix_arcmin: float, **kwargs: Any
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


def cmb_covariance_multiband(
    shape: tuple[int, int],
    pix_arcmin: float,
    fwhms: Sequence[float],
    fg_dl3000: float = 100.0,
    dl: np.ndarray | None = None,
) -> np.ndarray:
    """
    Joint covariance of the primary CMB in several bands on the same flat patch.

    The CMB is the same sky in every band (in uK_CMB) so the bands are perfectly correlated
    apart from their beams. Block `(i, j)` has the spectrum times `b_i(l) b_j(l)`, which for
    Gaussian beams is the single band form with `fwhm = sqrt((fwhm_i^2 + fwhm_j^2) / 2)`.
    The crude white foreground term is only added to the auto blocks.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the maps, which all share a pixelisation.
    pix_arcmin : float
        Pixel size in arcmin.
    fwhms : Sequence[float]
        Beam FWHM of each band in arcmin.
    fg_dl3000 : float, default: 100.0
        Foreground D_l at l=3000 in uK^2 for the auto blocks, see `cmb_cl`.
    dl : np.ndarray, optional
        Lensed CMB D_l indexed by l, see `cmb_cl`.

    Returns
    -------
    cov : np.ndarray
        The `(nband*npix, nband*npix)` covariance in uK^2, bands stacked and each map
        flattened in C order.
    """
    ell = np.arange(2, 10001, dtype=float)
    ix, iy = np.indices(shape)
    xy = np.stack([ix.ravel(), iy.ravel()], axis=1) * np.deg2rad(pix_arcmin / 60.0)
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    rgrid = np.linspace(0, dist.max(), 2000)
    jell = j0(rgrid[:, None] * ell) * ell / (2 * np.pi)
    nb, npix = len(fwhms), dist.shape[0]
    cov = np.zeros((nb * npix, nb * npix))
    for i in range(nb):
        for j in range(i, nb):
            fwhm = np.sqrt((fwhms[i] ** 2 + fwhms[j] ** 2) / 2)
            cl = cmb_cl(ell, fg_dl3000 * (i == j), fwhm, dl)
            block = np.interp(dist, rgrid, jell @ cl)
            cov[i * npix : (i + 1) * npix, j * npix : (j + 1) * npix] = block
            cov[j * npix : (j + 1) * npix, i * npix : (i + 1) * npix] = block.T
    return cov


def ring_cross_spectra(noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ring averaged cross power of the noise realisations of several bands.

    Parameters
    ----------
    noise : np.ndarray
        Shape `(nband, nreal, n, n)`.

    Returns
    -------
    power : np.ndarray
        The 2D power `P_ij(k)` of shape `(nband, nband, n, n)`, so that the covariance
        is `c_ij(dx) = ifft2(P_ij)`. The `k = 0` mode is set to 0.
    ring : np.ndarray
        Integer ring index of each Fourier mode, shape `(n, n)`.
    """
    nb, _, n, _ = noise.shape
    h = np.hanning(n + 2)[1:-1]
    taper = np.outer(h, h)
    ft = np.fft.fft2(taper * noise) / np.sqrt(np.mean(taper**2) * n * n)
    k = np.fft.fftfreq(n) * n
    ring = np.rint(np.hypot(*np.meshgrid(k, k, indexing="ij"))).astype(int)
    power = np.zeros((nb, nb, n, n))
    for r in range(1, ring.max() + 1):
        sel = ring == r
        f = ft[:, :, sel]  # (nb, nreal, nmode)
        pr = np.einsum("ajm,bjm->ab", f, f.conj()).real / (f.shape[1] * f.shape[2])
        power[:, :, sel] = pr[:, :, None]
    return power, ring


def noise_covariance(power: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """
    Stationary noise covariance of a set of pixels from the 2D power.

    Parameters
    ----------
    power : np.ndarray
        Output of `ring_cross_spectra`, `(nband, nband, n, n)`.
    ix, iy : np.ndarray
        Pixel indices of the pixels to use, flattened.

    Returns
    -------
    cov : np.ndarray
        `(nband*npix, nband*npix)` covariance in uK^2, bands stacked.
    """
    nb, _, n, _ = power.shape
    c = np.fft.ifft2(power, axes=(2, 3)).real
    dx = (ix[:, None] - ix[None]) % n
    dy = (iy[:, None] - iy[None]) % n
    npix = ix.size
    cov = np.zeros((nb * npix, nb * npix))
    for i in range(nb):
        for j in range(nb):
            cov[i * npix : (i + 1) * npix, j * npix : (j + 1) * npix] = c[i, j][dx, dy]
    return cov


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
        **kwargs: Any,
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
            Passed to `cmb_covariance`, e.g. `fwhm_arcmin` or `dl`.

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

    @classmethod
    def sample(
        cls,
        key: jax.Array,
        dat: jnp.ndarray,
        ivar: jnp.ndarray,
        pix_deg: float,
        cmb_scale: float = 1.0,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """
        Draw a noise realisation from the same model `compute` uses, minus the mean prior.
        The ivar weighted mean is removed to match `load_maps`, and pixels with no weight are zeroed.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        dat : jax.Array
            The map, only used for its shape.
        ivar : jax.Array
            Inverse variance of the white noise per pixel.
        pix_deg : float
            Pixel size in degrees (sign is ignored).
        cmb_scale : float, default: 1.0
            Multiplier on the CMB covariance.
        **kwargs
            Passed to `cmb_covariance`, e.g. `fwhm_arcmin` or `dl`.

        Returns
        -------
        noise : jax.Array
            A noise map with the shape of `dat`.
        """
        ivar = np.asarray(ivar, dtype=float).ravel()
        good = ivar > 0
        cov = cmb_scale * cmb_covariance(
            dat.shape, abs(float(pix_deg)) * 60.0, **kwargs
        )
        cov += np.diag(np.where(good, 1.0 / np.where(good, ivar, 1.0), 0.0))
        # eigh rather than cholesky since the smooth CMB covariance is close to singular
        vals, vecs = np.linalg.eigh(cov)
        white = np.asarray(jax.random.normal(key, (ivar.size,)))
        noise = vecs @ (np.sqrt(np.clip(vals, 0, None)) * white)
        noise -= np.sum(noise * ivar) / np.sum(ivar)
        return jnp.asarray(np.where(good, noise, 0.0).reshape(dat.shape))

    def tree_flatten(self) -> tuple[tuple, tuple[int, int]]:
        return (self.ninv,), self.shape

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(children[0], aux_data)


@register_pytree_node_class
class NoiseSpectrum(NoiseCMBMap):
    """
    Dense map noise model from an empirically measured stationary power spectrum.

    The spectrum is the output of `ring_cross_spectra` for a single band, measured for example
    on control fields of the same map. The mean of the map is marginalised over.
    Only suitable for small maps (the inverse is dense), the pixels must match the spectrum grid.

    Warning
    -------
    Interim model kept for comparisons: it assumes a stationary, periodic noise measured offline
    from control fields or splits. It is not the ACT noise estimate and should be replaced by it.
    """

    @classmethod
    def compute(
        cls,
        dat: jnp.ndarray,
        ivar: jnp.ndarray,
        pix_deg: float,
        power_path: str,
        mean_sigma: float = 1e-4,
    ) -> Self:
        """
        Build the noise model.

        Parameters
        ----------
        dat : jax.Array
            The map, only used for its shape which must be that of the spectrum.
        ivar : jax.Array
            Unused, kept so the signature matches `NoiseCMBMap`.
        pix_deg : float
            Unused, the spectrum is already in pixel units.
        power_path : str
            `.npy` file with the `(n, n)` 2D noise power.
        mean_sigma : float, default: 1e-4
            Prior width on the map offset in map units.

        Returns
        -------
        noise_model : NoiseSpectrum
            The noise model.
        """
        _ = (ivar, pix_deg)
        warnings.warn(
            "NoiseSpectrum is an interim, stationary-noise approximation kept for comparisons "
            "and is not the ACT noise model; do not use it for science results.",
            UserWarning,
            stacklevel=2,
        )
        power = np.load(power_path)
        if tuple(dat.shape) != power.shape:
            raise ValueError(f"Map {dat.shape} does not match spectrum {power.shape}")
        ix, iy = np.indices(dat.shape)
        cov = noise_covariance(power[None, None], ix.ravel(), iy.ravel())
        cov += mean_sigma**2
        return cls(jnp.asarray(np.linalg.inv(cov)), tuple(dat.shape))

    # Cannot draw noise, so map sims fall back to white noise rather than using the CMB model
    sample = None
