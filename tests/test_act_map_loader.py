"""Tests for the ACT map loader: array layout, WCS alignment and the zero-mean option."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from mpi4py import MPI

from witch.external.act_map import funcs as af

NY, NX = 24, 16  # Deliberately not square, FITS data is indexed (y, x)


def _wcs() -> WCS:
    """Build a small CAR WCS with a non-trivial reference pixel and pixel scale."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.crval = [44.6, 13.3]
    w.wcs.crpix = [NX / 2 + 0.5, NY / 2 + 0.5]
    w.wcs.cdelt = [-1.0 / 60, 1.0 / 60]
    w.wcs.cunit = ["deg", "deg"]
    return w


def _write(tmp_path: Path, dat: np.ndarray, ivar: np.ndarray) -> None:
    """Write a map and ivar pair with the `*_map.fits` and `*_ivar.fits` names the loader globs."""
    hdr = _wcs().to_header()
    fits.PrimaryHDU(dat, header=hdr).writeto(tmp_path / "t_map.fits")
    fits.PrimaryHDU(ivar, header=hdr).writeto(tmp_path / "t_ivar.fits")


def _load(tmp_path: Path, zero_mean: bool) -> af.maps.WCSMap:
    """Run `get_files` and `load_maps` on `tmp_path` and return the single map."""
    cfg = {
        "paths": {"maps": str(tmp_path)},
        "datasets": {"act": {"zero_mean": zero_mean}},
    }
    fnames = af.get_files("act", cfg)
    return af.load_maps("act", cfg, fnames, MPI.COMM_WORLD)[0]


@pytest.fixture
def raw(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write a map whose value encodes its FITS (y, x) index and return it with its ivar."""
    yy, xx = np.mgrid[:NY, :NX]
    dat = (yy * 100.0 + xx).astype(float)
    ivar = 1.0 + (xx % 3)
    _write(tmp_path, dat, ivar)
    return dat, ivar


def test_layout_is_x_first(tmp_path: Path, raw: tuple[np.ndarray, np.ndarray]) -> None:
    """The loaded array is transposed so that axis 0 is x, for non-square maps too."""
    dat, ivar = raw
    imap = _load(tmp_path, zero_mean=False)
    assert imap.data.shape == (NX, NY)
    np.testing.assert_array_equal(np.asarray(imap.data), dat.T)
    np.testing.assert_array_equal(np.asarray(imap.ivar), ivar.T)


def test_xy_matches_astropy(tmp_path: Path, raw: tuple[np.ndarray, np.ndarray]) -> None:
    """`imap.xy[i, j]` is the sky position of FITS pixel (x=i, y=j), in radians."""
    _ = raw
    imap = _load(tmp_path, zero_mean=False)
    ra, dec = (np.rad2deg(np.asarray(a)) for a in imap.xy)
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    ra_ref, dec_ref = _wcs().wcs_pix2world(ix, iy, 0)
    np.testing.assert_allclose(ra, ra_ref, atol=1e-4)
    np.testing.assert_allclose(dec, dec_ref, atol=1e-4)


def test_zero_mean(tmp_path: Path, raw: tuple[np.ndarray, np.ndarray]) -> None:
    """With `zero_mean` the ivar weighted mean is removed, without it the data is untouched."""
    dat, ivar = raw
    kept = _load(tmp_path, zero_mean=False)
    np.testing.assert_array_equal(np.asarray(kept.data), dat.T)
    sub = _load(tmp_path, zero_mean=True)
    mean = np.sum(dat * ivar) / np.sum(ivar)
    np.testing.assert_allclose(np.asarray(sub.data), dat.T - mean)
    assert abs(np.sum(np.asarray(sub.data) * np.asarray(sub.ivar))) < 1e-6 * np.sum(
        np.asarray(sub.ivar)
    )
