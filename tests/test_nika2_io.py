"""
Tests for NIKA2 TOD loading and the Jy/beam unit conversion.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import constants as const

import witch.utils as wu
from witch.external.nika2.io import _unwrap_deg, read_xdr

XDR = "/mnt/welch/NIKA2/id34/stru_scan20201029s119_2mm_tot.xdr"


def test_unwrap_deg_negative_ra():
    # RA stored as negative wrapped degrees, as in the id34 xdr files
    ra = np.array([-223.2, -223.0, -222.9])
    out = _unwrap_deg(ra)
    np.testing.assert_allclose(out, ra + 360)


def test_unwrap_deg_across_zero():
    ra = np.array([359.9, 0.05, 0.1])
    out = _unwrap_deg(ra)
    assert np.ptp(out) < 1.0


NIKA2_2MM_FWHMS = [17.4, 42.0, 99.0]
NIKA2_2MM_AMPS = [10 ** (-0.24 / 10), 10 ** (-12.8 / 10), 10 ** (-27 / 10)]


def test_gauss_solid_angle_single():
    fwhm = 17.6
    sigma = fwhm / np.sqrt(8 * np.log(2))
    assert wu.gauss_solid_angle([fwhm], [1.0]) == pytest.approx(2 * np.pi * sigma**2)
    # Amplitude normalization shouldn't matter for a single gaussian
    assert wu.gauss_solid_angle([fwhm], [3.0]) == pytest.approx(2 * np.pi * sigma**2)


def test_gauss_solid_angle_nika2():
    # Perotto et al. 2020 measure Omega_90 = 434 +- 13 arcsec^2 at 2mm (Table 9),
    # the 3-gaussian fit (Table 5) should be consistent with that
    omega = wu.gauss_solid_angle(NIKA2_2MM_FWHMS, NIKA2_2MM_AMPS)
    assert omega == pytest.approx(434, rel=0.08)


def test_beam_multi_gauss():
    dr = 2.0
    beam = np.array(wu.beam_multi_gauss(dr, NIKA2_2MM_FWHMS, NIKA2_2MM_AMPS))
    assert beam.sum() == pytest.approx(1.0)
    # Kernel covers 1.5x the widest component, has an even shape and peaks at shape // 2
    half = int(np.ceil(1.5 * 99.0 / dr))
    assert beam.shape == (2 * half, 2 * half)
    c = beam.shape[0] // 2
    assert np.unravel_index(np.argmax(beam), beam.shape) == (c, c)
    # Peak normalized beam integrates to the analytic solid angle (up to truncation)
    omega_num = beam.sum() / beam[c, c] * dr**2
    assert omega_num == pytest.approx(
        wu.gauss_solid_angle(NIKA2_2MM_FWHMS, NIKA2_2MM_AMPS), rel=0.01
    )


@pytest.mark.parametrize("dr", [1.0, 2.0])
@pytest.mark.parametrize("n", [180, 240])
def test_beam_multi_gauss_no_shift(dr, n):
    # Convolving a delta at the grid center must not move it (WITCH grids are even sized)
    beam = wu.beam_multi_gauss(dr, NIKA2_2MM_FWHMS, NIKA2_2MM_AMPS)
    if beam.shape[0] > n:
        pytest.skip("kernel larger than grid")
    img = jnp.zeros((n, n)).at[n // 2, n // 2].set(1.0)
    out = np.array(wu.beam_conv(img, beam))
    yy, xx = np.mgrid[:n, :n]
    assert (out * yy).sum() / out.sum() == pytest.approx(n // 2, abs=0.02)
    assert (out * xx).sum() / out.sum() == pytest.approx(n // 2, abs=0.02)
    assert np.unravel_index(np.argmax(out), out.shape) == (n // 2, n // 2)


def test_K_RJ2Jy_beam_analytic():
    freq = 150e9
    omega = wu.gauss_solid_angle([17.6], [1.0])
    omega_sr = omega / wu.rad_to_arcsec**2
    expected = 2 * const.k_B.value * freq**2 / const.c.value**2 * omega_sr * 1e26
    assert wu.K_RJ2Jy_beam(freq, omega) == pytest.approx(expected)
    # ~5.7 Jy/beam/K for a 17.6" gaussian
    assert wu.K_RJ2Jy_beam(freq, omega) == pytest.approx(5.70, rel=1e-2)


def test_y2Jy_beam_sign_and_scale():
    omega = wu.gauss_solid_angle([17.6], [1.0])
    # SZ decrement at 150 GHz, about -8.25 Jy/beam per unit y for a 17.6" gaussian
    val = wu.y2Jy_beam(150e9, 5.0, omega)
    assert val == pytest.approx(-8.25, rel=1e-2)
    assert val == pytest.approx(
        float(wu.y2K_RJ(150e9, 5.0)) * wu.K_RJ2Jy_beam(150e9, omega)
    )
    # Colour correction divides
    assert wu.y2Jy_beam(150e9, 5.0, omega, 0.972) == pytest.approx(val / 0.972)


@pytest.mark.skipif(not os.path.exists(XDR), reason="id34 NIKA2 data not available")
def test_read_xdr():
    dat = read_xdr(XDR)
    assert dat is not None
    ndet, nsamp = dat["dat_calib"].shape
    assert ndet > 100 and nsamp > 1000
    assert np.all(np.isfinite(dat["dat_calib"]))
    assert dat["dx"].shape == dat["dat_calib"].shape
    assert dat["dy"].shape == dat["dat_calib"].shape
    # id34 field is at RA ~137 deg, Dec ~0.17 deg
    assert np.rad2deg(np.median(dat["dx"])) == pytest.approx(136.9, abs=0.2)
    assert np.rad2deg(np.median(dat["dy"])) == pytest.approx(0.17, abs=0.2)
    assert dat["calinfo"]["units"] == "Jy/beam"
    assert dat["calinfo"]["band"] == "2mm"
    assert dat["dt"] > 0
