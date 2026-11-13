"""
Tests for NIKA2 TOD loading and the Jy/beam unit conversion.
"""

import os

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


def test_K_RJ2Jy_beam_analytic():
    freq = 150e9
    fwhm = 17.6
    omega = 2 * np.pi * (fwhm / wu.rad_to_arcsec / np.sqrt(8 * np.log(2))) ** 2
    expected = 2 * const.k_B.value * freq**2 / const.c.value**2 * omega * 1e26
    assert wu.K_RJ2Jy_beam(freq, fwhm) == pytest.approx(expected)
    # ~5.7 Jy/beam/K for the NIKA2 2mm beam
    assert wu.K_RJ2Jy_beam(freq, fwhm) == pytest.approx(5.70, rel=1e-2)


def test_y2Jy_beam_sign_and_scale():
    # SZ decrement at 150 GHz, of order -8 Jy/beam per unit y for a 17.6" beam
    val = wu.y2Jy_beam(150e9, 5.0, 17.6)
    assert -10 < val < -7
    assert val == pytest.approx(float(wu.y2K_RJ(150e9, 5.0)) * wu.K_RJ2Jy_beam(150e9, 17.6))


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
