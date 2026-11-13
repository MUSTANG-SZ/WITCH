"""
Tests for the gaussian beam helpers and their use with beam_conv.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import witch.utils as wu

# MUSTANG-2 double gaussian beam
M2 = (9.735, 0.9808, 32.627, 0.0192)


def _centroid_shift(beam, n):
    img = jnp.zeros((n, n)).at[n // 2, n // 2].set(1.0)
    out = np.array(wu.beam_conv(img, beam))
    yy, xx = np.mgrid[:n, :n]
    return (out * yy).sum() / out.sum() - n // 2, (out * xx).sum() / out.sum() - n // 2


def test_double_gauss_not_truncated():
    dr = 1.0
    beam = np.array(wu.beam_double_gauss(dr, *M2))
    # Kernel has to cover the secondary gaussian, not just the main beam
    assert beam.shape[0] >= 2 * 1.5 * M2[2] / dr
    assert beam.sum() == pytest.approx(1.0)
    # Peak normalized solid angle matches the analytic value (up to truncation at 1.5 FWHM2)
    c = beam.shape[0] // 2
    omega = beam.sum() / beam[c, c] * dr**2
    s1, s2 = (np.array([M2[0], M2[2]]) / np.sqrt(8 * np.log(2)))
    expected = 2 * np.pi * (M2[1] * s1**2 + M2[3] * s2**2) / (M2[1] + M2[3])
    assert omega == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("dr", [1.0, 2.0, 3.0])
def test_double_gauss_no_shift(dr):
    n = 2 * int(180 / dr)  # Same sizing as witch.grid.make_grid
    beam = wu.beam_double_gauss(dr, *M2)
    dy, dx = _centroid_shift(beam, n)
    assert dy == pytest.approx(0, abs=0.02)
    assert dx == pytest.approx(0, abs=0.02)


def test_double_gauss_zero_amp_component_ignored():
    # A zero amplitude secondary shouldn't make the kernel bigger
    a = np.array(wu.beam_double_gauss(10.0, 120.0, 1.0, 1800.0, 0.0))
    b = np.array(wu.beam_multi_gauss(10.0, [120.0], [1.0]))
    np.testing.assert_allclose(a, b)


def test_beam_cropped_to_grid(capsys):
    dr = 2.0
    n = 2 * int(300 / dr)
    beam = np.array(wu.beam_double_gauss(dr, 180.0, 0.9808, 270.0, 0.0192, max_npix=n))
    assert "cropping the beam" in capsys.readouterr().out
    assert beam.shape == (n, n)
    assert beam.sum() == pytest.approx(1.0)
    dy, dx = _centroid_shift(jnp.array(beam), n)
    assert dy == pytest.approx(0, abs=0.02)
    assert dx == pytest.approx(0, abs=0.02)


def test_multi_gauss_bad_input():
    with pytest.raises(ValueError):
        wu.beam_multi_gauss(1.0, [10.0, 20.0], [1.0])
    with pytest.raises(ValueError):
        wu.beam_multi_gauss(1.0, [10.0], [0.0])
