"""
Unit tests for structure.py - testing 3D models against analytical 2D solutions
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.analytical_2d import (
    gaussian_2d_integrated,
    isobeta_2d_analytical,
    sph_isobeta_2d_slice,
)
from witch import grid, structure


class TestStructureAnalytical:
    """Test structures with closed-form 2D solutions"""

    @pytest.fixture
    def test_grid(self):
        """Standard grid for most tests"""
        r_map = 60.0
        dr = 2.0
        dz = 2.0
        xyz = grid.make_grid(r_map, dr, dr, dz, 0.0, 0.0)
        return xyz, dz

    @pytest.fixture
    def wide_grid(self):
        """
        wide z-extent grid needed for the slowly decaying profiles
        """
        r_map = 300.0  # cylindrical_beta needs a lot of z extent
        dr = 2.0
        dz = 2.0
        xyz = grid.make_grid(r_map, dr, dr, dz, 0.0, 0.0)
        return xyz, dz

    def test_isobeta_spherical_vs_analytical_2d(self, test_grid):
        """
        Test isobeta 2D analytical formula matches minkasi C code.
        Compare structure.isobeta (3D) at a single z-slice against the minkasi closed-form 2D result.
        full z-integration is impractical due to very slow power-law decay.
        """
        xyz, dz = test_grid
        dx, dy, dz_offset = 0.0, 0.0, 0.0
        theta_core = 10.0
        beta = 0.7
        amp = 1.0

        # Get a single z=0 slice of the 3D model (center slice)
        model_3d = structure.isobeta(
            dx, dy, dz_offset, theta_core, theta_core, theta_core, 0.0, beta, amp, xyz
        )
        # Find the z slice closest to z=0
        z_vals = xyz[2][0, 0, :]
        iz = int(jnp.argmin(jnp.abs(z_vals)))
        z_actual = float(z_vals[iz])
        model_2d_from_3d = model_3d[..., iz]

        # Minkasi closed-form 2D result at that z
        model_2d_analytical = isobeta_2d_analytical(
            dx, dy, theta_core, beta, amp, xyz, z_val=z_actual
        )

        assert np.allclose(model_2d_from_3d, model_2d_analytical, rtol=1e-2, atol=1e-5)

    def test_gaussian_3d_vs_2d_integrated(self, wide_grid):
        """Test 3D gaussian integration against analytical 2D gaussian"""
        xyz, dz = wide_grid
        dx, dy, dz_offset = 0.0, 0.0, 0.0
        sigma = 10.0
        amp = 1.0

        model_3d = structure.egaussian(
            dx, dy, dz_offset, 1.0, 1.0, 1.0, 0.0, sigma, amp, xyz
        )

        model_2d_from_3d = jnp.sum(model_3d, axis=2) * dz

        model_2d_analytical = gaussian_2d_integrated(dx, dy, sigma, amp, xyz)

        assert np.allclose(model_2d_from_3d, model_2d_analytical, rtol=1e-2)

    def test_sph_isobeta_vs_analytical_slice(self, test_grid):
        """Test sph_isobeta 3D z-slice against analytical formula"""
        xyz, dz = test_grid
        dx, dy, dz_offset = 0.0, 0.0, 0.0
        r = 10.0
        beta = 0.7
        amp = 1.0

        model_3d = structure.sph_isobeta(dx, dy, dz_offset, r, 0.0, beta, amp, xyz)

        z_vals = xyz[2][0, 0, :]
        iz = int(jnp.argmin(jnp.abs(z_vals)))
        z_actual = float(z_vals[iz])
        model_2d_from_3d = model_3d[..., iz]

        model_2d_analytical = sph_isobeta_2d_slice(
            dx, dy, r, beta, amp, xyz, z_val=z_actual
        )

        assert np.allclose(model_2d_from_3d, model_2d_analytical, rtol=1e-2, atol=1e-5)

    def test_cylindrical_beta_3d_vs_2d(self, wide_grid):
        """
        Test cylindrical_beta 3D integration along z against cylindrical_beta_2d.
        Cylinder aligned along x-axis, phi=0 (no LoS tilt), so sec(phi)=1.
        The 3D model integrates P0*(1 + (y²+z²)/r_c²)^(-3β/2) along z,
        giving the same closed form as cylindrical_beta_2d.
        """
        xyz, dz = wide_grid
        dx, dy, dz_offset = 0.0, 0.0, 0.0
        L = 700.0  # so that cutoff never triggers within grid (bc > 2*r_map which equals 600)
        theta = 0.0
        phi = 0.0  # no tilt, sec(phi)=1
        P0 = 1.0
        r_c = 10.0
        beta = 0.7

        model_3d = structure.cylindrical_beta(
            dx, dy, dz_offset, L, theta, P0, r_c, beta, xyz
        )

        model_2d_from_3d = jnp.sum(model_3d, axis=2) * dz

        model_2d_analytical = structure.cylindrical_beta_2d(
            dx, dy, dz_offset, L, theta, phi, P0, r_c, beta, xyz
        )

        assert np.allclose(model_2d_from_3d, model_2d_analytical, rtol=1e-2)
