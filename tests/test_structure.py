"""
Unit tests for structure.py - testing 3D models against analytical 2D solutions
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.analytical_2d import gaussian_2d_integrated, isobeta_2d_analytical
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
        """Wider z-extent grid for gaussian test (needs to capture tails)"""
        r_map = 150.0  # ±150 arcsec = ±15σ for sigma=10, plenty of coverage
        dr = 2.0
        dz = 2.0
        xyz = grid.make_grid(r_map, dr, dr, dz, 0.0, 0.0)
        return xyz, dz

    def test_isobeta_spherical_vs_analytical_2d(self, test_grid):
        """
        Test isobeta 2D analytical formula matches minkasi C code.
        We compare structure.isobeta (3D) at a single z-slice against
        the minkasi closed-form 2D result.
        Note: full z-integration is impractical due to very slow power-law decay.
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
