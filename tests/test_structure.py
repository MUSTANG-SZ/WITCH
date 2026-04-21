"""
Unit tests for structure.py - testing 3D models against analytical 2D solutions
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.analytical_2d import (
    add_uniform_2d_analytical,
    gaussian_2d_integrated,
    isobeta_2d_analytical,
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

    def test_isobeta_vs_analytical_2d(self, wide_grid):
        """
        Test isobeta 3D numerical z-integration against closed-form 2D result.
        Uses theta_core << r_map so the LoS integral converges on the grid.
        (r_map=180 arcsec ~3arcmin, theta_core=10 arcsec per Jack's guidance)
        """
        xyz, dz = wide_grid
        dx, dy, dz_offset = 0.0, 0.0, 0.0
        theta_core = 10.0  # arcsec, well below r_map=180 arcsec
        beta = 0.7
        amp = 1.0

        model_3d = structure.isobeta(
            dx, dy, dz_offset, theta_core, theta_core, theta_core, 0.0, beta, amp, xyz
        )

        model_2d_from_3d = jnp.sum(model_3d, axis=2) * dz

        model_2d_analytical = isobeta_2d_analytical(dx, dy, theta_core, beta, amp, xyz)

        assert np.allclose(model_2d_from_3d, model_2d_analytical, rtol=1e-2)

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

    def test_sph_isobeta_vs_analytical_2d(self, wide_grid):
        """
        Test sph_isobeta 3D numerical z-integration against closed-form 2D result.
        """
        xyz, dz = wide_grid
        dx, dy, dz_offset = 0.0, 0.0, 0.0
        r = 10.0
        beta = 0.7
        amp = 1.0

        model_3d = structure.sph_isobeta(dx, dy, dz_offset, r, 0.0, beta, amp, xyz)

        model_2d_from_3d = jnp.sum(model_3d, axis=2) * dz

        model_2d_analytical = isobeta_2d_analytical(dx, dy, r, beta, amp, xyz)

        assert np.allclose(model_2d_from_3d, model_2d_analytical, rtol=1e-2)

    @pytest.mark.xfail(reason="Known bug in cylindrical_beta_2d: see issue #178")
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
        phi = 0.0
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

    def test_gnfw_matches_reference(self, test_grid, gnfw_reference):
        """
        Test gnfw output matches cached reference.
        """
        xyz, dz = test_grid

        model = structure.gnfw(
            dx=0.0,
            dy=0.0,
            dz=0.0,
            r=1.0,
            P0=8.403,
            c500=1.177,
            m500=1e15,
            gamma=0.3081,
            alpha=1.0510,
            beta=5.4905,
            z=0.5,
            xyz=xyz,
        )
        model_2d = jnp.sum(model, axis=2) * dz

        assert np.allclose(model_2d, gnfw_reference, rtol=1e-5)

    def test_a10_matches_reference(self, test_grid, a10_reference):
        """
        Test a10 output matches cached reference.
        """
        xyz, dz = test_grid

        model = structure.a10(
            dx=0.0,
            dy=0.0,
            dz=0.0,
            theta=1.0,
            P0=8.403,
            c500=1.177,
            m500=1e15,
            gamma=0.3081,
            alpha=1.0510,
            beta=5.4905,
            z=0.5,
            xyz=xyz,
        )
        model_2d = jnp.sum(model, axis=2) * dz

        assert np.allclose(model_2d, a10_reference, rtol=1e-5)

    def test_add_uniform_vs_analytical_2d(self):
        """
        Test add_uniform 3D integration against closed-form 2D result.
        Uses a fine grid to minimize boundary discretization error.
        """
        dr = 0.5  # finer resolution reduces boundary pixel error
        dz = 0.5
        r_map = 60.0
        xyz = grid.make_grid(r_map, dr, dr, dz, 0.0, 0.0)

        dx, dy, dz_offset = 0.0, 0.0, 0.0
        r = 20.0
        amp = 2.0

        n = xyz[0].shape[0]
        pressure = jnp.ones((n, n, n))

        model_3d = structure.add_uniform(
            pressure, xyz, dx, dy, dz_offset, r, r, r, 0.0, amp
        )

        diff_2d = jnp.sum(model_3d - pressure, axis=2) * dz

        model_2d_analytical = add_uniform_2d_analytical(dx, dy, r, amp, xyz)

        assert np.allclose(diff_2d, model_2d_analytical, rtol=1e-2, atol=1.0)
