"""
Analytical 2D solutions for structure profiles
"""

import jax.numpy as jnp
import numpy as np
from scipy.special import gamma

from witch.grid import transform_grid


def isobeta_2d_analytical(
    dx: float,
    dy: float,
    theta: float,
    beta: float,
    amp: float,
    xyz: tuple,
    z_val: float = 0.0,  # actual z coordinate of the slice
) -> jnp.ndarray:
    """
    3D isobeta evaluated at a given z slice.
    Matches structure.isobeta: amp * (1 + R²/θ² + z²/θ²)^(-3β/2)
    """
    x, y, *_ = transform_grid(dx, dy, 0, 1, 1, 1, 0, xyz)
    x_2d = x[..., 0]
    y_2d = y[..., 0]

    r_sq = x_2d**2 + y_2d**2
    rr = 1 + r_sq / theta**2 + z_val**2 / theta**2
    power = -1.5 * beta

    return amp * rr**power


def gaussian_2d_integrated(
    dx: float,
    dy: float,
    sigma: float,
    amp: float,
    xyz: tuple,
) -> jnp.ndarray:
    """
    Analytical result of integrating 3D gaussian along z.

    For P(x,y,z) = amp * exp(-(x²+y²+z²)/(2σ²))
    Integrating along z: amp * sqrt(2π) * σ * exp(-(x²+y²)/(2σ²))
    """
    x, y, *_ = transform_grid(dx, dy, 0, 1, 1, 1, 0, xyz)
    x_2d = x[..., 0]
    y_2d = y[..., 0]

    r_sq = x_2d**2 + y_2d**2
    amp_integrated = amp * jnp.sqrt(2 * jnp.pi) * sigma

    return amp_integrated * jnp.exp(-r_sq / (2 * sigma**2))


def cylindrical_beta_2d_analytical(
    dx: float,
    dy: float,
    dz: float,
    L: float,
    theta: float,
    phi: float,
    P0: float,
    r_c: float,
    beta: float,
    xyz: tuple,
) -> jnp.ndarray:
    """
    Wrapper around structure.cylindrical_beta_2d for use in tests.
    phi=0 means cylinder is along line of sight, sec(phi)=1.
    """
    from witch import structure

    return structure.cylindrical_beta_2d(dx, dy, dz, L, theta, phi, P0, r_c, beta, xyz)


def sph_isobeta_2d_slice(
    dx: float,
    dy: float,
    r: float,
    beta: float,
    amp: float,
    xyz: tuple,
    z_val: float = 0.0,
) -> jnp.ndarray:
    """
    3D sph_isobeta evaluated at a given z slice.
    structure.sph_isobeta: amp * (1 + r²)^(-3β/2)
    where r is already scaled by transform_grid with scale=r.
    So at z=z_val: amp * (1 + R²/r² + z_val²/r²)^(-3β/2)
    """
    x, y, *_ = transform_grid(dx, dy, 0, 1, 1, 1, 0, xyz)
    x_2d = x[..., 0]
    y_2d = y[..., 0]

    r_sq = x_2d**2 + y_2d**2
    rr = 1 + r_sq / r**2 + z_val**2 / r**2
    power = -1.5 * beta

    return amp * rr**power
