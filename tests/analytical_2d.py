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
) -> jnp.ndarray:
    """
    Definite integral of 3D isobeta from -r_map to +r_map along z.

    ∫_{-z_max}^{z_max} amp * (1 + R²/θ² + z²/θ²)^(-3β/2) dz
    = 2 * amp * θ * a^(1-3β) * T * 2F1(1/2, 3β/2; 3/2; -T²)

    where a = sqrt(1 + R²/θ²) and T = z_max / (θ * a)
    """
    from scipy.special import hyp2f1

    x, y, z_arr, *_ = transform_grid(dx, dy, 0, 1, 1, 1, 0, xyz)
    x_2d = np.array(x[..., 0])
    y_2d = np.array(y[..., 0])

    z_max = float(np.abs(z_arr[0, 0, -1]))

    n = 1.5 * beta
    r_sq = x_2d**2 + y_2d**2
    a = np.sqrt(1 + r_sq / theta**2)
    T = z_max / (theta * a)

    result = 2 * amp * theta * a ** (1 - 2 * n) * T * hyp2f1(0.5, n, 1.5, -(T**2))
    return jnp.array(result)


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
