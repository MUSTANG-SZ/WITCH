"""
Module for generating isobeta profiles with substructure and their gradients.
"""

from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from utils import fft_conv, add_shock, add_bubble

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

N_PAR_PROFILE = 6
N_PAR_SHOCK = 5
N_PAR_BUBBLE = 5


def isobeta_heper(params, tod, xyz, n_profiles, n_shocks, n_bubbles, dx, beam):
    """
    Helper function to be used when fitting with Minkasi.
    Use functools.partial to set all parameters but params and tod before passing to Minkasi.

    Arguments:

        params: 1D array of model parameters.

        tod: The TOD, assumed that idx and idy are in tod.info.

        xyz: Coordinate grid to compute profile on.

        n_profiles: Number of isobeta profiles to add.

        n_shocks: Number of shocks to add.

        n_bubbles: Number of bubbles to add.

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

    Returns:

        derivs: The gradient of the model with respect to the model parameters.
                Reshaped to be the correct format for Minkasi.

        pred: The isobeta model with the specified substructure.
    """
    idx = tod.info["idx"]
    idy = tod.info["idy"]

    profiles, shocks, bubbles = jnp.zeros((1, 1), dtype=float)
    start = 0
    if n_profiles:
        delta = n_profiles * N_PAR_PROFILE
        profiles = params[start : start + delta].reshape((n_profiles, N_PAR_PROFILE))
        start += delta
    if n_shocks:
        delta = n_shocks * N_PAR_SHOCK
        shocks = params[start : start + delta].reshape((n_shocks, N_PAR_SHOCK))
        start += delta
    if n_bubbles:
        delta = n_bubbles * N_PAR_BUBBLE
        bubbles = params[start : start + delta].reshape((n_bubbles, N_PAR_BUBBLE))
        start += delta

    pred, grad = isobeta_grad(
        xyz,
        n_profiles,
        profiles,
        n_shocks,
        shocks,
        n_bubbles,
        bubbles,
        dx,
        beam,
        idx,
        idy,
    )

    derivs = []
    if n_profiles:
        derivs.append(jnp.vstack(grad[0]))
    if n_shocks:
        derivs.append(jnp.vstack(grad[1]))
    if n_bubbles:
        derivs.append(jnp.vstack(grad[2]))
    derivs = jnp.vstack(derivs)

    return derivs, pred


@jax.jit
def _isobeta_elliptical(r_1, r_2, r_3, theta, beta, amp, xyz):
    """
    Elliptical isobeta pressure profile in 3d
    This function does not include smoothing or declination stretch
    which should be applied at the end.

    Arguments:

        r_1: Amount to scale along x-axis

        r_2: Amount to scale along y-axis

        r_3: Amount to scale along z-axis

        theta: Angle to rotate in xy-plane

        beta: Beta value of isobeta model

        amp: Amplitude of isobeta model

        xyz: Coordinte grid to calculate model on

    Returns:

        model: The isobeta model centered on the origin of xyz
    """
    # Rotate
    xx = xyz[0] * jnp.cos(theta) + xyz[1] * jnp.sin(theta)
    yy = xyz[1] * jnp.cos(theta) - xyz[0] * jnp.sin(theta)
    zz = xyz[2]

    # Apply ellipticity
    xfac = (xx / r_1) ** 2
    yfac = (yy / r_2) ** 2
    zfac = (zz / r_3) ** 2

    # Calculate pressure profile
    rr = 1 + xfac + yfac + zfac
    power = -1.5 * beta
    rrpow = rr**power

    return amp * rrpow


@partial(
    jax.jit,
    static_argnums=(
        1,
        3,
        5,
        7,
    ),
)
def isobeta(
    xyz, n_profiles, profiles, n_shocks, shocks, n_bubbles, bubbles, dx, beam, idx, idy
):
    """
    Generically create isobeta models with substructure.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_profiles: Number of isobeta profiles to add.

        profile: 2d array of profile parameters.
                 Each row is an isobeta profile with parameters:
                 (r_1, r_2, r_3, theta, beta, amp)

        n_shocks: Number of shocks to add.

        shocks: 2d array of shock parameters.
                Each row is a shock with parameters:
                (sr_1, sr_2, sr_3, s_theta, shock)

        n_bubbles: Number of bubbles to add.

        bubbles: 2d array of bubble parameters.
                 Each row is a bubble with parameters:
                 (xb, yb, zb, rb, sup)

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        idx: RA TOD in units of pixels.
             Should have Dec stretch applied.

        idy: Dec TOD in units of pixels.

    Returns:

        model: The isobeta model with the specified substructure.
    """
    pressure = jnp.zeros((xyz[0].shape[1], xyz[1].shape[0], xyz[2].shape[2]))
    for i in range(n_profiles):
        pressure = jnp.add(pressure, _isobeta_elliptical(*profiles[i], xyz))

    for i in range(n_shocks):
        pressure = add_shock(pressure, xyz, *shocks[i])

    for i in range(n_bubbles):
        pressure = add_bubble(pressure, xyz, *bubbles[i])

    # Integrate along line of site
    ip = jnp.trapz(pressure, dx=dx, axis=-1)

    bound0, bound1 = int((ip.shape[0] - beam.shape[0]) / 2), int(
        (ip.shape[1] - beam.shape[1]) / 2
    )
    beam = jnp.pad(
        beam,
        (
            (bound0, ip.shape[0] - beam.shape[0] - bound0),
            (bound1, ip.shape[1] - beam.shape[1] - bound1),
        ),
    )

    ip = fft_conv(ip, beam)

    return jsp.ndimage.map_coordinates(ip, (idy, idx), order=0)


# TODO: figure out how to skip parameters
@partial(
    jax.jit,
    static_argnums=(
        1,
        3,
        5,
        7,
    ),
)
def isobeta_grad(
    xyz, n_profiles, profiles, n_shocks, shocks, n_bubbles, bubbles, dx, beam, idx, idy
):
    """
    Generically create isobeta models with substructure and get their gradients.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_profiles: Number of isobeta profiles to add.

        profile: 2d array of profile parameters.
                 Each row is an isobeta profile with parameters:
                 (r_1, r_2, r_3, theta, beta, amp)

        n_shocks: Number of shocks to add.

        shocks: 2d array of shock parameters.
                Each row is a shock with parameters:
                (sr_1, sr_2, sr_3, s_theta, shock)

        n_bubbles: Number of bubbles to add.

        bubbles: 2d array of bubble parameters.
                 Each row is a bubble with parameters:
                 (xb, yb, zb, rb, sup, z)

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        idx: RA TOD in units of pixels.
             Should have Dec stretch applied.

        idy: Dec TOD in units of pixels.

    Returns:

        model: The isobeta model with the specified substructure.

        grad: The gradient of the model with respect to the model parameters.
    """
    pred = isobeta(
        xyz,
        n_profiles,
        profiles,
        n_shocks,
        shocks,
        n_bubbles,
        bubbles,
        dx,
        beam,
        idx,
        idy,
    )

    grad = jax.jacfwd(isobeta, argnums=(2, 4, 6))(
        xyz,
        n_profiles,
        profiles,
        n_shocks,
        shocks,
        n_bubbles,
        bubbles,
        dx,
        beam,
        idx,
        idy,
    )

    return pred, grad
