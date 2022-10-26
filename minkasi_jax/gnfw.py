import jax

jax.config.update("jax_enable_x64", true)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import jax.scipy as jsp

from astropy.cosmology import planck15 as cosmo
from astropy import constants as const
from astropy import units as u

import scipy as sp
import numpy as np
import timeit
import time

import matplotlib.pyplot as plt

from functools import partial
from utils import dzline, hzline, nzline, daline, h70, Mparsec, Xthom, me, y2K_RJ, fft_conv, make_grid, add_shock, add_bubble 

# gNFW Bubble
@partial(jax.jit, static_argnums=(8, 9 ,10, 15, 16, 17, 18, 19, 20))
def _gnfw_bubble(
    x0, y0, P0, c500, alpha, beta, gamma, m500, xb, yb, rb, sup,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    #A function for computing a gnfw+2 bubble model. The bubbles have pressure = b*P_gnfw, and are integrated along the line of sight
    #Inputs: 
    #    x0, y0 cluster center location
    #    P0, c500, alpha, beta, gamma, m500 gnfw params
    #    xb1, yb1, rb1: bubble location and radius for bubble
    #    sup: the supression factor, refered to as b above
    #    xi, yi: the xi, yi to evaluate at

    #Outputs:
    #    full_map, a 2D map of the sz signal from the gnfw_profile + 2 bubbles
    hz = jnp.interp(z, dzline, hzline)
    nz = jnp.interp(z, dzline, nzline)
    da = jnp.interp(z, dzline, daline)

    ap = 0.12

    #calculate some relevant contstants
    r500 = (m500 / (4.00 * np.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    P500 = (
        1.65e-03
        * (m500 / (3.00e14 / h70)) ** (2.00 / 3.00 + ap)
        * hz ** (8.00 / 3.00)
        * h70 ** 2
    )

    #Set up an r range at which to evaluate P_gnfw for interpolating the gnfw radial profile
    dR = max_R / 2e3
    r = jnp.arange(0.00, max_R, dR) + dR / 2.00

    #Compute the pressure as a function of radius
    x = r / r500
    pressure = (
        P500
        * P0
        / (
            (c500 * x) ** gamma
            * (1.00 + (c500 * x) ** alpha) ** ((beta - gamma) / alpha)
        )
    )

    XMpc = Xthom * Mparsec
    
    #Make a grid, centered on xb1, yb1 and size rb1, with resolution dr, and convert to Mpc
    # x_b = jnp.arange(-1*rb+xb, rb+xb, dr) * da
    # y_b = jnp.arange(-1*rb-yb, rb-yb, dr) * da
    # z_b = jnp.arange(-1*rb, rb, dr) * da
    x_b = jnp.linspace(-1*rb+xb, rb+xb, 2*int(rb/dr)) * da
    y_b = jnp.linspace(-1*rb-yb, rb-yb, 2*int(rb/dr)) * da
    z_b = jnp.linspace(-1*rb, rb, 2*int(rb/dr)) * da
    
    xyz_b = jnp.meshgrid(x_b, y_b, z_b)

    #Similar to above, make a 3d xyz cube with grid values = radius, and then interpolate with gnfw profile to get 3d gNFW profile
    rr_b = jnp.sqrt(xyz_b[0]**2 + xyz_b[1]**2 + xyz_b[2]**2)
    yy_b = jnp.interp(rr_b, r, pressure, right = 0.0) 

    #Set up a grid of points for computing distance from bubble center
    x_rb = jnp.linspace(-1,1, len(x_b))
    y_rb = jnp.linspace(-1,1, len(y_b))
    z_rb = jnp.linspace(-1,1, len(z_b))
    rb_grid = jnp.meshgrid(x_rb, y_rb, z_rb)

  
    #Zero out points outside bubble
    #outside_rb_flag = jnp.sqrt(rb_grid[0]**2 + rb_grid[1]**2+rb_grid[2]**2) >=1    
    #yy_b = jax.ops.index_update(yy_b, jax.ops.index[outside_rb_flag], 0.)

    yy_b = jnp.where(jnp.sqrt(rb_grid[0]**2 + rb_grid[1]**2+rb_grid[2]**2) >=1, 0., yy_b)

    #integrated along z/line of sight to get the 2D line of sight integral. Also missing it's dz term
    ip_b = -sup*jnp.trapz(yy_b, dx=dr*da, axis = -1) * XMpc / me

    return ip_b

# Beam-convolved gNFW profiel
# --------------------------------------------------------
@partial(jax.jit, static_argnums=(11, 12, 13, 14, 15, 16))
def _conv_int_gnfw(
    x0, y0, P0, c500, alpha, beta, gamma, m500,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    hz = jnp.interp(z, dzline, hzline)
    nz = jnp.interp(z, dzline, nzline)
    da = jnp.interp(z, dzline, daline)
    
    ap = 0.12

    r500 = (m500 / (4.00 * jnp.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    P500 = (
        1.65e-03
        * (m500 / (3.00e14 / h70)) ** (2.00 / 3.00 + ap)
        * hz ** (8.00 / 3.00)
        * h70 ** 2
    )

    dR = max_R / 2e3
    r = jnp.arange(0.00, max_R, dR) + dR / 2.00

    x = r / r500
    pressure = (
        P500
        * P0
        / (
            (c500 * x) ** gamma
            * (1.00 + (c500 * x) ** alpha) ** ((beta - gamma) / alpha)
        )
    )

    rmap = jnp.arange(1e-10, r_map, dr)
    r_in_Mpc = rmap * da
    rr = jnp.meshgrid(r_in_Mpc, r_in_Mpc)
    rr = jnp.sqrt(rr[0] ** 2 + rr[1] ** 2)
    yy = jnp.interp(rr, r, pressure, right=0.0)

    XMpc = Xthom * Mparsec

    ip = jnp.trapz(yy, dx=dr*da, axis=-1) * 2.0 * XMpc / me

    return rmap, ip


def conv_int_gnfw(
    x0, y0, P0, c500, alpha, beta, gamma, m500,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    rmap, ip = _conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500,
        xi,
        yi,
        z,
        max_R=max_R,
        fwhm=fwhm,
        freq=freq,
        T_electron=T_electron,
        r_map=r_map,
        dr=dr,
    )

    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam = jnp.exp(-4 * np.log(2) * x ** 2 / fwhm ** 2)
    beam = beam / jnp.sum(beam)

    nx = x.shape[0] // 2 + 1

    ipp = jnp.concatenate((ip[0:nx][::-1], ip))
    ip = jnp.convolve(ipp, beam, mode="same")[nx:]

    ip = ip * y2K_RJ(freq=freq, Te=T_electron)

    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0
    dr = jnp.sqrt(dx * dx + dy * dy) * 180.0 / np.pi * 3600.0

    return jnp.interp(dr, rmap, ip, right=0.0)

def _conv_int_gnfw_elliptical(
    e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    """
    Modification of conv_int_gnfw that adds ellipticity. This function does not include smoothing or declination stretch
    which should be applied at the end
   

    Arguments:
       Same as conv_int_gnfw except first 3 values of p are now:

           e: Eccentricity, fixing the semimajor

           theta: Angle to rotate profile by in radians

       remaining args are the same as conv_int_gnfw

    Returns:
       Elliptical gnfw profile
    """
   

    rmap, ip = _conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500,
        xi,
        yi,
        z,
        max_R=max_R,
        fwhm=fwhm,
        freq=freq,
        T_electron=T_electron,
        r_map=r_map,
        dr=dr,
    )

    dx = xi - x0
    dy = yi-y0

    dr = jnp.sqrt((dx*jnp.cos(theta) + dy * jnp.sin(theta))**2 + (dx * jnp.sin(theta) - dy * jnp.cos(theta))**2/(1-e**2)) * 180.0 / np.pi * 3600.0

    return jnp.interp(dr, rmap, ip, right=0.0)



def conv_int_gnfw_elliptical(
    e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    """
    Modification of conv_int_gnfw that adds ellipticity
    This is a somewhat crude implementation that could be improved in the future

    Arguments:
       Same as conv_int_gnfw except first 3 values of p are now:

           e: Eccentricity, fixing the semimajor

           theta: Angle to rotate profile by in radians

       remaining args are the same as conv_int_gnfw

    Returns:
       Elliptical gnfw profile
    """
    rmap, ip = _conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500,
        xi,
        yi,
        z,
        max_R=max_R,
        fwhm=fwhm,
        freq=freq,
        T_electron=T_electron,
        r_map=r_map,
        dr=dr,
    )
    
    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam = jnp.exp(-4 * np.log(2) * x ** 2 / fwhm ** 2)
    beam = beam / jnp.sum(beam)

    nx = x.shape[0] // 2 + 1

    ipp = jnp.concatenate((ip[0:nx][::-1], ip))
    ip = jnp.convolve(ipp, beam, mode="same")[nx:]

    ip = ip * y2K_RJ(freq=freq, Te=T_electron)

    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0
   
    dr = np.sqrt((dx*jnp.cos(theta) + dy * jnp.sin(theta))**2 + (dx * jnp.sin(theta) - dy * jnp.cos(theta))**2/(1-e**2)) * 180.0 / np.pi * 3600.0
    
    return jnp.interp(dr, rmap, ip, right=0.0)



def conv_int_gnfw_elliptical_two_bubbles(
    e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500,
    xb1, yb1, rb1, sup1,
    xb2, yb2, rb2, sup2,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    '''
    A hacky way of computing an eliptical gnfw + two bubble model. Currently computes a sphirically symmetric 
    gnfw model, the stretches it to make it eliptical. Then bubbles computed using that sphirically symmetric
    model are added at the right place. The issue here is that the profile under the bubbles is not the profile 
    used to compute the bubbles due to the eliptical stretching. Still, hopefully it's close enough.
    '''
    ''' 
    rmap, ip = _conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500,
        xi,
        yi,
        z,
        max_R=max_R,
        fwhm=fwhm,
        freq=freq,
        T_electron=T_electron,
        r_map=r_map,
        dr = dr,
    )
    '''
    da = jnp.interp(z, dzline, daline)

    #Set up a 2d xy map which we will interpolate over to get the 2D gnfw pressure profile
    x = jnp.arange(-1*r_map, r_map, dr) * jnp.pi / (180*3600)
    y = jnp.arange(-1*r_map, r_map, dr) * jnp.pi / (180*3600) 
    
    
    ''' 
    _x = jnp.array(x, copy=True)
    y = y / jnp.sqrt(1 - (abs(e)%1)**2)
    x = x * jnp.cos(theta) + y * jnp.sin(theta)
    y = -1 * _x *jnp.sin(theta) + y * jnp.cos(theta)
    print('jorlo x', x)
    '''
    xx, yy = jnp.meshgrid(x, y)
    
    #This might be inefficient?
 
    ip = _conv_int_gnfw_elliptical(
        e, theta, 0., 0., P0, c500, alpha, beta, gamma, m500,
        xx,
        yy,
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    ip_b = _gnfw_bubble(
        x0, y0, P0, c500, alpha, beta, gamma, m500, xb1, yb1, rb1, sup1,
        xi,
        yi,
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )
    
    idx = jax.ops.index[(int(ip.shape[1]/2)-int(rb1/dr)-int(yb1/dr)):(int(ip.shape[1]/2)+int(rb1/dr)-int(yb1/dr)), (int(ip.shape[0]/2)-int(rb1/dr)+int(xb1/dr)):(int(ip.shape[0]/2)+int(rb1/dr)+int(xb1/dr))]
    ip = jax.ops.index_add(ip, idx, ip_b)

    ip_b = _gnfw_bubble(
        x0, y0, P0, c500, alpha, beta, gamma, m500, xb2, yb2, rb2, sup2,
        xi,
        yi,
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    idx = jax.ops.index[(int(ip.shape[1]/2)-int(rb2/dr)-int(yb2/dr)):(int(ip.shape[1]/2)+int(rb2/dr)-int(yb2/dr)), (int(ip.shape[0]/2)-int(rb2/dr)+int(xb2/dr)):(int(ip.shape[0]/2)+int(rb2/dr)+int(xb2/dr))]
    ip = jax.ops.index_add(ip, idx, ip_b)

    #Sum of two gaussians with amp1, fwhm1, amp2, fwhm2
    amp1, fwhm1, amp2, fwhm2 = 9.735, 0.9808, 32.627, 0.0192
    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam_xx, beam_yy = jnp.meshgrid(x,x)
    beam_rr = jnp.sqrt(beam_xx**2 + beam_yy**2)
    beam = amp1*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm1 ** 2) + amp2*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm2 ** 2)
    beam = beam / jnp.sum(beam)

    bound0, bound1 = int((ip.shape[0]-beam.shape[0])/2), int((ip.shape[1] - beam.shape[1])/2)


    beam = jnp.pad(beam, ((bound0, ip.shape[0]-beam.shape[0]-bound0), (bound1, ip.shape[1] - beam.shape[1] - bound1)))

    ip = fft_conv(ip, beam)
    ip = ip * y2K_RJ(freq=freq, Te=T_electron) 
    
    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0
    
    dx *= (180*3600)/jnp.pi
    dy *= (180*3600)/jnp.pi
    full_rmap = jnp.arange(-1*r_map, r_map, dr) * da
    
    idx, idy = (dx + r_map)/(2*r_map)*len(full_rmap), (dy + r_map)/(2*r_map)*len(full_rmap)
    
    return jsp.ndimage.map_coordinates(ip, (idy,idx), order = 0)#, ip


                                                                     
@partial(
   jax.jit, 
   static_argnums=(8, 9, 10, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24)
)
def conv_int_gnfw_two_bubbles(
    x0, y0, P0, c500, alpha, beta, gamma, m500,
    xb1, yb1, rb1, sup1,
    xb2, yb2, rb2, sup2,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    rmap, ip = _conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500,
        xi,
        yi,
        z,
        max_R=max_R,
        fwhm=fwhm,
        freq=freq,
        T_electron=T_electron,
        r_map=r_map,
        dr=dr,
    )
    
    da = jnp.interp(z, dzline, daline)  
    
    #Set up a 2d xy map which we will interpolate over to get the 2D gnfw pressure profile
    full_rmap = jnp.arange(-1*r_map, r_map, dr) * da
    xx, yy = jnp.meshgrid(full_rmap, full_rmap)
    full_rr = jnp.sqrt(xx**2 + yy**2)
    ip = jnp.interp(full_rr, rmap*da, ip)
    
    ip_b = _gnfw_bubble(
        x0, y0, P0, c500, alpha, beta, gamma, m500, xb1, yb1, rb1, sup1,
        xi,
        yi,
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    ip = jax.ops.index_add(ip, jax.ops.index[int(ip.shape[1]/2+int((-1*rb1-yb1)/dr)):int(ip.shape[1]/2+int((rb1-yb1)/dr)),
       int(ip.shape[0]/2+int((-1*rb1+xb1)/dr)):int(ip.shape[0]/2+int((rb1+xb1)/dr))], ip_b)
    
    ip_b = _gnfw_bubble(
        x0, y0, P0, c500, alpha, beta, gamma, m500, xb2, yb2, rb2, sup2,
        xi,
        yi,
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    ip = jax.ops.index_add(ip, jax.ops.index[int(ip.shape[1]/2+int((-1*rb2-yb2)/dr)):int(ip.shape[1]/2+int((rb2-yb2)/dr)),
       int(ip.shape[0]/2+int((-1*rb2+xb2)/dr)):int(ip.shape[0]/2+int((rb2+xb2)/dr))], ip_b)
   
    #Sum of two gaussians with amp1, fwhm1, amp2, fwhm2
    amp1, fwhm1, amp2, fwhm2 = 9.735, 0.9808, 32.627, 0.0192 
    x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    beam_xx, beam_yy = jnp.meshgrid(x,x)
    beam_rr = jnp.sqrt(beam_xx**2 + beam_yy**2)
    beam = amp1*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm1 ** 2) + amp2*jnp.exp(-4 * jnp.log(2) * beam_rr ** 2 / fwhm2 ** 2)
    beam = beam / jnp.sum(beam)

    bound0, bound1 = int((ip.shape[0]-beam.shape[0])/2), int((ip.shape[1] - beam.shape[1])/2)
 

    beam = jnp.pad(beam, ((bound0, ip.shape[0]-beam.shape[0]-bound0), (bound1, ip.shape[1] - beam.shape[1] - bound1)))
 

    #ip = jsp.signal.convolve2d(ip, beam, mode = 'same')
    ip = fft_conv(ip, beam)
    ip = ip * y2K_RJ(freq=freq, Te=T_electron)

    dx = (xi - x0) * jnp.cos(yi)
    dy = yi - y0

    dx *= (180*3600)/jnp.pi
    dy *= (180*3600)/jnp.pi
    #Note this may  need to be changed for eliptical gnfws?
    idx, idy = (dx + r_map)/(2*r_map)*len(full_rmap), (dy + r_map)/(2*r_map)*len(full_rmap)
    return jsp.ndimage.map_coordinates(ip, (idx,idy), order = 0)#, ip 
   
   
    #temp = np.meshgrid(jnp.arange(-1*r_map, r_map, dr), jnp.arange(-1*r_map, r_map, dr))
    #print(np.arange(-1*r_map, r_map, dr).shape) 
    #bound = 10*np.pi/(180*3600)
    #x = np.linspace(-1*bound, bound, 200)
    #y = np.linspace(-1*bound, bound, 200)

    #return ip 

@partial(
    jax.jit,
    static_argnums=(
        3,
        4,
        5,
        6,
        7,
        8,
    ),
)
def val_conv_int_gnfw(
    p,
    tods,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
):
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p
    return conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )


@partial(
    jax.jit,
    static_argnums=(
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ),
)
def jac_conv_int_gnfw_fwd(
    p,
    tods,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7,)
):
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p
    grad = jax.jacfwd(conv_int_gnfw, argnums=argnums)(
        x0, y0, P0, c500, alpha, beta, gamma, m500, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jnp.array(grad)

    if len(argnums) != len(p):
        padded_grad = jnp.zeros(p.shape + grad[0].shape) + 1e-30
        grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return grad


@partial(
    jax.jit,
    static_argnums=(
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ),
)
def jit_conv_int_gnfw(
    p,
    tods,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7,)
    ):
    x0, y0, P0, c500, alpha, beta, gamma, m500 = p
    pred = conv_int_gnfw(
        x0, y0, P0, c500, alpha, beta, gamma, m500, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )

    if len(argnums) == 0:
        return pred, jnp.zeros(p.shape + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_gnfw, argnums=argnums)(
        x0, y0, P0, c500, alpha, beta, gamma, m500, tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jnp.array(grad)

    if len(argnums) != len(p):
        padded_grad = jnp.zeros(p.shape + pred.shape) + 1e-30
        grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad


@partial(
    jax.jit,
    static_argnums=(
        6,
        7,
        8,
        9,
        10,
        11,
        12
    ),
)
def jit_conv_int_gnfw_elliptical(
    p,
    tods,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,)
):
    e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500 = p
    pred = conv_int_gnfw_elliptical(
        e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500,
        tods[0],
        tods[1],
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )

    if len(argnums) == 0:
        return pred, jnp.zeros(p.shape + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_gnfw_elliptical, argnums=argnums)(
        e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500,
        tods[0],
        tods[1],
        z,
        max_R,
        fwhm,
        freq,
        T_electron,
        r_map,
        dr,
    )
    grad = jnp.array(grad)

    if len(argnums) != len(p):
        padded_grad = jnp.zeros(p.shape + pred.shape) + 1e-30
        grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad


@partial(
    jax.jit, 
    static_argnums=(
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13, 
        14,
        15
    ),
)

def jit_conv_int_gnfw_two_bubbles(
    p,
    tods,
    z,
    xb1, yb1, rb1,
    xb2, yb2, rb2,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    ):
   
    x0, y0, P0, c500, alpha, beta, gamma, m500, sup1, sup2 = p
    pred = conv_int_gnfw_two_bubbles(
        x0, y0, P0, c500, alpha, beta, gamma, m500, xb1, yb1, rb1, sup1, xb2, yb2, rb2, sup2 , tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    
    if len(argnums) == 0:
        return pred, jnp.zeros((len(p)+6,) + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_gnfw_two_bubbles, argnums=argnums)(
        x0, y0, P0, c500, alpha, beta, gamma, m500, xb1, yb1, rb1, sup1, xb2, yb2, rb2, sup2 , tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jnp.array(grad)
   
    padded_grad = jnp.zeros((len(p)+6,) + pred.shape) + 1e-30
    argnums = jnp.array(argnums)
    grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad

@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17
    ),
)

def jit_conv_int_gnfw_elliptical_two_bubbles(
    e,
    theta,
    p,
    tods,
    z,
    xb1, yb1, rb1,
    xb2, yb2, rb2,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.1,
    argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    ):

    x0, y0, P0, c500, alpha, beta, gamma, m500, sup1, sup2 = p

    pred = conv_int_gnfw_elliptical_two_bubbles(
        e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500, xb1, yb1, rb1, sup1, xb2, yb2, rb2, sup2 , tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    
    if len(argnums) == 0:
        return pred, jnp.zeros((len(p)+6,) + pred.shape) + 1e-30

    grad = jax.jacfwd(conv_int_gnfw_elliptical_two_bubbles, argnums=argnums)(
        e, theta, x0, y0, P0, c500, alpha, beta, gamma, m500, xb1, yb1, rb1, sup1, xb2, yb2, rb2, sup2 , tods[0], tods[1], z, max_R, fwhm, freq, T_electron, r_map, dr
    )
    grad = jnp.array(grad)

    padded_grad = jnp.zeros((len(p)+6,) + grad[0].shape) + 1e-30
    argnums = jnp.array(argnums)
    grad = padded_grad.at[jnp.array(argnums)].set(jnp.array(grad))

    return pred, grad
