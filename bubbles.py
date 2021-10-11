import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from astropy import units as u
from matplotlib import pyplot as plt

# Constants
# --------------------------------------------------------

h70 = cosmo.H0.value / 7.00e01

Tcmb = 2.7255
kb = const.k_B.value
me = ((const.m_e * const.c ** 2).to(u.keV)).value
h = const.h.value
Xthom = const.sigma_T.to(u.cm ** 2).value

Mparsec = u.Mpc.to(u.cm)

# Cosmology
# --------------------------------------------------------
dzline = np.linspace(0.00, 5.00, 1000)
daline = cosmo.angular_diameter_distance(dzline) / u.radian
nzline = cosmo.critical_density(dzline)
hzline = cosmo.H(dzline) / cosmo.H0

daline = daline.to(u.Mpc / u.arcsec)
nzline = nzline.to(u.Msun / u.Mpc ** 3)

dzline = np.array(dzline)
hzline = np.array(hzline.value)
nzline = np.array(nzline.value)
daline = np.array(daline.value)





def _gnfw_bubble(
    x0, y0, P0, c500, alpha, beta, gamma, m500, xb1, yb1, rb1, xb2, yb2, rb2, sup,
    xi,
    yi,
    z,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=15.0 * 60,
    dr=0.5,
):
    #A function for computing a gnfw+2 bubble model. The bubbles have pressure = b*P_gnfw, and are integrated along the line of sight
    #Inputs: 
    #    x0, y0 cluster center location
    #    P0, c500, alpha, beta, gamma, m500 gnfw params
    #    xb1, yb1, rb1: bubble location and radius for bubble 1
    #    xb2, yb2, rb2: same for bubble 2
    #    sup: the supression factor, refered to as b above
    #    xi, yi: the xi, yi to evaluate at

    #Outputs:
    #    full_map, a 2D map of the sz signal from the gnfw_profile + 2 bubbles
    hz = np.interp(z, dzline, hzline)
    nz = np.interp(z, dzline, nzline)

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
    r = np.arange(0.00, max_R, dR) + dR / 2.00

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

    #Set up 2D yz grid going out to the max radius of the map, with pixel size dr, and grid values = the radius from the center at that point
    rmap = np.arange(0, r_map, dr)
    r_in_Mpc = rmap * (np.interp(z, dzline, daline))
    rr = np.meshgrid(r_in_Mpc, r_in_Mpc)
    rr = np.sqrt(rr[0] ** 2 + rr[1] ** 2)

    #Interpolate to get the pressure at each grid point
    yy = np.interp(rr, r, pressure, right=0.0)
    
    XMpc = Xthom * Mparsec
    #integrate along the z-axis/line of sight to obtain the radial gnfw pressure proflie. Note we're missing the dz term here
    ip = np.sum(yy, axis=1) * 2.0 * XMpc / (me * 1000)
    
    #Set up a 2d xy map which we will interpolate over to get the 2D gnfw pressure profile
    full_rmap = np.arange(-1*r_map, r_map, dr) * (np.interp(z, dzline, daline))
    xx, yy = np.meshgrid(full_rmap, full_rmap)
    full_rr = np.sqrt(xx**2 + yy**2)
    full_map = np.interp(full_rr, r_in_Mpc, ip)




    #Make a grid, centered on xb1, yb1 and size rb1, with resolution dr, and convert to Mpc
    x_b1 = np.arange(-1*rb1+xb1, rb1+xb1, dr) * (np.interp(z, dzline, daline))
    y_b1 = np.arange(-1*rb1-yb1, rb1-yb1, dr) * (np.interp(z, dzline, daline))
    z_b1 = np.arange(-1*rb1, rb1, dr) * (np.interp(z, dzline, daline))

    xyz_b1 = np.meshgrid(x_b1, y_b1, z_b1)

    #Similar to above, make a 3d xyz cube with grid values = radius, and then interpolate with gnfw profile to get 3d gNFW profile
    rr_b1 = np.sqrt(xyz_b1[0]**2 + xyz_b1[1]**2 + xyz_b1[2]**2)
    yy_b1 = np.interp(rr_b1, r, pressure, right = 0.0) 

    #Set up a grid of points for computing distance from bubble center
    x_rb = np.linspace(-1,1, len(x_b1))
    y_rb = np.linspace(-1,1, len(y_b1))
    z_rb = np.linspace(-1,1, len(z_b1))
    rb_grid = np.meshgrid(x_rb, y_rb, z_rb)

  
    #Zero out points outside bubble
    outside_rb_flag = np.sqrt(rb_grid[0]**2 + rb_grid[1]**2+rb_grid[2]**2) >=1    
   
    yy_b1[outside_rb_flag] = 0

    #I make a 3D cube of just ones and apply the outside bubble filter and integrate along it's line of sight just to 
    #make sure the bubble filter works correctly
    test = sup*np.ones(yy_b1.shape)
    test[outside_rb_flag] = 0 
    
    #integrated along z/line of sight to get the 2D line of sight integral. Also missing it's dz term
    ip_b1 = -sup*np.sum(yy_b1, axis = -1) * 2.0 * XMpc / (me * 1000)

    #plot stuff for diagnostics
    plt.imshow(ip_b1)
    plt.colorbar()
    plt.savefig('bubble.png')
    plt.close()
    plt.imshow(np.sum(test, axis = -1))
    plt.colorbar()

 
    plt.savefig('unweight_bubble.png')
    plt.close()

    plt.imshow(full_map)
    plt.colorbar()
    plt.savefig('just_gnfw.png')
    plt.close() 
    print(np.amax(np.abs(full_map)), np.amax(np.abs(ip_b1)))
    
    #Add the bubble to the full 2d gnfw profile in the appropriate location. Note there is a small sub-pixel issue here:
    #if the bubble center and the gnfw profile center are not an integer * dr separated, then the two grids will be offset
    #by the remainder. This could be fixed but is likely not a huge issue. 
    full_map[int(full_map.shape[1]/2+int((-1*rb1-yb1)/dr)):int(full_map.shape[1]/2+int((rb1-yb1)/dr)),
                  int(full_map.shape[0]/2+int((-1*rb1+xb1)/dr)):int(full_map.shape[0]/2+int((rb1+xb1)/dr))] += ip_b1
    
    #full_map[int(full_map.shape[0]/2+int((-1*rb1+xb1-dr)/dr)):int(full_map.shape[0]/2+int((rb1+xb1+dr)/dr)),
    #             int(full_map.shape[1]/2+int((-1*rb1-yb1-dr)/dr)):int(full_map.shape[1]/2+int((rb1-yb1+dr)/dr))] += ip_b1
    
    #Not yet implemented, but would have to add the second bubble, which is essentially identical, as well as the 
    #convolution with the beam
    #x = jnp.arange(-1.5 * fwhm // (dr), 1.5 * fwhm // (dr)) * (dr)
    #beam = jnp.exp(-4 * np.log(2) * x ** 2 / fwhm ** 2)
    #beam = beam / jnp.sum(beam)

    #nx = x.shape[0] // 2 + 1

    #ipp = jnp.concatenate((ip[0:nx][::-1], ip))
    #ip = jnp.convolve(ipp, beam, mode="same")[nx:]

    #ip = ip * y2K_RJ(freq=freq, Te=T_electron)
    
    #rmap = np.meshgrid(x_b1, y_b1)
    #rr_b1[outside_rb_flag] = 0
    #print('rr shape' , rr_b1)
    #print(np.sum(rr_b1[outside_rb_flag], axis = -1).shape)
    #return rmap, full_map
    #test = sup*np.ones(rr_b1.shape)
    #test[outside_rb_flag] = 0

    #Return the full 2d gnfw+bubble profile, as well as the grid points for interpolation. Tods would then be evaluated using 
    #this interpolation, see e.g. conv_int_gnfw in luca_gnfw.py
    return rmap, full_map

bound = 10*np.pi/(180*3600)
x = np.linspace(-1*bound, bound, 20)
y = np.linspace(-1*bound, bound, 20)
xx, yy = np.meshgrid(x, y)

grid, ipmap =  _gnfw_bubble(0, 0, 8.403, 1.177, 1.2223, 5.49, 0.7736,3.2e14, 0, 0, 1*60, -25, 15, 20, 1.0,
    xx,
    yy,
    0.2,
    max_R=10.00,
    fwhm=9.0,
    freq=90e9,
    T_electron=5.0,
    r_map=1.0 * 60,
    dr=0.5,
)
#print(ipmap)
#print(ipmap.shape)
plt.imshow(np.log(np.abs(ipmap)))
plt.savefig('gnfw_bubble.png')
