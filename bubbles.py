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
    hz = np.interp(z, dzline, hzline)
    nz = np.interp(z, dzline, nzline)

    ap = 0.12

    r500 = (m500 / (4.00 * np.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
    P500 = (
        1.65e-03
        * (m500 / (3.00e14 / h70)) ** (2.00 / 3.00 + ap)
        * hz ** (8.00 / 3.00)
        * h70 ** 2
    )

    dR = max_R / 2e3
    r = np.arange(0.00, max_R, dR) + dR / 2.00

    x = r / r500
    pressure = (
        P500
        * P0
        / (
            (c500 * x) ** gamma
            * (1.00 + (c500 * x) ** alpha) ** ((beta - gamma) / alpha)
        )
    )

    rmap = np.arange(0, r_map, dr)
    r_in_Mpc = rmap * (np.interp(z, dzline, daline))
    rr = np.meshgrid(r_in_Mpc, r_in_Mpc)
    rr = np.sqrt(rr[0] ** 2 + rr[1] ** 2)
    yy = np.interp(rr, r, pressure, right=0.0)

    XMpc = Xthom * Mparsec

    ip = np.sum(yy, axis=1) * 2.0 * XMpc / (me * 1000)
    
    full_rmap = np.arange(-1*r_map, r_map, dr)
    xx, yy = np.meshgrid(full_rmap, full_rmap)
    full_rr = np.sqrt(xx**2 + yy**2)

    #xb1 *= ((3600*180)/np.pi)
    #yb1 *= ((3600*180)/np.pi)
    #rb1 *= 60 
    #Make a grid, centered on xb1, yb1 and size rb1, with resolution dr, and convert to Mpc
    x_b1 = np.arange(-1*rb1+xb1-dr, rb1+xb1+dr, dr) * (np.interp(z, dzline, daline))
    y_b1 = np.arange(-1*rb1-yb1-dr, rb1-yb1+dr, dr) * (np.interp(z, dzline, daline))
    z_b1 = np.arange(-1*rb1-dr, rb1+dr, dr) * (np.interp(z, dzline, daline))
    #print(dr * (np.interp(z, dzline, daline)))
    xyz_b1 = np.meshgrid(x_b1, y_b1, z_b1)
    rr_b1 = np.sqrt(xyz_b1[0]**2 + xyz_b1[1]**2 + xyz_b1[2]**2)
    yy_b1 = np.interp(rr_b1, rmap, ip)

    #Set up a grid of points for computing distance from bubble center
    x_rb = np.linspace(-1,1, len(x_b1))
    y_rb = np.linspace(-1,1, len(y_b1))
    z_rb = np.linspace(-1,1, len(z_b1))
    rb_grid = np.meshgrid(x_rb, y_rb, z_rb)

  
    #Zero out points outside bubble
    outside_rb_flag = np.sqrt(rb_grid[0]**2 + rb_grid[1]**2+rb_grid[2]**2) >=1    
   
    yy_b1[outside_rb_flag] = 0
    test = sup*np.ones(yy_b1.shape)
    test[outside_rb_flag] = 0 
    ip_b1 = -sup*np.sum(yy_b1, axis = -1)# * 2.0 * XMpc / (me * 1000)
    plt.imshow(ip_b1)
    plt.savefig('bubble.png')
    plt.close()
    plt.imshow(np.sum(test, axis = -1))
    #print(test.tolist())
    #print(np.sum(test, axis = ).tolist())
    plt.savefig('unweight_bubble.png')
    plt.close()
    full_map = np.interp(full_rr, rmap, ip)    
    print(np.amax(np.abs(full_map)), np.amax(np.abs(ip_b1)))
    full_map[int(full_map.shape[1]/2+int((-1*rb1-yb1-dr)/dr)):int(full_map.shape[1]/2+int((rb1-yb1+dr)/dr)),
                  int(full_map.shape[0]/2+int((-1*rb1+xb1-dr)/dr)):int(full_map.shape[0]/2+int((rb1+xb1+dr)/dr))] += ip_b1
    
    #full_map[int(full_map.shape[0]/2+int((-1*rb1+xb1-dr)/dr)):int(full_map.shape[0]/2+int((rb1+xb1+dr)/dr)),
    #             int(full_map.shape[1]/2+int((-1*rb1-yb1-dr)/dr)):int(full_map.shape[1]/2+int((rb1-yb1+dr)/dr))] += ip_b1
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
    return rmap, full_map

bound = 10*np.pi/(180*3600)
x = np.linspace(-1*bound, bound, 20)
y = np.linspace(-1*bound, bound, 20)
xx, yy = np.meshgrid(x, y)

grid, ipmap =  _gnfw_bubble(0, 0, 8.403, 1.177, 1.2223, 5.49, 0.7736,3.2e14, 0, 0, 15, -25, 15, 20, 1.0,
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
