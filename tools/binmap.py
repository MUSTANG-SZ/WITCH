import numpy as np
from astropy.io import fits
from witch.nonparametric import bin_map
import witch.utils as wu

hdu = fits.open(
    "/mnt/welch/USERS/jorlo/Reductions/MOOJ1142/gnfw_rs-ps_gauss/xray_peak_rs/P0-r_s-gamma-alpha-beta/mustang2/signal/niter_1_25.fits"
)

rbins = np.array([0, 1, 2, 3, 5, 7, 999999]) / wu.rad_to_arcsec
binmeans = [(rbins[i + 1] + rbins[i]) / 2 for i in range(len(rbins) - 1)]

bins = bin_map(hdu, binmeans)
