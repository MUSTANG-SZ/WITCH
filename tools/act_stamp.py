import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from pixell import enmap, reproject, utils

# ra = Angle('14:07:06.0 hours').to(u.rad).value
# dec = Angle('10:48:30.00 degrees').to(u.rad).value
ra = (175.6918169 * u.degree).to(u.radian).value
dec = (15.4532554 * u.degree).to(u.radian).value

r = 10 * utils.arcmin
box = np.array([[dec - r, ra - r], [dec + r, ra + r]])

stamp = enmap.read_map(
    "/mnt/welch/ACT/20240323_simple/act_planck_s08_s22_f090_daynight_map.fits", box=box
)
enmap.write_map(
    "/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/act/MOOJ1142_act_planck_s08_s22_f090_daynight_map.fits",
    stamp[0],
)
istamp = enmap.read_map(
    "/mnt/welch/ACT/20240323_simple/act_planck_s08_s22_f090_daynight_ivar.fits", box=box
)
enmap.write_map(
    "/mnt/welch/MUSTANG/M2-TODs/MOOJ1142/act/MOOJ1142_act_planck_s08_s22_f090_daynight_ivar.fits",
    istamp[0],
)
