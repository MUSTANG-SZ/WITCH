import numpy as np

import glob, os, yaml

import minkasi

import minkasi_jax.presets_by_source as pbs
from minkasi_jax.utils import *
from minkasi_jax.core import helper

from astropy.coordinates import Angle
from astropy import units as u

import matplotlib.pyplot as plt


# In[17]:


from functools import partial
import jax
import jax.numpy as jnp
from minkasi_jax.utils import fft_conv
from minkasi_jax.structure import (
    isobeta,
    gnfw,
    gaussian,
    add_uniform,
    add_exponential,
    add_powerlaw,
    add_powerlaw_cos,
)

import jax.lax as lax

N_PAR_ISOBETA = 9
N_PAR_GNFW = 14
N_PAR_GAUSSIAN = 9
N_PAR_UNIFORM = 8
N_PAR_EXPONENTIAL = 14
N_PAR_POWERLAW = 11

ARGNUM_SHIFT = 11

@partial(
    jax.jit,
    static_argnums=(1, 2, 3, 4, 5, 6, 7, 8),
)
def model(
    xyz,
    n_isobeta,
    n_gnfw,
    n_gaussian,
    n_uniform,
    n_exponential,
    n_powerlaw,
    n_powerlaw_cos,
    dx,
    beam,
    idx,
    idy,
    *params
):
    """
    Generically create models with substructure.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_isobeta: Number of isobeta profiles to add.

        n_gnfw: Number of gnfw profiles to add.

        n_gaussian: Number of gaussians to add.

        n_uniform: Number of uniform ellipsoids to add.

        n_exponential: Number of exponential ellipsoids to add.

        n_powerlaw: Number of power law ellipsoids to add.

        n_powerlaw_cos: Number of radial power law ellipsoids with angulas cos term to add.

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        idx: RA TOD in units of pixels.
             Should have Dec stretch applied.

        idy: Dec TOD in units of pixels.

        params: 1D array of model parameters.

    Returns:

        model: The model with the specified substructure.
    """
    params = jnp.array(params)
    params = jnp.ravel(params)
    isobetas = jnp.zeros((1, 1), dtype=float)
    gnfws = jnp.zeros((1, 1), dtype=float)
    gaussians = jnp.zeros((1, 1), dtype=float)
    uniforms = jnp.zeros((1, 1), dtype=float)
    exponentials = jnp.zeros((1, 1), dtype=float)
    powerlaws = jnp.zeros((1, 1), dtype=float)

    start = 0
    if n_isobeta:
        delta = n_isobeta * N_PAR_ISOBETA
        #isobetas = lax.slice(params, (1, start), (1, start + delta)).reshape((n_isobeta, N_PAR_ISOBETA))
        isobetas = params[start : start + delta].reshape((n_isobeta, N_PAR_ISOBETA))
        start += delta
    if n_gnfw:
        delta = n_gnfw * N_PAR_GNFW
        gnfws = params[start : start + delta].reshape((n_gnfw, N_PAR_GNFW))
        start += delta
    if n_gaussian:
        delta = n_gaussian * N_PAR_GAUSSIAN
        gaussians = params[start : start + delta].reshape((n_gaussian, N_PAR_GAUSSIAN))
        start += delta
    if n_uniform:
        delta = n_uniform * N_PAR_UNIFORM
        uniforms = params[start : start + delta].reshape((n_uniform, N_PAR_UNIFORM))
        start += delta
    if n_exponential:
        delta = n_exponential * N_PAR_EXPONENTIAL
        exponentials = params[start : start + delta].reshape(
            (n_exponential, N_PAR_EXPONENTIAL)
        )
        start += delta
    if n_powerlaw:
        delta = n_powerlaw * N_PAR_POWERLAW
        powerlaws = params[start : start + delta].reshape((n_powerlaw, N_PAR_POWERLAW))
        start += delta
    if n_powerlaw_cos:
        delta = n_powerlaw_cos * N_PAR_POWERLAW
        powerlaw_coses = params[start : start + delta].reshape(
            (n_powerlaw_cos, N_PAR_POWERLAW)
        )
        start += delta

    pressure = jnp.zeros((xyz[0].shape[1], xyz[1].shape[0], xyz[2].shape[2]))
    for i in range(n_isobeta):
        pressure = jnp.add(pressure, isobeta(*isobetas[i], xyz))

    for i in range(n_gnfw):
        pressure = jnp.add(pressure, gnfw(*gnfws[i], xyz))

    for i in range(n_gaussian):
        pressure = jnp.add(pressure, gaussian(*gaussians[i], xyz))

    for i in range(n_uniform):
        pressure = add_uniform(pressure, xyz, *uniforms[i])

    for i in range(n_exponential):
        pressure = add_exponential(pressure, xyz, *exponentials[i])

    for i in range(n_powerlaw):
        pressure = add_powerlaw(pressure, xyz, *powerlaws[i])

    for i in range(n_powerlaw_cos):
        pressure = add_powerlaw_cos(pressure, xyz, *powerlaw_coses[i])

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

    # return jsp.ndimage.map_coordinates(ip, (idy, idx), order=0)
    return ip[idy.ravel(), idx.ravel()].reshape(idx.shape)


# In[2]:


with open('/home/r/rbond/jorlo/dev/minkasi_jax/configs/ms0735_noSub.yaml', "r") as file:
    cfg = yaml.safe_load(file)
#with open('/home/r/rbond/jorlo/dev/minkasi_jax/configs/ms0735/ms0735.yaml', "r") as file:
#    cfg = yaml.safe_load(file)
fit = True

# Setup coordindate stuff
z = eval(str(cfg["coords"]["z"]))
da = get_da(z)
r_map = eval(str(cfg["coords"]["r_map"]))
dr = eval(str(cfg["coords"]["dr"]))
xyz = make_grid(r_map, dr)
coord_conv = eval(str(cfg["coords"]["conv_factor"]))
x0 = eval(str(cfg["coords"]["x0"]))
y0 = eval(str(cfg["coords"]["y0"]))

# Load TODs
tod_names = glob.glob(os.path.join(cfg["paths"]["tods"], cfg["paths"]["glob"]))
bad_tod, addtag = pbs.get_bad_tods(
    cfg["cluster"]["name"], ndo=cfg["paths"]["ndo"], odo=cfg["paths"]["odo"]
)
tod_names = minkasi.cut_blacklist(tod_names, bad_tod)
tod_names.sort()
tod_names = tod_names[minkasi.myrank :: minkasi.nproc]
print('tod #: ', len(tod_names))
minkasi.barrier()  # Is this needed?


# In[3]:


todvec = minkasi.TodVec()
n_tod = 2
for i, fname in enumerate(tod_names):
    if i >= n_tod: break
    dat = minkasi.read_tod_from_fits(fname)
    minkasi.truncate_tod(dat)

    
    # figure out a guess at common mode and (assumed) linear detector drifts/offset
    # drifts/offsets are removed, which is important for mode finding.  CM is *not* removed.
    dd, pred2, cm = minkasi.fit_cm_plus_poly(dat["dat_calib"], cm_ord=3, full_out=True)
    dat["dat_calib"] = dd
    dat["pred2"] = pred2
    dat["cm"] = cm

    # Make pixelized RA/Dec TODs
    idx, idy = tod_to_index(dat["dx"], dat["dy"], x0, y0, r_map, dr, coord_conv)
    idu, id_inv = np.unique(
        np.vstack((idx.ravel(), idy.ravel())), axis=1, return_inverse=True
    )
    dat["idx"] = idu[0]
    dat["idy"] = idu[1]
    dat["id_inv"] = id_inv
    dat["model_idx"] = idx
    dat["model_idy"] = idy

    tod = minkasi.Tod(dat)
    todvec.add_tod(tod)


# In[4]:


lims = todvec.lims()
pixsize = 2.0 / 3600 * np.pi / 180
skymap = minkasi.SkyMap(lims, pixsize)

Te = eval(str(cfg["cluster"]["Te"]))
freq = eval(str(cfg["cluster"]["freq"]))
beam = beam_double_gauss(
    dr,
    eval(str(cfg["beam"]["fwhm1"])),
    eval(str(cfg["beam"]["amp1"])),
    eval(str(cfg["beam"]["fwhm2"])),
    eval(str(cfg["beam"]["amp2"])),
)

funs = []
npars = []
labels = []
params = []
to_fit = []
priors = []
prior_vals = []
re_eval = []
par_idx = {}
for cur_model in cfg["models"].values():
    npars.append(len(cur_model["parameters"]))
    for name, par in cur_model["parameters"].items():
        labels.append(name)
        par_idx[name] = len(params)
        params.append(eval(str(par["value"])))
        to_fit.append(eval(str(par["to_fit"])))
        if "priors" in par:
            priors.append(par["priors"]["type"])
            prior_vals.append(eval(str(par["priors"]["value"])))
        else:
            priors.append(None)
            prior_vals.append(None)
        if "re_eval" in par and par["re_eval"]:
            re_eval.append(str(par["value"]))
        else:
            re_eval.append(False)
        2.627 * da, funs.append(eval(str(cur_model["func"])))

npars = np.array(npars)
labels = np.array(labels)
params = np.array(params)
to_fit = np.array(to_fit, dtype=bool)
priors = np.array(priors)

noise_class = eval(str(cfg["minkasi"]["noise"]["class"]))
noise_args = eval(str(cfg["minkasi"]["noise"]["args"]))
noise_kwargs = eval(str(cfg["minkasi"]["noise"]["kwargs"]))


# In[5]:


#TODO: Implement tsBowl here 
if "bowling" in cfg:
    sub_poly = cfg["bowling"]["sub_poly"]

sim = False #This script is for simming, the option to turn off is here only for debugging
#TODO: Write this to use minkasi_jax.core.model
for i, tod in enumerate(todvec.tods):
        
    ipix = skymap.get_pix(tod)
    tod.info["ipix"] = ipix
    tod.set_noise(noise_class, *noise_args, **noise_kwargs)


# In[6]:


idxs = np.array([])
idys = np.array([])

dxs = np.array([])
dys = np.array([])

lens = np.array([])

for tod in todvec.tods:
    idxs = np.append(idxs, tod.info["idx"])
    idys = np.append(idys, tod.info["idy"])
    lens = np.append(lens, len(tod.info["dx"].ravel()))
    dxs = np.append(dxs, tod.info["model_idx"])
    dys = np.append(dys, tod.info["model_idy"])

lens = np.array(lens, dtype=int)

vis_pars = params[:42]
vis_pars[8], vis_pars[17] = -1e-5, -1e-5


# In[7]:


#idx_min, idx_max = max(np.min(idxs),0), min(np.max(idxs), 750)
#idy_min, idy_max = max(np.min(idys),0), min(np.max(idys), 750)

idx_min, idx_max = np.min(idxs), np.max(idxs)
idy_min, idy_max = np.min(idys), np.max(idys)


# In[8]:


x = np.arange(idx_min, idx_max, dtype = int)
y = np.arange(idy_min, idy_max, dtype = int)

X, Y = np.meshgrid(x, y)


# In[20]:


r_map = eval(str(cfg["coords"]["r_map"]))
dr = eval(str(cfg["coords"]["dr"]))*2 #Things look wrong even without this but they look really wrong with it
xyz = make_grid(r_map, dr)

dx = float(y2K_RJ(freq, Te)*dr*XMpc/me)


# In[21]:




vis_model = model(xyz, 2, 0, 0, 3, 0, 0, 0, float(y2K_RJ(freq, Te)*dr*XMpc/me), beam, X, Y, vis_pars)

plt.imshow(vis_model, extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)], origin = 'lower')
plt.colorbar()


# # Ignore stuff below here

# In[ ]:





# In[ ]:





# In[ ]:





# In[159]:


r_map*60


# In[162]:


plt.imshow(vis_model[300], vmin=-5e-6, vmax =0)
plt.colorbar()


# In[154]:


import matplotlib.animation as animation

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the initial image
im = ax.imshow(vis_model[0], animated=True, vmin=-5e-6, vmax =0)

def update(i):
    im.set_array(vis_model[i])
    return im, 

# Create the animation object
animation_fig = animation.FuncAnimation(fig, update, frames=len(vis_model), interval=10, blit=True,repeat_delay=10,)

# Show the animation
plt.show()

animation_fig.save("ms0735.gif")


# In[133]:


vis_model.shape


# In[160]:


import scipy
int_model = scipy.integrate.trapz(vis_model, dx=dx, axis=-1)


# In[161]:


plt.imshow(int_model)
plt.colorbar()


# In[116]:


plt.imshow(test)
plt.colorbar()


# In[228]:


with open('ms0735_beam.pk', 'wb') as f:
    pk.dump(full_beam, f)


# In[69]:


with open('ms0735.pk', 'rb') as f:
    test = pk.load(f)


# In[164]:


plt.clf();plt.plot(int_model[:,750//5]);plt.show()


# In[167]:


plt.clf();plt.plot(vis_model[:,750//5]);plt.show()


# In[220]:



plt.clf();plt.plot(((conv_img-conv_img2)/conv_img)[:,750//5]);plt.show()


# In[168]:


plt.clf();plt.plot(test[:,750//5]);plt.show()


# In[61]:


jax_conv_model = model(xyz, 2, 0, 0, 3, 0, 0, 0, float(y2K_RJ(freq, Te)*dr*XMpc/me), beam, X, Y, params[:42])


# In[62]:


plt.clf();plt.plot(jax_conv_model[:,750//5]);plt.show()


# In[56]:


from astropy.convolution import convolve_fft, Gaussian2DKernel

#kernel = Gaussian2DKernel(x_stddev=15)
full_beam = model(xyz, 1, 0, 0, 0, 0, 0, 0, float(y2K_RJ(freq, Te)*dr*XMpc/me), beam, X, Y, params[:42])

conv_img = convolve_fft(vis_model, full_beam)
conv_img2 = fft_conv(vis_model, full_beam)


# In[ ]:


plt.imshow(full_beam)


# In[63]:


@jax.jit
def fft_conv(image, kernel):
    """
    Perform a convolution using FFTs for speed.

    Arguments:

        image: Data to be convolved

        kernel: Convolution kernel

    Returns:

        convolved_map: Image convolved with kernel.
    """
    Fmap = jnp.fft.rfft2(image)
    Fkernel = jnp.fft.rfft2(kernel)
    convolved_map = jnp.fft.fftshift(jnp.real(jnp.fft.irfft2(Fmap * Fkernel)))

    return convolved_map


# In[64]:


import jax.scipy as jsp
#from minkasi_jax.utils import fft_conv

@partial(
    jax.jit,
    static_argnums=(1, 2, 3, 4, 5, 6, 7, 8),
)
def model(
    xyz,
    n_isobeta,
    n_gnfw,
    n_gaussian,
    n_uniform,
    n_exponential,
    n_powerlaw,
    n_powerlaw_cos,
    dx,
    beam,
    idx,
    idy,
    *params
):
    """
    Generically create models with substructure.

    Arguments:

        xyz: Coordinate grid to compute profile on.

        n_isobeta: Number of isobeta profiles to add.

        n_gnfw: Number of gnfw profiles to add.

        n_gaussian: Number of gaussians to add.

        n_uniform: Number of uniform ellipsoids to add.

        n_exponential: Number of exponential ellipsoids to add.

        n_powerlaw: Number of power law ellipsoids to add.

        n_powerlaw_cos: Number of radial power law ellipsoids with angulas cos term to add.

        dx: Factor to scale by while integrating.
            Since it is a global factor it can contain unit conversions.
            Historically equal to y2K_RJ * dr * da * XMpc / me.

        beam: Beam to convolve by, should be a 2d array.

        idx: RA TOD in units of pixels.
             Should have Dec stretch applied.

        idy: Dec TOD in units of pixels.

        params: 1D array of model parameters.

    Returns:

        model: The model with the specified substructure.
    """
    params = jnp.array(params)
    params = jnp.ravel(params)
    isobetas = jnp.zeros((1, 1), dtype=float)
    gnfws = jnp.zeros((1, 1), dtype=float)
    gaussians = jnp.zeros((1, 1), dtype=float)
    uniforms = jnp.zeros((1, 1), dtype=float)
    exponentials = jnp.zeros((1, 1), dtype=float)
    powerlaws = jnp.zeros((1, 1), dtype=float)

    start = 0
    if n_isobeta:
        delta = n_isobeta * N_PAR_ISOBETA
        #isobetas = lax.slice(params, (1, start), (1, start + delta)).reshape((n_isobeta, N_PAR_ISOBETA))
        isobetas = params[start : start + delta].reshape((n_isobeta, N_PAR_ISOBETA))
        start += delta
    if n_gnfw:
        delta = n_gnfw * N_PAR_GNFW
        gnfws = params[start : start + delta].reshape((n_gnfw, N_PAR_GNFW))
        start += delta
    if n_gaussian:
        delta = n_gaussian * N_PAR_GAUSSIAN
        gaussians = params[start : start + delta].reshape((n_gaussian, N_PAR_GAUSSIAN))
        start += delta
    if n_uniform:
        delta = n_uniform * N_PAR_UNIFORM
        uniforms = params[start : start + delta].reshape((n_uniform, N_PAR_UNIFORM))
        start += delta
    if n_exponential:
        delta = n_exponential * N_PAR_EXPONENTIAL
        exponentials = params[start : start + delta].reshape(
            (n_exponential, N_PAR_EXPONENTIAL)
        )
        start += delta
    if n_powerlaw:
        delta = n_powerlaw * N_PAR_POWERLAW
        powerlaws = params[start : start + delta].reshape((n_powerlaw, N_PAR_POWERLAW))
        start += delta
    if n_powerlaw_cos:
        delta = n_powerlaw_cos * N_PAR_POWERLAW
        powerlaw_coses = params[start : start + delta].reshape(
            (n_powerlaw_cos, N_PAR_POWERLAW)
        )
        start += delta

    pressure = jnp.zeros((xyz[0].shape[1], xyz[1].shape[0], xyz[2].shape[2]))
    for i in range(n_isobeta):
        pressure = jnp.add(pressure, isobeta(*isobetas[i], xyz))

    for i in range(n_gnfw):
        pressure = jnp.add(pressure, gnfw(*gnfws[i], xyz))

    for i in range(n_gaussian):
        pressure = jnp.add(pressure, gaussian(*gaussians[i], xyz))

    for i in range(n_uniform):
        pressure = add_uniform(pressure, xyz, *uniforms[i])

    for i in range(n_exponential):
        pressure = add_exponential(pressure, xyz, *exponentials[i])

    for i in range(n_powerlaw):
        pressure = add_powerlaw(pressure, xyz, *powerlaws[i])

    for i in range(n_powerlaw_cos):
        pressure = add_powerlaw_cos(pressure, xyz, *powerlaw_coses[i])
    #return pressure
    # Integrate along line of site
    ip = jnp.trapz(pressure, dx=dx, axis=-1)
    #return ip
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
    #return beam
    ip = fft_conv(ip, beam)
    #ip = jax.scipy.signal.fftconvolve(ip, beam, mode='same')

    # return jsp.ndimage.map_coordinates(ip, (idy, idx), order=0)
    return ip[idy.ravel(), idx.ravel()].reshape(idx.shape)


# In[79]:


750//5


# In[ ]:





# In[ ]:





# In[ ]:


pars_fit = params
if fit:
    t1 = time.time()
    print_once("Started actual fitting")
    pars_fit, chisq, curve, errs = minkasi.fit_timestreams_with_derivs_manyfun(
        funs,
        params,
        npars,
        todvec,
        to_fit,
        maxiter=cfg["minkasi"]["maxiter"],
        priors=priors,
        prior_vals=prior_vals,
    )
    minkasi.comm.barrier()
    t2 = time.time()
    print_once("Took", t2 - t1, "seconds to fit")

    for i, re in enumerate(re_eval):
        if not re:
            continue
        pars_fit[i] = eval(re)

    print_once("Fit parameters:")
    for l, pf, err in zip(labels, pars_fit, errs):
        print_once("\t", l, "=", pf, "+/-", err)
    print_once("chisq =", chisq)

    if minkasi.myrank == 0:
        res_path = os.path.join(outdir, "results")
        print_once("Saving results to", res_path + ".npz")
        np.savez_compressed(
            res_path, pars_fit=pars_fit, chisq=chisq, errs=errs, curve=curve
        )


# In[ ]:




