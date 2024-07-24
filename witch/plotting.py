import os

import aplpy
import astropy.units as u
import dill as pk
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from matplotlib.colors import ListedColormap

from .fitter import load_config
from .utils import get_da, get_nz, rad_to_arcsec


def plot_cluster(
    name,
    fits_path,
    root=None,
    pix_size=None,
    ra=None,
    dec=None,
    units="mJy",
    bound=None,
    radius=2.0,
    plot_r=True,
    figsize=(5, 5),
    ncontours=0,
    hdu=0,
    downsample=1,
    smooth=9.0,
    convention="calabretta",
):
    """
    Function for doing core plotting. TODO: This function could probably use an args/kwargs, but there are an enourmous number of keyword args within so that might be difficult.

    Parameters:
    -----------
    Name : str
        Name of the cluster
    fits_path : str
        Path to the fits file to be plotted.
    root : None | str, default: None
        Path to the output root. If none, then it will assume WITCH output formating.
    pix_size : None | float, default: None
        Pixel size. If None, then will be computed from results file.
    ra : None | float, default: None
        RA of center of plot, in degrees. If none, will be taken from config
    dec : None | float, dfault: None
        Dec of center of plot, in degrees. If none, will be taken from config
    units : str, default: mJy
        String to be used as units. If snr, then it will autoformat to sigma
    bound : None | float, default: None
        Bounds for the colormap. If none, reasonable bounds will be computed.
    radius : float, default: 2.0
        Radius, in arcmin, of figure
    plot_r : bool | str, default: True
        If true, plot r500. If a str, plot a related critical radius
    figsize : tuple[float, float], default: (5,5)
        Width and height of plot in inches.
    ncontours : int, default = 0
        Number of countours to be plotted
    hdu : int, default: 0
        Fits hdu corresponding to the image to be plotted
    downsample : int, default: 1
        Factor by which to downsample the image.
    smooth : float, default: 9.0
        Scale, in arcminutes, at which to smooth the image.
    convention : str, default: calabretta
        Determines interpretation of abigious fits headers. See aplpy.FITSFigure documentation

    Returns:
    --------
    img: aplpy.FITSFigure
        FITSFigure plot of the cluster
    """
    fits_path = os.path.abspath(fits_path)
    if root is None:
        root = os.path.split(os.path.split(fits_path)[0])[0]

    if pix_size is None:
        res_path = (
            root
            + "/"
            + str(sorted([file for file in os.listdir(root) if ".dill" in file])[-1])
        )
        with open(res_path, "rb") as f:
            results = pk.load(f)
        pix_size = results.pix_size * rad_to_arcsec

    if ra is None or dec is None:
        cfg_path = root + "/" + "config.yaml"
        cfg = load_config({}, cfg_path)
        ra = eval(cfg["coords"]["x0"])
        dec = eval(cfg["coords"]["y0"])
        ra, dec = np.rad2deg(
            [ra, dec]
        )  # TODO: Currently center on config center, which is fine but should probably be fit center

    smooth = max(
        1, int(smooth / pix_size)
    )  # FITSfigure smoothing is in pixels, so convert arcsec to pixels

    kernel = Gaussian2DKernel(x_stddev=smooth * 5)

    fig = plt.figure(figsize=figsize)
    img = aplpy.FITSFigure(
        fits_path,
        hdu=hdu,
        figure=fig,
        downsample=downsample,
        smooth=False,
        convention=convention,
    )  # Smooth here does something whack
    img.set_theme("publication")

    if units is not None:
        if units == "snr":
            cbar_label = r"$\sigma$"
        elif units == "uK_cmb":
            img._data /= 1.28
            img._data *= 1e6
            cbar_label = r"$uK_{CMB}$"
        elif units == "uK_RJ":
            img._data *= 1e6
            cbar_label = r"$uK_{RJ}"
        else:
            cbar_label = str(units)


    ## make and register a divergent blue-orange colormap:
    cmap = "mymap"
    try:
        matplotlib.colormaps.get_cmap(
            cmap
        )  # Stops these anoying messages if you've already registered mymap

    except:
        mymap = matplotlib.colors.LinearSegmentedColormap.from_list(cmap, ["Blue", "White", "Red"]) 
        matplotlib.colormaps.register(cmap=mymap)


    if bound is None:
        nx, ny = img._data.shape
        lims = int(radius * 60 / pix_size)
        xmin = int(nx / 2 - lims)
        xmax = int(nx / 2 + lims)
        ymin = int(ny / 2 - lims)
        ymax = int(ny / 2 + lims)
        bound = np.amax(np.abs(img._data[xmin:xmax, ymin:ymax]))
        order = int(np.floor(np.log10(bound)))
        bound = np.round(bound, -1 * order) / 2

    img.show_colorscale(cmap=cmap, stretch="linear", vmin=-bound, vmax=bound, smooth=3)
    img.recenter(ra, dec, radius=radius / 60.0)
    img.ax.tick_params(axis="both", which="both", direction="in")

    matplotlib.rcParams["lines.linewidth"] = 3.0
    img.add_scalebar(
        0.5 / 60.0, '30"', color="black"
    )  # Adds a 30 arcsec scalebar to the image

    matplotlib.rcParams["lines.linewidth"] = 2.0

    img.add_beam(
        major=9.0 / 3600.0, minor=9.0 / 3600.0, angle=0
    )  # TODO: For now hard-coded to M2 beam but may want some flexibility later
    img.beam.set_color("white")
    img.beam.set_edgecolor("green")
    img.beam.set_facecolor("white")
    img.beam.set_corner("bottom left")

    img.show_markers(
        ra,
        dec,
        facecolor="black",
        edgecolor=None,
        marker="+",
        s=50,
        linewidths=2,
        alpha=0.5,
    )
    if units is not None:
        img.add_colorbar("right")
        img.colorbar.set_width(0.12)
        img.colorbar.set_axis_label_text(cbar_label)

    if ncontours:
        matplotlib.rcParams["lines.linewidth"] = 0.5
        clevels = np.linspace(-bound, bound, ncontours)
        img.show_contour(
            fits_path,
            colors="gray",
            levels=clevels,
            returnlevels=True,
            convention="calabretta",
            smooth=3,
        )

    if plot_r:  # TODO: Allow passing of r500 values, make this a subfunction
        if "a10" in cfg["model"]["structures"].keys():
            mod_type = "a10"
        elif "ea10" in cfg["model"]["structures"].keys():
            mod_type = "ea10"
        else:
            raise ModelError("For R500, must have structure type A10 or EA10")

        for i in range(len(results.structures)):
            if str(results.structures[i].name) == mod_type:
                break

        for parameter in results.structures[i].parameters:
            if str(parameter.name.lower()) == "m500":
                m500 = parameter.val
                break

        z = float(cfg["constants"]["z"])
        nz = get_nz(z)

        r500 = (m500 / (4.00 * np.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
        da = get_da(z)
        r500 /= da
        if plot_r == "rs":
            r500 /= float(
                cfg["model"]["structures"][mod_type]["parameters"]["c500"]["value"]
            )  # Convert to rs
        img.show_circles(
            ra, dec, radius=r500 / 3600, coords_frame="world", color="green"
        )

    return img


def plot_cluster_act(
    name,
    fits_path,
    cfg_path=None,
    ra=None,
    dec=None,
    units="mJy",
    bound=None,
    radius=2.0,
    plot_r=True,
    figsize=(5, 5),
    ncontours=0,
    hdu=0,
    downsample=1,
    smooth=60.0,
    convention="calabretta",
):
    """
    Function for doing core plotting. TODO: This function could probably use an args/kwargs, but there are an enourmous number of keyword args within so that might be difficult.

    Parameters:
    -----------
    Name : str
        Name of the cluster
    fits_path : str
        Path to the fits file to be plotted.
    cfg_path : None | str, default: None
        Path to WITCH config file corresponding to same cluster
    ra : None | float, default: None
        RA of center of plot, in degrees. If none, will be taken from config
    dec : None | float, dfault: None
        Dec of center of plot, in degrees. If none, will be taken from config
    units : str, default: mJy
        String to be used as units. If snr, then it will autoformat to sigma
    bound : None | float, default: None
        Bounds for the colormap. If none, reasonable bounds will be computed.
    radius : float, default: 2.0
        Radius, in arcmin, of figure
    plot_r : bool | str, default: True
        If true, plot r500. If a str, plot a related critical radius
    figsize : tuple[float, float], default: (5,5)
        Width and height of plot in inches.
    ncontours : int, default = 0
        Number of countours to be plotted
    hdu : int, default: 0
        Fits hdu corresponding to the image to be plotted
    downsample : int, default: 1
        Factor by which to downsample the image.
    smooth : float, default: 9.0
        Scale, in arcminutes, at which to smooth the image.
    convention : str, default: calabretta
        Determines interpretation of abigious fits headers. See aplpy.FITSFigure documentation

    Returns:
    --------
    img: aplpy.FITSFigure
        FITSFigure plot of the cluster
    """
    if cfg_path is not None:
        cfg = load_config({}, cfg_path)
        ra = eval(cfg["coords"]["x0"])
        dec = eval(cfg["coords"]["y0"])
        ra, dec = np.rad2deg(
            [ra, dec]
        )  # TODO: Currently center on config center, which is fine but should probably be fit center
    elif ra is None or dec is None:
        raise ValueError("Either cfg_path or both ra and dec must be specified.")
    cur_hdu = fits.open(fits_path)
    pix_size = cur_hdu[0].header["CDELT1"] * 3600

    smooth = max(
        1, int(smooth / pix_size)
    )  # FITSfigure smoothing is in pixels, so convert arcsec to pixels

    kernel = Gaussian2DKernel(x_stddev=smooth * 5)

    fig = plt.figure(figsize=figsize)
    img = aplpy.FITSFigure(
        fits_path,
        hdu=hdu,
        figure=fig,
        downsample=downsample,
        smooth=False,
        convention=convention,
    )  # Smooth here does something whack
    img.set_theme("publication")

    beam_fwhm = 2.2
    fwhm_to_sigma = 1.0 / (8 * np.log(2)) ** 0.5
    beam_sigma = beam_fwhm * fwhm_to_sigma
    omega_B = 2 * np.pi * beam_sigma**2

    if units == "snr":
        cbar_label = r"$\sigma$"
    elif units == "uK":
        img._data *= np.sqrt(omega_B)
        cbar_label = r"$uK_{CMB}$"
    else:
        cbar_label = str(units)

    cmap = "mymap"
    try:
        cm.get_cmap(
            cmap
        )  # Stops these anoying messages if you've already registered mymap

    except:
        bottom = cm.get_cmap("Oranges", 128)
        top = cm.get_cmap("Blues_r", 128)
        newcolors = np.vstack(
            (top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128)))
        )
        cm.register_cmap(cmap, cmap=ListedColormap(newcolors))

    if bound is None:
        nx, ny = img._data.shape
        lims = int(radius * 60 / pix_size)
        xmin = int(nx / 2 - lims)
        xmax = int(nx / 2 + lims)
        ymin = int(ny / 2 - lims)
        ymax = int(ny / 2 + lims)
        bound = np.amax(np.abs(img._data[xmin:xmax, ymin:ymax]))
        order = int(np.floor(np.log10(bound)))
        bound = np.round(bound, -1 * order) / 2

    img.show_colorscale(cmap=cmap, stretch="linear", vmin=-bound, vmax=bound, smooth=1)

    img.recenter(ra, dec, radius=radius / 60.0)
    img.ax.tick_params(axis="both", which="both", direction="in")

    matplotlib.rcParams["lines.linewidth"] = 3.0
    img.add_scalebar(
        0.5 / 60.0, '30"', color="black"
    )  # Adds a 30 arcsec scalebar to the image

    matplotlib.rcParams["lines.linewidth"] = 2.0

    img.add_beam(
        major=120.0 / 3600.0, minor=120.0 / 3600.0, angle=0
    )  # TODO: For now hard-coded to M2 beam but may want some flexibility later
    img.beam.set_color("white")
    img.beam.set_edgecolor("green")
    img.beam.set_facecolor("white")
    img.beam.set_corner("bottom left")

    img.show_markers(
        ra,
        dec,
        facecolor="black",
        edgecolor=None,
        marker="+",
        s=50,
        linewidths=2,
        alpha=0.5,
    )

    img.add_colorbar("right")
    img.colorbar.set_width(0.12)
    img.colorbar.set_axis_label_text(cbar_label)

    if ncontours:
        matplotlib.rcParams["lines.linewidth"] = 0.5
        clevels = np.linspace(-bound, bound, ncontours)
        img.show_contour(
            fits_path,
            colors="gray",
            levels=clevels,
            returnlevels=True,
            convention="calabretta",
            smooth=3,
        )

    if plot_r:  # TODO: Allow passing of r500 values, make this a subfunction
        if "a10" in cfg["model"]["structures"].keys():
            mod_type = "a10"
        elif "ea10" in cfg["model"]["structures"].keys():
            mod_type = "ea10"
        else:
            raise ModelError("For R500, must have structure type A10 or EA10")

        for i in range(len(results.structures)):
            if str(results.structures[i].name) == mod_type:
                break

        for parameter in results.structures[i].parameters:
            if str(parameter.name.lower()) == "m500":
                m500 = parameter.val
                break

        z = float(cfg["constants"]["z"])
        nz = get_nz(z)

        r500 = (m500 / (4.00 * np.pi / 3.00) / 5.00e02 / nz) ** (1.00 / 3.00)
        da = get_da(z)
        r500 /= da
        if plot_r == "rs":
            r500 /= float(
                cfg["model"]["structures"][mod_type]["parameters"]["c500"]["value"]
            )  # Convert to rs
        img.show_circles(
            ra, dec, radius=r500 / 3600, coords_frame="world", color="green"
        )

    return img
