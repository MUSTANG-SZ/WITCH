import aplpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colros import ListedColormap

from .fitting import load_config
from .utils import rad_to_arcsec

from astropy.io import fits
from astropy import wcs
import astropy.units as u

import numpy as np


def plot_cluster(
    name,
    fits_path,
    config_path,
    units="mJy",
    bound=None,
    radius=2.0,
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
    config_path : str
        Path to the config file used to make the fits file.
    units : str, default: mJy
        String to be used as units. If snr, then it will autoformat to sigma
    bound : None | float, default: None
        Bounds for the colormap. If none, reasonable bounds will be computed.
    radius : float, default: 2.0
        Radius, in arcmin, of figure
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
    fig = pyplot.figure(figsize=figsize)
    cfg = load_config({}, config_path)

    wcs_img = fits.open(fits_path)
    w = wcs.WCS(wcs_img[hdu].header)

    pix_size = w.wcs.cdelt[0] * rad_to_arcsec  # Mustang always uses square pixels
    smooth = min(
        1, int(smooth / pix_size)
    )  # FITSfigure smoothing is in pixels, so convert arcsec to pixels

    img = aplpy.FITSFigure(
        fits_path,
        hdu=hdu,
        figure=fig,
        downsample=downsample,
        smooth=False,
        convention=convention,
    )  # Smooth here does something whack
    img.set_theme("publication")

    ## make and register a divergent blue-orange colormap:
    bottom = cm.get_cmap("Oranges", 128)
    top = cm.get_cmap("Blues_r", 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    cm.register_cmap("mymap", cmap=ListedColormap(newcolors))
    cmap = "mymap"

    if bound is None:
        bound = 3.53e-4
        order = int(np.floor(np.log10(bound)))
        bound = np.round(bound, -1 * order)

    img.show_colorscale(
        cmap=cmap, stretch="linear", vmin=-bound, vmax=bound, smooth=smooth
    )

    ra = eval(cfg["coords"]["x0"])
    dec = eval(cfg["coords"]["y0"])
    ra, dec = np.rad2deg([ra, dec])

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

    img.add_colorbar("right")
    img.colorbar.set_width(0.12)

    if units == "snr":
        cbar_label = "$\sigma$"
    else:
        cbar_label = str(units)
    img.colorbar.set_axis_label_text(cbar_label)

    if ncontours:
        matplotlib.rcParams["lines.linewidth"] = 0.5
        clevels = np.linspace(-bound, bound, ncontours)
        img.show_contour(
            cluster,
            colors="gray",
            levels=clevels,
            returnlevels=True,
            convention="calabretta",
            smooth=smooth,
        )

    return img
