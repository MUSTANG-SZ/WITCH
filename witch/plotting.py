import os
from importlib import import_module
from typing import Optional, Union

import aplpy
import dill as pk
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from matplotlib.colors import ListedColormap

from .fitter import load_config
from .utils import get_da, get_nz, rad_to_arcsec

here, this_filename = os.path.split(__file__)

# from https://gist.github.com/zonca/6515744
cmb_cmap = ListedColormap(
    np.loadtxt(f"{here}/Planck_Parchment_RGB.txt") / 255.0, name="cmb"
)
cmb_cmap.set_bad("white")

matplotlib.colormaps.register(cmb_cmap)

cmap = "mustang"
try:
    matplotlib.colormaps.get_cmap(
        cmap
    )  # Stops these anoying messages if you've already registered mymap

except:
    mustang_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap, ["Blue", "White", "Red"]
    )
    matplotlib.colormaps.register(cmap=mustang_cmap)


def plot_cluster(
    name: str,
    fits_path: str,
    root: Optional[str] = None,
    pix_size: Optional[float] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    units: str = "mJy",
    scale: float = 1.0,
    cmap: str = "mustang",
    bound: Optional[float] = None,
    radius: float = 2.0,
    plot_r=True,
    figsize: tuple[float, float] = (6, 5),
    ncontours: int = 0,
    hdu_int: int = 0,
    downsample: int = 1,
    smooth: float = 9.0,
    convention: str = "calabretta",
):
    """
    Function for doing core plotting.
    TODO: This function could probably use an args/kwargs, but there are an enourmous number of keyword args within so that might be difficult.

    Parameters
    ----------
    name : str
        name of the cluster
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
    scale : float, default: 1
        Amount to scale data by
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
    hdu_int : int, default: 0
        Fits hdu corresponding to the image to be plotted
    downsample : int, default: 1
        Factor by which to downsample the image.
    smooth : float, default: 9.0
        Scale, in arcminutes, at which to smooth the image.
    convention : str, default: calabretta
        Determines interpretation of abigious fits headers. See aplpy.FITSFigure documentation

    Returns
    -------
    img: aplpy.FITSFigure
        FITSFigure plot of the cluster
    """

    fits_path = os.path.abspath(fits_path)
    if root is None:
        root = os.path.split(os.path.split(os.path.split(fits_path)[0])[0])[
            0
        ]  # TODO: There's gotta be a better way!

    cfg_path = root + "/" + "config.yaml"
    cfg = load_config({}, cfg_path)
    # Do imports
    for module, name in cfg.get("imports", {}).items():
        mod = import_module(module)
        if isinstance(name, str):
            locals()[name] = mod
        elif isinstance(name, list):
            for n in name:
                locals()[n] = getattr(mod, n)
        else:
            raise TypeError("Expect import name to be a string or a list")

    res_path = (
        root
        + "/"
        + str(sorted([file for file in os.listdir(root) if ".dill" in file])[-1])
    )
    with open(res_path, "rb") as f:
        results = pk.load(f)

    if pix_size is None:
        pix_size = results.pix_size * rad_to_arcsec

    if ra is None or dec is None:
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

    hdu = fits.open(fits_path)[0]
    hdu.data *= scale

    plot_hdu = fits.PrimaryHDU(data=hdu.data, header=hdu.header)

    img = aplpy.FITSFigure(
        plot_hdu,
        hdu=hdu_int,
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
            img._data *= 1e6
            cbar_label = r"$uK_{CMB}$"
        elif units == "uK_RJ":
            img._data *= 1e6
            cbar_label = r"$uK_{RJ}$"
        elif units == "uJy/beam":
            img._data *= 0.7 * 1e6
            cbar_label = r"$\mu Jy/beam$"
        else:
            cbar_label = str(units)

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
        elif "gnfw_rs" in cfg["model"]["structures"].keys():
            mod_type = "gnfw_rs"
        else:
            raise ValueError("For R500, must have structure type gnfw_rs, A10, or EA10")

        # Get index of structure of interest. TODO: currently only plots first
        for i in range(len(results.structures)):
            if str(results.structures[i].name) == mod_type:
                break

        if mod_type == "ea10" or mod_type == "a10":
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

        elif mod_type == "gnfw_rs":
            for parameter in results.structures[i].parameters:
                if (
                    str(parameter.name.lower()) == "rs"
                    or str(parameter.name.lower()) == "r_s"
                ):
                    rs = parameter.val
                    break
            img.show_circles(
                ra, dec, radius=rs / 3600, coords_frame="world", color="green"
            )

    return img


def plot_cluster_act(
    name: str,
    fits_path: str,
    cfg_path: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    units: str = "mJy",
    bound: Optional[float] = None,
    radius: float = 2.0,
    plot_r=True,
    figsize: tuple[float, float] = (5, 5),
    ncontours: int = 0,
    hdu: int = 0,
    downsample: int = 1,
    smooth: float = 60.0,
    convention: str = "calabretta",
):
    """
    Function for doing core plotting.
    TODO: This function could probably use an args/kwargs, but there are an enourmous number of keyword args within so that might be difficult.

    Parameters
    ----------
    name : str
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
    smooth : float, default: 60.0
        Scale, in arcminutes, at which to smooth the image.
    convention : str, default: calabretta
        Determines interpretation of abigious fits headers. See aplpy.FITSFigure documentation

    Returns
    -------
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
