import os
import warnings
from itertools import product
from pickle import load as pickle_load
from typing import Optional, Tuple, Union, overload

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colorbar, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import structures as struct
from ..core import header as hdr

__all__ = [
    "default_cmap",
    "plot_channel",
    "plot_hist",
    "plot_hist2d",
    "plot_map",
    "plot_pixel",
    "plot_profile",
    "save_channel_plot",
    "save_hist",
    "save_hist2d",
    "save_map_plot",
    "save_pixel_plot",
    "save_profile_plot",
    "show_channel",
    "show_hist",
    "show_hist2d",
    "show_map",
    "show_pixel",
    "show_profile",
]

path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(path, "astro_cmap.pickle"), "rb") as f:
    default_cmap = pickle_load(f)


# Loading and saving


def _load_fits(
    filename: str, path: Optional[str] = None
) -> Tuple[np.ndarray, fits.Header]:
    """Load FITS file. The extension of the file can be ommited. `.fits` and `.fits.gz` extension are both checked."""
    file = filename if path is None else os.path.join(path, filename)
    file = file.replace(".fits", "").replace(".gz", "")
    if os.path.exists(file + ".fits"):
        ext = ".fits"
    elif os.path.exists(file + ".fits.gz"):
        ext = ".fits.gz"
    else:
        raise FileNotFoundError(
            "No files with a .fits or .fits.gz extension were found for the path {}".format(
                file
            )
        )
    hdu = fits.open(file + ext)[0]
    return hdu.data, hdu.header


def _save_fits(
    input: "struct.Struct", filename: str, path: Optional[str] = None, overwrite=False
) -> None:
    """Save `input` structure in a FITS file. The extension of the file can be ommited."""
    if path is not None and not os.path.exists(path):
        os.makedirs(path)
    file = filename if path is None else os.path.join(path, filename)
    file = file.replace(".fits", "").replace(".gz", "") + ".fits"
    hdu = fits.PrimaryHDU(input.data, input.header)
    hdu.writeto(file, overwrite=overwrite)


# Visualization


def _colorbar(
    mappable: img.AxesImage, label: Optional[str] = None, **kwargs
) -> colorbar.Colorbar:
    """
    Creates and returns a nice looking colorbar.

    Parameters
    ----------
    mappable : img.AxesImage
        Image (basically returned by a function like imshow).
    label : Optional[str], optional
        Colorbar label. If None, no label is added, by default None.

    Returns
    -------
    colorbar.Colorbar
        Plotted colorbar.
    """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.10)
    cbar = fig.colorbar(mappable, cax=cax, label=label, **kwargs)
    plt.sca(last_axes)
    return cbar


def plot_map(
    map: "struct.Map",
    ax: Optional[plt.Axes] = None,
    label_unit: str = "angle",
    no_logical=False,
    norm: Optional[colors.Normalize] = None,
    cmap: Union[str, colors.Colormap] = default_cmap,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plots a map. Returns the figure axis and the colorbar."""
    if not isinstance(map, struct.Map):
        raise ValueError(f"map must be an instance of Map, not {type(map)}")
    label_unit = label_unit.lower()
    if label_unit not in ["index", "angle"]:
        raise ValueError(f"label_unit must be 'index' or 'angle', not {label_unit}")

    if cmap is None:
        cmap = default_cmap

    imshow_kwargs = {"norm": norm, "vmin": vmin, "vmax": vmax, "origin": "lower"}

    if (
        map.header["CTYPE1"].upper() == "INDICES"
        or map.header["CTYPE2"].upper() == "INDICES"
    ):
        label_unit = "index"

    if label_unit == "angle":
        bounds = hdr.bound_coordinates(map.header)
        imshow_kwargs.update(
            {
                "extent": bounds[0] + bounds[1],
                "aspect": map.shape[0]
                / map.shape[1]
                * abs((bounds[0][1] - bounds[0][0]) / (bounds[1][1] - bounds[1][0])),
            }
        )
    else:
        imshow_kwargs.update(
            {"extent": (1, map.header["NAXIS1"], 1, map.header["NAXIS2"])}
        )

    if map.is_logical() and not no_logical:
        cmap = colors.LinearSegmentedColormap.from_list(
            "binary cmap", [plt.cm.gray(0.0), plt.cm.gray(1.0)], 2
        )
        imshow_kwargs.update({"vmin": 0, "vmax": 1})
        colorbar_kwargs = {"ticks": [0.25, 0.75]}
    else:
        colorbar_kwargs = {}

    if ax is not None:
        im = ax.imshow(map.data, cmap=cmap, **imshow_kwargs)
    else:
        im = plt.imshow(map.data, cmap=cmap, **imshow_kwargs)
        ax = plt.gca()

    cbar = _colorbar(im, label=None, **colorbar_kwargs)
    if map.is_logical() and not no_logical:
        cbar.ax.set_yticklabels([0, 1])

    if label_unit == "index":
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    else:
        ax.set_xlabel("$\delta x$ (')")
        ax.set_ylabel("$\delta y$ (')")

    return im, ax, cbar


def plot_profile(
    profile: "struct.Profile",
    ax: plt.Axes = None,
    label_unit: str = "velocity",
    logy: bool = False,
    linestyle: Optional[str] = "solid",
    color: Optional[str] = "k",
    linewidth: Optional[float] = 1.5,
    label: Optional[str] = None,
):
    """Plot a profile. Returns the figure axis."""
    if not isinstance(profile, struct.Profile):
        raise ValueError(f"profile must be an instance of Profile, not {type(profile)}")

    plot_kwargs = {
        "linestyle": linestyle,
        "color": color,
        "linewidth": linewidth,
        "label": label,
    }

    if logy:
        plot = ax.semilogy if ax is not None else plt.semilogy
    else:
        plot = ax.plot if ax is not None else plt.plot

    plot_label = ax.set_xlabel if ax is not None else plt.xlabel

    if profile.header["CTYPE1"].upper() == "INDICES":
        label_unit = "index"

    idx = np.arange(profile.data.size)
    if label_unit == "velocity":
        line = plot(
            hdr.index_to_velocity(profile.header, idx), profile.data, **plot_kwargs
        )
        plot_label("Velocity (km/s)")
    elif label_unit == "frequency":
        line = plot(
            hdr.index_to_frequency(profile.header, idx), profile.data, **plot_kwargs
        )
        plot_label("Frequency (GHz)")
    else:
        line = plot(idx + 1, profile.data, **plot_kwargs)
        plot_label("Channel index")

    if ax is None:
        ax = plt.gca()

    return line, ax


def plot_channel(
    cube: "struct.Cube", z: Union[int, float], unit: str = "index", **kwargs
):
    """TODO"""
    return cube.get_channel(z, unit=unit).plot(**kwargs)


def plot_pixel(
    cube: "struct.Cube", xy: tuple[int, float], unit: str = "index", **kwargs
):
    """TODO"""
    return cube.get_pixel(xy, unit=unit).plot(**kwargs)


def save_map_plot(
    map: "struct.Map", filename: str, path: Optional[str] = None, **kwargs
) -> None:
    """TODO"""
    plot_map(map, **kwargs)
    if path is None:
        plt.savefig(filename)
    else:
        plt.savefig(os.path.join(path, filename))
    plt.clf()


def save_profile_plot(
    profile: "struct.Profile", filename: str, path: Optional[str] = None, **kwargs
) -> None:
    """TODO"""
    plot_profile(profile, **kwargs)
    if path is None:
        plt.savefig(filename)
    else:
        plt.savefig(os.path.join(path, filename))
    plt.clf()


def save_channel_plot(
    cube: "struct.Cube",
    z: Union[int, float],
    filename: str,
    unit: str = "index",
    path: Optional[str] = None,
    **kwargs,
) -> None:
    """TODO"""
    map = cube.get_channel(z, unit=unit)
    save_map_plot(map, filename, path, **kwargs)


def save_pixel_plot(
    cube: "struct.Cube",
    xy: Union[tuple[int], tuple[float]],
    filename: str,
    unit: str = "index",
    path: Optional[str] = None,
    **kwargs,
) -> None:
    """TODO"""
    profile = cube.get_pixel(xy, unit=unit)
    save_profile_plot(profile, filename, path, **kwargs)


def show_map(map: "struct.Map", **kwargs) -> None:
    """TODO"""
    plot_map(map, **kwargs)
    plt.show()


def show_profile(profile: "struct.Profile", **kwargs) -> None:
    """TODO"""
    plot_profile(profile, **kwargs)
    plt.show()


def show_channel(
    cube: "struct.Cube", z: Union[int, float], unit: str = "index", **kwargs
) -> None:
    """TODO"""
    map = cube.get_channel(z, unit=unit)
    show_map(map, **kwargs)


def show_pixel(
    cube: "struct.Cube",
    xy: Union[tuple[int], tuple[float]],
    unit: str = "index",
    **kwargs,
) -> None:
    """TODO"""
    profile = cube.get_pixel(xy, unit=unit)
    show_profile(profile, **kwargs)


# Histograms


def plot_hist(
    input: "struct.Struct",
    bins: Union[int, str] = "auto",
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    logy: bool = False,
    density: bool = False,
    color: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
) -> Tuple[np.ndarray, np.ndarray, plt.Axes]:
    """TODO"""
    a = input.data.flatten()
    a = a[~np.isnan(a)]

    if logx:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            _a = np.log10(a)
        bins = 10 ** np.histogram_bin_edges(_a[np.isfinite(_a)], bins)
    else:
        bins = np.histogram_bin_edges(a, bins)

    if ax is None:
        ax = plt.gca()

    values, bins, _ = ax.hist(a, bins=bins, log=logy, density=density, color=color)

    if logx:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return values, bins, ax


def plot_hist2d(
    input1: "struct.Struct",
    input2: "struct.Struct",
    bins: Union[int, tuple, str] = "auto",
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    logy: bool = False,
    logz: bool = False,
    density: bool = False,
    plot_identity: bool = False,
    cmap: Union[str, colors.Colormap] = default_cmap,
    scatter: bool = False,
    scatter_threshold: int = 1,
    zeros_to_nans: bool = True,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    vmin: float = None,
    vmax: float = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], plt.Axes]:
    """
    Plot a 2-dimensional histogram of two structures of same shape `input1` and `input2`.

    Parameters
    ----------
    input1 : `Cube | Map | Profile`.
        Structure whose data will be plotted on the x axis.
    input2 : `Cube | Map | Profile`.
        Structure whose data will be plotted on the y axis.
    bins : `int | tuple | str`, optional.
        Number of bins. If it is an integer, then the same number of bins is used for each axis.
        If it is a tuple, then we assume that the format is `(bins_x, bins_y)`.
        If bins is a string, then the number of bins is automatically computed (default: `'auto'`).
        For more information, refer to `numpy.histogram_bin_edges`.
    ax : `Axes`, optional.
        Matplotlib axis to plot the histogram. If `ax` is None, then `matplotlib.pyplot.gca()` is used.
        Default: `None`.
    logx : `bool`, optional.
        If `True`, then the x axis is plotted in log scale. Default: False.
    logy : `bool`, optional.
        If `True`, then the y axis is plotted in log scale. Default: False.
    logz : `bool`, optional.
        If `True`, then the colorbar is in log scale. Default: False.
    density : `bool`, optional.
        If `True`, then the density is plotted. Else, the standard counting of samples by bin is plotted.
        Default: `False`.
    plot_identity : `bool`, optional.
        If `True`, then the y=x line is plotted on the figure. Default: `False`.
    cmap : `str`, optional
        Default: `default_colormap`.
    scatter : `bool`, optional.
        If `True`, the samples that are less than 2 per bins are displayed individually
        (using `matplotlib.pyplot.scatter`) instead of being included in the histogram. Default: `False`.
    scatter_threshold : `int`, optional.
        The minimum number of pixel by bins necessary to be plotted as an histogram and not as a scatter plot.
        Must be positive. Default: 1.
    zeros_to_nans : `bool`, optional.
        If `True`, then the bins without samples are plotted as `nan`. Default: `True`.
    xlim : `tuple | None`, optional.
        X axis limits for `matplotlib.pyplot.xlim`. Default: `None`.
    ylim : `tuple | None`, optional.
        Y axis limits for `matplotlib.pyplot.ylim`. Default: `None`.
    vmin : `float`, optional.
        Minimum value for colormap. Default: `None`.
    vmax : `float`, optional.
        Maximum value for colormap. Default: `None`.

    Returns
    -------
    H : `ndarray`
        Matrix of 2-dimensional histogram.
    edges : `tuple` of `ndarray`
        Tuple of x and y axis bins `(xedges, yedges)`
    ax : `Axes`
        Matplotlib axis.
    """
    if input1.shape != input2.shape:
        raise ValueError(
            f"input1 and input2 must have the same shape (here {input1.shape} and {input2.shape})"
        )

    a = input1.data.flatten()
    b = input2.data.flatten()

    indices = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[indices], b[indices]

    if xlim is None:
        _xlim = (a.min(), a.max())
    else:
        _xlim = (
            a.min() if xlim[0] is None else xlim[0],
            a.max() if xlim[1] is None else xlim[1],
        )
    if ylim is None:
        _ylim = (b.min(), b.max())
    else:
        _ylim = (
            b.min() if ylim[0] is None else ylim[0],
            b.max() if ylim[1] is None else ylim[1],
        )

    indices = (a >= _xlim[0]) & (a <= _xlim[1]) & (b >= _ylim[0]) & (b <= _ylim[1])
    a, b = a[indices], b[indices]

    if isinstance(bins, str):
        if logx:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                _a = np.log10(a)
        else:
            _a = a
        if logy:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                _b = np.log10(b)
        else:
            _b = b
        _binsx = np.histogram_bin_edges(_a[np.isfinite(_a)], bins).size - 1
        _binsy = np.histogram_bin_edges(_b[np.isfinite(_b)], bins).size - 1
        _bins = round((_binsx * _binsy) ** 0.25)
        bins = (_bins, _bins)
    elif isinstance(bins, int):
        if bins <= 0:
            raise ValueError(
                "bins must be a positive integer, a tuple of two positive integers or a string"
            )
        bins = (bins, bins)
    elif isinstance(bins, tuple):
        if len(bins) != 2 or bins[0] <= 0 or bins[1] <= 0:
            raise ValueError(
                "bins must be a positive integer, a tuple of two positive integers or a string"
            )
    else:
        raise TypeError(
            f"bins must be an integer, a tuple of integer or a string, not {type(bins)}"
        )

    if logx:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            _a = np.log10(a)
        bin_edges_a = 10 ** np.histogram_bin_edges(_a[np.isfinite(_a)], bins[0])
    else:
        bin_edges_a = np.histogram_bin_edges(a, bins[0])

    if logy:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            _b = np.log10(b)
        bin_edges_b = 10 ** np.histogram_bin_edges(_b[np.isfinite(_b)], bins[1])
    else:
        bin_edges_b = np.histogram_bin_edges(b, bins[1])

    bin_edges = (bin_edges_a, bin_edges_b)

    H, xedges, yedges = np.histogram2d(a, b, bins=bin_edges, density=density)
    X, Y = np.meshgrid(xedges, yedges)

    if scatter or zeros_to_nans:
        H_count, _, __ = np.histogram2d(a, b, bins=bin_edges, density=False)

    if scatter:
        a_scatter, b_scatter = [], []
        for i, j in product(range(xedges.size - 1), range(yedges.size - 1)):
            if (0 < H_count[i, j]) and (H_count[i, j] <= scatter_threshold):
                _cond = (
                    (xedges[i] <= a)
                    & (a <= xedges[i + 1])
                    & (yedges[j] <= b)
                    & (b <= yedges[j + 1])
                )
                a_scatter += a[_cond].tolist()
                b_scatter += b[_cond].tolist()
                H[i, j] = 0

    if zeros_to_nans:
        H[H_count == 0] *= np.nan

    if vmin is not None:
        H[H < vmin] *= np.nan
    if vmax is not None:
        H[H > vmax] *= np.nan

    if ax is None:
        ax = plt.gca()

    im = ax.pcolormesh(
        X, Y, H.T, norm=colors.LogNorm(vmin, vmax) if logz else None, cmap=cmap
    )
    ax.get_figure().colorbar(im, ax=ax)

    if scatter:
        print("len", a_scatter.__len__(), b_scatter.__len__())
        ax.scatter(a_scatter, b_scatter, s=5, color="k")

    if plot_identity:
        _start = max(xedges[0], yedges[0])
        _end = min(xedges[-1], yedges[-1])
        ax.plot([_start, _end], [_start, _end], "--k")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    return H.T, (xedges, yedges), ax


def save_hist(
    input: "struct.Struct", filename: str, path: Optional[str] = None, **kwargs
):
    """TODO"""
    plot_hist(input, **kwargs)
    plt.savefig(os.path.join(path, filename))


def save_hist2d(
    input1: "struct.Struct",
    input2: "struct.Struct",
    filename: str,
    path: Optional[str] = None,
    **kwargs,
):
    """TODO"""
    plot_hist2d(input1, input2, **kwargs)
    plt.savefig(os.path.join(path, filename))


def show_hist(input: "struct.Struct", **kwargs):
    """TODO"""
    plot_hist(input, **kwargs)
    plt.show()


def show_hist2d(input1: "struct.Struct", input2: "struct.Struct", **kwargs):
    """TODO"""
    plot_hist2d(input1, input2, **kwargs)
    plt.show()
