import os
import typing
from functools import partial
from itertools import product
from typing import Literal, Optional

import astropy.units as u
import matplotlib
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.figure
import matplotlib.gridspec as gridspec
import numpy as np
from astropy import visualization as astroviz
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import *

__all__ = (
    "init",
    "gcf",
    "gca",
    "add_colorbar",
    "imshow",
    "matshow",
    "image_grid",
    "show_diff",
    "three_panel_diff_plot",
    "norm",
    "zscale",
    "contrast_limits_plot",
    "inferno_k",
    "inferno_g",
    "magma_k",
    "magma_g",
    "gray_k",
    "gray_g",
    "RdBu_k",
    "RdBu_g",
    "RdBu_r_k",
    "RdBu_r_g",
    "RdYlBu_k",
    "RdYlBu_g",
    "RdYlBu_r_k",
    "RdYlBu_r_g",
    "twilight_k",
    "twilight_g",
    "complex_color",
)
inferno_k: Colormap = matplotlib.cm.inferno.copy()
inferno_k.set_bad("k")
inferno_g: Colormap = matplotlib.cm.inferno.copy()
inferno_g.set_bad("0.5")

magma_k: Colormap = matplotlib.cm.magma.copy()
magma_k.set_bad("k")
magma_g: Colormap = matplotlib.cm.magma.copy()
magma_g.set_bad("0.5")
gray_k: Colormap = matplotlib.cm.gray.copy()
gray_k.set_bad("k")
gray_g: Colormap = matplotlib.cm.gray.copy()
gray_g.set_bad("0.5")
RdBu_k: Colormap = matplotlib.cm.RdBu.copy()
RdBu_k.set_bad("k")
RdBu_g: Colormap = matplotlib.cm.RdBu.copy()
RdBu_g.set_bad("0.5")
RdBu_r_k: Colormap = matplotlib.cm.RdBu_r.copy()
RdBu_r_k.set_bad("k")
RdBu_r_g: Colormap = matplotlib.cm.RdBu_r.copy()
RdBu_r_g.set_bad("0.5")
RdYlBu_k: Colormap = matplotlib.cm.RdYlBu.copy()
RdYlBu_k.set_bad("k")
RdYlBu_g: Colormap = matplotlib.cm.RdYlBu.copy()
RdYlBu_g.set_bad("0.5")
RdYlBu_r_k: Colormap = matplotlib.cm.RdYlBu_r.copy()
RdYlBu_r_k.set_bad("k")
RdYlBu_r_g: Colormap = matplotlib.cm.RdYlBu_r.copy()
RdYlBu_r_g.set_bad("0.5")
twilight_k: Colormap = matplotlib.cm.twilight.copy()
twilight_k.set_bad("k")
twilight_g: Colormap = matplotlib.cm.twilight.copy()
twilight_g.set_bad("0.5")

DEFAULT_CMAP: Colormap = magma_g
DEFAULT_MATRIX_CMAP: Colormap = magma_g
DEFAULT_DIVERGING_CMAP: Colormap = RdBu_r_g

_tableau_colorblind_10 = [
    [0, 107, 164],
    [255, 128, 14],
    [171, 171, 171],
    [89, 89, 89],
    [95, 158, 209],
    [200, 82, 0],
    [137, 137, 137],
    [162, 200, 236],
    [255, 188, 121],
    [207, 207, 207],
]

tableau_colorblind_10 = [
    [r / 256, g / 256, b / 256] for r, g, b in _tableau_colorblind_10
]

# https://colorcyclepicker.mpetroff.net/
custom_color_cycle = (
    "#1f8efd, #ee6f31, #c40127, #245f95, #919190, #99d0a2, #fcb6f3".split(", ")
)


def init():
    matplotlib.rcParams.update(
        {
            "image.origin": "lower",
            "image.interpolation": "nearest",
            "image.cmap": "Greys_r",
            "font.family": "serif",
            "axes.prop_cycle": matplotlib.cycler(color=custom_color_cycle),
        }
    )
    from astropy.visualization import quantity_support

    quantity_support()


def gcf() -> matplotlib.figure.Figure:
    import matplotlib.pyplot as plt

    return plt.gcf()


def gca() -> matplotlib.axes.Axes:
    import matplotlib.pyplot as plt

    return plt.gca()


def add_colorbar(
    mappable, colorbar_label=None, ax=None
) -> matplotlib.colorbar.Colorbar:
    import matplotlib.pyplot as plt

    last_axes = plt.gca()
    ax: matplotlib.axis.Axes = mappable.axes if ax is None else ax
    fig: matplotlib.figure.Figure = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar: matplotlib.colorbar.Colorbar = fig.colorbar(mappable, cax=cax)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)
    plt.sca(last_axes)
    return cbar


def image_extent(shape, units_per_px):
    """Produce an extent tuple to pass to `plt.imshow`
    that places 0,0 at the center of the image rather than
    the corner.

    Parameters
    ----------
    shape : 2-tuple of integers
    units_per_px : float
        scale factor multiplied with the pixel extent values

    Returns
    -------
    (max_x, min_x, max_y, min_y) : tuple
        When origin='lower' (after `init()`) this is
        the right, left, top, bottom coordinate
        for the array
    """
    units_per_px = units_per_px if units_per_px is not None else 1.0
    # left, right, bottom, top
    # -> when origin='lower':
    #     right, left, top, bottom
    npix_y, npix_x = shape
    # note this is the full covered extent, not exactly the coordinates
    # of the center of the corner pixel. so for pixels that range from
    # 0.0 to 1.0 with center at 0.5, you don't need to subtract the half-pixel here
    min_y = npix_y / 2
    max_y = -min_y
    min_x = npix_x / 2
    max_x = -min_x
    return (
        units_per_px * max_x,
        units_per_px * min_x,
        units_per_px * max_y,
        units_per_px * min_y,
    )


def _cmap_and_norm_for_image(
    arr: np.ndarray,
    cmap=None,
    log=False,
    symmetric=False,
    pct_min=None,
    pct_max=None,
    vmin=None,
    vmax=None,
):
    if "complex" in arr.dtype.name:
        arr = (arr * arr.conj()).real

    # Do the range calculations on |arr| if we want the
    # final result symmetric about zero
    if symmetric:
        calc_range_arr = np.abs(arr)
    else:
        calc_range_arr = arr

    if not symmetric:
        # use percentile min if provided, or vmin, or nanmin(arr)
        if pct_min is not None and vmin is not None:
            raise ValueError("Supply either pct_min or vmin, not both")
        elif pct_min is not None:
            vmin = np.nanpercentile(arr, pct_min)
        elif vmin is None:
            vmin = np.nanmin(arr)
        # final case is vmin is not None, -> vmin = vmin
    elif symmetric and (pct_min is not None or vmin is not None):
        raise ValueError("Cannot supply pct_min or vmin when symmetric=True")

    # use percentile max if provided, or vmax, or nanmax(arr)
    if pct_max is not None and vmax is not None:
        raise ValueError("Supply either pct_max or vmax, not both")
    elif pct_max is not None:
        vmax = np.nanpercentile(calc_range_arr, pct_max)
    elif vmax is None:
        vmax = np.nanmax(calc_range_arr)
    # final case is vmax is not None, -> vmax = vmax

    if symmetric:
        if log:
            norm_cls = matplotlib.colors.AsinhNorm
        else:
            norm_cls = astroviz.simple_norm
        norm = norm_cls(arr, vmin=-vmax, vmax=vmax)
        cmap = cmap or DEFAULT_DIVERGING_CMAP
    else:
        if log:
            norm = astroviz.simple_norm(arr, stretch="log", vmin=vmin, vmax=vmax)
        else:
            norm = astroviz.simple_norm(arr, vmin=vmin, vmax=vmax)
        cmap = cmap or DEFAULT_CMAP
    return cmap, norm


def imsave(
    fname: str | os.PathLike | typing.BinaryIO,
    im: np.ndarray,
    *args,
    colorize_complex=True,
    **kwargs,
):
    from matplotlib.image import imsave

    if "complex" in im.dtype.name:
        if colorize_complex:
            im = complex_color(im, log=kwargs.get("log", False))
        else:
            im = (im * im.conj()).real
    if "norm" in kwargs:
        # skip the whole _cmap_and_norm_for_image thing
        kwargs["cmap"] = kwargs.get("cmap", DEFAULT_CMAP)
    else:
        cmap, norm = _cmap_and_norm_for_image(
            im,
            cmap=kwargs.pop("cmap", None),
            log=kwargs.pop("log", False),
            symmetric=kwargs.pop("symmetric", False),
            pct_min=kwargs.pop("pct_min", None),
            pct_max=kwargs.pop("pct_max", None),
            vmin=kwargs.pop("vmin", None),
            vmax=kwargs.pop("vmax", None),
        )
        kwargs["norm"] = norm
        kwargs["cmap"] = cmap
    imsave(fname, im, *args, **kwargs)


@supply_argument(ax=lambda: gca())
def imshow(
    im,
    *args,
    ax: matplotlib.axes.Axes = None,
    log: bool = False,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    colorize_complex: bool = True,
    title: Optional[str] = None,
    origin: Literal["center", "upper", "lower"] = "center",
    units_per_px: Optional[float] = None,
    crop: Optional[float] = None,
    symmetric: Optional[bool] = False,
    cmap: Optional[Colormap] = None,
    pct_min: Optional[float] = None,
    pct_max: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
):
    """
    Parameters
    ----------
    ax : axes.Axes
    log : bool
    colorbar : bool
    colorbar_label : str
    colorize_complex : bool (default: False)
        Whether to plot complex phase as hue angle and magnitude as brightness
    title : str
    origin : str
        default: center
    units_per_px : float
        scale factor multiplied with the pixel extent values
    crop : float
        show central crop x crop cutout of the image
    symmetric : bool
        Whether to use a vmin/vmax range symmetric about zero and a diverging colormap (overridable)
    cmap : matplotlib.cm.Color

    Returns
    -------
    mappable
    """
    if origin == "center" and "extent" not in kwargs:
        kwargs.update(
            {
                "extent": image_extent(im.shape, units_per_px),
                "origin": "lower",  # always explicit
            }
        )
    elif origin == "center":  # extent is given explicitly but origin was not
        kwargs.update(
            {
                "origin": "lower",  # always explicit
            }
        )
    else:
        kwargs["origin"] = origin
    if "complex" in im.dtype.name:
        if colorize_complex:
            im = complex_color(im, log=log)
        else:
            im = (im * im.conj()).real
    if "norm" in kwargs:
        # skip the whole _cmap_and_norm_for_image thing
        kwargs["cmap"] = kwargs.get("cmap", DEFAULT_CMAP)
    else:
        cmap, norm = _cmap_and_norm_for_image(
            im,
            cmap=cmap,
            log=log,
            symmetric=symmetric,
            pct_min=pct_min,
            pct_max=pct_max,
            vmin=vmin,
            vmax=vmax,
        )
        kwargs["norm"] = norm
        kwargs["cmap"] = cmap
    mappable = ax.imshow(im, *args, **kwargs)

    if colorbar:
        add_colorbar(mappable, colorbar_label=colorbar_label)
    ax.set_title(title)
    if crop is not None:
        if origin == "center":
            ax.set(xlim=(-crop, crop), ylim=(-crop, crop))
        else:
            npix_y, npix_x = im.shape
            ctr_x, ctr_y = (npix_x - 1) / 2, (npix_y - 1) / 2
            ax.set(xlim=(ctr_x - crop, ctr_x + crop), ylim=(ctr_y - crop, ctr_y + crop))
    return mappable


def viewarr(arr, *args, **kwargs):
    import pyviewarr

    pyviewarr.viewarr(arr, *args, **kwargs)


@supply_argument(ax=lambda: gca())
def matshow(im, *args, **kwargs):
    kwargs.update({"origin": "upper"})
    if "cmap" not in kwargs:
        kwargs["cmap"] = (
            DEFAULT_MATRIX_CMAP
            if not kwargs.get("symmetric")
            else DEFAULT_DIVERGING_CMAP
        )
    if np.isscalar(im):
        im = [[im]]
    elif len(im.shape) == 1:
        im = im[:, np.newaxis]
    return imshow(im, *args, **kwargs)


@supply_argument(fig=lambda: gcf())
def image_grid(
    cube,
    columns,
    colorbar=False,
    cmap=None,
    fig=None,
    log=False,
    match=True,
    vmin=None,
    vmax=None,
):
    if match:
        if vmax is None:
            vmax = np.nanmax(cube)
        if vmin is None:
            vmin = np.nanmin(cube)
    rows = (
        cube.shape[0] // columns
        if cube.shape[0] % columns == 0
        else cube.shape[0] // columns + 1
    )
    gs = gridspec.GridSpec(rows, columns)
    for idx, (row, col) in enumerate(product(range(rows), range(columns))):
        if idx >= cube.shape[0]:
            break
        ax = fig.add_subplot(gs[row, col])
        im = imshow(cube[idx], cmap=cmap, vmin=vmin, vmax=vmax, log=log, ax=ax)
        if colorbar:
            add_colorbar(im)
    return fig


@supply_argument(ax=lambda: gca())
def show_diff(
    im1,
    im2,
    ax=None,
    log=False,
    vmin=None,
    vmax=None,
    cmap=matplotlib.cm.RdBu_r,
    as_percent=False,
    colorbar=False,
    colorbar_label=None,
    clip_percentile=None,
    norm_class=None,
    **kwargs,
):
    """
    Plot (observed) - (expected) for 2D images. Optionally, show percent error
    (i.e. (observed - expected) / expected) with `as_percent`.

    Parameters
    ---------

    im1 : array (2d)
        Observed values
    im2 : array (2d)
        Expected values
    ax : matplotlib.axes.Axes
        Axes into which to plot (default: current Axes)
    log : bool
        Whether to use a symmetric pseudo-logarithmic (asinh) stretch
        about zero
    vmax : float
        Value corresponding to endpoints of colorbar (because
        vmin = -vmax). (default: np.nanmax(np.abs(im1 - im2)))
    cmap : matplotlib colormap instance
        Colormap to pass to `imshow` (default: matplotlib.cm.RdBu_r)
    as_percent : bool
        Whether to divide the difference by im2 before displaying
    colorbar : bool
        Whether to add a colorbar
    colorbar_label : str
        What label to apply to the colorbar
    clip_percentile : float
        Set vmin/vmax based on a percentile of the absolute differences
        (ignored when `vmax` is not None)
    """
    diff = im1 - im2
    if as_percent:
        diff /= im2
        diff *= 100

    if vmax is not None:
        clim = vmax
    else:
        absdiffs = np.abs(diff)
        if clip_percentile is not None:
            clim = np.nanpercentile(absdiffs, clip_percentile)
        else:
            clim = np.nanmax(absdiffs)
    if vmin is not None:
        clim_min = vmin
    else:
        clim_min = -clim
    if log and norm_class is None:
        norm_class = matplotlib.colors.AsinhNorm
    if norm_class is not None:
        norm_instance = norm_class(vmin=clim_min, vmax=clim)
        # can't supply norm and vmin/vmax, so:
        clim_min = clim = None
        kwargs["norm"] = norm_instance
    im = imshow(
        diff, vmin=clim_min, vmax=clim, cmap=cmap, ax=ax, colorbar=False, **kwargs
    )  # pylint: disable=invalid-unary-operand-type
    if colorbar:
        if colorbar_label is None:
            if as_percent:
                colorbar_label = "% difference"
            else:
                colorbar_label = "difference"
        cbar: matplotlib.colorbar.Colorbar = add_colorbar(
            im, colorbar_label=colorbar_label
        )
    return im


def three_panel_diff_plot(
    image_a,
    image_b,
    title_a="",
    title_b="",
    title_aminusb="",
    as_percent=True,
    diff_kwargs=None,
    log=False,
    ax_a=None,
    ax_b=None,
    ax_aminusb=None,
    match_clim=True,
    **kwargs,
) -> tuple[list, list]:
    """
    Three panel plot of image_a, image_b, (image_a-image_b) optionally scaled to percent difference

    Returns
    -------
    [mappable_a, mappable_b, diffim] : list[matplotlib.image.AxesImage]
    [ax_a, ax_b, ax_aminusb] : list[matplotlib Axes]
    """
    updated_diff_kwargs = kwargs.copy()  # keep a pristine copy
    import matplotlib.pyplot as plt

    missing_axes = [x is None for x in (ax_a, ax_b, ax_aminusb)]
    if any(missing_axes):
        present_axes = [x is not None for x in (ax_a, ax_b, ax_aminusb)]
        if any(present_axes):
            raise ValueError("Supply all axes or none")
        fig, (ax_a, ax_b, ax_aminusb) = plt.subplots(ncols=3, figsize=(12, 4))
    else:
        fig = ax_a.figure
    if match_clim and ("vmin" not in kwargs) and ("vmax" not in kwargs):
        kwargs.update(
            {
                "vmin": np.nanmin([image_a, image_b]),
                "vmax": np.nanmax([image_a, image_b]),
            }
        )
    imshow(image_a, ax=ax_a, log=log, **kwargs)
    imshow(image_b, ax=ax_b, log=log, **kwargs)
    ax_a.set_title(title_a)
    ax_b.set_title(title_b)
    ax_aminusb.set_title(title_aminusb)
    updated_diff_kwargs.update({"colorbar": True, "as_percent": as_percent})
    if diff_kwargs is not None:
        updated_diff_kwargs.update(diff_kwargs)
    show_diff(image_a, image_b, ax=ax_aminusb, **updated_diff_kwargs)
    return fig, [ax_a, ax_b, ax_aminusb]


def norm(image, interval="minmax", stretch="linear"):
    interval_kinds = {
        "zscale": astroviz.ZScaleInterval,
        "minmax": astroviz.MinMaxInterval,
    }
    stretch_kinds = {
        "linear": astroviz.LinearStretch,
        "log": astroviz.LogStretch,
    }
    norm = astroviz.ImageNormalize(
        image, interval=interval_kinds[interval](), stretch=stretch_kinds[stretch]()
    )
    return norm


zscale = partial(norm, interval="zscale")


@supply_argument(as_ax=lambda: gca())
def contrast_limits_plot(r_arcsec, contrast_ratios, distance, as_ax=None):
    """ """
    from .modeling.astrometry import arcsec_to_au
    from .modeling.photometry import contrast_to_deltamag

    as_ax.plot(r_arcsec, contrast_ratios)
    as_ax.set(
        xlabel="separation [arcsec]",
        ylabel=r"estimated $5\sigma$ contrast",
        yscale="log",
    )
    as_ax.grid(which="both")

    au_ax = as_ax.twiny()
    xlim_arcsec = as_ax.get_xlim()
    au_ax.set(
        xlim=(
            arcsec_to_au(xlim_arcsec[0] * u.arcsec, distance).value,
            arcsec_to_au(xlim_arcsec[1] * u.arcsec, distance).value,
        ),
        xlabel="Separation [AU]",
    )

    dmag_ax = as_ax.twinx()
    ylim_contrast = as_ax.get_ylim()
    dmag_ax.set(
        ylim=(
            contrast_to_deltamag(ylim_contrast[0]),
            contrast_to_deltamag(ylim_contrast[1]),
        ),
        ylabel=r"estimated $5\sigma$ $\Delta m$",
    )
    return as_ax, au_ax, dmag_ax


def complex_color(z, log=False):
    # https://stackoverflow.com/a/20958684
    from colorsys import hls_to_rgb

    r = np.abs(z)
    if log:
        r = np.log10(r)
    arg = np.angle(z)

    hue = (arg + np.pi) / (2 * np.pi) + 0.5
    luminance = 1.0 - 1.0 / (1.0 + r**0.3)
    saturation = 0.8

    c = np.vectorize(hls_to_rgb)(hue, luminance, saturation)
    c = np.array(c)
    c = c.transpose(1, 2, 0)
    return c
