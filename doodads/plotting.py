from itertools import product
from functools import partial
import matplotlib.figure
import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from astropy import visualization as astroviz
import astropy.units as u

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
inferno_k = matplotlib.cm.inferno.copy()
inferno_k.set_bad("k")
inferno_g = matplotlib.cm.inferno.copy()
inferno_g.set_bad("0.5")

magma_k = matplotlib.cm.magma.copy()
magma_k.set_bad("k")
magma_g = matplotlib.cm.magma.copy()
magma_g.set_bad("0.5")
gray_k = matplotlib.cm.gray.copy()
gray_k.set_bad("k")
gray_g = matplotlib.cm.gray.copy()
gray_g.set_bad("0.5")
RdBu_k = matplotlib.cm.RdBu.copy()
RdBu_k.set_bad("k")
RdBu_g = matplotlib.cm.RdBu.copy()
RdBu_g.set_bad("0.5")
RdBu_r_k = matplotlib.cm.RdBu_r.copy()
RdBu_r_k.set_bad("k")
RdBu_r_g = matplotlib.cm.RdBu_r.copy()
RdBu_r_g.set_bad("0.5")
RdYlBu_k = matplotlib.cm.RdYlBu.copy()
RdYlBu_k.set_bad("k")
RdYlBu_g = matplotlib.cm.RdYlBu.copy()
RdYlBu_g.set_bad("0.5")
RdYlBu_r_k = matplotlib.cm.RdYlBu_r.copy()
RdYlBu_r_k.set_bad("k")
RdYlBu_r_g = matplotlib.cm.RdYlBu_r.copy()
RdYlBu_r_g.set_bad("0.5")
twilight_k = matplotlib.cm.twilight.copy()
twilight_k.set_bad("k")
twilight_g = matplotlib.cm.twilight.copy()
twilight_g.set_bad("0.5")

DEFAULT_MATRIX_CMAP = magma_g

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
custom_color_cycle = '#1f8efd, #ee6f31, #c40127, #245f95, #919190, #99d0a2, #fcb6f3'.split(', ')

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


def add_colorbar(mappable, colorbar_label=None, ax=None) -> matplotlib.colorbar.Colorbar:
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


def image_extent(shape, units_per_px, origin):
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
    #     left, right, top, bottom
    npix_y, npix_x = shape
    # note this is the full covered extent, not exactly the coordinates
    # of the center of the corner pixel. so for pixels that range from
    # 0.0 to 1.0 with center at 0.5, you don't need to subtract the half-pixel here
    if origin == 'center':
        ctr_y = (npix_y - 1) / 2
        ctr_x = (npix_x - 1) / 2
        return (
            units_per_px * -ctr_x,
            units_per_px * ctr_x,
            units_per_px * -ctr_y,
            units_per_px * ctr_y,
        )
    elif origin == 'lower':
        return (
            units_per_px * -0.5,
            units_per_px * (npix_x - 0.5),
            units_per_px * -0.5,
            units_per_px * (npix_y - 0.5),
        )
    elif origin == 'upper':
        return (
            units_per_px * -0.5,
            units_per_px * (npix_x - 0.5),
            units_per_px * (npix_y - 0.5),
            units_per_px * -0.5
        )




@supply_argument(ax=lambda: gca())
def imshow(
    im,
    *args,
    ax=None,
    log=False,
    colorbar=True,
    colorbar_label=None,
    title=None,
    origin="center",
    units_per_px=None,
    crop=None,
    **kwargs,
):
    """
    Parameters
    ----------
    ax : axes.Axes
    log : bool
    colorbar : bool
    colorbar_label : str
    title : str
    origin : str
        default: center
    units_per_px : float
        scale factor multiplied with the pixel extent values
    crop : float
        show central crop x crop cutout of the image

    Returns
    -------
    mappable
    """
    if 'extent' not in kwargs:
        kwargs.update({'extent': image_extent(im.shape, units_per_px, origin)})
    # Ensure 'origin' is set explicitly so extent is oriented correctly
    kwargs['origin'] = 'lower' if origin == 'center' else origin

    if "complex" in str(im.dtype):
        im = complex_to_rgb(im, log=log)
    if log:
        vmin = kwargs.pop("vmin") if "vmin" in kwargs else None
        vmax = kwargs.pop("vmax") if "vmax" in kwargs else None
        norm = astroviz.simple_norm(im, stretch="log", vmin=vmin, vmax=vmax)
        kwargs.update({"norm": norm})
        mappable = ax.imshow(im, *args, **kwargs)
    else:
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


@supply_argument(ax=lambda: gca())
def matshow(im, *args, **kwargs):
    kwargs.update({"origin": "upper"})
    if 'cmap' not in kwargs:
        kwargs['cmap'] = DEFAULT_MATRIX_CMAP
    if np.isscalar(im):
        im = [[im]]
    elif len(im.shape) == 1:
        im = im[:, np.newaxis]
    return imshow(im, *args, **kwargs)


@supply_argument(fig=lambda: gcf())
def image_grid(
    cube, columns, colorbar=False, cmap=None, fig=None, log=False, match=True,
    vmin=None, vmax=None,
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
        cbar: matplotlib.colorbar.Colorbar = add_colorbar(im, colorbar_label=colorbar_label)
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
            {"vmin": np.min([image_a, image_b]), "vmax": np.max([image_a, image_b])}
        )
    imshow(image_a, ax=ax_a, log=log, **kwargs)
    imshow(image_b, ax=ax_b, log=log, **kwargs)
    ax_a.set_title(title_a)
    ax_b.set_title(title_b)
    updated_diff_kwargs.update({"colorbar": True, "as_percent": as_percent})
    if diff_kwargs is not None:
        updated_diff_kwargs.update(diff_kwargs)
    show_diff(image_a, image_b, ax=ax_aminusb, **updated_diff_kwargs)
    ax_aminusb.set_title(title_aminusb)
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

import numpy as np
from matplotlib.colors import hsv_to_rgb

def complex_to_rgb(z, r_min=None, r_max=None, hue_start_deg=0, log=False):
    amp = np.abs(z)
    if r_min is not None:
        amp = np.where(amp < r_min, r_min, amp)
    if r_max is not None:
        amp = np.where(amp > r_max, r_max, amp)
    if log:
        amp = np.log10(amp)
    ph = np.angle(z, deg=True) + hue_start_deg
    # HSV are values in range [0,1]
    h = ((ph + 180) % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))
    result = hsv_to_rgb(np.dstack((h,s,v)))
    result[np.isinf(z)] = (1.0, 1.0, 1.0)
    result[np.isnan(z)] = (0.5, 0.5, 0.5)
    return result
