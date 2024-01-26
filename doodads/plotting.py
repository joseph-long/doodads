from itertools import product
from functools import partial
import numpy as np
import matplotlib
import matplotlib.cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from astropy import visualization as astroviz
import astropy.units as u

from .utils import *
__all__ = (
    'init',
    'gcf',
    'gca',
    'add_colorbar',
    'imshow',
    'matshow',
    'logimshow',
    'image_grid',
    'show_diff',
    'three_panel_diff_plot',
    'norm',
    'zscale',
    'contrast_limits_plot',
    'magma_k',
    'magma_g',
    'gray_k',
    'gray_g',
    'complex_color'
)

magma_k = matplotlib.cm.magma.copy()
magma_k.set_bad('k')
magma_g = matplotlib.cm.magma.copy()
magma_g.set_bad('0.5')
gray_k = matplotlib.cm.magma.copy()
gray_k.set_bad('k')
gray_g = matplotlib.cm.magma.copy()
gray_g.set_bad('0.5')

def init():
    matplotlib.rcParams.update({
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'Greys_r',
    })
    from astropy.visualization import quantity_support
    quantity_support()


def gcf() -> matplotlib.figure.Figure:
    import matplotlib.pyplot as plt
    return plt.gcf()


def gca() -> matplotlib.axes.Axes:
    import matplotlib.pyplot as plt
    return plt.gca()


def add_colorbar(mappable):
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def image_extent(shape, units_per_px):
    '''Produce an extent tuple to pass to `plt.imshow`
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
    '''
    units_per_px = units_per_px if units_per_px is not None else 1.0
    # left, right, bottom, top
    # -> when origin='lower':
    #     right, left, top, bottom
    npix_y, npix_x = shape
    min_y = (npix_y - 1) / 2
    max_y = -min_y
    min_x = (npix_x - 1) / 2
    max_x = -min_x
    return units_per_px * max_x, units_per_px * min_x, units_per_px * max_y, units_per_px * min_y


@supply_argument(ax=lambda: gca())
def imshow(im, *args, ax=None, log=False, colorbar=True, title=None, origin='center', units_per_px=None, **kwargs):
    '''
    Parameters
    ----------
    ax : axes.Axes
    log : bool
    colorbar : bool
    title : str
    origin : str
        default: center
    units_per_px : float
        scale factor multiplied with the pixel extent values

    Returns
    -------
    mappable
    '''
    if origin == 'center' and 'extent' not in kwargs:
        kwargs.update({
            'extent': image_extent(im.shape, units_per_px),
            'origin': 'lower',  # always explicit
        })
    elif origin == 'center':  # extent is given explicitly but origin was not
        kwargs.update({
            'origin': 'lower',  # always explicit
        })
    else:
        kwargs['origin'] = origin

    if 'complex' in str(im.dtype):
        if log:
            raise NotImplementedError("No log=True for complex images (yet)")
        im = complex_color(im)
    if log:
        mappable = logimshow(im, *args, ax=ax, **kwargs)
    else:
        mappable = ax.imshow(im, *args, **kwargs)
    if colorbar:
        add_colorbar(mappable)
    ax.set_title(title)
    return mappable

@supply_argument(ax=lambda: gca())
def logimshow(im, *args, ax=None, **kwargs):
    vmin = kwargs.pop('vmin') if 'vmin' in kwargs else None
    vmax = kwargs.pop('vmax') if 'vmax' in kwargs else None
    norm = astroviz.simple_norm(im, stretch='log', min_cut=vmin, max_cut=vmax)
    kwargs.update({
        'norm': norm
    })
    return ax.imshow(im, *args, **kwargs)

@supply_argument(ax=lambda: gca())
def matshow(im, *args, **kwargs):
    kwargs.update({'origin': 'upper'})
    if np.isscalar(im):
        im = [[im]]
    elif len(im.shape) == 1:
        im = im[:,np.newaxis]
    return imshow(im, *args, **kwargs)

@supply_argument(fig=lambda: gcf())
def image_grid(cube, columns, colorbar=False, cmap=None, fig=None, log=False, match=False):
    vmin = None
    vmax = None
    if match:
        vmin = np.nanmin(cube)
        vmax = np.nanmax(cube)
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
        if log:
            im = logimshow(cube[idx], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(cube[idx], cmap=cmap, vmin=vmin, vmax=vmax)
        if colorbar:
            add_colorbar(im)
    return fig


@supply_argument(ax=lambda: gca())
def show_diff(im1, im2, ax=None, vmin=None, vmax=None, cmap=matplotlib.cm.RdBu_r,
              as_percent=False, colorbar=False, clip_percentile=None, **kwargs):
    '''
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
    vmax : float
        Value corresponding to endpoints of colorbar (because
        vmin = -vmax). (default: np.nanmax(np.abs(im1 - im2)))
    cmap : matplotlib colormap instance
        Colormap to pass to `imshow` (default: matplotlib.cm.RdBu_r)
    as_percent : bool
        Whether to divide the difference by im2 before displaying
    clip_percentile : float
        Set vmin/vmax based on a percentile of the absolute differences
        (ignored when `vmax` is not None)
    '''
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
    im = ax.imshow(diff, vmin=clim_min, vmax=clim, cmap=cmap, **kwargs) # pylint: disable=invalid-unary-operand-type
    if colorbar:
        cbar = add_colorbar(im)
        if as_percent:
            cbar.set_label('% difference')
        else:
            cbar.set_label('difference')
    return im

def three_panel_diff_plot(image_a, image_b, title_a='', title_b='',
    title_diff='', as_percent=True, diff_kwargs=None, log=False,
    ax_a=None, ax_b=None, ax_aminusb=None, **kwargs
) -> tuple[list, list]:
    '''
    Three panel plot of image_a, image_b, (image_a-image_b) optionally scaled to percent difference

    Returns
    -------
    [mappable_a, mappable_b, diffim] : list[matplotlib.image.AxesImage]
    [ax_a, ax_b, ax_aminusb] : list[matplotlib Axes]
    '''
    import matplotlib.pyplot as plt
    missing_axes = [x is None for x in (ax_a, ax_b, ax_aminusb)]
    if any(missing_axes):
        present_axes = [x is not None for x in (ax_a, ax_b, ax_aminusb)]
        if any(present_axes):
            raise ValueError("Supply all axes or none")
        fig, (ax_a, ax_b, ax_aminusb) = plt.subplots(ncols=3, figsize=(12, 4))
    else:
        fig = ax_a.figure
    if log:
        mappable_a = logimshow(image_a, ax=ax_a, **kwargs)
        mappable_b = logimshow(image_b, ax=ax_b, **kwargs)
    else:
        mappable_a = ax_a.imshow(image_a, **kwargs)
        mappable_b = ax_b.imshow(image_b, **kwargs)
    add_colorbar(mappable_a)
    add_colorbar(mappable_b)
    ax_a.set_title(title_a)
    ax_b.set_title(title_b)
    ax_aminusb.set_title(title_diff)
    updated_diff_kwargs = kwargs.copy()
    updated_diff_kwargs.update({'colorbar': True, 'as_percent': as_percent})
    if diff_kwargs is not None:
        updated_diff_kwargs.update(diff_kwargs)
    show_diff(image_a, image_b, ax=ax_aminusb, **updated_diff_kwargs)
    return fig, [ax_a, ax_b, ax_aminusb]

def norm(image, interval='minmax', stretch='linear'):
    interval_kinds = {
        'zscale': astroviz.ZScaleInterval,
        'minmax': astroviz.MinMaxInterval,
    }
    stretch_kinds = {
        'linear': astroviz.LinearStretch,
        'log': astroviz.LogStretch,
    }
    norm = astroviz.ImageNormalize(
        image,
        interval=interval_kinds[interval](),
        stretch=stretch_kinds[stretch]()
    )
    return norm

zscale = partial(norm, interval='zscale')

@supply_argument(as_ax=lambda: gca())
def contrast_limits_plot(r_arcsec, contrast_ratios, distance, as_ax=None):
    '''
    '''
    from .modeling.astrometry import arcsec_to_au
    from .modeling.photometry import contrast_to_deltamag
    as_ax.plot(r_arcsec, contrast_ratios)
    as_ax.set(
        xlabel='separation [arcsec]',
        ylabel='estimated $5\sigma$ contrast',
        yscale='log',
    )
    as_ax.grid(which='both')

    au_ax = as_ax.twiny()
    xlim_arcsec = as_ax.get_xlim()
    au_ax.set(
        xlim=(arcsec_to_au(xlim_arcsec[0] * u.arcsec, distance).value, arcsec_to_au(xlim_arcsec[1] * u.arcsec, distance).value),
        xlabel='Separation [AU]'
    )

    dmag_ax = as_ax.twinx()
    ylim_contrast = as_ax.get_ylim()
    dmag_ax.set(
        ylim=(contrast_to_deltamag(ylim_contrast[0]), contrast_to_deltamag(ylim_contrast[1])),
        ylabel=r"estimated $5\sigma$ $\Delta m$"
    )
    return as_ax, au_ax, dmag_ax

def complex_color(z, log=False):
    # https://stackoverflow.com/a/20958684
    from colorsys import hls_to_rgb
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h,l,s)
    c = np.array(c)
    c = c.swapaxes(0,2)
    return c