from itertools import product
import numpy as np
import matplotlib
import matplotlib.cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
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
)

def init():
    matplotlib.rcParams.update({
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'Greys_r'
    })
    from astropy.visualization import quantity_support
    quantity_support()


def gcf():
    import matplotlib.pyplot as plt
    return plt.gcf()


def gca():
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


def image_extent(shape):
    '''Produce an extent tuple to pass to `plt.imshow`
    that places 0,0 at the center of the image rather than
    the corner.

    Parameters
    ----------
    shape : 2-tuple of integers

    Returns
    -------
    (max_x, min_x, max_y, min_y) : tuple
        When origin='lower' (after `init()`) this is
        the right, left, top, bottom coordinate
        for the array
    '''
    # left, right, bottom, top
    # -> when origin='lower':
    #     right, left, top, bottom
    npix_y, npix_x = shape
    min_y = (npix_y - 1) / 2
    max_y = -min_y
    min_x = (npix_x - 1) / 2
    max_x = -min_x
    return max_x, min_x, max_y, min_y


@supply_argument(ax=lambda: gca())
def imshow(im, *args, ax=None, log=False, colorbar=True, title=None, origin='center', **kwargs):
    if origin == 'center':
        kwargs.update({
            'extent': image_extent(im.shape)
        })
    else:
        kwargs['origin'] = origin
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
    kwargs.update({
        'norm': LogNorm()
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
        for plane in range(cube.shape[0]):
            new_vmin = np.nanmin(cube[plane])
            vmin = new_vmin if new_vmin < vmin else vmin
            new_vmax = np.nanmax(cube[plane])
            vmax = new_vmax if new_vmax < vmax else vmax
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
def show_diff(im1, im2, ax=None, vmax=None, cmap=matplotlib.cm.RdBu_r,
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
    im = ax.imshow(diff, vmin=-clim, vmax=clim, cmap=cmap, **kwargs) # pylint: disable=invalid-unary-operand-type
    if colorbar:
        cbar = add_colorbar(im)
        if as_percent:
            cbar.set_label('% difference')
        else:
            cbar.set_label('difference')
    return im

def three_panel_diff_plot(image_a, image_b, title_a='', title_b='', title_diff='', as_percent=True, diff_kwargs=None, log=False, clip_percentile=None, **kwargs):
    '''
    Three panel plot of image_a, image_b, (image_a-image_b)/image_b
    '''
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    if log:
        mappable_a = logimshow(image_a, ax=axes[0], **kwargs)
        mappable_b = logimshow(image_b, ax=axes[1], **kwargs)
    else:
        mappable_a = axes[0].imshow(image_a, **kwargs)
        mappable_b = axes[1].imshow(image_b, **kwargs)
    add_colorbar(mappable_a)
    add_colorbar(mappable_b)
    axes[0].set_title(title_a)
    axes[1].set_title(title_b)
    axes[2].set_title(title_diff)
    updated_diff_kwargs = kwargs.copy()
    updated_diff_kwargs.update({'colorbar': True, 'as_percent': as_percent})
    if diff_kwargs is not None:
        updated_diff_kwargs.update(diff_kwargs)
    diffim = show_diff(image_a, image_b, ax=axes[2], **updated_diff_kwargs)
    fig.tight_layout()
    return fig, axes
