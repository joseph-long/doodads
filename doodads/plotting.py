import matplotlib
from .utils import *

__all__ = (
    'init',
    'gcf',
    'gca',
    'add_colorbar',
    'imshow',
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

@supply_argument(ax=lambda: gca())
def imshow(im, *args, ax=None, log=False, colorbar=False, **kwargs):
    kwargs.update({
        'extent': image_extent(im.shape)
    })
    if log:
        mappable = logimshow(im, *args, ax=ax, **kwargs)
    else:
        mappable = ax.imshow(im, *args, **kwargs)
    if colorbar:
        add_colorbar(mappable)
    return mappable

@supply_argument(ax=lambda: gca())
def logimshow(im, *args, ax=None, **kwargs):
    kwargs.update({
        'norm': LogNorm()
        # 'norm': simple_norm(im, 'log')
    })
    return ax.imshow(im, *args, **kwargs)

@supply_argument(fig=lambda: gcf())
def image_grid(cube, columns, colorbar=False, cmap=None, fig=None, log=False, match=False):
    vmin = None
    vmax = None
    if match:
        for plane in range(cube.shape[0]):
            new_vmin = np.nanmin(vmin)
            vmin = new_vmin if new_vmin < vmin else vmin
            new_vmax = np.nanmax(vmax)
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
def show_diff(im1, im2, ax=None, vmax=None, cmap=matplotlib.cm.RdBu_r, as_percent=False):
    '''
    Plot (observed) - (expected) for 2D images. Optionally, show percent error
    (i.e. observed - expected / expected) with `as_percent`.
    Arguments
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
    '''
    diff = im1 - im2
    if as_percent:
        diff /= im2
        diff *= 100
    if vmax is not None:
        clim = vmax
    else:
        clim = np.nanmax(np.abs(diff))
    im = ax.imshow(diff, vmin=-clim, vmax=clim, cmap=cmap)
    return im

def three_panel_diff_plot(image_a, image_b, diff_kwargs=None, log=False, **kwargs):
    '''
    Three panel plot of image_a, image_b, (image_a-image_b)/image_b
    '''
    default_diff_kwargs = {'as_percent': True}
    if diff_kwargs is not None:
        default_diff_kwargs.update(diff_kwargs)
    diff_kwargs = default_diff_kwargs
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    if log:
        mappable_a = logimshow(image_a, ax=axes[0], **kwargs)
        mappable_b = logimshow(image_b, ax=axes[1], **kwargs)
    else:
        mappable_a = axes[0].imshow(image_a, **kwargs)
        mappable_b = axes[1].imshow(image_b, **kwargs)

    diffim = show_diff(image_a, image_b, ax=axes[2], **diff_kwargs)
    cbar = add_colorbar(diffim)
    cbar.set_label('% difference')
    fig.tight_layout()
    return fig
