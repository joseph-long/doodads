from functools import wraps

def supply_argument(**override_kwargs):
    '''
    Decorator to supply a keyword argument using a callable if
    it is not provided.
    '''

    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            for kwarg in override_kwargs:
                if kwarg not in kwargs:
                    kwargs[kwarg] = override_kwargs[kwarg]()
            return f(*args, **kwargs)
        return inner
    return decorator
import matplotlib
from matplotlib.colors import LogNorm
import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.modeling import fitting
from astropy.nddata.utils import Cutout2D
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel
from photutils import centroid_com, centroid_1dg, centroid_2dg

from mpl_toolkits.axes_grid1 import make_axes_locatable

def gcf():
    import matplotlib.pyplot as plt
    return plt.gcf()
def gca():
    import matplotlib.pyplot as plt
    return plt.gca()

def add_colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

import matplotlib.gridspec as gridspec
from itertools import product

@supply_argument(fig=lambda: gcf())
def image_grid(cube, columns, colorbar=False, cmap=None, fig=None):
    rows = (
        cube.shape[0] // columns
        if cube.shape[0] % columns == 0
        else cube.shape[0] // columns + 1
    )
    gs = gridspec.GridSpec(rows, columns)
    for idx, (row, col) in enumerate(product(range(rows), range(columns))):
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(cube[idx], cmap=cmap)
        if colorbar:
            add_colorbar(im)
    return fig

@supply_argument(ax=lambda: gca())
def show_diff(im1, im2, ax=None, vmax=None, cmap=matplotlib.cm.RdBu_r, scale='linear'):
    diff = im1 - im2
    if vmax is not None:
        clim = vmax
    else:
        clim = np.nanmax(np.abs(diff))
    if scale == 'linear':
        im = ax.imshow(diff, vmin=-clim, vmax=clim, cmap=cmap)
    elif scale == 'log':
        im = ax.imshow(diff, vmin=-clim, vmax=clim, cmap=cmap, norm=LogNorm())
    else:
        raise ValueError("Unknown scale: {}".format(scale))
    return im

def rotated_sigmoid_2d(x, y, maximum=1, theta=0, x_0=0, y_0=0, k=1):
    xs = x - x_0
    ys = y - y_0
    radii = np.sqrt(xs**2 + ys**2)
    thetae = np.arctan2(ys, xs)
    xx2 = (radii * np.cos(thetae - theta)).reshape([y.shape[0], x.shape[1]])
    return maximum / (1 + np.exp(-k * xx2))

def fwhm_to_stddev(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def stddev_to_fwhm(stddev):
    return (2 * np.sqrt(2 * np.log(2))) * stddev

def gaussian_2d(x, y, amplitude, center, fwhm):
    x_0, y_0 = center if center is not None else (0, 0)
    stddev = fwhm_to_stddev(fwhm)
    xs = (x - x_0)**2
    ys = (y - y_0)**2
    return amplitude * np.exp(- ((x - x_0)**2 + (y - y_0)**2)/(2 * stddev**2))

def half_gaussian_2d(x, y, amplitude, center, fwhm, sigmoid_theta=0, sigmoid_x_0=0, sigmoid_k=1.0):
    x_0, y_0 = center
    return (
        rotated_sigmoid_2d(x, y, 1.0, sigmoid_theta, sigmoid_x_0 + x_0, y_0, sigmoid_k) *
        gaussian_2d(x, y, amplitude, center, fwhm)
    )
