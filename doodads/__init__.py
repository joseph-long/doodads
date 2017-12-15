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
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import Gaussian2D
from astropy.modeling import fitting
from astropy.nddata.utils import Cutout2D
from astropy.convolution import convolve_fft
from astropy.convolution.kernels import Gaussian2DKernel
from photutils import centroid_com, centroid_1dg, centroid_2dg

from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

import matplotlib.gridspec as gridspec
from itertools import product

@supply_argument(fig=lambda: plt.gcf())
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

@supply_argument(ax=lambda: plt.gca())
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


class CoupledCutout:
    LEAK_BOX_SIZE = 45
    PSF_BOX_SIZE = 180
    SMOOTH_WIDTH_PX = 3
    INITIAL_GUESS_LEAK_CENTER = 250, 287
    PSF_OFFSET = -60, 130

    def __init__(self, data, center_leak=None, center_top=None, center_bottom=None, refine=False, verbose=False):
        self.data = data
        if center_leak is not None:
            self.center_leak = center_leak
        else:
            self.center_leak = self.INITIAL_GUESS_LEAK_CENTER

        if refine:
            self.center_leak = self._centroid(
                initial_guess=self.center_leak,
                data=self.data,
                size=self.LEAK_BOX_SIZE,
                verbose=verbose
            )

        leak_x, leak_y = self.center_leak
        offset_x, offset_y = self.PSF_OFFSET
        if center_top is not None:
            self.center_top = center_top
        else:
            self.center_top = leak_x + offset_x, leak_y + offset_y
        if center_bottom is not None:
            self.center_bottom = center_bottom
        else:
            self.center_bottom = leak_x - offset_x, leak_y - offset_y

        if refine:
            self.center_top = self._centroid(
                initial_guess=self.center_top,
                data=self.data,
                size=self.PSF_BOX_SIZE,
                verbose=verbose
            )
            self.center_bottom = self._centroid(
                initial_guess=self.center_bottom,
                data=self.data,
                size=self.PSF_BOX_SIZE,
                verbose=verbose
            )
        if verbose:
            print("Leak term center:", self.center_leak)
            print("Top PSF center:", self.center_top)
            print("Bottom PSF center:", self.center_bottom)
        self.top_cutout = Cutout2D(
            self.data,
            position=self.center_top,
            size=self.PSF_BOX_SIZE,
        )
        self.bottom_cutout = Cutout2D(
            self.data,
            position=self.center_bottom,
            size=self.PSF_BOX_SIZE,
        )
        self.leak_cutout = Cutout2D(
            self.data,
            position=self.center_leak,
            size=self.LEAK_BOX_SIZE,
        )
    @classmethod
    def _centroid(cls, initial_guess, data, size, verbose=False):
        cutout = Cutout2D(
            data,
            copy=True,
            position=initial_guess,
            size=size,
        )
        cutout.data = convolve_fft(
            cutout.data,
            Gaussian2DKernel(cls.SMOOTH_WIDTH_PX),
            boundary='wrap'
        )
        centroid = centroid_1dg(cutout.data)
        if verbose:
            #fig, ax = plt.subplots()
            #ax.imshow(cutout.data)
            #ax.scatter(*centroid_com(cutout.data), label="COM")
            #ax.scatter(*centroid_1dg(cutout.data), label="1D G.")
            #ax.scatter(*centroid_2dg(cutout.data), label="2D G.")
            #ax.legend()
            print("Centroid:", centroid)
        return cutout.to_original_position(centroid)

    @supply_argument(ax=lambda: plt.gca())
    def display(self, ax=None, colorbar=False, vmin=None, vmax=None):
        img = ax.imshow(self.data, norm=LogNorm(), vmin=vmin, vmax=vmax)
        add_colorbar(img)
        ax.add_artist(plt.Circle(
            self.center_leak,
            radius=self.LEAK_BOX_SIZE / 2,
            facecolor="none",
            edgecolor="C0"
        ))
        for loc in (self.center_top, self.center_bottom):
            ax.add_artist(plt.Rectangle(
                xy=(
                    loc[0] - self.PSF_BOX_SIZE / 2,
                    loc[1] - self.PSF_BOX_SIZE / 2
                ),
                width=self.PSF_BOX_SIZE,
                height=self.PSF_BOX_SIZE,
                fc='none',
                ec='C1'
            ))
            ax.plot(*list(zip(self.center_leak, loc)))
        return ax

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
