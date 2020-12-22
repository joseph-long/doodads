import numpy as np
from astropy import wcs
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve_fft
from astropy.nddata.utils import Cutout2D
from astropy.modeling import fitting
from astropy.modeling.models import Gaussian2D
from matplotlib.colors import LogNorm, SymLogNorm
from functools import wraps

from . import utils
from .utils import *
from . import plotting
from .plotting import *
from . import math
from .math import *
from . import modeling
from .modeling import *
from . import ref
from .ref import *
from . import analysis
from .analysis import *

def f_test(npix):
    '''Create a square npix x npix array of zeros and draw a capital F
    that is upright and facing right when plotted with (0,0) at lower left
    as regions of ones'''
    f_test = np.zeros((npix, npix))
    mid = npix // 2
    stem = (slice(mid//8, npix - mid//8), slice((mid - mid // 4) - mid//8, (mid - mid//4) + mid // 8))
    f_test[stem] = 1
    bottom = (slice(mid - mid//8, mid + mid//8), slice((mid - mid // 4) - mid//8, (mid - mid//4) + 2*mid//3))
    f_test[bottom] = 1
    top = (slice(npix - mid//8 - mid // 4, npix - mid//8), slice((mid - mid // 4) - mid//8, (mid - mid//4) + mid))
    f_test[top] = 1
    return f_test

def construct_centered_wcs(npix, ref_ra, ref_dec, rot_deg, deg_per_px):
    '''
    Parameters
    ----------
    npix : int
        number of pixels for a square cutout
    ref_ra : float
        right ascension in degrees
    ref_dec : float
        declination in degrees
    rot_deg : float
       angle between +Y pixel and +Dec sky axes

    Note: FITS images are 1-indexed, with (x,y) = (1,1)
    placed at the lower left when displayed. To place North up,
    the data should be rotated clockwise by `rot_deg`.
    '''
    # +X should be -RA when N is up and E is left
    # +Y should be +Dec when N is up and E is left
    scale_m = np.matrix([
        [-deg_per_px, 0],
        [0, deg_per_px]
    ])
    theta_rad = np.deg2rad(rot_deg)
    rotation_m = np.matrix([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    the_wcs = wcs.WCS(naxis=2)
    the_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    the_wcs.wcs.crpix = [npix / 2 + 1, npix / 2 + 1]  # FITS is 1-indexed
    the_wcs.wcs.crval = [ref_ra, ref_dec]
    the_wcs.wcs.cd = rotation_m @ scale_m
    return the_wcs

def count_nans(arr):
    '''Shorthand to count NaN values in an array'''
    return np.count_nonzero(np.isnan(arr))

from skimage.feature import register_translation

def find_shift(model, data, upsample_factor=100):
    if count_nans(model) != 0 or count_nans(data) != 0:
        raise ValueError("Can't compute subpixel shifts because NaN values present in inputs")
    return register_translation(model, data, upsample_factor=upsample_factor)
