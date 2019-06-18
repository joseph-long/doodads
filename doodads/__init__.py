import matplotlib
import matplotlib.gridspec as gridspec
from itertools import product
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils import centroid_com, centroid_1dg, centroid_2dg
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve_fft
from astropy.nddata.utils import Cutout2D
from astropy.modeling import fitting
from astropy.modeling.models import Gaussian2D
import numpy as np
from matplotlib.colors import LogNorm
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


def gcf():
    import matplotlib.pyplot as plt
    return plt.gcf()


def gca():
    import matplotlib.pyplot as plt
    return plt.gca()


def init():
    import matplotlib.pyplot as plt
    plt.rc('image', origin='lower', interpolation='nearest')


def add_colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


@supply_argument(fig=lambda: gcf())
def image_grid(cube, columns, colorbar=False, cmap=None, fig=None):
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


def shift2(image, dx, dy, flux_tol=1e-15):
    '''
    Fast Fourier subpixel shifting

    Parameters
    ----------
    dx : float
        Translation in +X direction (i.e. a feature at (x, y) moves to (x + dx, y))
    dy : float
        Translation in +Y direction (i.e. a feature at (x, y) moves to (x, y + dy))
    flux_tol : float
        Fractional flux change permissible
        ``(sum(output) - sum(image)) / sum(image) < flux_tol``
    '''
    xfreqs = np.fft.fftfreq(image.shape[1])
    yfreqs = np.fft.fftfreq(image.shape[0])
    xform = np.fft.fft2(image)
    modified_xform = xform * \
        np.exp(2j*np.pi*((-dx*xfreqs)[np.newaxis,
                                      :] + (-dy*yfreqs)[:, np.newaxis]))
    new_image = np.fft.ifft2(modified_xform)
    assert (np.sum(image) - np.sum(new_image.real)) / \
        np.sum(image) < flux_tol, "Flux conservation violated by more than {}".format(
            flux_tol)
    return new_image.real


def mask_arc(data_shape, center, from_radius, to_radius, from_radians, to_radians, overall_rotation_radians):
    rho, phi = polar_coords(center, data_shape)
    phi = (phi + np.pi + overall_rotation_radians) % (2 * np.pi)
    mask = (from_radius <= rho) & (rho <= to_radius)
    from_radians %= (2 * np.pi)
    to_radians %= (2 * np.pi)
    if from_radians != to_radians:
        mask &= (from_radians <= phi) & (phi <= to_radians)
    return mask


def cartesian_coords(center, data_shape):
    '''returns xx, yy'''
    yy, xx = np.indices(data_shape, dtype=float)
    center_x, center_y = center
    yy -= center_y
    xx -= center_x
    return xx, yy


def polar_coords(center, data_shape):
    xx, yy = cartesian_coords(center, data_shape)
    rho = np.sqrt(yy**2 + xx**2)
    phi = np.arctan2(yy, xx)
    return rho, phi


def max_radius(center, data_shape):
    bottom_left = np.sqrt(center[0]**2 + center[1]**2)
    data_height, data_width = data_shape
    top_right = np.sqrt(
        (data_height - center[0])**2 + (data_width - center[1])**2)
    return min(bottom_left, top_right)


def encircled_energy_and_profile(data, center, dq=None, arcsec_per_px=None, normalize=None,
                                 display=False, ee_ax=None, profile_ax=None, label=None):
    '''Compute encircled energies and profiles

    Returns
    -------
    ee_rho_steps
    encircled_energy_at_rho
    profile_bin_centers_rho
    profile_value_at_rho
    '''
    rho, phi = polar_coords(center, data.shape)
    max_radius_px = int(max_radius(center, data.shape))

    ee_rho_steps = []
    profile_bin_centers_rho = []

    encircled_energy_at_rho = []
    profile_value_at_rho = []

    for n in np.arange(1, max_radius_px):
        interior_mask = rho < n - 1
        exterior_mask = rho < n
        ring_mask = exterior_mask & ~interior_mask
        ring_contains_saturated = False
        if dq is not None:
            ring_contains_saturated = np.count_nonzero(
                dq[ring_mask] & 0b00000010) > 0
            ring_mask &= dq == 0
            exterior_mask &= dq == 0

        # EE
        ee_npix = np.count_nonzero(exterior_mask)
        if ee_npix > 0:
            ee = np.nansum(data[exterior_mask])
            encircled_energy_at_rho.append(ee)
            ee_rho_steps.append(
                n * arcsec_per_px if arcsec_per_px is not None else n)

        # profile
        profile_npix = np.count_nonzero(ring_mask)
        if profile_npix > 0:
            profile_bin_centers_rho.append(
                (n - 0.5) * arcsec_per_px if arcsec_per_px is not None else n - 0.5)
            profile_value = np.nansum(data[ring_mask]) / profile_npix
            profile_value /= profile_npix
            # if not ring_contains_saturated:
            profile_value_at_rho.append(profile_value)
            # else:
            # profile_value_at_rho.append(np.nan)

    ee_rho_steps, encircled_energy_at_rho, profile_bin_centers_rho, profile_value_at_rho = (
        np.asarray(ee_rho_steps),
        np.asarray(encircled_energy_at_rho),
        np.asarray(profile_bin_centers_rho),
        np.asarray(profile_value_at_rho)
    )
    if normalize is not None:
        if normalize is True:
            ee_normalize_at_mask = profile_normalize_at_mask = None
        else:
            ee_normalize_at_mask = ee_rho_steps < normalize
            profile_normalize_at_mask = profile_bin_centers_rho < normalize
        encircled_energy_at_rho /= np.nanmax(
            encircled_energy_at_rho[ee_normalize_at_mask])
        profile_value_at_rho /= np.nanmax(
            profile_value_at_rho[profile_normalize_at_mask])

    if display:
        import matplotlib.pyplot as plt
        if ee_ax is None or profile_ax is None:
            _, (ee_ax, profile_ax) = plt.subplots(figsize=(8, 6), nrows=2)
        xlabel = r'$\rho$ [arcsec]' if arcsec_per_px is not None else r'$\rho$ [pixel]'
        ee_ax.set_xlabel(xlabel)
        ee_ax.set_ylabel('Encircled Energy')
        ee_ax.plot(ee_rho_steps, encircled_energy_at_rho, label=label)
        profile_ax.set_xlabel(xlabel)
        profile_ax.set_ylabel('Radial Profile')
        profile_ax.set_yscale('log')
        profile_ax.plot(profile_bin_centers_rho,
                        profile_value_at_rho, label=label)
        plt.tight_layout()
    return ee_rho_steps, encircled_energy_at_rho, profile_bin_centers_rho, profile_value_at_rho


def clipped_zoom(img, zoom_factor, **kwargs):
    '''https://stackoverflow.com/a/37121993/421355'''

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def rebin_1d(a, factor):
    assert a.shape[0] % factor == 0
    sh = a.shape[0] // factor, factor
    return a.reshape(sh).sum(-1)

def test_rebin():
    x = np.array([0, 1, 2, 3])
    assert np.all(rebin_1d(x, 2) == np.array([1, 5]))

STDDEV_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
FWHM_TO_STDDEV = 1. / STDDEV_TO_FWHM
