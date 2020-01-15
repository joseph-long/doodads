import matplotlib
import matplotlib.gridspec as gridspec
from itertools import product
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import wcs
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve_fft
from astropy.nddata.utils import Cutout2D
from astropy.modeling import fitting
from astropy.modeling.models import Gaussian2D
from astropy.visualization import simple_norm
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
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
    plt.rc('image', origin='lower', interpolation='nearest', cmap='Greys_r')


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
def logimshow(im, *args, ax=None, **kwargs):
    kwargs.update({
        # 'norm': LogNorm()
        'norm': simple_norm(im, 'log')
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

def three_panel_diff_plot(image_a, image_b, diff_kwargs=None, **kwargs):
    '''
    Three panel plot of image_a, image_b, image_a-image_b/image_b
    '''
    default_diff_kwargs = {'as_percent': True}
    if diff_kwargs is not None:
        default_diff_kwargs.update(diff_kwargs)
    diff_kwargs = default_diff_kwargs
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    add_colorbar(axes[0].imshow(image_a, **kwargs))
    add_colorbar(axes[1].imshow(image_b, **kwargs))
    diffim = show_diff(image_a, image_b, ax=axes[2])
    cbar = add_colorbar(diffim)
    cbar.set_label('% difference')
    fig.tight_layout()
    return fig

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
    '''center in x,y order; returns coord arrays xx, yy of data_shape'''
    yy, xx = np.indices(data_shape, dtype=float)
    center_x, center_y = center
    yy -= center_y
    xx -= center_x
    return xx, yy


def polar_coords(center, data_shape):
    '''center in x,y order; returns coord arrays rho, phi of data_shape'''
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

def describe(ndarray):
    return {
        'min': np.nanmin(ndarray),
        'median': np.median(ndarray.flat),
        'mean': np.nanmean(ndarray),
        'max': np.nanmax(ndarray),
        'nonfinite': np.count_nonzero(~np.isfinite(ndarray)),
    }

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
    Arguments
    ---------
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
