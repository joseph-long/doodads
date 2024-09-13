from dataclasses import dataclass
from collections import namedtuple
import numpy as np
from scipy.optimize import curve_fit
import astropy.units as u
from matplotlib import patches

from . import plotting
from . import utils

__all__ = (
    'calc_snr_mawet',
    'cartesian_coords',
    'clipped_zoom',
    'polar_coords',
    'describe',
    'downsample_first_axis',
    'encircled_energy_and_profile',
    'fwhm_to_stddev',
    'fwhm1d',
    'gaussian_2d',
    'gaussian_2d_symmetric',
    'plane_background',
    'gaussian_2d_with_background',
    'gaussian_2d_symmetric_with_background',
    'quick_fit_point_source_gaussian',
    'mask_arc',
    'max_radius',
    'polar_coords',
    'powerlaw_fit',
    'rebin_1d',
    'reduce_apertures',
    'shift2',
    'show_simple_aperture_locations',
    'simple_aperture_locations',
    'stddev_to_fwhm',
    'measure_snr',
    'lambda_over_d_to_arcsec',
    'arcsec_to_lambda_over_d',
    'moffat',
    'moffat_with_background',
    'quick_fit_point_source_moffat',
)


def lambda_over_d_to_arcsec(lambda_over_d, wavelength, d):
    unit_lambda_over_d = (wavelength.to(u.m) / d.to(u.m)).si.value * u.radian
    return (lambda_over_d * unit_lambda_over_d).to(u.arcsec)

def arcsec_to_lambda_over_d(arcsec, wavelength, d):
    unit_lambda_over_d = ((wavelength.to(u.m) / d.to(u.m)).si.value * u.radian).to(
        u.arcsec
    )
    lambda_over_d = (arcsec / unit_lambda_over_d).si
    return lambda_over_d


def simple_aperture_locations(r_px, pa_deg, resolution_element_px,
                              exclude_nearest=0, exclude_planet=False):
    '''Generator yielding aperture centers (x, y) in a ring of radius `r_px`
    and starting at angle `pa_deg` E of N. Unless `exclude_planet` is True,
    the first (x, y) pair gives the planet location (signal aperture).

    Specifying `exclude_nearest` > 0 will skip that many apertures
    from either side of the signal aperture's location'''
    circumference = 2 * r_px * np.pi
    aperture_pixel_diameter = resolution_element_px
    n_apertures = int(circumference / aperture_pixel_diameter)
    start_theta = np.deg2rad(pa_deg + 90)
    delta_theta = np.deg2rad(360 / n_apertures)
    if not exclude_planet:
        yield (r_px * np.cos(start_theta),
               r_px * np.sin(start_theta))
    for i in range(1 + exclude_nearest, n_apertures - exclude_nearest):
        offset_x, offset_y = (
            r_px * np.cos(start_theta + i * delta_theta),
            r_px * np.sin(start_theta + i * delta_theta)
        )
        yield offset_x, offset_y

@utils.supply_argument(ax=plotting.gca)
def show_simple_aperture_locations(image, resolution_element_px, r_px, pa_deg,
                                   exclude_nearest=0, exclude_planet=False, ax=None):
    '''Plot `image` and overplot the circular apertures of diameter
    `resolution_element_px` in a ring at radius `r_px`
    starting at `pa_deg` E of N.
    '''
    ctr = (image.shape[0] - 1) / 2
    im = ax.imshow(image)
    ax.axhline(ctr, color='w', linestyle=':')
    ax.axvline(ctr, color='w', linestyle=':')
    planet_dx, planet_dy = r_px * np.cos(np.deg2rad(90 + pa_deg)), r_px * np.sin(np.deg2rad(90 + pa_deg))
    ax.arrow(ctr, ctr, planet_dx, planet_dy, color='w', lw=2)
    for offset_x, offset_y in simple_aperture_locations(r_px, pa_deg, resolution_element_px, exclude_nearest=exclude_nearest, exclude_planet=exclude_planet):
        ax.add_artist(patches.Circle(
            (ctr + offset_x, ctr + offset_y),
            radius=resolution_element_px/2,
            edgecolor='orange',
            facecolor='none',
        ))

def reduce_apertures(image, r_px, starting_pa_deg, resolution_element_px, operation,
                     exclude_nearest=0, exclude_planet=False):
    '''apply `operation` to the pixels within radius `resolution_element_px`/2 of the centers
    of the simple aperture locations for a planet at `r_px` and `starting_pa_deg`, returning
    the locations and the results as a tuple with the first location and result corresponding
    to the planet aperture'''
    center = (image.shape[0] - 1) / 2, (image.shape[0] - 1) / 2
    xx, yy = cartesian_coords(center, image.shape)
    locations = list(simple_aperture_locations(r_px, starting_pa_deg, resolution_element_px, exclude_nearest=exclude_nearest, exclude_planet=exclude_planet))
    simple_aperture_radius = resolution_element_px / 2
    results = []
    for offset_x, offset_y in locations:
        dist = np.sqrt((xx - offset_x)**2 + (yy - offset_y)**2)
        mask = dist <= simple_aperture_radius
        results.append(operation(image[mask] / np.count_nonzero(mask & np.isfinite(image))))
    return locations, results

def calc_snr_mawet(signal, noises):
    '''Calculate signal to noise following the
    two-sample t test as defined in Mawet 2014'''
    return (
        signal - np.average(noises)
    ) / (
        np.std(noises) * np.sqrt(1 + 1/len(noises))
    )

def measure_snr(image, r_px, pa_deg, resolution_element_px, exclude_nearest=0, display=False):
    if display:
        show_simple_aperture_locations(image, resolution_element_px, r_px, pa_deg, exclude_nearest=exclude_nearest)
    _, results = reduce_apertures(image, r_px, pa_deg, resolution_element_px, np.sum, exclude_nearest=exclude_nearest)
    signal = results[0]
    noises = results[1:]
    return calc_snr_mawet(signal, noises)

def fwhm1d(values, locations=None):
    '''Compute the full width at half maximum for a 1D sequence of values
    and return FWHM, location of minimum, location of maximum. When
    `locations` is not supplied these locations are indices into the
    `values` array, otherwise they are taken from `locations` at the
    appropriate indices.

    Parameters
    ---------
    values : 1D array of length N
        Sequence of values to compute FWHM from
    locations : 1D array of length N
        Sequence of locations for each value in `values`

    Returns
    -------
    fwhm : int or locations.dtype
    min_location : int or locations.dtype
    max_location : int or locations.dtype
    '''
    mask = values > 0.5 * np.nanmax(values)
    idxs = np.arange(len(values))[mask]
    min_idx, max_idx = np.min(idxs), np.max(idxs)
    if locations is not None:
        return locations[max_idx] - locations[min_idx], locations[min_idx], locations[max_idx]
    return max_idx - min_idx, min_idx, max_idx

STDDEV_TO_FWHM = (2 * np.sqrt(2 * np.log(2)))

def fwhm_to_stddev(fwhm):
    return fwhm / STDDEV_TO_FWHM

def stddev_to_fwhm(stddev):
    return STDDEV_TO_FWHM * stddev

def gaussian_2d(x, y, amplitude=1, sigma_x=1, sigma_y=1, theta=0, x_center=0, y_center=0, xp=np):
    # following the modeling.Gaussian2D in Astropy
    a = (
        (xp.cos(theta)**2 / (2 * sigma_x**2))
        +
        (xp.sin(theta)**2 / (2 * sigma_y**2))
    )
    b = (
        (xp.sin(2 * theta) / (2 * sigma_x**2))
        -
        (xp.sin(2 * theta) / (2 * sigma_y**2))
    )
    c = (
        (xp.sin(theta)**2 / (2 * sigma_x**2))
        +
        (xp.cos(theta)**2 / (2 * sigma_y**2))
    )
    return amplitude * xp.exp(
        -a * (x - x_center)**2
        - b * (y - y_center) * (x - x_center)
        - c * (y - y_center)**2
    )

def plane_background(x, y, slope_x, slope_y, offset):
    return (slope_x * x) + (slope_y * y) + offset

def gaussian_2d_with_background(x, y, amplitude=1, sigma_x=1, sigma_y=1, theta=0, x_center=0, y_center=0, slope_x=0, slope_y=0, offset=0, xp=np):
    return gaussian_2d(x, y, amplitude, x_center, y_center, sigma_x, sigma_y, theta, xp=np) + plane_background(x, y, slope_x, slope_y, offset)

def gaussian_2d_symmetric_radial(r, amplitude=1, sigma=1, normalize=True, xp=np):
    norm_fact = 1.0
    if normalize:
        norm_fact = 1 / xp.sqrt(2 * xp.pi * sigma**2)
    return (
        norm_fact *
        amplitude *
        xp.exp(
            -1 * (r ** 2 / (2 * sigma**2))
        )
    )

def gaussian_2d_symmetric(x, y, amplitude=1, sigma=1, x_center=0, y_center=0, normalize=True, xp=np):
    rr = xp.hypot(x - x_center, y - y_center)
    return gaussian_2d_symmetric_radial(rr, amplitude=amplitude, sigma=sigma, normalize=normalize, xp=xp)

def gaussian_2d_symmetric_with_background(x, y, amplitude=1, sigma=1, x_center=0, y_center=0, slope_x=0, slope_y=0, offset=0, xp=np):
    return gaussian_2d_symmetric(x, y, amplitude, sigma, x_center, y_center, xp=xp) + plane_background(x, y, slope_x, slope_y, offset)

Gaussian2DSymmetricWithBackgroundFit = namedtuple('Gaussian2DSymmetricWithBackgroundFit', ('amplitude', 'sigma', 'x_center', 'y_center',  'slope_x', 'slope_y', 'offset'))

def _display_quick_fit(model, image, x_center, y_center, crop_limit, title='model'):
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    axs = fig.subplots(ncols=3)
    plotting.three_panel_diff_plot(
        model, image, title_a=title, title_b='image', origin='lower', as_percent=False,
        ax_a=axs[0], ax_b=axs[1], ax_aminusb=axs[2],
    )
    for ax in axs:
        ax.set(
            xlim=(x_center - crop_limit, x_center + crop_limit),
            ylim=(y_center - crop_limit, y_center + crop_limit),
        )
        ax.scatter(x_center, y_center, marker='+', alpha=0.5)
    plt.show(fig)

def quick_fit_point_source_gaussian(image, initial_x, initial_y, display=False):
    xs = np.arange(image.shape[1]) + 0.5
    ys = np.arange(image.shape[0]) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    coords = np.vstack((xx.ravel(), yy.ravel()))
    def _fit(coords, amplitude, sigma, x_center, y_center, slope_x, slope_y, offset):
        x, y = coords
        model = gaussian_2d_symmetric_with_background(x, y, amplitude, abs(sigma), x_center, y_center, slope_x, slope_y, offset)
        return model
    guess_bg, guess_peak = np.percentile(image, [1, 99])
    guess_amplitude = guess_peak - guess_bg
    popt, pcov = curve_fit(
        _fit,
        coords,
        image.ravel(),
        p0=(guess_amplitude, 1.0, initial_x, initial_y, 0.0, 0.0, guess_bg),
    )
    amplitude, sigma, x_center, y_center, slope_x, slope_y, offset = popt
    if display:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(ncols=3)
        limits = 1 * sigma
        model = gaussian_2d_symmetric_with_background(xx, yy, amplitude, sigma, x_center, y_center, slope_x, slope_y, offset)

        plotting.three_panel_diff_plot(
            model, image, title_a='model', title_b='image', origin='lower', as_percent=False,
            ax_a=axs[0], ax_b=axs[1], ax_aminusb=axs[2],
        )
        for ax in axs:
            # ax.set(
            #     xlim=(x_center - limits, x_center + limits),
            #     ylim=(y_center - limits, y_center + limits),
            # )
            ax.scatter(x_center, y_center, marker='+', alpha=0.5)
        plt.show(fig)
    return Gaussian2DSymmetricWithBackgroundFit(amplitude, abs(sigma), x_center, y_center, slope_x, slope_y, offset)




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

    Parameters
    ----------
    data
    center
    dq
    arcsec_per_px
    normalize
        if `normalize` is `True`, the `nanmax` of the EE and profile values
        are used respectively to normalize the peak of the EE and profile.
        otherwise, if normalize is a number, this is used as a radius in pixels
        within which

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
            profile_value_at_rho.append(profile_value)

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

    from scipy.ndimage import zoom
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

def downsample_first_axis(data, chunk_size):
    '''Downsample chunks of `chunk_size` along axis 0 with median combination'''
    ndata = data.shape[0]
    nchunks = ndata // chunk_size
    if ndata % chunk_size != 0:
        nchunks += 1
    output = np.zeros((nchunks,) + data.shape[1:])
    for chunk_idx in range(nchunks):
        if (chunk_idx + 1) * chunk_size > ndata:
            output[chunk_idx] = np.median(data[chunk_idx * chunk_size:], axis=0)
        else:
            output[chunk_idx] = np.median(data[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size], axis=0)
    return output

def rebin_1d(a, factor):
    assert a.shape[0] % factor == 0
    sh = a.shape[0] // factor, factor
    return a.reshape(sh).sum(-1)

def test_rebin():
    x = np.array([0, 1, 2, 3])
    assert np.all(rebin_1d(x, 2) == np.array([1, 5]))

def describe(arr):
    '''Describe contents of an array with useful statistics'''
    arr = np.asarray(arr)
    return {
        'dtype': arr.dtype,
        'shape': arr.shape,
        'min': np.nanmin(arr),
        'median': np.nanmedian(arr.flat),
        'mean': np.nanmean(arr),
        'max': np.nanmax(arr),
        'nonfinite': np.count_nonzero(~np.isfinite(arr)),
        'nan': np.count_nonzero(np.isnan(arr)),
        'std': np.nanstd(arr),
        'p50': np.nanpercentile(arr, 50),
        'p90': np.nanpercentile(arr, 90),
        'p99': np.nanpercentile(arr, 99),
    }

from scipy import optimize

def powerlaw_fit(xdata, ydata):
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    pinit = [1, 1]
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy))
    pfinal = out[0]
    index = pfinal[1]
    amp = 10**pfinal[0]
    return amp, index

def moffat(r, amplitude=1.0, core_width=1.0, power=1.5, normalize=True, xp=np):
    r'''Moffat distribution evaluated at (x, y)

    .. math::

        f(x, y) = A \left(1 + \frac{\left(x - x_{0}\right)^{2} + \left(y - y_{0}\right)^{2}}{\gamma^{2}}\right)^{- \alpha}

    Parameters
    ----------
    x : ndarray
    y
    core_width : float
        called gamma in Astropy, alpha in IRAF
    power : float
        called alpha in Astropy, beta in IRAF
    x_center : float
    y_center : float
    '''
    norm_const = 1.0
    if normalize:
        norm_const = (power - 1) / (xp.pi * core_width**2)
    return amplitude * norm_const * (
        1 + r**2/core_width**2
    )**(-power)

def moffat_xy(x, y, amplitude=1.0, core_width=1.0, power=1.5, x_center=0.0, y_center=0.0, normalize=True, xp=np):
    r'''Moffat distribution evaluated at (x, y)

    .. math::

        f(x, y) = A \left(1 + \frac{\left(x - x_{0}\right)^{2} + \left(y - y_{0}\right)^{2}}{\gamma^{2}}\right)^{- \alpha}

    Parameters
    ----------
    x : ndarray
    y
    core_width : float
        called gamma in Astropy, alpha in IRAF
    power : float
        called alpha in Astropy, beta in IRAF
    x_center : float
    y_center : float
    '''
    r = xp.sqrt((x - x_center)**2 + (y - y_center)**2)
    return moffat(r, amplitude=amplitude, core_width=core_width, power=power, normalize=normalize)


def moffat_with_background(x, y, amplitude=1.0, core_width=1.0, power=1.5, x_center=0.0, y_center=0.0, slope_x=0, slope_y=0, offset=0, xp=np):
    return (
        moffat_xy(x, y, amplitude=amplitude, core_width=core_width, power=power, x_center=x_center, y_center=y_center, xp=xp)
        +
        plane_background(x, y, slope_x, slope_y, offset)
    )

MoffatWithBackgroundFit = namedtuple('MoffatWithBackgroundFit', ('amplitude', 'x_center', 'y_center', 'core_width', 'power', 'slope_x', 'slope_y', 'offset'))

@dataclass
class MoffatWithBackgroundFit:
    amplitude : float
    x_center : float
    y_center : float
    core_width : float
    power : float
    slope_x : float
    slope_y : float
    offset : float

    def evaluate(self, xx, yy, xp=np):
        return moffat_with_background(xx, yy, self.amplitude, self.core_width, self.power, self.x_center, self.y_center, self.slope_x, self.slope_y, self.offset, xp=xp)

def quick_fit_point_source_moffat(image, initial_x, initial_y, display=False):
    xs = np.arange(image.shape[1]) + 0.5
    ys = np.arange(image.shape[0]) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    coords = np.vstack((xx.ravel(), yy.ravel()))
    def _fit(coords, amplitude, x_center, y_center, width, power, slope_x, slope_y, offset):
        x, y = coords
        model = moffat_with_background(x, y, amplitude, width, power, x_center, y_center, slope_x, slope_y, offset)
        return model
    guess_bg, guess_peak = np.percentile(image, [1, 99])
    guess_amplitude = guess_peak - guess_bg
    p0 = (guess_amplitude, initial_x, initial_y, 1.0, 1.5, 0.0, 0.0, guess_bg)
    # bounds = (
    #     np.array([0.01 * guess_amplitude, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf]),
    #     np.array([np.inf, np.max(xs), np.max(ys), max(*image.shape), np.inf, np.inf, np.inf, np.inf])
    # )
    print(f"{p0=}")
    # print(f"{bounds=}")
    popt, pcov = curve_fit(
        _fit, coords, image.ravel(),
        p0=p0,
        # bounds=bounds
    )
    amplitude, x_center, y_center, core_width, power, slope_x, slope_y, offset = popt
    model = MoffatWithBackgroundFit(amplitude, x_center, y_center, core_width, power, slope_x, slope_y, offset)
    if display:
        _display_quick_fit(model.evaluate(xx, yy), image, x_center, y_center, crop_limit=3 * model.core_width)
    return model
