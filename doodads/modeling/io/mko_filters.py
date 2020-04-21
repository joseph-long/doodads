import os.path
import logging
import numpy as np
from scipy.interpolate import interp1d

from astropy.io import fits
import astropy.units as u

from ..units import WAVELENGTH_UNITS, FLUX_UNITS, COMMON_WAVELENGTH_START, COMMON_WAVELENGTH_END
from ... import utils
from .. import spectra

log = logging.getLogger(__name__)

# MKO filter system
# compiled from http://irtfweb.ifa.hawaii.edu/IRrefdata/iwafdv.html
# and http://irtfweb.ifa.hawaii.edu/~nsfcam/filters.html

# "Isophotal wavelengths, isophotal frequencies, and flux densities for Vega for the MKO-NIR filters"
# Tokunaga & Vacca 2005
# Table 1:  Isophotal wavelengths and flux densities for Vega
# (note: only rows for MKO filters)
MKO_FILTERS_VEGA = {
    'J':      {'lambda_iso': 1.2500 * u.um, 'F_lambda': 3.01e-09 * u.W / u.m**2 / u.um},
    'H':      {'lambda_iso': 1.6440 * u.um, 'F_lambda': 1.18e-09 * u.W / u.m**2 / u.um},
    'Kprime': {'lambda_iso': 2.1210 * u.um, 'F_lambda': 4.57e-10 * u.W / u.m**2 / u.um},
    'Ks':     {'lambda_iso': 2.1490 * u.um, 'F_lambda': 4.35e-10 * u.W / u.m**2 / u.um},
    'K':      {'lambda_iso': 2.1980 * u.um, 'F_lambda': 4.00e-10 * u.W / u.m**2 / u.um},
    'Lprime': {'lambda_iso': 3.7540 * u.um, 'F_lambda': 5.31e-11 * u.W / u.m**2 / u.um},
    'Mprime': {'lambda_iso': 4.7020 * u.um, 'F_lambda': 2.22e-11 * u.W / u.m**2 / u.um},
}
MKO_FILTER_NAMES = tuple(MKO_FILTERS_VEGA.keys())
MKO_FILTERS_FITS = utils.generated_path('mko_filters.fits')

def plot_filters(ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    table = fits.getdata(MKO_FILTERS_FITS)
    for name in MKO_FILTER_NAMES:
        ax.plot(table['wavelength'], table[name], label=name)
    ax.legend()
    ax.set(xlabel='Wavelength [m]', ylabel='Transmission')
    return ax

# Filter Transmission Profiles
def download_and_convert_mko_filters(overwrite=False):
    import matplotlib
    matplotlib.use('Agg')
    logging.basicConfig(level='INFO')
    if not overwrite and os.path.exists(MKO_FILTERS_FITS):
        log.info(f"Output exists: {MKO_FILTERS_FITS}")
        return
    urls = [
        ('J', 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_jmk_trans.dat'),
        ('H', 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_hmk_trans.dat',),
        ('Kprime', 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_kpmk_trans.dat',),
        ('Ks', 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_ksmk_trans.dat',),
        ('K', 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_kmk_trans.dat',),
        ('Lprime', 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_lpmk_trans.dat',),
        ('Mprime', 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_mpmk_trans.dat',),
    ]
    from urllib.parse import urlparse
    curves = {}
    for name, url in urls:
        log.info(f"Processing MKO filter scan for {name}")
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        filepath = utils.download(url, os.path.join('mko_filter_data', filename))
        table = np.genfromtxt(filepath, skip_header=1, names=['wavelength', 'transmission'])
        table['transmission'] /= 100
        # note: wavelengths are in um in the data files, not using astropy units here
        table = table[np.argsort(table['wavelength'])]
        curves[name] = table

    # again, using floats as meters, not astropy units for ease of FITS storage
    common_wl_start, common_wl_end = COMMON_WAVELENGTH_START.to(WAVELENGTH_UNITS).value, COMMON_WAVELENGTH_END.to(WAVELENGTH_UNITS).value

    # different filters have different and even inconsistent sampling but
    # H, K, Ks, Kprime all sampled approximately this finely
    common_wl_step = (0.0005 * u.um).to(WAVELENGTH_UNITS).value
    common_wl = np.arange(common_wl_start, common_wl_end, common_wl_step)
    # scans are in um, so interpolation locations will be too:
    common_wl_um = (common_wl * u.m).to(u.um).value

    columns = []
    columns.append(fits.Column(name='wavelength', format='E', array=common_wl))
    for name in curves:
        wls = curves[name]['wavelength']
        trans = curves[name]['transmission']
        mask = trans >= 0
        # note: discarding negative values in transmission
        # and extrapolating zeroes beyond the filter scan
        regrid_trans = interp1d(wls[mask], trans[mask], bounds_error=False, fill_value=0.0)(common_wl_um)
        columns.append(fits.Column(name=name, format='E', array=regrid_trans))

    log.info(f"Writing {MKO_FILTERS_FITS}")
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(MKO_FILTERS_FITS, overwrite=True)
    log.info(f"Done writing {MKO_FILTERS_FITS}")

    ax = plot_filters()
    ax.figure.savefig(utils.generated_path('mko_filters.png'))
