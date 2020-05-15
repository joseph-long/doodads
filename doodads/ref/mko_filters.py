import os.path
import logging
import numpy as np
from scipy.interpolate import interp1d

from astropy.io import fits
import astropy.units as u

from ..modeling.units import WAVELENGTH_UNITS, COMMON_WAVELENGTH_START, COMMON_WAVELENGTH_END
from .. import utils
from ..modeling import photometry, spectra

__all__ = (
    'MKO',
)

log = logging.getLogger(__name__)

# MKO filter system
# compiled from http://irtfweb.ifa.hawaii.edu/IRrefdata/iwafdv.html
# and http://irtfweb.ifa.hawaii.edu/~nsfcam/filters.html

# "Isophotal wavelengths, isophotal frequencies, and flux densities for Vega for the MKO-NIR filters"
# Tokunaga & Vacca 2005
# Table 1:  Isophotal wavelengths and flux densities for Vega
# (note: only rows for MKO filters)
VEGA_F_LAMBDA = {
    'J':      {'lambda_iso': 1.2500 * u.um, 'F_lambda': 3.01e-09 * u.W / u.m**2 / u.um},
    'H':      {'lambda_iso': 1.6440 * u.um, 'F_lambda': 1.18e-09 * u.W / u.m**2 / u.um},
    'Kprime': {'lambda_iso': 2.1210 * u.um, 'F_lambda': 4.57e-10 * u.W / u.m**2 / u.um},
    'Ks':     {'lambda_iso': 2.1490 * u.um, 'F_lambda': 4.35e-10 * u.W / u.m**2 / u.um},
    'K':      {'lambda_iso': 2.1980 * u.um, 'F_lambda': 4.00e-10 * u.W / u.m**2 / u.um},
    'Lprime': {'lambda_iso': 3.7540 * u.um, 'F_lambda': 5.31e-11 * u.W / u.m**2 / u.um},
    'Mprime': {'lambda_iso': 4.7020 * u.um, 'F_lambda': 2.22e-11 * u.W / u.m**2 / u.um},
}

def _convert_mko_filter(download_filepath, output_filepath):
    table = np.genfromtxt(download_filepath, skip_header=1, names=['wavelength', 'transmission'])
    table['transmission'] /= 100
    # note: wavelengths are in um in the data files, not using astropy units here
    table = table[np.argsort(table['wavelength'])]
    mask = table['transmission'] > 0  # ignore negative transmission as unphysical
    columns = [
        fits.Column(name='wavelength', format='E', array=table[mask]['wavelength']),
        fits.Column(name='transmission', format='E', array=table[mask]['transmission']),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(output_filepath, overwrite=True)

J_FITS = utils.REMOTE_RESOURCES.add(
    url='http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_jmk_trans.dat',
    converter_function=_convert_mko_filter,
    output_filename='MKO_J_filter.fits',
).output_filepath
H_FITS = utils.REMOTE_RESOURCES.add(
    url='http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_hmk_trans.dat',
    converter_function=_convert_mko_filter,
    output_filename='MKO_H_filter.fits',
).output_filepath
KPRIME_FITS = utils.REMOTE_RESOURCES.add(
    url='http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_kpmk_trans.dat',
    converter_function=_convert_mko_filter,
    output_filename='MKO_Kprime_filter.fits',
).output_filepath
KS_FITS = utils.REMOTE_RESOURCES.add(
    url='http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_ksmk_trans.dat',
    converter_function=_convert_mko_filter,
    output_filename='MKO_Ks_filter.fits',
).output_filepath
K_FITS = utils.REMOTE_RESOURCES.add(
    url='http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_kmk_trans.dat',
    converter_function=_convert_mko_filter,
    output_filename='MKO_K_filter.fits',
).output_filepath
LPRIME_FITS = utils.REMOTE_RESOURCES.add(
    url='http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_lpmk_trans.dat',
    converter_function=_convert_mko_filter,
    output_filename='MKO_Lprime_filter.fits',
).output_filepath
MPRIME_FITS = utils.REMOTE_RESOURCES.add(
    url='http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_mpmk_trans.dat',
    converter_function=_convert_mko_filter,
    output_filename='MKO_Mprime_filter.fits',
).output_filepath

def _filter_from_fits(filepath, name):
    return spectra.FITSSpectrum(
        filepath,
        wavelength_column='wavelength',
        wavelength_units=u.um,
        value_column='transmission',
        value_units=u.dimensionless_unscaled,
        name=name
    )

MKO = photometry.FilterSet([
    _filter_from_fits(J_FITS, 'J'),
    _filter_from_fits(H_FITS, 'H'),
    _filter_from_fits(KPRIME_FITS, 'Kprime'),
    _filter_from_fits(KS_FITS, 'Ks'),
    _filter_from_fits(K_FITS, 'K'),
    _filter_from_fits(LPRIME_FITS, 'Lprime'),
    _filter_from_fits(MPRIME_FITS, 'Mprime'),
])

def plot_all():
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    MKO.plot_all(ax=ax)
    ax.figure.savefig(utils.generated_path('mko_filters.png'))

utils.DIAGNOSTICS.add(plot_all)
