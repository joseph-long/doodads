'''
MKO filter system
=================

compiled from http://irtfweb.ifa.hawaii.edu/IRrefdata/iwafdv.html
and http://irtfweb.ifa.hawaii.edu/~nsfcam/filters.html
'''
import os.path
import logging
import numpy as np
from functools import partial
from scipy.interpolate import interp1d

from astropy.io import fits
import astropy.units as u

from ..modeling.units import WAVELENGTH_UNITS, COMMON_WAVELENGTH_START, COMMON_WAVELENGTH_END
from .. import utils
from ..modeling import photometry, spectra

from .helpers import filter_from_fits, generate_filter_set_diagnostic_plot

__all__ = (
    'MKO',
    'VEGA_F_LAMBDA'
)

log = logging.getLogger(__name__)

#: Taken from Table 1 "Isophotal wavelengths and flux densities for Vega" in
#: "Isophotal wavelengths, isophotal frequencies, and flux densities for Vega for the MKO-NIR filters"
#: by Tokunaga & Vacca (2005)
#: (note: only values for MKO filters)
VEGA_F_LAMBDA = {
    'J':      {'lambda_iso': 1.2500 * u.um, 'F_lambda': 3.01e-09 * u.W / u.m**2 / u.um},
    'H':      {'lambda_iso': 1.6440 * u.um, 'F_lambda': 1.18e-09 * u.W / u.m**2 / u.um},
    'Kprime': {'lambda_iso': 2.1210 * u.um, 'F_lambda': 4.57e-10 * u.W / u.m**2 / u.um},
    'Ks':     {'lambda_iso': 2.1490 * u.um, 'F_lambda': 4.35e-10 * u.W / u.m**2 / u.um},
    'K':      {'lambda_iso': 2.1980 * u.um, 'F_lambda': 4.00e-10 * u.W / u.m**2 / u.um},
    'Lprime': {'lambda_iso': 3.7540 * u.um, 'F_lambda': 5.31e-11 * u.W / u.m**2 / u.um},
    'Mprime': {'lambda_iso': 4.7020 * u.um, 'F_lambda': 2.22e-11 * u.W / u.m**2 / u.um},
}

def _convert_mko_filter(download_filepath, output_filepath, percent_transmission=True):
    table = np.genfromtxt(download_filepath, skip_header=1, names=['wavelength', 'transmission'])
    if percent_transmission:
        table['transmission'] /= 100
    # note: wavelengths are in um in the data files, not using astropy units here
    table = table[np.argsort(table['wavelength'])]
    wl = (table['wavelength'] * u.um).to(WAVELENGTH_UNITS)
    trans = table['transmission']
    mask = trans > 0  # ignore negative transmission as unphysical
    columns = [
        fits.Column(name='wavelength', format='E', array=wl[mask]),
        fits.Column(name='transmission', format='E', array=trans[mask]),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(output_filepath, overwrite=True)

_mko_filter_urls = {
    'J': 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_jmk_trans.dat',
    'H': 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_hmk_trans.dat',
    'Kprime': 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_kpmk_trans.dat',
    'Ks': 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_ksmk_trans.dat',
    'K': 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_kmk_trans.dat',
    'Lprime': 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_lpmk_trans.dat',
    'legacy_Lprime': 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_lp_trans.dat',
    'Mprime': 'http://irtfweb.ifa.hawaii.edu/~nsfcam/filters/nsfcam_mpmk_trans.dat',
}

_filters = []
for name in _mko_filter_urls:
    converter = _convert_mko_filter
    if 'legacy' in name:
        converter = partial(_convert_mko_filter, percent_transmission=False)
    res = utils.REMOTE_RESOURCES.add(
        module=__name__,
        url=_mko_filter_urls[name],
        converter_function=converter,
        output_filename=f'MKO_{name}_filter.fits',
    )
    _filters.append(filter_from_fits(res.output_filepath, name))

MKO = photometry.FilterSet(_filters)

utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, MKO, 'MKO'))
