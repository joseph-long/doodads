import numpy as np
from astropy.io import fits
import astropy.units as u
from ..modeling.units import WAVELENGTH_UNITS
from .helpers import filter_from_fits
from .. import utils

__all__ = [
    'GEMINI_ATMOSPHERES',
]


def _convert_gemini_atmosphere(download_filepath, output_filepath):
    table = np.genfromtxt(download_filepath, names=['wavelength_um', 'transmission'])
    # note: wavelengths are in um in the data files, not using astropy units here
    table = table[np.argsort(table['wavelength_um'])]
    wl = (table['wavelength_um'] * u.um).to(WAVELENGTH_UNITS)
    trans = table['transmission']
    mask = trans > 0  # ignore negative transmission as unphysical
    columns = [
        fits.Column(name='wavelength', format='E', array=wl[mask]),
        fits.Column(name='transmission', format='E', array=trans[mask]),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(output_filepath, overwrite=True)

# The raw data files generated by ATRAN for 0.9-5.6 microns are available via
# the following table. Any use of these data should reference Lord, S. D.,
# 1992, NASA Technical Memorandum 103957, and acknowledge Gemini Observatory.
# The models are in ASCII two-column format: (a) wavelength with a sampling of
# 0.00002µm and a resolution of 0.00004µm (0.04nm) and (b) transmission.
# The numbers in the titles of the files are 10X the water vapor in mm and 10X the airmass.

_base_url = "http://www.gemini.edu/sciops/ObsProcess/obsConstraints/atm-models/"
_filename_template = "{site}trans_zm_{pwv}_{airmass}.dat"
GEMINI_ATMOSPHERES = {
    'Mauna Kea': {},
    'Cerro Pachon': {}
}

# Mauna Kea
for airmass in [1.0, 1.5, 2.0]:
    for pwv_mm in [1.0, 1.6, 3.0, 5.0]:
        fn = _filename_template.format(
            site="mk",
            airmass=int(10 * airmass),
            pwv=int(10 * pwv_mm)
        )
        res = utils.REMOTE_RESOURCES.add(
            module=__name__,
            url=_base_url + fn,
            converter_function=_convert_gemini_atmosphere,
            output_filename=fn.replace('.dat', '.fits'),
        )
        name = f"Gemini ATRAN Mauna Kea airmass {airmass} pwv {pwv_mm} mm"
        GEMINI_ATMOSPHERES['Mauna Kea'][fn.replace('.dat', '')] = filter_from_fits(res.output_filepath, name)

# Cerro Pachon
for airmass in [1.0, 1.5, 2.0]:
    for pwv_mm in [2.3, 4.3, 7.6, 10.0]:
        fn = _filename_template.format(
            site="cp",
            airmass=int(10 * airmass),
            pwv=int(10 * pwv_mm)
        )
        res = utils.REMOTE_RESOURCES.add(
            module=__name__,
            url=_base_url + fn,
            converter_function=_convert_gemini_atmosphere,
            output_filename=fn.replace('.dat', '.fits'),
        )
        name = f"Gemini ATRAN Cerro Pachon airmass {airmass} pwv {pwv_mm} mm"
        GEMINI_ATMOSPHERES['Cerro Pachon'][fn.replace('.dat', '')] = filter_from_fits(res.output_filepath, name)