from functools import partial
import numpy as np

from astropy.io import fits
import astropy.units as u

from ..modeling import photometry
from ..modeling.units import WAVELENGTH_UNITS
from .. import utils
from .helpers import filter_from_fits, generate_filter_set_diagnostic_plot

__all__ = [
    'IRCS'
]

def _convert_filter(download_filepath, output_filepath):
    table = np.genfromtxt(download_filepath, names=['wavelength', 'transmission'])
    wl = (table['wavelength'] * u.Angstrom).to(WAVELENGTH_UNITS)
    trans = table['transmission']
    columns = [
        fits.Column(name='wavelength', format='E', array=wl),
        fits.Column(name='transmission', format='E', array=trans),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(output_filepath, overwrite=True)

BR_ALPHA_URL = 'https://web.archive.org/web/20151012173309if_/http://www.subarutelescope.org/Observing/Instruments/IRCS/camera/txt/TR_Br_A.dat'

res = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url=BR_ALPHA_URL,
    converter_function=_convert_filter,
    output_filename=f'IRCS_Br_alpha.fits',
)

IRCS = photometry.FilterSet({'Br_alpha': filter_from_fits(res.output_filepath, f"IRCS Br-alpha")})
utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, IRCS, 'Subaru IRCS'))
