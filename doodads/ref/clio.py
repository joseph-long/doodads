import astropy.units as u
from ..modeling import spectra
from ..modeling import photometry
import os.path

__all__ = [
    'CLIO_3_9_FILTER',
    'CLIO',
]

CLIO_3_9_FILTER_DAT = os.path.join(os.path.dirname(__file__), '3.9um_Clio.dat')
CLIO_3_9_FILTER = spectra.TableSpectrum(CLIO_3_9_FILTER_DAT, u.um, u.dimensionless_unscaled, name='Clio [3.95]')

CLIO = photometry.FilterSet({'NARROWBAND_3_95_UM': CLIO_3_9_FILTER})
