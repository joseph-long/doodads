import astropy.units as u
from ..modeling import spectra
import os.path

__all__ = [
    'CLIO_3_9_FILTER',
]

CLIO_3_9_FILTER_DAT = os.path.join(os.path.dirname(__file__), '3.9um_Clio.dat')
CLIO_3_9_FILTER = spectra.TableSpectrum(CLIO_3_9_FILTER_DAT, u.um, u.dimensionless_unscaled, name='Clio [3.95]')
