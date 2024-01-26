import astropy.units as u
from ..modeling import spectra
from ..modeling import photometry
import os.path

__all__ = [
    'MAGAOX',
    'H_ALPHA',
]

H_ALPHA_DAT = os.path.join(os.path.dirname(__file__), 'Alluxa_656.3-1_OD4_Ultra_Narrow_Bandpass_Filter_7019_T.dat')
H_ALPHA = spectra.TableSpectrum(H_ALPHA_DAT, u.nm, u.dimensionless_unscaled, name=r'[H$\alpha$] (MagAO-X)')

MAGAOX = photometry.FilterSet({'H_ALPHA': H_ALPHA})
