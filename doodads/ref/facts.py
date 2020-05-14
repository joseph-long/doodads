import astropy.units as u
import astropy.constants as c

from ..modeling.photometry import absolute_mag

__all__ = [
    'SiriusFacts'
]

class SiriusFacts:
    d = 2.64 * u.pc
    # Magnitudes from IRAS calib paper
    # Cohen et al. "Spectra Irradiance Calibration" (1992)
    # Table 1
    m_J = -1.39
    M_J = absolute_mag(m_J, d)
    m_H = -1.40
    M_H = absolute_mag(m_H, d)
    m_K = -1.37
    M_K = absolute_mag(m_K, d)
    m_L = -1.36
    M_L = absolute_mag(m_L, d)
    m_Lprime = -1.36
    M_Lprime = absolute_mag(m_Lprime, d)
    m_M = -1.36
    M_M = absolute_mag(m_M, d)
