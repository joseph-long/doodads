import astropy.units as u
import astropy.constants as c

from ..modeling.photometry import absolute_mag

__all__ = [
    'SiriusFacts'
]

class SiriusFacts:
    d = 2.64 * u.pc
    # Age from Bond et al. (2017)
    age = 242 * u.Myr
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
    # Temperature from
    # Adelman, Saul J. (8–13 July 2004). "The Physical Properties of normal A stars".
    # Proceedings of the International Astronomical Union. Poprad, Slovakia: Cambridge University Press. pp. 1–11.
    # Bibcode:2004IAUS..224....1A. doi:10.1017/S1743921304004314.
    T_eff = 9940 * u.K
    # Radius from
    # Liebert, J.; Young, P. A.; Arnett, D.; Holberg, J. B.; Williams, K. A. (2005).
    # "The Age and Progenitor Mass of Sirius B". The Astrophysical Journal. 630 (1): L69–L72.
    # arXiv:astro-ph/0507523. Bibcode:2005ApJ...630L..69L. doi:10.1086/462419. S2CID 8792889.
    R = 1.711 * u.R_sun
