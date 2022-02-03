import astropy.units as u
import astropy.constants as c
import numpy as np

from ..modeling.photometry import absolute_mag

__all__ = [
    'SiriusFacts',
    'MagellanFacts',
]

class SiriusFacts:
    d = 2.64 * u.pc
    # Spectral type from Gray, R. O.; Corbally, C. J.; Garrison, R. F.;
    # McFadden, M T.; Robinson, P. E. (2003). "Contributions to the
    # Nearby Stars (NStars) Project: Spectroscopy of Stars Earlier than
    # M0 within 40 Parsecs: The Northern Sample. I."
    # Astronomical Journal. 126 (4): 2048–2059.
    # Bibcode:2003AJ....126.2048G. doi:10.1086/378365
    spectral_type = 'A0mA1 Va'
    # Luminosity from Liebert, James; Young, P. A.; Arnett, David;
    # Holberg, J. B.; Williams, Kurtis A. (2005). "The Age and
    # Progenitor Mass of Sirius B". The Astrophysical Journal.
    # 630 (1): L69–L72.
    # Bibcode:2005ApJ...630L..69L. doi:10.1086/462419. S2CID 8792889.
    L = 25.4 * u.Lsun
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


class MagellanFacts:
    # From LCO telescope information page
    PRIMARY_MIRROR_DIAMETER = 6502.4 * u.mm
    PRIMARY_STOP_DIAMETER = 6478.4 * u.mm
    SECONDARY_AREA_FRACTION = 0.074
    # computed from the above
    SECONDARY_DIAMETER = 2 * np.sqrt(
        ((PRIMARY_STOP_DIAMETER / 2) ** 2) * SECONDARY_AREA_FRACTION
    )
    # from MagAO-X Pupil Definition doc
    SPIDERS_OFFSET = 0.34 * u.m
