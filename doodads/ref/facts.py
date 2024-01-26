import astropy.units as u
from astropy.units import imperial
import astropy.constants as c
import numpy as np

from ..modeling.photometry import absolute_mag

__all__ = [
    'SiriusFacts',
    'MagellanFacts',
    'MagAOXFacts',
    'VLTFacts',
    'SubaruFacts',
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
    # "The Sirius System and Its Astrophysical Puzzles: Hubble Space Telescope and Ground-based Astrometry"
    # Bond, Howard et al. ApJ 2017
    mass = 2.063 * u.Msun

    # Radius from
    # Liebert, J.; Young, P. A.; Arnett, D.; Holberg, J. B.; Williams, K. A. (2005).
    # "The Age and Progenitor Mass of Sirius B". The Astrophysical Journal. 630 (1): L69–L72.
    # arXiv:astro-ph/0507523. Bibcode:2005ApJ...630L..69L. doi:10.1086/462419. S2CID 8792889.
    R = 1.711 * u.R_sun
    # derived from Liebert's L and R values
    T_eff = 9910 * u.K
    sigma_T_eff = 132 * u.K

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
    h = (8254 * imperial.foot).to(u.m)  # height above sea level

class MagAOXFacts:
    # from coron_pupil_v3.pdf (Jared Males, 2019) slide "MagAO-X Coronagraph Pupil Mask"
    CORONAGRAPH_PUPIL_DIAMETER = 9 * u.mm
    _pupil_to_primary = MagellanFacts.PRIMARY_STOP_DIAMETER / CORONAGRAPH_PUPIL_DIAMETER
    PUPIL_STOP_DIAMETER_PROJECTED = (8.6040 * u.mm * _pupil_to_primary).to(u.m)
    PUPIL_STOP_SECONDARY_DIAMETER_PROJECTED = (2.7900 * u.mm * _pupil_to_primary).to(u.m)
    SPIDER_ORIGIN_OFFSET = 0.4707 * u.mm
    SPIDER_ORIGIN_OFFSET_PROJECTED = (0.4707 * u.mm * _pupil_to_primary).to(u.mm)
    SPIDER_ANGLE_BETWEEN = 45 * u.deg
    SPIDER_ROTATION_ANGLE = 38.775 * u.deg
    SPIDER_WIDTH = 0.1917 * u.mm
    SPIDER_WIDTH_PROJECTED = (0.1917 * u.mm * _pupil_to_primary).to(u.m)
    BUMP_MASK_OFFSET_X, BUMP_MASK_OFFSET_Y = 2.853 * u.mm, -0.6705 * u.mm
    BUMP_MASK_OFFSET_X_PROJECTED, BUMP_MASK_OFFSET_Y_PROJECTED = (
        (2.853 * u.mm * _pupil_to_primary).to(u.m),
        (-0.6705 * u.mm * _pupil_to_primary).to(u.m)
    )
    BUMP_MASK_DIAMETER = 0.5742 * u.mm
    BUMP_MASK_DIAMETER_PROJECTED = (BUMP_MASK_DIAMETER * _pupil_to_primary).to(u.m)

class VLTFacts:
    PRIMARY_MIRROR_DIAMETER = 8.2 * u.m  # https://www.hq.eso.org/public/teles-instr/paranal-observatory/vlt/

class SubaruFacts:
    PRIMARY_STOP_DIAMETER = 8.2 * u.m  # https://www.naoj.org/Observing/Telescope/Parameters/
