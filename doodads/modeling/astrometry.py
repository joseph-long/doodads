import numpy as np
import astropy.units as u

__all__ = ('arcsec_to_au', 'au_to_arcsec')

def arcsec_to_au(sep, d):
    return (np.tan(sep.to(u.rad)) * d).to(u.AU)

def au_to_arcsec(semimaj, d):
    return (np.arctan(((semimaj.to(u.AU)) / d.to(u.pc)).si.value) * u.rad).to(u.arcsec)
