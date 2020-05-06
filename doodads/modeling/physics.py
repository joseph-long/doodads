import numpy as np
import astropy.units as u
import astropy.constants as c
from .units import FLUX_UNITS

__all__ = [
    'mass_log_g_to_radius',
    'blackbody_flux',
    'wien_peak'
]

def mass_log_g_to_radius(mass, log_g):
    grav_accel = 10**log_g * u.cm / u.s**2
    # g = (G M) / r^2
    # r^2 = (G M) / g
    # r = sqrt((G M) / g)
    return np.sqrt((c.G * mass) / grav_accel).si

def blackbody_flux(wavelength, temperature, radius, distance):
    '''Blackbody flux at `wavelength` scaled by object radius and
    distance
    '''
    return ((
        ((2 * np.pi * c.h * c.c**2) / wavelength**5)
        /
        (np.exp((c.h*c.c) / (wavelength * c.k_B * temperature)) - 1)
    ) * (radius / distance) ** 2).to(FLUX_UNITS)

def wien_peak(T):
    return ((2898 * u.um * u.K) / T).to(u.um)
