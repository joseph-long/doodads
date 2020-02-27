import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as c

WAVELENGTH_UNITS = u.m
FLUX_UNITS = u.W * u.m**-3

def integrate(wavelengths, fluxes, filter_transmission, filter_wavelengths=None):
    if filter_wavelengths is None:
        filter_wavelengths = wavelengths
    if len(filter_transmission) != len(filter_wavelengths):
        raise ValueError(
            f"Filter transmission (shape: {filter_transmission.shape}) "
            f"mismatched with wavelengths (shape: {filter_wavelengths.shape})"
        )
    wavelength_bins = np.diff(wavelengths)
    if len(np.unique(wavelength_bins)) == 1:
        np.append(wavelength_bins, wavelength_bins[-1])
    else:
        print('Dropping last wavelength with unknown bin width')
        wavelengths = wavelengths[:-1]
        fluxes = fluxes[:-1]

    # regrid if needed
    if filter_wavelengths is not wavelengths:
        # force to zero at ends of input
        min_wave_idx, max_wave_idx = np.argmin(filter_wavelengths), np.argmax(filter_wavelengths)
        filter_transmission[min_wave_idx] = 0
        filter_transmission[max_wave_idx] = 0
        filter_transmission = interp1d(filter_wavelengths, filter_transmission, bounds_error=False, fill_value=0.0)(wavelengths)

    # apply filter transmission at each wavelength
    fluxes *= filter_transmission

    # numerically integrate
    integrated_flux = np.trapz(fluxes, wavelengths)
    return integrated_flux


def blackbody_flux(wavelength, temperature, radius, distance):
    '''Blackbody flux at `wavelength` scaled by object radius and
    distance
    '''
    return (
        ((2 * np.pi * c.h * c.c**2) / wavelength**5)
        /
        (np.exp((c.h*c.c) / (wavelength * c.k_B * temperature)) - 1)
    ) * (radius / distance) ** 2


def wien_peak(T):
    return ((2898 * u.um * u.K) / T).to(u.um)
