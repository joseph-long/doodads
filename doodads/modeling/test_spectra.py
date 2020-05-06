import numpy as np
import astropy.units as u
import astropy.constants as c
import pytest
from . import spectra, physics


def test_integrate_simple():
    wavelengths = np.asarray([1,2,3,4]) * u.m
    fluxes = np.asarray([1, 1, 1, 1]) * u.W * u.m**-3
    spec = spectra.Spectrum(wavelengths, fluxes)
    filter_transmissions = np.asarray([0, 1, 1, 0])
    filt = spectra.Spectrum(wavelengths, filter_transmissions)

    spec = spec.multiply(filt)
    assert np.isclose(
        spec.integrate().si.value,
        (2 * u.W * u.m**-2).si.value
    ), "Numerical integration gave unexpected answer for flat fluxes, top hat filter"

def test_integrate_subset():
    wavelengths = np.asarray([1,2,3,4,5,6]) * u.m
    fluxes = np.asarray([1, 1, 1, 1, 1, 1]) * u.W * u.m**-3
    spec = spectra.Spectrum(wavelengths, fluxes)

    filter_transmissions = np.asarray([0, 1, 0])
    filter_wavelengths = np.asarray([2,3,4]) * u.m
    filt = spectra.Spectrum(filter_wavelengths, filter_transmissions)

    spec = spec.multiply(filt)

    assert np.isclose(
        spec.integrate().value,
        (1 * u.W * u.m**-2).value
    ), "Numerical integration of filter with smaller wavelength range gave unexpected answer for flat fluxes, top hat filter"

def test_blackbody_flux():
    wls = np.logspace(np.log10(100), np.log10(20500), num=1000) * u.nm
    temperature = 5772 * u.K
    bb = spectra.Blackbody(temperature, 1 * u.Rsun, 1 * u.AU, wavelengths=wls)
    # is peak consistent?
    discretized_peak = bb.wavelengths[np.argmax(bb.values)].to(u.um).value
    analytic_peak = physics.wien_peak(temperature).to(u.um).value
    # note threshold depends on the sampling, so we pick coarse sampling
    # to keep tests lightweight
    threshold = 0.01
    assert np.abs(discretized_peak - analytic_peak) / analytic_peak < 0.01, "Discrepancy with solar peak wavelength"
    # is integrated flux ~Lsun scaled to 1 AU about the same as the solar constant?
    analytic_solar_constant = (c.L_sun / (4 * np.pi * (1 * u.AU)**2)).to(u.W / u.m**2)
    numerical_solar_constant = bb.integrate().to(u.W * u.m**-2)
    assert np.abs(numerical_solar_constant - analytic_solar_constant) / analytic_solar_constant < threshold, "Discrepancy with solar constant"
