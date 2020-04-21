import numpy as np
import astropy.units as u
import astropy.constants as c
import pytest
import doodads as dd
from . import spectra

try:
    import pysynphot
except ImportError:
    pysynphot = None

def test_wien_peak():
    peak_um = spectra.wien_peak(5772 * u.K).to(u.um).value
    ref_um = 0.502079
    assert np.isclose(peak_um, ref_um), "You've donked up the Wien peak calculation somehow!"

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

def test_integrate_resampling():
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
    ), "Numerical integration of resampled filter gave unexpected answer for flat fluxes, top hat filter"

def test_blackbody_flux():
    wls = np.logspace(np.log10(100), np.log10(20500), num=1000) * u.nm
    temperature = 5772 * u.K
    fluxes = spectra.blackbody_flux(wls, temperature, 1 * u.Rsun, 1 * u.AU).to(u.W * u.m**-3)
    # is peak consistent?
    discretized_peak = wls[np.argmax(fluxes)].to(u.um).value
    analytic_peak = spectra.wien_peak(temperature).to(u.um).value
    # note threshold depends on the sampling, so we pick coarse sampling
    # to keep tests lightweight
    threshold = 0.01
    assert np.abs(discretized_peak - analytic_peak) / analytic_peak < 0.01, "Discrepancy with solar peak wavelength"
    # is integrated flux ~Lsun scaled to 1 AU about the same as the solar constant?
    analytic_solar_constant = (c.L_sun / (4 * np.pi * (1 * u.AU)**2)).to(u.W / u.m**2)
    numerical_solar_constant = spectra.integrate(wls, fluxes, np.ones_like(fluxes)).to(u.W * u.m**-2)
    assert np.abs(numerical_solar_constant - analytic_solar_constant) / analytic_solar_constant < threshold, "Discrepancy with solar constant"

@pytest.mark.skipif(
    pysynphot is None,
    reason='Install pysynphot to test agreement'
)
@pytest.mark.filterwarnings("ignore")
def test_blackbody_pysynphot():
    bb = pysynphot.BlackBody(5000)
    bb.convert('flam')
    my_bb = spectra.blackbody_flux(bb.wave * u.AA, 5000 * u.K, 1 * u.Rsun, 1 * u.kpc)
    my_bb_vals = my_bb.to(u.erg / u.s / u.cm**2 / u.AA).value
    assert np.all((my_bb_vals - bb.flux) / bb.flux < 0.005), 'Blackbody flux discrepancy exceeds 0.5% between PySynphot and ours'
