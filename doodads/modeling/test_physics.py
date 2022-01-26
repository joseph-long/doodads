import numpy as np
import astropy.units as u
import astropy.constants as c
import pytest

from . import physics

def test_mass_surface_gravity_to_radius():
    '''Earth mass and earth gravity should produce earth radius'''
    radius = physics.mass_surface_gravity_to_radius(1 * u.M_earth, c.g0.cgs).to(u.R_earth)
    assert (radius - c.R_earth) / c.R_earth < 0.005, 'Radius from mass & log g incorrect'

def test_wien_peak():
    peak_um = physics.wien_peak(5772 * u.K).to(u.um).value
    ref_um = 0.502079
    assert np.isclose(peak_um, ref_um), "You've donked up the Wien peak calculation somehow!"

try:
    import pysynphot
except ImportError:
    pysynphot = None

@pytest.mark.skipif(
    pysynphot is None,
    reason='Install pysynphot to test agreement'
)
@pytest.mark.filterwarnings("ignore")
def test_blackbody_pysynphot():
    bb = pysynphot.BlackBody(5000)
    bb.convert('flam')
    my_bb = physics.blackbody_flux(bb.wave * u.AA, 5000 * u.K, 1 * u.Rsun, 1 * u.kpc)
    my_bb_vals = my_bb.to(u.erg / u.s / u.cm**2 / u.AA).value
    assert np.all((my_bb_vals - bb.flux) / bb.flux < 0.005), 'Blackbody flux discrepancy exceeds 0.5% between PySynphot and ours'
