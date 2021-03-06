from os.path import exists
import numpy as np
import pytest
import astropy.units as u

from . import hst_calspec
from ..modeling import spectra

@pytest.mark.skipif(
    not hst_calspec.SUN.exists,
    reason='Download CALSPEC model spectra to run tests'
)
def test_solar_luminosity():
    solar_spec = hst_calspec.SUN
    solar_luminosity = solar_spec.integrate() * 4 * np.pi * (1 * u.AU)**2
    diff = (solar_luminosity.to(u.Lsun) - 1 * u.Lsun) / (1 * u.Lsun)
    assert diff < 0.005, (
        "Integrated solar model spectrum differs from Lsun by >0.5%"
    )
