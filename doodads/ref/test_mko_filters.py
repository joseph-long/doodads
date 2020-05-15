from os.path import exists
import pytest
import numpy as np
import astropy.units as u

from . import mko_filters, hst_calspec

@pytest.mark.skipif(
    not hst_calspec.ALPHA_LYR_CALSPEC.exists,
    reason='Download HST CALSPEC data to test MKO F_lambda'
)
def test_f_lambda():
    # HST CALSPEC's model Vega spectrum has changed by 1-2%
    # and we're not sure what atmosphere model was used for the MKO
    # IR filter calculations. Using the up-to-date Vega model produces
    # F_lambda values within +/- 2%.
    VEGA = hst_calspec.VEGA
    for filt_name in mko_filters.VEGA_F_LAMBDA:
        filt_spec = getattr(mko_filters.MKO, filt_name)
        my_F_lambda = (VEGA.multiply(filt_spec).integrate() / filt_spec.integrate()).to(u.W / u.m**2 / u.um)
        F_lambda = mko_filters.VEGA_F_LAMBDA[filt_name]['F_lambda']
        print(filt_name, F_lambda, my_F_lambda, np.abs((my_F_lambda - F_lambda) / F_lambda))
        assert np.abs((my_F_lambda - F_lambda) / F_lambda) < 0.02
