from os.path import exists
import pytest

from . import mko_filters, hst_calspec
from ..spectra import FITSSpectrum
from ..photometry import FilterSet

@pytest.mark.skipif(
    not exists(hst_calspec.OLD_ALPHA_LYR_FITS),
    reason='Download HST CALSPEC data to test MKO F_lambda'
)
def test_F_lambda():
    # It looks like the 1-2% change in Vega from recent (winter 2019)
    # CALSPEC changes results in a change to the F_lambda values we find
    # for the MKO filters relative to those given in Table 1 of
    # http://irtfweb.ifa.hawaii.edu/IRrefdata/iwafdv.html.
    #
    # If we use `alpha_lyr_stis_005` like Jared did, we reproduce the values within
    # <0.5%.
    vega = FITSSpectrum(hst_calspec.OLD_ALPHA_LYR_FITS)
