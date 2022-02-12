from os.path import exists
import pytest
import numpy as np
import astropy.units as u

from . import mko_filters, hst_calspec, gemini_atmospheres
from .. import math

@pytest.mark.skipif(
    (
        not hst_calspec.ALPHA_LYR_CALSPEC.exists or
        not mko_filters.MKO.exists or
        not gemini_atmospheres.GEMINI_ATMOSPHERES['Mauna Kea']['mktrans_zm_10_10']
    ),
    reason="""
Checking MKO F_lambda requires Vega from HST calspec,
MKO filter curves, and a representative Mauna Kea atmosphere from Gemini
"""
)
def test_f_lambda():
    # Note that this is an older HST VEGA, and that the order of resampling
    # seems to matter. This is probably not the most correct zeropoint
    # calculation but it's done this way to reproduce the tabulated values
    # from table 1.
    VEGA = hst_calspec.VEGA_BOHLIN_GILLILAND_2004.multiply(1.019)  # there's mention in Tokunaga & Vacca 2005 of adjusting the model up 1.9%
    for filt_name in mko_filters.VEGA_F_LAMBDA:
        filt_spec = getattr(mko_filters.MKO, filt_name)
        filt_spec = filt_spec.multiply(gemini_atmospheres.GEMINI_ATMOSPHERES['Mauna Kea']['mktrans_zm_10_10'])
        _, my_F_lambda = VEGA.flux_density(filt_spec)

        F_lambda = mko_filters.VEGA_F_LAMBDA[filt_name]['F_lambda']
        my_F_lambda = my_F_lambda.to(F_lambda.unit)
        print(filt_name, '\n\t',
            "\tF_lambda\n\t",
            f"mko:\t{F_lambda:1.3e}\n\t",
            f"mine:\t{my_F_lambda:1.3e}\n\t",
            f"diff:\t{((my_F_lambda - F_lambda) / F_lambda).si*100:1.1f}%")
        assert np.abs((my_F_lambda - F_lambda) / F_lambda) < 0.007
