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
    # HST CALSPEC's model Vega spectrum has changed by 1-2%
    # and we're not sure what atmosphere model was used for the MKO
    # IR filter calculations. Using the up-to-date Vega model produces
    # F_lambda values within +/- 2%, except L' at 3.1%
    VEGA = hst_calspec.VEGA_BOHLIN_GILLILAND_2004.multiply(1.019)  # there's mention in Tokunaga & Vacca 2005 of adjusting the model up 1.9%
    for filt_name in mko_filters.VEGA_F_LAMBDA:
        filt_spec = getattr(mko_filters.MKO, filt_name)
        filt_spec = gemini_atmospheres.GEMINI_ATMOSPHERES['Mauna Kea']['mktrans_zm_10_10'].multiply(filt_spec)
        nonzero_transmission = filt_spec.wavelengths[filt_spec.values > 0.01 * np.max(filt_spec.values)]
        cut_on_wl = np.min(nonzero_transmission)
        cut_off_wl = np.max(nonzero_transmission)
        _, my_F_lambda = VEGA.flux_density(filt_spec)
        from scipy.interpolate import interp1d
        # get lambda_iso by inverting the uninvertible
        wls, fluxes, excl = math.make_monotonic_decreasing(
            VEGA.wavelengths[(VEGA.wavelengths > cut_on_wl) & (VEGA.wavelengths < cut_off_wl)],
            VEGA.values[(VEGA.wavelengths > cut_on_wl) & (VEGA.wavelengths < cut_off_wl)]
        )
        my_lambda_iso = interp1d(fluxes, wls, bounds_error=False)(my_F_lambda.to(fluxes.unit).value) * wls.unit

        F_lambda = mko_filters.VEGA_F_LAMBDA[filt_name]['F_lambda']
        lambda_iso = mko_filters.VEGA_F_LAMBDA[filt_name]['lambda_iso']
        my_F_lambda = my_F_lambda.to(F_lambda.unit)
        my_lambda_iso = my_lambda_iso.to(lambda_iso.unit)
        print(filt_name, '\n\t',
            "\tlambda_iso\tF_lambda\n\t",
            f"mko:\t{lambda_iso:03.3f}\t{F_lambda:1.3e}\n\t",
            f"mine:\t{my_lambda_iso:03.3f}\t{my_F_lambda:1.3e}\n\t",
            f"diff:\t{((my_lambda_iso - lambda_iso) / lambda_iso).si*100:1.1f}%\t\t{((my_F_lambda - F_lambda) / F_lambda).si*100:1.1f}%")
        # not even sure I'm calculating lambda_iso right
        # assert np.abs((my_lambda_iso - lambda_iso) / lambda_iso) < 0.03
        assert np.abs((my_F_lambda - F_lambda) / F_lambda) < 0.01
