import requests
import re
import numpy as np
import logging
from astropy.io import fits
import astropy.units as u
import os.path
from ..modeling.units import WAVELENGTH_UNITS
from .helpers import filter_from_fits
from .facts import MagellanFacts
from . import model_grids
from .. import utils

log = logging.getLogger(__name__)

__all__ = [
    'LA_SILLA_ATMOSPHERES',
]

_EXAMPLE_POST_DICT = {
    "INS.NAME": "SKYCALC",
    "INS.MODE": "swspectr",
    "POSTFILE.FLAG": 0,
    "TEL.SITE.HEIGHT": 2400,
    "SKYMODEL.TARGET.ALT": 90.0,
    "SKYMODEL.TARGET.AIRMASS": 1,
    "SKYMODEL.SEASON": 0,
    "SKYMODEL.TIME": 0,
    "SKYMODEL.PWV": 0.50,
    "SKYMODEL.MSOLFLUX": 130.00,
    "SKYMODEL.VACAIR": "vac",
    "SKYMODEL.WAVELENGTH.MIN": 500.00,
    "SKYMODEL.WAVELENGTH.MAX": 15000.00,
    "SKYMODEL.WAVELENGTH.GRID.MODE": "fixed_spectral_resolution",
    "SKYMODEL.WAVELENGTH.RESOLUTION": 20000,
    "SKYMODEL.LSF.KERNEL.TYPE": "none",
}

def retrieve_and_convert(output_filepath):
    postdata = _EXAMPLE_POST_DICT.copy()
    wavelengths = None
    spectra = None
    zenith_angles_deg = [0, 20, 40, 60]
    pwv_mm_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    n_spec = len(zenith_angles_deg) * len(pwv_mm_values)
    params = np.zeros(n_spec, dtype=[('airmass', float), ('pwv_mm', float)])

    idx = 0
    for zenith_angle_deg in zenith_angles_deg:
        for pwv_mm in pwv_mm_values:
            postdata["SKYMODEL.TARGET.ALT"] = 90 - zenith_angle_deg
            airmass = 1 / np.cos(np.deg2rad(zenith_angle_deg))
            postdata["SKYMODEL.TARGET.AIRMASS"] = airmass
            postdata["SKYMODEL.PWV"] = f"{pwv_mm:3.1f}"
            params[idx]['airmass'] = airmass
            params[idx]['pwv_mm'] = pwv_mm
            try:
                resp = requests.post("https://www.eso.org/observing/etc/bin/simu/skycalc", data=postdata)
                table_path = re.findall(r'/observing/etc/tmp/.+/skytable.fits', resp.text)[0]
                hdul = fits.open('https://www.eso.org' + table_path, cache=False)
            except Exception as e:
                print(resp.text)
                log.exception(e)
                raise RuntimeError(f"Unable to retrieve ESO SkyCalc atmosphere for {zenith_angle_deg=} {pwv_mm=}")
            if wavelengths is None:
                wavelengths = hdul[1].data['lam'] * u.nm
                spectra = np.zeros((n_spec, len(hdul[1].data)))
            spectra[idx] = hdul[1].data['trans']
            idx += 1

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU(params, name='PARAMS'),
        fits.ImageHDU(wavelengths.to(WAVELENGTH_UNITS).value, name='WAVELENGTHS'),
        fits.ImageHDU(spectra, name='MODEL_SPECTRA'),
    ])
    hdul.writeto(output_filepath, overwrite=True)

LA_SILLA_ATMOSPHERES_DATA = utils.REMOTE_RESOURCES.add(
    __name__,
    utils.CustomRemoteResource(retrieve_and_convert, 'eso_la_silla_atmospheres.fits'),
)
LA_SILLA_ATMOSPHERES = model_grids.ModelAtmosphereGrid(LA_SILLA_ATMOSPHERES_DATA.output_filepath, name="ESO La Silla")
