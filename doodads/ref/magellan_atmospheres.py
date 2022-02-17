import re
import zipfile
import numpy as np
from astropy.io import fits
import astropy.units as u
from doodads.modeling.units import WAVELENGTH_UNITS
from .. import utils
from . import model_grids

__all__ = ['MANQUI_ATMOSPHERES']

_FILENAME_RE = re.compile(r'^manqui_zd([\d\.]+)_pwv([\d\.]+).txt$')

def _convert_manqui_atmospheres(download_filepath, output_filepath):
    z = zipfile.ZipFile(download_filepath)
    wavelengths_um = None
    spectra = None
    params = None
    for idx, fileinfo in enumerate(sorted(z.filelist, key=lambda x: x.filename)):
        name = fileinfo.filename
        match = _FILENAME_RE.match(name)
        if match is None:
            raise ValueError(f"Unexpected filename in {download_filepath} zipfile: {name}")
        zd, pwv = match.groups()
        zd = float(zd)
        airmass = 1 / np.cos(np.deg2rad(zd))
        pwv = float(pwv)
        with z.open(fileinfo, mode='r') as fh:
            wls, trans = np.genfromtxt(fh, unpack=True)
            if spectra is None:
                spectra = np.zeros((len(z.filelist), len(trans)))
                wavelengths_um = wls
                params = np.zeros(len(z.filelist), dtype=[('airmass', float), ('pwv_mm', float)])
            if wavelengths_um is not None:
                if not np.all(wls == wavelengths_um):
                    raise RuntimeError("Inconsistent sampling")
            spectra[idx] = trans
            params[idx]['airmass'] = airmass
            params[idx]['pwv_mm'] = pwv

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU(params, name='PARAMS'),
        fits.ImageHDU((wavelengths_um * u.um).to(WAVELENGTH_UNITS).value, name='WAVELENGTHS'),
        fits.ImageHDU(spectra, name='MODEL_SPECTRA'),
    ])
    hdul.writeto(output_filepath, overwrite=True)

MANQUI_ATMOSPHERES_DATA = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url='https://magao-x.org/docs/handbook/_static/ref/atm/magaox_manqui_atm.zip',
    converter_function=_convert_manqui_atmospheres,
    output_filename='magaox_cerro_manqui_atmosphere_grid.fits',
)
MANQUI_ATMOSPHERES_FITS = MANQUI_ATMOSPHERES_DATA.output_filepath
MANQUI_ATMOSPHERES = model_grids.ModelAtmosphereGrid(MANQUI_ATMOSPHERES_FITS, name="LCO Cerro Manqui")
