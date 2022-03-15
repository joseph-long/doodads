import logging
import os.path
from astropy.io import fits
import astropy.units as u
from .. import utils
from ..modeling import units, spectra

log = logging.getLogger(__name__)

__all__ = [
    'VEGA',
    'VEGA_BOHLIN_GILLILAND_2004',
    'SIRIUS',
    'SUN',
]

def _convert_calspec(orig_fits, outpath, overwrite=False):
    '''Convert CALSPEC spectra from Angstroms and
    erg / s / cm^2 / Angstrom to meters and W / m^3
    '''
    if not overwrite and os.path.exists(outpath):
        log.info(f"Found existing output: {outpath}")
        return outpath
    with open(orig_fits, 'rb') as f:
        log.info(f"Converting {orig_fits}...")
        hdul = fits.open(f)
        wl, flux = hdul[1].data['WAVELENGTH'].astype('=f8'), hdul[1].data['FLUX'].astype('=f8')
        wl *= u.AA
        wl = wl.to(units.WAVELENGTH_UNITS)

        # From the header of sirius_stis_003.fits:
        #
        # HISTORY Units: Angstroms(A) and erg s-1 cm-2 A-1
        # HISTORY  All wavelengths are in vacuum.
        flux *= u.erg / u.s / u.cm ** 2 / u.AA
        flux = flux.to(units.FLUX_UNITS)
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='wavelength', format='D', array=wl),
        fits.Column(name='flux', format='D', array=flux),
    ])
    hdu.writeto(outpath, overwrite=overwrite)
    log.info(f"...saved to {outpath}.")

def _plot_all_standards(all_standards):
    import matplotlib.pyplot as plt
    plt.clf()
    num_std = len(all_standards)
    fig, axes = plt.subplots(nrows=num_std, figsize=(6, 4 * num_std))
    for i in range(num_std):
        all_standards[i].display(ax=axes[i])
        axes[i].set(
            xscale='log',
            yscale='log',
        )
    plt.tight_layout()
    outpath = utils.generated_path('hst_calspec.png')
    fig.savefig(outpath)
    log.info(f'Saved plot of HST CALSPEC standards to {outpath}')

CURRENT_CALSPEC_URL = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/'

HST_ALPHA_LYR_NAME = 'alpha_lyr_mod_004.fits'
HST_SIRIUS_NAME = 'sirius_mod_003.fits'
HST_SUN_NAME = 'sun_mod_001.fits'

VEGA_BOHLIN_GILLILAND_2004_CALSPEC = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url=CURRENT_CALSPEC_URL + 'alpha_lyr_stis_002.fits',
    converter_function=_convert_calspec,
)
VEGA_BOHLIN_GILLILAND_2004 = spectra.FITSSpectrum(
    VEGA_BOHLIN_GILLILAND_2004_CALSPEC.output_filepath,
    name='Vega (Bohlin & Gilliland 2004)'
)

ALPHA_LYR_CALSPEC = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url=CURRENT_CALSPEC_URL + HST_ALPHA_LYR_NAME,
    converter_function=_convert_calspec,
)
VEGA = spectra.FITSSpectrum(ALPHA_LYR_CALSPEC.output_filepath, name='Vega (Bohlin, Gordon, & Tremblay 2014)')

SIRIUS_CALSPEC = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url=CURRENT_CALSPEC_URL + HST_SIRIUS_NAME,
    converter_function=_convert_calspec,
)
SIRIUS = spectra.FITSSpectrum(SIRIUS_CALSPEC.output_filepath, name='Sirius')

SUN_CALSPEC = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url=CURRENT_CALSPEC_URL + HST_SUN_NAME,
    converter_function=_convert_calspec,
)
SUN = spectra.FITSSpectrum(SUN_CALSPEC.output_filepath, name='Sun')


utils.DIAGNOSTICS.add(_plot_all_standards, [VEGA, SIRIUS, SUN])
