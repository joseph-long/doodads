import logging
import os.path
from astropy.io import fits
import astropy.units as u
from . import hst_calspec
from .. import utils
from ..modeling import units, spectra

log = logging.getLogger(__name__)

CURRENT_CALSPEC_URL = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/calspec/'

HST_CALSPEC_DIR = 'hst_calspec/'
HST_ALPHA_LYR_NAME = 'alpha_lyr_mod_004.fits'
HST_OLD_ALPHA_LYR_NAME = 'alpha_lyr_mod_002.fits'
HST_SIRIUS_NAME = 'sirius_mod_003.fits'
HST_SUN_NAME = 'sun_mod_001.fits'

ALPHA_LYR_FITS = utils.generated_path(HST_CALSPEC_DIR + HST_ALPHA_LYR_NAME.replace('.fits', '_converted.fits'))
VEGA = spectra.FITSSpectrum(hst_calspec.ALPHA_LYR_FITS, name='Vega')
OLD_ALPHA_LYR_FITS = utils.generated_path(HST_CALSPEC_DIR + HST_OLD_ALPHA_LYR_NAME.replace('.fits', '_converted.fits'))
OLD_VEGA = spectra.FITSSpectrum(hst_calspec.OLD_ALPHA_LYR_FITS, name='Vega (old)')
SIRIUS_FITS = utils.generated_path(HST_CALSPEC_DIR + HST_SIRIUS_NAME.replace('.fits', '_converted.fits'))
SUN_FITS = utils.generated_path(HST_CALSPEC_DIR + HST_SUN_NAME.replace('.fits', '_converted.fits'))

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
        wl, flux = hdul[1].data['WAVELENGTH'], hdul[1].data['FLUX']
        wl *= u.AA
        wl = wl.to(units.WAVELENGTH_UNITS)

        # From the header of sirius_stis_003.fits:
        #
        # HISTORY Units: Angstroms(A) and erg s-1 cm-2 A-1
        # HISTORY  All wavelengths are in vacuum.
        flux *= u.erg / u.s / u.cm ** 2 / u.AA
        flux = flux.to(units.FLUX_UNITS)
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='wavelength', format='E', array=wl),
        fits.Column(name='flux', format='E', array=flux),
    ])
    hdu.writeto(outpath, overwrite=overwrite)
    log.info(f"...saved to {outpath}.")

def _plot_standard(converted_fits, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    name = os.path.basename(converted_fits).replace('.fits', '')
    with open(converted_fits, 'rb') as f:
        table = fits.getdata(f)
        ax.plot(table['wavelength'], table['flux'], label=name)
    ax.set(
        xlabel=f'Wavelength [{units.WAVELENGTH_UNITS}]',
        ylabel=f'Flux [{units.FLUX_UNITS}]',
        xscale='log',
        yscale='log',
        title=name
    )
    return ax

def _plot_all_standards(all_standards):
    import matplotlib.pyplot as plt
    num_std = len(all_standards)
    fig, axes = plt.subplots(nrows=num_std, figsize=(6, 4 * num_std))
    for i in range(num_std):
        _plot_standard(all_standards[i], ax=axes[i])
    plt.tight_layout()
    fig.savefig(utils.generated_path('hst_calspec.png'))

def download_and_convert_hst_standards(overwrite=False):
    # download
    orig_vega = utils.download(CURRENT_CALSPEC_URL + HST_ALPHA_LYR_NAME, HST_CALSPEC_DIR + HST_ALPHA_LYR_NAME)
    orig_old_vega = utils.download(CURRENT_CALSPEC_URL + HST_OLD_ALPHA_LYR_NAME, HST_CALSPEC_DIR + HST_OLD_ALPHA_LYR_NAME)
    orig_sirius = utils.download(CURRENT_CALSPEC_URL + HST_SIRIUS_NAME, HST_CALSPEC_DIR + HST_SIRIUS_NAME)
    orig_sun = utils.download(CURRENT_CALSPEC_URL + HST_SUN_NAME, HST_CALSPEC_DIR + HST_SUN_NAME)
    # convert
    _convert_calspec(orig_vega, ALPHA_LYR_FITS, overwrite=overwrite)
    _convert_calspec(orig_old_vega, OLD_ALPHA_LYR_FITS, overwrite=overwrite)
    _convert_calspec(orig_sirius, SIRIUS_FITS, overwrite=overwrite)
    _convert_calspec(orig_sun, SUN_FITS, overwrite=overwrite)
    # save plot
    _plot_all_standards([ALPHA_LYR_FITS, OLD_ALPHA_LYR_FITS, SIRIUS_FITS, SUN_FITS])
