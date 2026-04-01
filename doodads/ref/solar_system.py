'''(Some) Solar System disk-integrated spectra for modeling

From https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/AA_README_SOLSYS

The spectra here for Jupiter, Saturn, Unanus, and Neptune have wavelength
ranges from the visible to mid-infrared, typically 0.53 to 28.75 microns,
while for the solar spectrum, the wavelenght range is 0.2 to 30 microns.

Because the NIR spectra of the planets are from observations, and the MIR
spectra are largely models, the two wavelength regions do not match up neatly
in the ~5 micron range. Additionally, there are no available spectra that
adequately represent the region from ~4 to 8 microns.

To piece the NIR and MIR spectra together, a Gaussian was extended from the
shortest wavelengths of the MIR spectrum until it met up with the NIR spectrum.
Random Gaussian noise was added to mimic the behavior at the shortest MIR
wavelengths. The NIR wavelengths longward of the meet-up point were discarded
in favor of the extended MIR data.

The wavelengths of the planetary spectra are in Angstroms, while the fluxes are
tabulated in units of erg/s/cm^2/A/sq. arcsec. The spectra are in surface
brightness units, which are F_lambda units divided by square arcseconds.
This is a more appropriate choice than disk-integrated flux because it enables
a source of any size to be specified in the ETC with the appropriate flux,
as long as the normalization is in units of surface brightness.

The empirical solar spectrum uses observations from Thuillier et al., (2003)
from 0.2-2.4 microns. Longer wavelengths are represented by a local thermal
equilibrium (LTE) model from Holweger & Muller (1974). The region from 4-6.5
microns was modified to more accurately represent the CO absorption features,
bringing the spectrum into agreement with the observations of
Wallace & Livingston (2003). This spectrum was interpolated onto a regular
wavelength grid with approximately the same mean spacing as the original.
Normalized to 1 au heliocentric distance. The wavelengths are in Angstroms, and
the fluxes were tabulated in units of erg/s/cm^2/A.

References:
Planets:
Vis-NIR: Clark & McCord (1979)
Vis-1 micron geometric albedo: Karkoschka (1994)
2.5-5 micron from ISO, cross-check: Encrenaz et al. (1997)
MIR, Cassini/CIRS: Fletcher et al. (2009)
MIR models for wavelengths <6.5 microns and >17 microns were privided by L. Fletcher (private communciation).

Solar:
Rieke et al., 2008, AJ 135, 2245

==================================+=========================================+
Filename                          |          Wavelength Range (microns)     |
----------------------------------+-----------------------------------------|
jupiter_solsys_surfbright_001.fits|                 0.53 - 28.75            |
----------------------------------+-----------------------------------------|
saturn_solsys_surfbright_001.fits |                 0.53 - 28.75            |
----------------------------------+-----------------------------------------|
uranus_solsys_surfbright_001.fits |                 0.54 - 28.75            |
----------------------------------+-----------------------------------------|
neptune_solsys_surfbright_001.fits|                 0.54 - 28.75            |
----------------------------------+-----------------------------------------|
solar_spec.fits                   |                 0.2  - 30               |
----------------------------------+-----------------------------------------|
'''
import logging
import os

import astropy.units as u
from astropy.io import fits

from .. import utils
from ..modeling import spectra, units

log = logging.getLogger(__name__)

BASE_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/"

JUPITER_FILENAME = "jupiter_solsys_surfbright_001.fits"
SATURN_FILENAME = "saturn_solsys_surfbright_001.fits"
URANUS_FILENAME = "uranus_solsys_surfbright_001.fits"
NEPTUNE_FILENAME = "neptune_solsys_surfbright_001.fits"
# note that solar spectrum exists as `doodads.ref.hst_calspec.SUN` or `dd.SUN` for short.

ORIG_VALUE_UNIT = u.erg / u.s / u.cm**2 / u.Angstrom / u.arcsec**2
ORIG_WL_UNIT = u.Angstrom

JUPITER_SURFBRIGHT_FILE = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url=BASE_URL + JUPITER_FILENAME,
    converter_function=os.symlink,
)

JUPITER_SURFBRIGHT = spectra.FITSSpectrum(
    JUPITER_SURFBRIGHT_FILE.output_filepath,
    name="Jupiter surface brightness (meas., composite)",
    wavelength_units=ORIG_WL_UNIT,
    value_units=ORIG_VALUE_UNIT,
)

def _plot_all_standards(all_standards):
    fig = spectra.plot_spectra(all_standards)
    outpath = utils.generated_path('cdbs_solsys.png')
    fig.savefig(outpath)
    log.info(f'Saved plot of HST CALSPEC standards to {outpath}')

JUPITER_PERIHELION = 4.95 * u.au

# def get_jupiter_like_spectrum(radius=1 * u.Rjup, distance=JUPITER_PERIHELION - 1 * u.au):
    
#     4 * np.pi**2 
#     JUPITER_SURFBRIGHT.multiply()

utils.DIAGNOSTICS.add(_plot_all_standards, [JUPITER_SURFBRIGHT])
