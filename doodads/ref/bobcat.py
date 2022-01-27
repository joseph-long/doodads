from collections import defaultdict
import gzip
import re
from functools import partial
import tarfile
import logging
import numpy as np
from astropy.io import fits
import astropy.units as u
from scipy import interpolate

from ..modeling.units import WAVELENGTH_UNITS, FLUX_UNITS, FLUX_PER_FREQUENCY_UNITS
from ..modeling.physics import f_nu_to_f_lambda
from . import model_grids
from .. import utils

__all__ = [
    'BOBCAT_EVOLUTION_AGE_COLS',
    'BOBCAT_PHOTOMETRY_COLS',
    'read_bobcat',
    'bobcat_mass_age_to_temp_grav',
    'bobcat_mass_age_to_mag',
    'load_bobcat_evolution_age',
    'load_bobcat_photometry',
    'BOBCAT_SPECTRA_M0',
]

FLOAT_PART = r'([+\-]?[\d.E+]+)\*?'

FLOAT_RE = re.compile(FLOAT_PART)

log = logging.getLogger(__name__)

BOBCAT_EVOLUTION_AGE_COLS = [
    'age_Gyr', 'mass_M_sun', 'log_L_L_sun', 'T_eff_K', 'log_g_cm_per_s2', 'radius_R_sun']

BOBCAT_PHOTOMETRY_COLS = [
    'T_eff_K',
    'log_g_cm_per_s2',
    'mass_M_jup',
    'radius_R_sun',
    'helium_frac_Y',
    'log_Kzz',
    'mag_MKO_Y',
    'mag_MKO_Z',
    'mag_MKO_J',
    'mag_MKO_H',
    'mag_MKO_K',
    'mag_MKO_Lprime',
    'mag_MKO_Mprime',
    'mag_2MASS_J',
    'mag_2MASS_H',
    'mag_2MASS_Ks',
    'mag_Keck_Ks',
    'mag_Keck_Lprime',
    'mag_Keck_Ms',
    'mag_SDSS_gprime',
    'mag_SDSS_rprime',
    'mag_SDSS_iprime',
    'mag_SDSS_zprime',
    'mag_IRAC_3_6_um',
    'mag_IRAC_4_5_um',
    'mag_IRAC_5_7_um',
    'mag_IRAC_7_9_um',
    'mag_WISE_W1',
    'mag_WISE_W2',
    'mag_WISE_W3',
    'mag_WISE_W4',
]


def read_bobcat(fh, colnames, first_header_line_contains):
    cols = defaultdict(list)
    line = next(fh)
    if isinstance(line, bytes):
        decode = lambda x: x.decode('utf8')
        line = decode(line)
    else:
        decode = lambda x: x
    while first_header_line_contains not in line:
        line = decode(next(fh))
    rows = 0
    for line in fh:
        line = decode(line)
        parts = line.split()
        if len(parts) != len(colnames):
            log.debug(f"Line column number mismatch: got {len(parts)=} and expected {len(colnames)=}\nLine was {line=}")
            continue
        for idx, col in enumerate(colnames):
            try:
                val = FLOAT_RE.findall(parts[idx])[0]
                val = float(val)
            except (ValueError, IndexError):
                val = np.nan
                print(f'{parts[idx]=} -> NaN')
            cols[col].append(val)
        rows += 1
    tbl = np.zeros((rows,), dtype=[(name, '=f4') for name in colnames])
    for name in colnames:
        tbl[name] = cols[name]
    return tbl

def bobcat_mass_age_to_temp_grav(evol_tbl, eq_temp=None):
    mass_age_points = np.stack([evol_tbl['mass_M_sun'], evol_tbl['age_Gyr']], axis=-1)
    if eq_temp is not None:
        T_eff_K = np.power(evol_tbl['T_eff_K']**4 + eq_temp.to(u.K).value**4, 1/4)
    else:
        T_eff_K = evol_tbl['T_eff_K']
    mass_age_vals = np.stack([T_eff_K, evol_tbl['log_g_cm_per_s2']], axis=-1)
    interp = interpolate.LinearNDInterpolator(mass_age_points, mass_age_vals)
    def mass_age_to_temp_grav(mass, age):
        mass_vals = mass.to(u.M_sun).value
        age_val = age.to(u.Gyr).value
        if np.isscalar(age_val):
            age_val = np.repeat(age_val, len(mass_vals))
        interp_at_pts = np.stack([mass_vals, age_val], axis=-1)
        temps, gravs = interp(interp_at_pts).T
        return temps * u.K, gravs
    return mass_age_to_temp_grav

def bobcat_temp_grav_to_mag(phot_tbl, mag_col):
    temp_grav_points = np.stack([phot_tbl['T_eff_K'], phot_tbl['log_g_cm_per_s2']], axis=-1)
    temp_grav_vals = phot_tbl[mag_col]
    interp = interpolate.LinearNDInterpolator(temp_grav_points, temp_grav_vals)
    def temp_grav_to_mag(temps, gravs):
        temps_K = temps.to(u.K).value
        return interp(np.stack([temps_K, gravs], axis=-1))
    return temp_grav_to_mag

def bobcat_mass_age_to_mag(evol_tbl, phot_tbl, mag_col='mag_MKO_Lprime', eq_temp=0*u.K):
    mass_age_to_temp_grav = bobcat_mass_age_to_temp_grav(evol_tbl, eq_temp)
    temp_grav_to_mag = bobcat_temp_grav_to_mag(phot_tbl, mag_col)
    def interpolator(mass, age):
        '''Take mass and age as unitful quantities,
        return in-bounds masses from the original input
        and magnitudes for them
        '''
        mass_vals = mass.to(u.M_sun)
        ages = np.repeat(age, len(mass_vals))
        temps, gravs = mass_age_to_temp_grav(mass, ages)
        oob_temp_grav = np.isnan(temps)
        mags = temp_grav_to_mag(temps[~oob_temp_grav], gravs[~oob_temp_grav])
        oob_mags = np.isnan(mags)
        final_masses = mass[~oob_temp_grav][~oob_mags].to(u.M_jup)
        final_mags = mags[~oob_mags]
        assert final_masses.shape[0] == final_mags.shape[0]
        return final_masses, final_mags
    return interpolator


def _mag_to_mass(masses, mags, take_smaller=True):
    '''Given a set of masses and corresponding magnitudes, return the
    minimum mass reachable at each magnitude—even if the mass to
    magnitude relationship is not monotonic. Ranges of masses where
    lower masses are attainable at the same or brighter limiting
    magnitude are returned.

    Parameters
    ----------
    masses : array
    mags : array
    take_smaller : bool
        Whether to pick the lower of two masses where the mag
        to mass relationship is double-valued

    Returns
    -------
    mags : array
        magnitudes in order from brightest (smallest)
        to dimmest (largest)
    masses : array
        minimum masses reachable corresponding to `mags`
    excluded_ranges : list[tuple[float,float,float]]
        list of ``(begin, end, limiting mag)`` tuples defining the
        ranges where the inverse relationship was double-valued
        and the smaller mass was chosen for `masses` (empty
        if `take_smaller` is False). If the ultimate limiting magnitude
        less than `limiting_mag` for a particular mass interval, it's
        safe to say that range is included (and would have been
        detected)
    '''

    masses = masses[np.isfinite(mags)]
    mags = mags[np.isfinite(mags)]
    n_points = mags.shape[0]
    min_masses = np.zeros_like(masses)
    mask_good = np.ones(n_points, dtype=bool)
    mag_order = np.argsort(mags)
    mags, masses = mags[mag_order], masses[mag_order]
    excluded_masses = []
    if not take_smaller:
        return mags, masses, []

    for idx in range(n_points):
        mask = mags <= mags[idx]
        min_mass_at_mag = np.min(masses[mask])
        if min_mass_at_mag != masses[idx]:
            excluded_masses.append(masses[idx])
            mask_good[idx] = False
        else:
            min_masses[idx] = min_mass_at_mag
            mask_good[idx] = True

    # now turn it around and process the excluded ranges
    mass_order = np.argsort(masses)
    masses_mass_order = masses[mass_order]
    mags_mass_order = mags[mass_order]
    excluded_ranges = []
    range_start = None
    for idx in range(n_points):
        if masses_mass_order[idx] in excluded_masses and range_start is None:
            range_start = masses_mass_order[idx-1]
        elif masses_mass_order[idx] not in excluded_masses and range_start is not None:
            excluded_ranges.append((range_start, masses_mass_order[idx]))
            range_start = None

    # need to say where the excluded range bottoms out (largest mag);
    # if final mag limit is below that value then
    # we are actually sensitive to the excluded range
    excluded_ranges_and_limits = []
    for begin_range, end_range in excluded_ranges:
        excluded_ranges_and_limits.append((
            begin_range,
            end_range,
            np.max(mags_mass_order[(masses_mass_order > begin_range) & (masses_mass_order < end_range)])
        ))

    good_min_masses, good_mags = min_masses[mask_good], mags[mask_good]
    return good_mags, good_min_masses, excluded_ranges_and_limits

MIN_SENTINEL = -np.inf
MAX_SENTINEL = np.inf

def bobcat_mag_to_mass(evol_tbl, mass_age_to_mag, age, take_smaller=True, masses=None):
    if masses is None:
        masses = np.unique(evol_tbl['mass_M_sun'] * u.M_sun)
    masses, mags = mass_age_to_mag(masses, age)
    mags, min_masses, excluded_mass_ranges = _mag_to_mass(masses, mags, take_smaller=take_smaller)
    min_mag, max_mag = np.min(mags), np.max(mags)
    log.debug(f"Initializing interpolator for magnitude values in [{min_mag}, {max_mag}]")
    interpolator = interpolate.interp1d(mags, min_masses, fill_value=(MIN_SENTINEL, MAX_SENTINEL), bounds_error=False)
    min_mag_mass_mjup, max_mag_mass_mjup = interpolator(min_mag), interpolator(max_mag)
    log.debug(f"Interpolator covering mass ranges [{min_mag_mass_mjup} M_jup, {max_mag_mass_mjup} M_jup]")
    def mag_to_mass(mag):
        '''
        Returns
        -------
        mass
        too_bright
            whether value was replaced with mag_to_mass(MIN_MAG)
            because it would have gone out of bounds in the low
            (brighter) direction
        too_faint
            whether value was replaced with mag_to_mass(MAX_MAG)
            because it would have gone out of bounds in the high
            (dimmer) direction
        '''
        mass = interpolator(mag)
        too_bright = too_faint = False
        if mass == MIN_SENTINEL:
            too_bright = True
            mass = min_mag_mass_mjup
        if mass == MAX_SENTINEL:
            too_faint = True
            mass = max_mag_mass_mjup
        return mass * u.M_jup, too_bright, too_faint
    return mag_to_mass, excluded_mass_ranges

from .. import utils

BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA = utils.REMOTE_RESOURCES.add(
    module=__name__,
    url='https://zenodo.org/record/5063476/files/evolution_and_photometery.tar.gz?download=1',
    output_filename='bobcat_2021_evolution_and_photometry.tar.gz',
)
BOBCAT_2021_EVOLUTION_PHOTOMETRY_ARCHIVE = BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA.output_filepath

def _load_from_resource(columns, first_header_line_contains, path_in_archive):
    archive_tarfile = tarfile.open(BOBCAT_2021_EVOLUTION_PHOTOMETRY_ARCHIVE)
    with archive_tarfile.extractfile(path_in_archive) as fh:
        tbl = read_bobcat(fh, columns, first_header_line_contains)
    return tbl

load_bobcat_evolution_age = partial(_load_from_resource, BOBCAT_EVOLUTION_AGE_COLS, 'age(Gyr)')
load_bobcat_photometry = partial(_load_from_resource, BOBCAT_PHOTOMETRY_COLS, 'MKO')


SPECTRA_PARAMS_COLS = ['T_eff_K', 'gravity_m_per_s2', 'Y', 'f_rain', 'Kzz', 'Fe_over_H', 'C_over_O', 'f_hole']

class InconsistentSamplingException(Exception):
    pass

def load_bobcat_spectrum(fh,
                         source_wavelength_unit=u.um, source_flux_per_frequency_unit=(u.erg / u.cm**2 / u.s / u.Hz),
                         wavelengths=None, wavelength_order_sorter=None):
    header = next(fh)
    (
        T_eff_K, gravity_m_per_s2, Y, f_rain, Kzz, Fe_over_H, C_over_O, f_hole
    ) = tuple(map(float, FLOAT_RE.findall(header)))

    params = {}
    params['T_eff_K'] = T_eff_K
    params['gravity_m_per_s2'] = gravity_m_per_s2
    params['Y'] = Y
    params['f_rain'] = f_rain
    params['Kzz'] = Kzz
    params['Fe_over_H'] = Fe_over_H
    params['C_over_O'] = C_over_O
    params['f_hole'] = f_hole
    _ = next(fh)
    these_wavelengths, these_fluxes = np.genfromtxt(fh, unpack=True)
    if wavelength_order_sorter is None:
        wavelength_order_sorter = np.argsort(these_wavelengths)
    these_wavelengths = (these_wavelengths[wavelength_order_sorter] * source_wavelength_unit).to(WAVELENGTH_UNITS)
    if wavelengths is not None and not np.all(these_wavelengths == wavelengths):
        raise InconsistentSamplingException(f"Inconsistent wavelength sampling")
    elif wavelengths is None:
        wavelengths = these_wavelengths
    these_fluxes = these_fluxes[wavelength_order_sorter] * source_flux_per_frequency_unit
    fluxes = f_nu_to_f_lambda(these_fluxes, wavelengths)
    return params, wavelengths, fluxes, wavelength_order_sorter

BOBCAT_SPECTRA_FILENAMES = re.compile(r'spectra/sp_(.+)\.gz')
def _convert_spectra(tarfile_filepath, output_filepath, match_pattern=BOBCAT_SPECTRA_FILENAMES):
    archive_tarfile = tarfile.open(tarfile_filepath)
    spectra_paths = [name for name in archive_tarfile.getnames() if match_pattern.match(name)]
    params_tbl = np.zeros(
        len(spectra_paths),
        dtype=list((param, float) for param in SPECTRA_PARAMS_COLS)
    )
    wavelengths, wavelength_order_sorter = None, None
    spectra = None
    for idx, name in enumerate(spectra_paths):
        with gzip.open(archive_tarfile.extractfile(name), mode='rt', encoding='utf8') as fh:
            try:
                params, wavelengths, f_lambda, wavelength_order_sorter = load_bobcat_spectrum(
                    fh,
                    wavelengths=wavelengths,
                    wavelength_order_sorter=wavelength_order_sorter
                )
            except InconsistentSamplingException:
                raise InconsistentSamplingException(f"Inconsistent wavelength sampling in {name}")
        if spectra is None:
            # have to create the empty array after we know how long f_lambda is
            spectra : u.Quantity = np.zeros((len(spectra_paths), len(f_lambda))) * FLUX_UNITS
        spectra[idx] = f_lambda
        row = params_tbl[idx]
        for colname in SPECTRA_PARAMS_COLS:
            row[colname] = params[colname]
    with open(output_filepath, 'wb') as fh:
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.BinTableHDU(data=params_tbl, name='PARAMS'),
            fits.ImageHDU(data=wavelengths.to(WAVELENGTH_UNITS).value, name='WAVELENGTHS'),
            fits.ImageHDU(data=spectra.to(FLUX_UNITS).value, name='MODEL_SPECTRA'),
        ])
        hdul.writeto(fh)

BOBCAT_2021_SPECTRA_M0_DATA = utils.REMOTE_RESOURCES.add(
    module=__name__,
    url='https://zenodo.org/record/5063476/files/spectra_m%2B0.0.tar.gz?download=1',
    converter_function=_convert_spectra,
    output_filename='bobcat_2021_spectra_m0.fits',
)
BOBCAT_2021_SPECTRA_M0_FITS = BOBCAT_2021_SPECTRA_M0_DATA.output_filepath

BOBCAT_SPECTRA_M0 = (
    model_grids.ModelSpectraGrid(BOBCAT_2021_SPECTRA_M0_FITS)
    if BOBCAT_2021_SPECTRA_M0_DATA.exists
    else utils.YellingProxy("Use ddx get_reference_data to download Bobcat spectra")
)
