from collections import defaultdict
from genericpath import exists
import gzip
import re
from functools import partial
import tarfile
import logging
import typing
import numpy as np
from astropy.io import fits
import astropy.units as u
import warnings
from scipy import interpolate
from numpy.lib.recfunctions import append_fields

from .hst_calspec import VEGA_BOHLIN_GILLILAND_2004
from ..modeling.units import WAVELENGTH_UNITS, FLUX_UNITS, FLUX_PER_FREQUENCY_UNITS
from ..modeling.physics import f_nu_to_f_lambda
from ..modeling import spectra
from . import model_grids
from .. import utils

__all__ = [
    'BOBCAT_EVOLUTION_AGE_COLS',
    'BOBCAT_PHOTOMETRY_COLS',
    'read_bobcat',
    'bobcat_mass_age_to_temp_grav',
    'bobcat_mass_age_to_mag',
    'load_bobcat_evolution_age',
    'load_bobcat_evolution_mass',
    'load_bobcat_photometry',
    'BOBCAT_SPECTRA_M0',
    'BOBCAT_EVOLUTION_M0',
    'BOBCAT_EVOLUTION_TABLES_M0',
]

BOBCAT_LSUN_REFERENCE = 10**33.5827 * u.erg / u.s

FLOAT_PART = r'([+\-]?[\d.E+]+)\*?'

FLOAT_RE = re.compile(FLOAT_PART)

log = logging.getLogger(__name__)

BOBCAT_EVOLUTION_AGE_COLS = [
    'age_Gyr',
    'mass_Msun',
    'log_L_Lsun',
    'T_eff_K',
    'log_g_cm_per_s2',
    'radius_Rsun'
]

BOBCAT_EVOLUTION_MASS_COLS = [
    'mass_Msun',
    'age_Gyr',
    'log_L_Lsun',
    'T_eff_K',
    'log_g_cm_per_s2',
    'radius_Rsun',
    'log_I_g_cm2',
]

BOBCAT_EVOLUTION_LBOL_COLS = [
    'log_L_Lsun',
    'mass_Msun',
    'age_Gyr',
    'T_eff_K',
    'log_g_cm_per_s2',
    'radius_Rsun',
]

BOBCAT_EVOLUTION_MASS_AGE_COLS = [
    'T_eff_K',
    'log_g_cm_per_s2',
    'mass_Msun',
    'radius_Rsun',
    'log_L_Lsun',
    'log_age_yr',
]

BOBCAT_PHOTOMETRY_COLS = [
    'T_eff_K',
    'log_g_cm_per_s2',
    'mass_Mjup',
    'radius_Rsun',
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
            cols[col].append(val)
        rows += 1
    tbl = np.zeros((rows,), dtype=[(name, '=f4') for name in colnames])
    for name in colnames:
        tbl[name] = cols[name]
    return tbl

def bobcat_mass_age_to_temp_grav(evol_tbl, eq_temp=None):
    mass_age_points = np.stack([evol_tbl['mass_Msun'], evol_tbl['age_Gyr']], axis=-1)
    if eq_temp is not None:
        T_eff_K = np.power(evol_tbl['T_eff_K']**4 + eq_temp.to(u.K).value**4, 1/4)
    else:
        T_eff_K = evol_tbl['T_eff_K']
    mass_age_vals = np.stack([T_eff_K, evol_tbl['log_g_cm_per_s2']], axis=-1)
    interp = interpolate.LinearNDInterpolator(mass_age_points, mass_age_vals)
    def mass_age_to_temp_grav(mass, age):
        mass_vals = mass.to(u.Msun).value
        age_vals = age.to(u.Gyr).value
        age_scalar = np.isscalar(age_vals)
        mass_scalar = np.isscalar(mass_vals)
        if mass_scalar:
            if not age_scalar:
                repeat = len(age_vals)
            else:
                repeat = 1
            mass_vals = np.repeat(mass_vals, repeat)
        if age_scalar:
            age_vals = np.repeat(age_vals, len(mass_vals))
        interp_at_pts = np.stack([mass_vals, age_vals], axis=-1)
        temps, gravs = interp(interp_at_pts).T
        gravs = 10**gravs * u.cm / u.s**2
        return temps * u.K, gravs
    return mass_age_to_temp_grav

def bobcat_mass_age_to_temp_grav2(evol_tbl, eq_temp=None):
    mass_age_points = np.stack([evol_tbl['mass_Msun'], evol_tbl['age_Gyr']], axis=-1)
    if eq_temp is not None:
        T_eff_K = np.power(evol_tbl['T_eff_K']**4 + eq_temp.to(u.K).value**4, 1/4)
    else:
        T_eff_K = evol_tbl['T_eff_K']
    temperature_interp = interpolate.LinearNDInterpolator(mass_age_points, T_eff_K)
    unique_masses = np.unique(evol_tbl['mass_Msun'])
    log_gs = np.zeros(len(evol_tbl))
    for mass_Msun in unique_masses:
        selector = evol_tbl['mass_Msun'] == mass_Msun
        interp = interpolate.interp1d(evol_tbl[selector]['T_eff_K'], evol_tbl[selector]['log_g_cm_per_s2'], fill_value='extrapolate')
        log_gs[selector] = interp(T_eff_K[selector])

    log_g_interp = interpolate.LinearNDInterpolator(
        np.stack([evol_tbl['mass_Msun'], evol_tbl['age_Gyr']], axis=-1),
        log_gs
    )
    def mass_age_to_temp_grav(mass, age):
        mass_vals = mass.to(u.Msun).value
        age_vals = age.to(u.Gyr).value
        age_scalar = np.isscalar(age_vals)
        mass_scalar = np.isscalar(mass_vals)
        if mass_scalar:
            if not age_scalar:
                repeat = len(age_vals)
            else:
                repeat = 1
            mass_vals = np.repeat(mass_vals, repeat)
        if age_scalar:
            age_vals = np.repeat(age_vals, len(mass_vals))
        interp_at_pts = np.stack([mass_vals, age_vals], axis=-1)
        temps_K = temperature_interp(interp_at_pts)
        gravs = log_g_interp(mass_vals, age_vals)
        temps = temps_K * u.K
        gravs = 10**gravs * u.cm / u.s**2
        if age_scalar and mass_scalar:
            return temps[0], gravs[0]
        return temps, gravs
    return mass_age_to_temp_grav


def bobcat_temp_grav_to_mag(phot_tbl, filter_name):
    temp_grav_points = np.stack([phot_tbl['T_eff_K'], phot_tbl['log_g_cm_per_s2'], phot_tbl['mass_Msun']], axis=-1)
    corresponding_mags = phot_tbl[filter_name]
    interp = interpolate.LinearNDInterpolator(temp_grav_points, corresponding_mags)
    def temp_grav_to_mag(temps, gravs):
        temps_K = temps.to(u.K).value
        gravs_cm_per_s2 = gravs.to(u.cm/u.s**2).value
        log_gs = np.log10(gravs_cm_per_s2)
        return interp(np.stack([temps_K, log_gs], axis=-1))
    return temp_grav_to_mag

def bobcat_mass_age_to_mag(phot_tbl, mass_age_to_temp_grav, filter_name):
    temp_grav_to_mag = bobcat_temp_grav_to_mag(phot_tbl, filter_name)
    def interpolator(mass, age):
        '''Take mass and age as unitful quantities,
        return in-bounds masses from the original input
        and magnitudes for them
        '''
        mass_vals = mass.to(u.Msun)
        ages = np.repeat(age, len(mass_vals))
        temps, gravs = mass_age_to_temp_grav(mass, ages)
        oob_temp_grav = np.isnan(temps)
        mags = temp_grav_to_mag(temps[~oob_temp_grav], gravs[~oob_temp_grav])
        oob_mags = np.isnan(mags)
        final_masses = mass[~oob_temp_grav][~oob_mags].to(u.Mjup)
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

def bobcat_mag_to_mass(tabulated_masses, mass_age_to_mag, age, take_smaller=True):
    masses, mags = mass_age_to_mag(tabulated_masses, age)
    mags, min_masses, excluded_mass_ranges = _mag_to_mass(masses, mags, take_smaller=take_smaller)
    min_mag, max_mag = np.min(mags), np.max(mags)
    log.debug(f"Initializing interpolator for magnitude values in [{min_mag}, {max_mag}]")
    interpolator = interpolate.interp1d(mags, min_masses, fill_value=(MIN_SENTINEL, MAX_SENTINEL), bounds_error=False)
    min_mag_mass_mjup, max_mag_mass_mjup = interpolator(min_mag), interpolator(max_mag)
    log.debug(f"Interpolator covering mass ranges [{min_mag_mass_mjup} Mjup, {max_mag_mass_mjup} Mjup]")
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
        if np.isscalar(mass):
            mass = np.repeat(mass, 1)
        too_bright = np.zeros(len(mass), dtype=bool)
        too_faint = np.zeros(len(mass), dtype=bool)
        too_bright[mass == MIN_SENTINEL] = True
        too_faint[mass == MAX_SENTINEL] = True
        mass[too_faint] = max_mag_mass_mjup
        mass[too_bright] = min_mag_mass_mjup
        return mass * u.Mjup, too_bright, too_faint
    return mag_to_mass, excluded_mass_ranges

def bobcat_add_mags_from_filter(phot_tbl, filter_spectrum):
    mags = np.zeros(len(phot_tbl))
    for idx, row in enumerate(phot_tbl):
        T_eff_K = row['T_eff_K']
        mass = row['mass_Mjup'] * u.Mjup
        log_g_cm_per_s2 = row['log_g_cm_per_s2']
        surface_gravity = 10**log_g_cm_per_s2 * u.cm/u.s**2
        try:
            model_spec = BOBCAT_SPECTRA_M0.get(
                temperature=T_eff_K*u.K,
                surface_gravity=surface_gravity,
                mass=mass,
            )
            mags[idx] = VEGA_BOHLIN_GILLILAND_2004.magnitude(model_spec, filter_spectrum=filter_spectrum)
        except model_grids.BoundsError:
            mags[idx] = np.nan
    mask = ~np.isnan(mags)
    if len(phot_tbl[mask]) < 0.9*len(phot_tbl):
        raise ValueError("Out of bounds points in photometry table are out of control")
    phot_tbl = phot_tbl[mask]
    mags = mags[mask]
    phot_tbl = append_fields(phot_tbl, filter_spectrum.name, mags)
    return phot_tbl

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
load_bobcat_evolution_mass = partial(_load_from_resource, BOBCAT_EVOLUTION_MASS_COLS, 'age(Gyr)')
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

class BobcatModelSpectraGrid(model_grids.ModelSpectraGrid):
    fractional_param_err = 0.001
    @property
    def bounds(self):
        out = {}
        for name in self._real_param_names:
            out[name] = np.min(self.params[name]), np.max(self.params[name])
        # hack: single outlier 2401 K point messing everything up, pretend
        # it doesn't exist
        if out['T_eff_K'][1] == 2401:
            out['T_eff_K'] = out['T_eff_K'][0], 2400
        return out
    def _args_to_params(self, temperature, surface_gravity, extra_args):
        # There's some quirks in the data (using cm/s^2 here, m/s^2 in the
        # spectra archive) that mean we should replace barely out-of-bounds
        # values with the boundary values
        min_T_K, max_T_K = self.bounds['T_eff_K']
        T_K = temperature.to(u.K).value
        if T_K < min_T_K and np.abs((T_K - min_T_K)/min_T_K) < self.fractional_param_err:
            temperature = min_T_K * u.K
        if T_K > max_T_K and np.abs((T_K - max_T_K)/max_T_K) < self.fractional_param_err:
            temperature = max_T_K * u.K
        min_g_m_per_s2, max_g_m_per_s2 = self.bounds['gravity_m_per_s2']
        g_m_per_s2 = surface_gravity.to(u.m/u.s**2).value
        if g_m_per_s2 < min_g_m_per_s2 and np.abs((g_m_per_s2 - min_g_m_per_s2)/min_g_m_per_s2) < self.fractional_param_err:
            surface_gravity = min_g_m_per_s2 * u.m / u.s**2
        if g_m_per_s2 > max_g_m_per_s2 and np.abs((g_m_per_s2 - max_g_m_per_s2)/max_g_m_per_s2) < self.fractional_param_err:
            surface_gravity = max_g_m_per_s2 * u.m / u.s**2
        return super()._args_to_params(temperature, surface_gravity, extra_args)

BOBCAT_SPECTRA_M0 = BobcatModelSpectraGrid(BOBCAT_2021_SPECTRA_M0_FITS)

class BobcatEvolutionTables(utils.LazyLoadable):
    age : np.ndarray
    lbol : np.ndarray
    mass : np.ndarray
    mass_age : np.ndarray
    _metallicity_to_string = {
        -0.5: '-0.5',
        0.0: '+0.0',
        0.5: '+0.5',
    }
    _table_mapping = [
        # attrname, columns, first_header_line_contains
        ('age', BOBCAT_EVOLUTION_AGE_COLS, 'age(Gyr)'),
        ('mass', BOBCAT_EVOLUTION_MASS_COLS, 'age(Gyr)'),
        ('lbol', BOBCAT_EVOLUTION_LBOL_COLS, 'age(Gyr)'),
        ('mass_age', BOBCAT_EVOLUTION_MASS_AGE_COLS, '(yr)'),
    ]
    def __init__(self, *args, metallicity=0.0, **kwargs):
        self.metallicity = metallicity
        super().__init__(*args, **kwargs)
    def _lazy_load(self):
        self._archive_tarfile = tarfile.open(self.filepath)
        if not np.any(np.isclose(self.metallicity, [k for k in self._metallicity_to_string])):
            raise ValueError(f"Available metallicities are {[v for v in self._metallicity_to_string.values()]}")
        for attrname, columns, first_header_line_contains in self._table_mapping:
            metallicity_str = self._metallicity_to_string[self.metallicity]
            path_in_archive = f'evolution_tables/evo_tables{metallicity_str}/nc{metallicity_str}_co1.0_{attrname}'
            with self._archive_tarfile.extractfile(path_in_archive) as fh:
                tbl = read_bobcat(fh, columns, first_header_line_contains)
                tbl.setflags(write=0)
                setattr(self, attrname, tbl)

BOBCAT_EVOLUTION_TABLES_M0 = BobcatEvolutionTables(BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA.output_filepath)

class BobcatEvolutionModel:
    _interp_mass_temp_to_log_g : typing.Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    _interp_mass_age_to_T_evol : typing.Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    def __init__(self, evolution_tables, spectra_library):
        self.evolution_tables : BobcatEvolutionTables = evolution_tables
        self.spectra_library : model_grids.ModelSpectraGrid = spectra_library

    @property
    def _mass_temp_to_log_g(self):
        if self._interp_mass_temp_to_log_g is None:
            self._interp_mass_temp_to_log_g = interpolate.LinearNDInterpolator(
                np.stack(
                    [self.evolution_tables.mass['mass_Msun'],
                     self.evolution_tables.mass['T_eff_K']], axis=-1),
                self.evolution_tables.mass['log_g_cm_per_s2'],
                rescale=True
            )
        return self._interp_mass_temp_to_log_g

    @property
    def _mass_age_to_T_evol(self):
        if self._interp_mass_age_to_T_evol is None:
            self._interp_mass_age_to_T_evol = interpolate.LinearNDInterpolator(
                np.stack(
                    [self.evolution_tables.mass['mass_Msun'],
                     self.evolution_tables.mass['age_Gyr']], axis=-1),
                self.evolution_tables.mass['T_eff_K'],
                rescale=True
            )
        return self._interp_mass_age_to_T_evol

    def mass_age_to_properties(
        self,
        mass : u.Quantity,
        age : u.Quantity,
        filter_spectrum : spectra.Spectrum,
        T_eq : u.Quantity=None,
        magnitude_reference : spectra.Spectrum=VEGA_BOHLIN_GILLILAND_2004,
    ):
        '''Given mass, age, and filter spectrum produce evolutionary temperature,
        surface gravity, and magnitude

        Parameters
        ----------
        mass : `astropy.units.Quantity`
            One or more masses to model
        age : `astropy.units.Quantity`
            One or more ages to model (one per mass, if multiple mass
            values provided)
        filter_spectrum : `doodads.modeling.spectra.Spectrum`
            Filter spectrum in which to compute magnitude
        T_eq : `Optional[astropy.units.Quantity]`
            Equilibrium temperature set by host star, if any
        magnitude_reference : `doodads.modeling.spectra.Spectrum`
            Reference spectrum with which to compute magnitude
            (default: `doodads.ref.hst_calspec.VEGA`)

        Returns
        -------
        T_evol : `astropy.units.Quantity`
        T_eff : `astropy.units.Quantity`
        surface_gravity : `astropy.units.Quantity`
        mags : np.ndarray
            Astronomical magnitudes relative to `magnitude_reference`
            in `filter_spectrum`
        '''
        mass_Msun_vals = mass.to(u.Msun).value
        age_Gyr_vals = age.to(u.Gyr).value
        age_scalar = np.isscalar(age_Gyr_vals)
        mass_scalar = np.isscalar(mass_Msun_vals)
        if mass_scalar:
            if not age_scalar:
                repeat = len(age_Gyr_vals)
            else:
                repeat = 1
            mass_Msun_vals = np.repeat(mass_Msun_vals, repeat)
        if age_scalar:
            age_Gyr_vals = np.repeat(age_Gyr_vals, len(mass_Msun_vals))
        T_evol_K_vals = self._mass_age_to_T_evol(mass_Msun_vals, age_Gyr_vals)
        if T_eq is not None:
            T_eff_K_vals = (T_evol_K_vals**4 + T_eq.to(u.K).value**4)**(1/4)
        else:
            T_eff_K_vals = T_evol_K_vals
        log_g_cm_per_s2_vals = self._mass_temp_to_log_g(mass_Msun_vals, T_eff_K_vals)
        surface_gravity_m_per_s2 = (10**log_g_cm_per_s2_vals * u.cm/u.s**2).to(u.m/u.s**2).value
        surface_gravity = surface_gravity_m_per_s2 * u.m/u.s**2

        mags = np.zeros(len(mass_Msun_vals))
        for idx in range(len(mass_Msun_vals)):
            if not np.isnan(T_eff_K_vals[idx]) and not np.isnan(surface_gravity[idx]):
                try:
                    spec = self.spectra_library.get(
                        temperature=T_eff_K_vals[idx] * u.K,
                        surface_gravity=surface_gravity[idx],
                        mass=mass_Msun_vals[idx] * u.Msun,
                    )
                    mag_value = magnitude_reference.magnitude(spec, filter_spectrum)
                except model_grids.BoundsError as e:
                    log.debug(f"Out of bounds {T_eff_K_vals[idx]} K, {surface_gravity[idx]}, {mass_Msun_vals[idx]} Msun")
                    mag_value = np.nan
                mags[idx] = mag_value
            else:
                log.debug(f"Nonfinite interpolated values {T_eff_K_vals[idx]} K, {surface_gravity[idx]}, {mass_Msun_vals[idx]} Msun")
                mags[idx] = np.nan
        if mass_scalar and age_scalar:
            return T_evol_K_vals[0] * u.K, T_eff_K_vals[0] * u.K, surface_gravity[0], mags[0]
        return T_evol_K_vals * u.K, T_eff_K_vals * u.K, surface_gravity, mags

BOBCAT_EVOLUTION_M0 = BobcatEvolutionModel(BOBCAT_EVOLUTION_TABLES_M0, BOBCAT_SPECTRA_M0)