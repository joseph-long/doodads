from collections import defaultdict
import io
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
from .. import utils, math

__all__ = [
    'BOBCAT_EVOLUTION_AGE_COLS',
    'BOBCAT_PHOTOMETRY_MAG_COLS',
    'BOBCAT_PHOTOMETRY_FLUX_COLS',
    'read_bobcat',
    'load_bobcat_photometry_flux',
    'load_bobcat_photometry_mag',
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

_BOBCAT_PHOTOMETRY_COLS = [
    'T_eff_K',
    'log_g_cm_per_s2',
    'mass_Mjup',
    'radius_Rsun',
    'helium_frac_Y',
    'log_Kzz',
]

_BOBCAT_PHOTOMETRY_BANDS = [
    'MKO_Y',
    'MKO_Z',
    'MKO_J',
    'MKO_H',
    'MKO_K',
    'MKO_Lprime',
    'MKO_Mprime',
    '2MASS_J',
    '2MASS_H',
    '2MASS_Ks',
    'Keck_Ks',
    'Keck_Lprime',
    'Keck_Ms',
    'SDSS_gprime',
    'SDSS_rprime',
    'SDSS_iprime',
    'SDSS_zprime',
    'IRAC_3_6_um',
    'IRAC_4_5_um',
    'IRAC_5_7_um',
    'IRAC_7_9_um',
    'WISE_W1',
    'WISE_W2',
    'WISE_W3',
    'WISE_W4',
]

BOBCAT_PHOTOMETRY_MAG_COLS = _BOBCAT_PHOTOMETRY_COLS + [f"mag_{band}" for band in _BOBCAT_PHOTOMETRY_BANDS]
BOBCAT_PHOTOMETRY_FLUX_COLS = _BOBCAT_PHOTOMETRY_COLS + [f"log_flux_mJy_{band}" for band in _BOBCAT_PHOTOMETRY_BANDS]

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
            if len(parts) == 1:
                # lines containing only an integer saying how many lines were written so far
                continue
            else:
                raise ValueError(f"Line column number mismatch: got {len(parts)=} and expected {len(colnames)=}\nLine was {line=}")
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

BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA = utils.REMOTE_RESOURCES.add_from_url(
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
load_bobcat_photometry_mag = partial(_load_from_resource, BOBCAT_PHOTOMETRY_MAG_COLS, 'MKO')
load_bobcat_photometry_flux = partial(_load_from_resource, BOBCAT_PHOTOMETRY_FLUX_COLS, 'MKO')


SPECTRA_PARAMS_COLS = ['T_eff_K', 'gravity_m_per_s2', 'Y', 'f_rain', 'Kzz', 'Fe_over_H', 'C_over_O', 'f_hole']

class InconsistentSamplingException(Exception):
    pass

def load_bobcat_spectrum(fh,
                         source_wavelength_unit=u.um, source_flux_per_frequency_unit=(u.erg / u.cm**2 / u.s / u.Hz),
                         wavelengths=None):
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
    wavelength_order_sorter = np.argsort(these_wavelengths)
    these_wavelengths = (these_wavelengths[wavelength_order_sorter] * source_wavelength_unit).to(WAVELENGTH_UNITS)
    if wavelengths is not None and not np.all((these_wavelengths - wavelengths) / wavelengths < 5e-7):
        raise InconsistentSamplingException(f"Inconsistent wavelength sampling")
    elif wavelengths is None:
        wavelengths = these_wavelengths
    these_fluxes = these_fluxes[wavelength_order_sorter] * source_flux_per_frequency_unit
    fluxes = f_nu_to_f_lambda(these_fluxes, wavelengths)
    return params, wavelengths, fluxes

BOBCAT_SPECTRA_FILENAMES = re.compile(r'(?:spectra/)?sp_(.+)(?:\.gz)?')
def _convert_spectra(tarfile_filepath, output_filepath, match_pattern=BOBCAT_SPECTRA_FILENAMES):
    archive_tarfile = tarfile.open(tarfile_filepath)
    spectra_paths = [name for name in archive_tarfile.getnames() if match_pattern.match(name)]
    if len(spectra_paths) == 0:
        raise RuntimeError("Could not match any spectra names")
    params_tbl = np.zeros(
        len(spectra_paths),
        dtype=list((param, float) for param in SPECTRA_PARAMS_COLS)
    )
    wavelengths = None
    spectra = None
    for idx, name in enumerate(spectra_paths):
        if name.endswith('.gz'):
            rawfh = gzip.open(archive_tarfile.extractfile(name), mode='r')
        else:
            rawfh = archive_tarfile.extractfile(name)
        fh = io.TextIOWrapper(rawfh, encoding='utf8')
        try:
            params, wavelengths, f_lambda = load_bobcat_spectrum(
                fh,
                wavelengths=wavelengths,
            )
        except InconsistentSamplingException:
            raise InconsistentSamplingException(f"Inconsistent wavelength sampling in {name}")
        finally:
            rawfh.close()
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

BOBCAT_2021_SPECTRA_M0_DATA = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url='https://zenodo.org/record/5063476/files/spectra_m%2B0.0.tar.gz?download=1',
    converter_function=_convert_spectra,
    output_filename='bobcat_2021_spectra_m0.fits',
)
BOBCAT_2021_SPECTRA_M0_FITS = BOBCAT_2021_SPECTRA_M0_DATA.output_filepath

BOBCAT_2021_SPECTRA_Mplus0_5_DATA = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url='https://zenodo.org/record/5063476/files/spectra_m%2B0.5.tar.gz?download=1',
    converter_function=_convert_spectra,
    output_filename='bobcat_2021_spectra_m+0.5.fits',
)

BOBCAT_2021_SPECTRA_Mminus0_5_DATA = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url='https://zenodo.org/record/5063476/files/spectra_m-0.5.tar.gz?download=1',
    converter_function=_convert_spectra,
    output_filename='bobcat_2021_spectra_m-0.5.fits',
)

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
    def _interpolate(self, T_eff_K=None, gravity_m_per_s2=None, **kwargs):
        # There's some quirks in the data (using cm/s^2 here, m/s^2 in the
        # spectra archive) that mean we should replace barely out-of-bounds
        # values with the boundary values
        min_T_K, max_T_K = self.bounds['T_eff_K']
        if T_eff_K < min_T_K and np.abs((T_eff_K - min_T_K)/min_T_K) < self.fractional_param_err:
            print(f"replacing {T_eff_K} with {min_T_K}")
            kwargs['T_eff_K'] = min_T_K
        elif T_eff_K > max_T_K and np.abs((T_eff_K - max_T_K)/max_T_K) < self.fractional_param_err:
            kwargs['T_eff_K'] = max_T_K
        else:
            kwargs['T_eff_K'] = T_eff_K

        min_g_m_per_s2, max_g_m_per_s2 = self.bounds['gravity_m_per_s2']
        if gravity_m_per_s2 < min_g_m_per_s2 and np.abs((gravity_m_per_s2 - min_g_m_per_s2)/min_g_m_per_s2) < self.fractional_param_err:
            kwargs['gravity_m_per_s2'] = min_g_m_per_s2
        elif gravity_m_per_s2 > max_g_m_per_s2 and np.abs((gravity_m_per_s2 - max_g_m_per_s2)/max_g_m_per_s2) < self.fractional_param_err:
            kwargs['gravity_m_per_s2'] = max_g_m_per_s2
        else:
            kwargs['gravity_m_per_s2'] = gravity_m_per_s2
        return super()._interpolate(**kwargs)

BOBCAT_SPECTRA_M0 = BobcatModelSpectraGrid(BOBCAT_2021_SPECTRA_M0_FITS)
BOBCAT_SPECTRA_Mplus0_5 = BobcatModelSpectraGrid(BOBCAT_2021_SPECTRA_Mplus0_5_DATA.output_filepath)
BOBCAT_SPECTRA_Mminus0_5 = BobcatModelSpectraGrid(BOBCAT_2021_SPECTRA_Mminus0_5_DATA.output_filepath)

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
        archive_tarfile = tarfile.open(self.filepath)
        if not np.any(np.isclose(self.metallicity, [k for k in self._metallicity_to_string])):
            raise ValueError(f"Available metallicities are {[v for v in self._metallicity_to_string.values()]}")
        for attrname, columns, first_header_line_contains in self._table_mapping:
            metallicity_str = self._metallicity_to_string[self.metallicity]
            path_in_archive = f'evolution_tables/evo_tables{metallicity_str}/nc{metallicity_str}_co1.0_{attrname}'
            with archive_tarfile.extractfile(path_in_archive) as fh:
                tbl = read_bobcat(fh, columns, first_header_line_contains)
                tbl.setflags(write=0)
                setattr(self, attrname, tbl)

BOBCAT_EVOLUTION_TABLES_M0 = BobcatEvolutionTables(
    BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA.output_filepath,
    metallicity=0.0
)
BOBCAT_EVOLUTION_TABLES_Mplus0_5 = BobcatEvolutionTables(
    BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA.output_filepath,
    metallicity=0.5
)
BOBCAT_EVOLUTION_TABLES_Mminus0_5 = BobcatEvolutionTables(
    BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA.output_filepath,
    metallicity=-0.5
)

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

    def _match_arg_length(self, arg1, arg2):
        arg2_scalar = utils.is_scalar(arg2)
        arg1_scalar = utils.is_scalar(arg1)
        if arg1_scalar:
            if not arg2_scalar:
                repeat = len(arg2)
            else:
                repeat = 1
            arg1 = np.repeat(arg1, repeat)
        if arg2_scalar:
            arg2 = np.repeat(arg2, len(arg1))
        return arg1, arg1_scalar, arg2, arg2_scalar

    def mass_age_to_properties(
        self,
        mass : u.Quantity,
        age : u.Quantity,
        T_eq : u.Quantity=None,
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
        T_eq : `Optional[astropy.units.Quantity]`
            Equilibrium temperature set by host star, if any

        Returns
        -------
        T_evol : `astropy.units.Quantity`
            Evolutionary temperature based on mass and age
        T_eff : `astropy.units.Quantity`
            (T_evol^4 + T_eq^4)^(1/4)
        surface_gravity : `astropy.units.Quantity`
        '''
        mass, mass_scalar, age, age_scalar = self._match_arg_length(mass, age)
        mass_Msun_vals = mass.to(u.Msun).value
        age_Gyr_vals = age.to(u.Gyr).value
        T_evol_K_vals = self._mass_age_to_T_evol(mass_Msun_vals, age_Gyr_vals)
        if T_eq is not None:
            T_eff_K_vals = (T_evol_K_vals**4 + T_eq.to(u.K).value**4)**(1/4)
        else:
            T_eff_K_vals = T_evol_K_vals
        log_g_cm_per_s2_vals = self._mass_temp_to_log_g(mass_Msun_vals, T_eff_K_vals)
        surface_gravity_m_per_s2 = (10**log_g_cm_per_s2_vals * u.cm/u.s**2).to(u.m/u.s**2).value
        surface_gravity = surface_gravity_m_per_s2 * u.m/u.s**2
        if mass_scalar and age_scalar:
            return T_evol_K_vals[0] * u.K, T_eff_K_vals[0] * u.K, surface_gravity[0]
        return T_evol_K_vals * u.K, T_eff_K_vals * u.K, surface_gravity

    def mass_age_to_magnitude(
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
            (default: `doodads.ref.hst_calspec.VEGA_BOHLIN_GILLILAND_2004`)

        Returns
        -------
        T_evol : `astropy.units.Quantity`
            Evolutionary temperature based on mass and age
        T_eff : `astropy.units.Quantity`
            (T_evol^4 + T_eq^4)^(1/4)
        surface_gravity : `astropy.units.Quantity`
        mags : np.ndarray
            Astronomical magnitudes relative to `magnitude_reference`
            in `filter_spectrum`
        '''
        mass, mass_scalar, age, age_scalar = self._match_arg_length(mass, age)
        T_evol, T_eff, surface_gravity = self.mass_age_to_properties(
            mass,
            age,
            T_eq=T_eq,
        )
        convolved_mag_ref = magnitude_reference.multiply(filter_spectrum)
        mags = np.zeros(len(T_eff))
        for idx in range(len(T_eff)):
            if not np.isnan(T_eff[idx]) and not np.isnan(surface_gravity[idx]):
                try:
                    spec = self.spectra_library.get(
                        temperature=T_eff[idx],
                        surface_gravity=surface_gravity[idx],
                        mass=mass[idx],
                    )
                    mag_value = convolved_mag_ref.magnitude(spec.multiply(filter_spectrum))
                except model_grids.BoundsError as e:
                    log.debug(f"Out of bounds {T_eff[idx]}, {surface_gravity[idx]}, {mass[idx]}")
                    mag_value = np.nan
                mags[idx] = mag_value
            else:
                log.debug(f"Nonfinite interpolated values {T_eff[idx]}, {surface_gravity[idx]}, {mass[idx]}")
                mags[idx] = np.nan
        if mass_scalar and age_scalar:
            T_evol, T_eff, surface_gravity, mags = T_evol[0], T_eff[0], surface_gravity[0], mags[0]
        return T_evol, T_eff, surface_gravity, mags

    def magnitude_age_to_mass(
        self,
        magnitude : np.ndarray,
        age : u.Quantity,
        filter_spectrum : spectra.Spectrum,
        T_eq : u.Quantity=None,
        magnitude_reference : spectra.Spectrum=VEGA_BOHLIN_GILLILAND_2004,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[math.ExcludedRange]]:
        """
        Returns
        -------
        mass, too_faint, too_bright, excluded_ranges
        """
        magnitude_scalar = utils.is_scalar(magnitude)
        tabulated_masses = (np.unique(self.evolution_tables.age['mass_Msun']) * u.Msun).to(u.Mjup)
        T_evol, T_eff, surface_gravity, mags = self.mass_age_to_magnitude(
            tabulated_masses,
            age,
            filter_spectrum,
            T_eq=T_eq,
            magnitude_reference=magnitude_reference,
        )
        subset_masses, subset_mags, excluded_ranges = math.make_monotonic_decreasing(tabulated_masses, mags)
        TOO_FAINT = -np.inf
        TOO_BRIGHT = np.inf
        mass_Mjup = interpolate.interp1d(subset_mags, subset_masses.to(u.Mjup).value, bounds_error=False, fill_value=(TOO_BRIGHT, TOO_FAINT))(magnitude)
        too_faint = mass_Mjup == TOO_FAINT
        too_bright = mass_Mjup == TOO_BRIGHT
        mass = mass_Mjup * u.Mjup
        if not magnitude_scalar:
            mass[too_faint] = np.min(subset_masses)
            mass[too_bright] = np.max(subset_masses)
        elif too_faint:
            mass = np.min(subset_masses)
        elif too_bright:
            mass = np.max(subset_masses)
        return mass, too_faint, too_bright, excluded_ranges

BOBCAT_EVOLUTION_M0 = BobcatEvolutionModel(BOBCAT_EVOLUTION_TABLES_M0, BOBCAT_SPECTRA_M0)
BOBCAT_EVOLUTION_Mplus0_5 = BobcatEvolutionModel(BOBCAT_EVOLUTION_TABLES_Mplus0_5, BOBCAT_SPECTRA_Mplus0_5)
BOBCAT_EVOLUTION_Mminus0_5 = BobcatEvolutionModel(BOBCAT_EVOLUTION_TABLES_Mminus0_5, BOBCAT_SPECTRA_Mminus0_5)
