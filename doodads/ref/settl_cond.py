import tarfile
import lzma
import gzip
import re
import os.path
import logging

import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
from astropy.io import fits
import astropy.units as u

from joblib import Parallel, delayed
from functools import partial
from ..modeling.units import WAVELENGTH_UNITS, FLUX_UNITS
from ..modeling import spectra, physics
from .. import utils

__all__ = [
    'ModelSpectraGrid',
    'load_ames_cond_model',
    'load_bt_settl_model',
    'AMES_COND',
    'AMES_COND_MKO_ISOCHRONES',
    'BT_SETTL',
    'BT_SETTL_MKO_ISOCHRONES',
]

log = logging.getLogger(__name__)

# > The file names contain the main parameters of the models:
# lte{Teff/10}-{Logg}{[M/H]}a[alpha/H].GRIDNAME.7.spec.gz/bz2/xz
# is the synthetic spectrum for the requested effective temperature
# (Teff),surface gravity (Logg), metallicity by log10 number density with
# respect to solar values ([M/H]), and alpha element enhencement relative
# to solar values [alpha/H]. The model grid is also mentioned in the name.
#
# — [README](https://phoenix.ens-lyon.fr/Grids/FORMAT)

# It appears (based on the included blackbody flux spectrum peak) that
# the filename contains Teff / 100, not Teff / 10.

AMES_COND_NAME_RE = re.compile(r'SPECTRA/lte(?P<t_eff_over_100>[^-]+)-(?P<log_g>[\d\.]+)-(?P<M_over_H>[\d.]+).AMES-Cond-2000.(?:spec|7).gz')

BT_SETTL_NAME_RE = re.compile(r'lte(?P<t_eff_over_100>[^-]+)-(?P<log_g>[\d\.]+)-(?P<M_over_H>[\d.]+)a(?P<alpha_over_H>[-+\d.]+).BT-Settl.spec.7.xz')

def filepath_to_params(filepath, compiled_regex):
    match = compiled_regex.match(filepath)
    if match is None:
        return None
    parts = match.groupdict()
    t_eff = 100 * float(parts['t_eff_over_100'])
    log_g = float(parts['log_g'])
    M_over_H = float(parts['M_over_H'])
    return t_eff, log_g, M_over_H

# For regridding onto a regular reduced wavelength grid
ORIG_WL_UNITS = u.AA
MODEL_WL_START = (0.5 * u.um).to(ORIG_WL_UNITS)
MODEL_WL_END = (6 * u.um).to(ORIG_WL_UNITS)
# chosen based on the coarsest sampling out of both AMES-Cond and
# BT-Settl, found for wavelengths at the long end of the range of
# interest in Cond
MODEL_WL_STEP = (5e-4 * u.um).to(ORIG_WL_UNITS)
MODEL_WL = np.arange(MODEL_WL_START.value, MODEL_WL_END.value, MODEL_WL_STEP.value) * ORIG_WL_UNITS

def make_filepath_lookup(archive_tarfile, name_regex):
    '''Loop through all files in a tarfile of spectra
    and return `lookup` mapping (T_eff, log_g, M_over_H) tuples
    to filenames, `all_params` a dict of sets for 'T_eff',
    'log_g', 'M_over_H' values present
    '''
    filepath_lookup = {}
    all_params = {
        'T_eff': set(),
        'log_g': set(),
        'M_over_H': set(),
    }
    for name in archive_tarfile.getnames():
        parsed_params = filepath_to_params(name, name_regex)
        if parsed_params is None:
            continue  # skip entry for the 'SPECTRA' dir itself
        filepath_lookup[parsed_params] = name
        T_eff, log_g, M_over_H = parsed_params
        all_params['T_eff'].add(T_eff)
        all_params['log_g'].add(log_g)
        all_params['M_over_H'].add(M_over_H)
    return filepath_lookup, all_params

def parse_float(val):
    try:
        return float(val)
    except ValueError:
        return float(val.replace(b'D', b'e'))

BT_SETTL_DILUTION_FACTOR = -8
# See test_settl_cond.test_bt_settl_grid_magic_number for how this was
# computed from the data
BT_SETTL_MAGIC_SCALE_FACTOR = 5.059761904761906e-18
AMES_COND_DILUTION_FACTOR = -26.9007901434

# column1: wavelength in Angstroem
# column2: 10**(F_lam + DF) to convert to Ergs/sec/cm**2/A
# column3: 10**(B_lam + DF) i.e. the blackbody fluxes of same Teff in same units.
# -- https://phoenix.ens-lyon.fr/Grids/FORMAT
# Looks like this means column2 is F_lam in the equation in the README, column3 is B_lam
FORTRAN_FLOAT_REGEX = re.compile(rb'([-+]?[\d.]+[De][-+][\d]+|Inf)')
BT_SETTL_REGEX = re.compile(rb'^\s*([\d.]+)\s*([-+]?[\d.]+D[-+][\d]+)\s*([-+]?[\d.]+D[-+][\d]+).+$')
AMES_COND_REGEX = re.compile(rb'^\s*([-+]?[\d.]+D[-+][\d]+)\s*([-+]?[\d.]+D[-+][\d]+)\s*([-+]?[\d.]+D[-+][\d]+).+$')

ORIG_FLUX_UNITS = u.erg * u.s**-1 * u.cm**-2 * u.AA**-1

def parse_bt_settl_stacked_format(file_handle):
    raise NotImplementedError("No stacked-format BT-Settl spectra, right?")

def parse_bt_settl_row(row):
    match = BT_SETTL_REGEX.match(row)
    if match is None:
        raise ValueError("Unparseable row: {}".format(row))
    wavelength_bytes, flux_bytes, bb_flux_bytes = match.groups()
    wavelength_aa = parse_float(wavelength_bytes)
    flux = np.power(10, parse_float(flux_bytes) + BT_SETTL_DILUTION_FACTOR)
    bb_flux = np.power(10, parse_float(bb_flux_bytes) + BT_SETTL_DILUTION_FACTOR)
    return wavelength_aa, flux, bb_flux

def parse_ames_cond_stacked_format(file_handle):
    params_line = next(file_handle)
    assert b'Teff, logg, [M/H]' in params_line, "file doesn't start with params"
    n_wls_line = next(file_handle)
    assert b'number of wavelength points' in n_wls_line, "file doesn't include # wl points"
    numbers = re.findall(rb'(\d+)', n_wls_line)
    assert len(numbers) == 1, "More than 1 number found for # of wls"
    number_of_wls = int(numbers[0])
    wavelengths = []
    fluxes = []
    bb_fluxes = []
    wls_to_go = number_of_wls
    list_to_populate = wavelengths
    for line in file_handle:
        if wls_to_go < 1:
            raise Exception("Too much data?")
        numbers = FORTRAN_FLOAT_REGEX.findall(line)
        n_parts = len(line.split())
        if n_parts != len(numbers):
            raise RuntimeError(f'Got {n_parts} parts, only {len(numbers)} matched as floats')
        list_to_populate.extend([float(numstr.replace(b'D', b'e').replace(b'Inf', b'0')) for numstr in numbers])
        wls_to_go -= len(numbers)
        if wls_to_go == 0:
            if list_to_populate is wavelengths:
                list_to_populate = fluxes
            elif list_to_populate is fluxes:
                list_to_populate = bb_fluxes
            wls_to_go = number_of_wls
        elif wls_to_go < 0:
            if list_to_populate is wavelengths:
                raise RuntimeError(f'Parser overshot on wavelengths in {file_handle}')
            elif list_to_populate is fluxes:
                raise RuntimeError(f'Parser overshot on fluxes in {file_handle}')
            else:
                raise RuntimeError(f'Parser overshot on blackbody fluxes in {file_handle}')
    if not len(wavelengths) == len(fluxes) == len(bb_fluxes):
        raise RuntimeError(f'Mismatched lengths: {len(wavelengths)} wavelengths, {len(fluxes)} fluxes, {len(bb_fluxes)} BB fluxes in {file_handle}')
    wavelengths = np.asarray(wavelengths)
    fluxes = np.asarray(fluxes) * np.power(10, AMES_COND_DILUTION_FACTOR)
    bb_fluxes = np.asarray(bb_fluxes) * np.power(10, AMES_COND_DILUTION_FACTOR)
    return wavelengths, fluxes, bb_fluxes

def parse_ames_cond_row(row):
    match = AMES_COND_REGEX.match(row)
    if match is None:
        raise ValueError("Unparseable row: {}".format(row))
    wavelength_bytes, flux_bytes, bb_flux_bytes = match.groups()
    wavelength_aa = parse_float(wavelength_bytes)
    flux = np.power(10, parse_float(flux_bytes) + AMES_COND_DILUTION_FACTOR)
    bb_flux = np.power(10, parse_float(bb_flux_bytes) + AMES_COND_DILUTION_FACTOR)
    return wavelength_aa, flux, bb_flux

def apply_ordering_and_units(wls, fluxes, bb_fluxes):
    wls = np.asarray(wls)
    fluxes = np.asarray(fluxes)
    bb_fluxes = np.asarray(bb_fluxes)
    sorter = np.argsort(wls)
    wls = wls[sorter]
    fluxes = fluxes[sorter]
    bb_fluxes = bb_fluxes[sorter]
    return (
        wls * ORIG_WL_UNITS,
        fluxes * ORIG_FLUX_UNITS,
        bb_fluxes * ORIG_FLUX_UNITS
    )

def resample_spectrum(orig_wls, orig_fluxes, new_wls):
    unit = orig_fluxes.unit
    wls = orig_wls.to(new_wls.unit).value
    return interp1d(wls, orig_fluxes.value)(new_wls.value) * unit

STACKED_FILENAMES_REGEX = re.compile(r'.*\.spec(\.gz)?$')

def _parse_one_spectrum(name, file_handle, row_parser_function, stacked_parser_function):
    if STACKED_FILENAMES_REGEX.match(name):
        try:
            wls, fluxes, bb_fluxes = stacked_parser_function(file_handle)
        except Exception as e:
            print(name, e)
            raise
    else:
        wls, fluxes, bb_fluxes = [], [], []
        for row_bytes in file_handle:
            try:
                wl, f, bb = row_parser_function(row_bytes)
            except ValueError as e:
                print(e)
                continue
            wls.append(wl)
            fluxes.append(f)
            bb_fluxes.append(bb)
    return wls, fluxes, bb_fluxes

def _load_one_spectrum(name, file_handle, row_parser_function, stacked_parser_function):
    wls, fluxes, bb_fluxes = _parse_one_spectrum(name, file_handle, row_parser_function, stacked_parser_function)
    model_wls, model_fluxes, model_bb_fluxes = apply_ordering_and_units(wls, fluxes, bb_fluxes)

    resampled_fluxes = resample_spectrum(model_wls, model_fluxes, MODEL_WL)
    resampled_bb_fluxes = resample_spectrum(model_wls, model_bb_fluxes, MODEL_WL)
    return resampled_fluxes, resampled_bb_fluxes

def _load_grid_spectrum(archive_filename, filepath_lookup, idx, params,
                        row_parser_function, stacked_parser_function,
                        decompressor):
    archive_tarfile = tarfile.open(archive_filename)
    T_eff, log_g, M_over_H = params
    filepath = filepath_lookup[params]
    n_spectra = len(filepath_lookup)
    print(f'{idx+1}/{n_spectra} T_eff={T_eff} log g={log_g} M/H={M_over_H}: {filepath}')
    specfile = decompressor(archive_tarfile.extractfile(filepath))
    try:
        resampled_fluxes, resampled_bb_fluxes = _load_one_spectrum(filepath, specfile, row_parser_function, stacked_parser_function)
    except Exception as e:
        print(f'Exception {e} processing {filepath}')
        return None, None
    return resampled_fluxes, resampled_bb_fluxes

def load_bt_settl_model(filepath):
    with open(filepath, 'rb') as file_handle:
        resampled_fluxes, resampled_bb_fluxes = _load_one_spectrum(
            filepath,
            file_handle,
            parse_bt_settl_row,
            parse_bt_settl_stacked_format
        )
    return MODEL_WL.copy(), resampled_fluxes, resampled_bb_fluxes

def load_ames_cond_model(filepath):
    with open(filepath, 'rb') as file_handle:
        resampled_fluxes, resampled_bb_fluxes = _load_one_spectrum(
            filepath,
            file_handle,
            parse_ames_cond_row,
            parse_ames_cond_stacked_format
        )
    return MODEL_WL.copy(), resampled_fluxes, resampled_bb_fluxes

def _load_all_spectra(archive_filename, sorted_params, filepath_lookup,
                      row_parser_function, stacked_parser_function,
                      decompressor):
    n_spectra = len(sorted_params)
    all_spectra = np.zeros((n_spectra,) + MODEL_WL.shape) * ORIG_FLUX_UNITS
    all_bb_spectra = np.zeros((n_spectra,) + MODEL_WL.shape) * ORIG_FLUX_UNITS
    loader = partial(_load_grid_spectrum,
        archive_filename=archive_filename,
        filepath_lookup=filepath_lookup,
        row_parser_function=row_parser_function,
        stacked_parser_function=stacked_parser_function,
        decompressor=decompressor
    )
    results = Parallel(n_jobs=-1)(
        delayed(loader)(idx=idx, params=params) for idx, params in enumerate(sorted_params)
    )
    bad_indices = []
    for idx, (fluxes, bb_fluxes) in enumerate(results):
        if fluxes is not None:  # sentinel value for unparseable / undersized arrays
            all_spectra[idx] = fluxes
            all_bb_spectra[idx] = bb_fluxes
        else:
            bad_indices.append(idx)
    for idx in bad_indices:
        sorted_params.pop(idx)
    all_spectra = np.delete(all_spectra, np.asarray(bad_indices, dtype=int), axis=0)
    all_bb_spectra = np.delete(all_bb_spectra, np.asarray(bad_indices, dtype=int), axis=0)
    return sorted_params, all_spectra, all_bb_spectra


def convert_grid(archive_filename, filename_regex, row_parser_function, stacked_parser_function, decompressor, _debug_first_n=None):
    archive_tarfile = tarfile.open(archive_filename)
    filepath_lookup, all_params = make_filepath_lookup(archive_tarfile, filename_regex)
    sorted_params = list(sorted(filepath_lookup.keys()))
    if _debug_first_n is not None:
        sorted_params = sorted_params[:_debug_first_n]
    # Some spectra have missing data or are otherwise unusable
    # so we re-assign sorted_params to contain only the
    # ones we could successfully load
    sorted_params, all_spectra, all_bb_spectra = _load_all_spectra(
        archive_filename,
        sorted_params,
        filepath_lookup,
        row_parser_function,
        stacked_parser_function,
        decompressor
    )

    hdulist = fits.HDUList([fits.PrimaryHDU(),])

    T_eff = [row[0] for row in sorted_params]
    log_g = [row[1] for row in sorted_params]
    M_over_H = [row[2] for row in sorted_params]
    params_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='T_eff', format='E', array=T_eff),
        fits.Column(name='log_g', format='E', array=log_g),
        fits.Column(name='M_over_H', format='E', array=M_over_H),
        fits.Column(name='index', format='J', array=np.arange(len(sorted_params)))
    ])
    params_hdu.header['EXTNAME'] = 'PARAMS'
    hdulist.append(params_hdu)

    wls_hdu = fits.ImageHDU(MODEL_WL.value)
    wls_hdu.header['EXTNAME'] = 'WAVELENGTHS'
    wls_hdu.header['UNIT'] = str(MODEL_WL.unit)
    hdulist.append(wls_hdu)

    flux_hdu = fits.ImageHDU(all_spectra.value)
    flux_hdu.header['EXTNAME'] = 'MODEL_SPECTRA'
    flux_hdu.header['UNIT'] = str(all_spectra.unit)
    hdulist.append(flux_hdu)

    bb_hdu = fits.ImageHDU(all_bb_spectra.value)
    bb_hdu.header['EXTNAME'] = 'BLACKBODY_SPECTRA'
    bb_hdu.header['UNIT'] = str(all_bb_spectra.unit)
    hdulist.append(bb_hdu)

    return hdulist

ISOCHRONE_COLUMNS = (
    'age_Gyr',
    'M_Msun',
    'T_eff_K',
    'L_Lsun',
    'log_g',
    'R_Gcm',
    'D',
    'Li',
    'J',
    'H',
    'Ks',
    "Lprime",
    "Mprime"
)

def _convert_isochrones(original_path, output_path):
    with open(original_path, 'r') as fh:
        isochrones = fh.read()
    chunks = isochrones.split('\n\n\n\n')
    parsed_chunks = []
    with open(output_path, 'w') as fh:
        # isochrone column names are a taken from a list here because
        # the column format in the file runs the names together...
        fh.write(','.join(ISOCHRONE_COLUMNS) + '\n')
        for ch in chunks:
            age, header, data, _ = ch.split('-'*113)
            age_Gyr = float(re.match(r't \(Gyr\) =\s+([\d.]+)', age.strip()).groups()[0])
            for line in data.strip().split('\n'):
                cols = line.strip().split()
                if cols:
                    outseq = [str(age_Gyr),] + cols
                    fh.write(','.join(outseq) + '\n')


def _convert_bt_settl(settl_filepath, output_filepath):
    settl_hdul = convert_grid(
            settl_filepath,
            BT_SETTL_NAME_RE,
            parse_bt_settl_row,
            parse_bt_settl_stacked_format,
            lzma.open
        )
    settl_hdul.writeto(output_filepath, overwrite=True)

def _convert_ames_cond(cond_filepath, output_filepath):
    cond_hdul = convert_grid(
            cond_filepath,
            AMES_COND_NAME_RE,
            parse_ames_cond_row,
            parse_ames_cond_stacked_format,
            gzip.open
        )
    cond_hdul.writeto(output_filepath, overwrite=True)

class ModelSpectraGrid(utils.LazyLoadable):
    def __init__(self, filepath, magic_scale_factor=1.0):
        super().__init__(filepath)
        self.name = os.path.basename(filepath)
        self.magic_scale_factor = magic_scale_factor
        # populated by _lazy_load():
        self.hdu_list = None
        self.params = None
        self.param_names = None
        self.wavelengths = None
        self.model_spectra = None
        self.blackbody_spectra = None

    def _lazy_load(self):
        self.hdu_list = fits.open(self.filepath)
        self.params = np.asarray(self.hdu_list['PARAMS'].data)
        self.param_names = self.params.dtype.fields.keys() - {'index'}
        self.wavelengths = self.hdu_list['WAVELENGTHS'].data
        self.model_spectra = self.hdu_list['MODEL_SPECTRA'].data
        self.blackbody_spectra = self.hdu_list['BLACKBODY_SPECTRA'].data

        # some params don't vary in all libraries, exclude those
        # so qhull doesn't refuse to interpolate
        self._real_param_names = self.param_names.copy()
        for name in self.param_names:
            if len(np.unique(self.params[name])) == 1:
                self._real_param_names.remove(name)
                log.debug(f'Discarding {name} because all grid points have {name} == {np.unique(self.params[name])[0]}')
        # coerce to sequence because we can't depend on iteration order
        self._real_param_names = list(sorted(self._real_param_names))

        params_grid = np.stack([self.params[name] for name in self._real_param_names]).T
        self._interpolator = LinearNDInterpolator(
            params_grid,
            self.model_spectra,
            rescale=True
        )
    @property
    def bounds(self):
        out = {}
        for name in self._real_param_names:
            out[name] = np.min(self.params[name]), np.max(self.params[name])
        return out
    def get(self, mass=None, distance=10*u.pc, **kwargs):
        '''Look up or interpolate a spectrum for given parameters, scaled
        appropriately for mass and distance. (To disable scaling and get
        the values from the FITS file, omit mass.)

        Parameters
        ----------
        mass : units.Quantity or None
            If mass is provided, scale returned Spectrum correctly for
            `distance`. Otherwise, return as-is and ignore `distance`.
        distance : units.Quantity or None
            If mass is provided, scale returned Spectrum correctly for
            `distance`. Ignored otherwise.
        **kwargs : number
            Values for grid parameters listed in the `param_names` attribute.
        '''
        # kwargs: all true params required, all incl. non-varying params accepted
        if (
            (not self.param_names.issuperset(kwargs.keys()))
            or
            (not all(name in kwargs for name in self._real_param_names))
        ):
            raise ValueError(f"Valid kwargs (from grid params) are {self.param_names}")

        interpolator_args = []
        for name in self._real_param_names:
            interpolator_args.append(kwargs[name])
        model_fluxes = self._interpolator(*interpolator_args) * ORIG_FLUX_UNITS
        if np.any(np.isnan(model_fluxes)):
            raise ValueError(f"Parameters {kwargs} are out of bounds for this model grid")
        wl = self.wavelengths * ORIG_WL_UNITS
        model_spec = spectra.Spectrum(wl, model_fluxes)
        # with great effort, it was determined that the correct scaling
        # to make the flux in the models reproduce the right MKO mags in
        # the isochrones is given by a magic number multiplied by the
        # radius (obtained from log g) squared
        if mass is not None:
            radius = physics.mass_log_g_to_radius(mass, kwargs['log_g'])
            radius_Rsun = radius.to(u.Rsun).value
            scale_factor = self.magic_scale_factor * radius_Rsun**2 * (((10 * u.pc) / distance)**2).si
            model_spec = model_spec.multiply(scale_factor)

        return model_spec


class Isochrones(utils.LazyLoadable):
    def __init__(self, filepath, name):
        self.name = name
        self.data = None
        super().__init__(filepath)
    def _lazy_load(self):
        self.data = np.genfromtxt(self.filepath, delimiter=',', names=True)
    @property
    def masses(self):
        return np.unique(self.data['M_Msun']) * u.Msun
    @property
    def ages(self):
        return np.unique(self.data['age_Gyr']) * u.Gyr
    def __getitem__(self, name):
        self._ensure_loaded()
        return self.data[name]

BT_SETTL_CIFIST2011_2015_DATA = utils.REMOTE_RESOURCES.add(
    module=__name__,
    url='https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/SPECTRA/BT-Settl_M-0.0a+0.0.tar',
    converter_function=_convert_bt_settl,
    output_filename='BT-Settl_CIFIST2011_2015_spectra.fits',
)
BT_SETTL_CIFIST2011_2015_ARCHIVE = BT_SETTL_CIFIST2011_2015_DATA.download_filepath
BT_SETTL_CIFIST2011_2015_FITS = BT_SETTL_CIFIST2011_2015_DATA.output_filepath

AMES_COND_DATA = utils.REMOTE_RESOURCES.add(
    module=__name__,
    url='https://phoenix.ens-lyon.fr/Grids/AMES-Cond/SPECTRA.tar',
    converter_function=_convert_ames_cond,
    output_filename='AMES-Cond_spectra.fits',
)
AMES_COND_ARCHIVE = AMES_COND_DATA.download_filepath
AMES_COND_FITS = AMES_COND_DATA.output_filepath

AMES_COND_MKO_ISOCHRONES = utils.REMOTE_RESOURCES.add(
    module=__name__,
    url='https://phoenix.ens-lyon.fr/Grids/AMES-Cond/ISOCHRONES/model.AMES-Cond-2000.M-0.0.MKO.Vega',
    converter_function=_convert_isochrones,
    output_filename='AMES-Cond_MKO_isochrones.csv'
)
BT_SETTL_CIFIST2011_2015_MKO_ISOCHRONES = utils.REMOTE_RESOURCES.add(
    module=__name__,
    url='https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/ISOCHRONES/model.BT-Settl.M-0.0.MKO.Vega',
    converter_function=_convert_isochrones,
    output_filename='BT-Settl_CIFIST2011_2015_isochrones.csv'
)

AMES_COND = (
    ModelSpectraGrid(AMES_COND_FITS)
    if os.path.exists(AMES_COND_FITS) else None
)
AMES_COND_MKO_ISOCHRONES = Isochrones(AMES_COND_MKO_ISOCHRONES.output_filepath, name='AMES-Cond (2000) MKO')

BT_SETTL = (
    ModelSpectraGrid(BT_SETTL_CIFIST2011_2015_FITS, magic_scale_factor=BT_SETTL_MAGIC_SCALE_FACTOR)
    if os.path.exists(BT_SETTL_CIFIST2011_2015_FITS) else None
)
BT_SETTL_MKO_ISOCHRONES = Isochrones(BT_SETTL_CIFIST2011_2015_MKO_ISOCHRONES.output_filepath, name='BT-Settl CIFIST2011 (2015) MKO')
