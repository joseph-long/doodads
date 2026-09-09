import gzip
import io
import logging
import re
import tarfile
import typing
import warnings
import zipfile
from collections import defaultdict
from functools import partial
from genericpath import exists

import astropy.units as u
import numpy as np
import tqdm
from astropy.io import fits
from numpy.lib.recfunctions import append_fields
from scipy import interpolate

from ... import analysis, math, utils
from ...modeling import spectra
from ...modeling.physics import f_nu_to_f_lambda
from .. import model_grids

__all__ = (
    "DIAMONDBACK_EVOLUTION_HYBRID",
    "DIAMONDBACK_EVOLUTION_NC",
    "DIAMONDBACK_EVOLUTION_HYBRID_GRAV",
    "DIAMONDBACK_SPECTRA",
)

log = logging.getLogger(__name__)

BASE_EVOLUTION_AGE_COLS = [
    "age/Gyr",
    "mass/Msun",
    "log(L/Lsun)",
    "T_eff/K",
    "log(g/cm/s^2)",
    "radius/Rsun",
]

EVOLUTION_AGE_COLS = BASE_EVOLUTION_AGE_COLS + ["metallicity/[M/H]"]


def read_diamondback(fh, first_header_line_contains):
    cols = defaultdict(list)
    line = next(fh)
    if isinstance(line, bytes):
        decode = lambda x: x.decode("utf8").strip()
        line = decode(line)
    else:
        decode = lambda x: x.strip()
    while first_header_line_contains not in line:
        line = decode(next(fh))
    rows = 0
    for line in fh:
        line = decode(line)
        parts = line.split()
        if len(parts) == 0:
            continue
        if len(parts) != len(BASE_EVOLUTION_AGE_COLS):
            # matches lines containing the age for the following block
            continue
        for idx, col in enumerate(BASE_EVOLUTION_AGE_COLS):
            try:
                val = utils.FLOAT_RE.findall(parts[idx])[0]
                val = float(val)
            except (ValueError, IndexError):
                val = np.nan
            cols[col].append(val)
        rows += 1
    tbl = np.zeros((rows,), dtype=[(name, "=f4") for name in BASE_EVOLUTION_AGE_COLS])
    for name in BASE_EVOLUTION_AGE_COLS:
        tbl[name] = cols[name]
    return tbl


_isochrone_groups = {
    "hybrid": "hybrid_f2_",
    "nc": "nc_",
    "hybrid-grav": "hybrid-grav_f2_",
}


def _convert_isochrones(archive_filepath, output_filepath):
    hdus = [fits.PrimaryHDU()]
    for method in _isochrone_groups:
        base_fn = _isochrone_groups[method]
        tbls_all_metallicities = []
        for mpart, mval in [("m-0.5", -0.5), ("m0.0", 0.0), ("m+0.5", 0.5)]:
            path_in_archive = "evolution/" + base_fn + mpart + "_age"
            log.info(f"Processing {path_in_archive}")
            tbl = load_diamondback_evolution_age(archive_filepath, path_in_archive)
            tbl = utils.append_fields(
                tbl, "metallicity/[M/H]", mval * np.ones(len(tbl))
            )
            tbls_all_metallicities.append(tbl)
        tbl_for_group = np.concatenate(tbls_all_metallicities)
        hdus.append(fits.BinTableHDU(tbl_for_group, name=method))
    hdul = fits.HDUList(hdus)
    hdul.writeto(output_filepath, overwrite=True)
    return output_filepath


EVOLUTION_DATA = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url="https://zenodo.org/records/12735103/files/evolution.zip?download=1",
    converter_function=_convert_isochrones,
    output_filename="sonora_diamondback_2024_evolution.fits",
)


def _load_from_resource(archive_filepath, path_in_archive, first_header_line_contains):
    archive = zipfile.ZipFile(archive_filepath)

    with archive.open(path_in_archive) as fh:
        tbl = read_diamondback(fh, first_header_line_contains)
    return tbl


load_diamondback_evolution_age = partial(
    _load_from_resource,
    first_header_line_contains="age(Gyr)",
)


_raw_to_unitful_mapping = {
    "age/Gyr": ("age", u.Gyr),
    "mass/Msun": ("mass", u.Msun),
    "log(L/Lsun)": ("luminosity", u.dex(u.Lsun)),
    "T_eff/K": ("temperature", u.K),
    "log(g/cm/s^2)": ("log_g", u.dex(u.cm / u.s**2)),
    "radius/Rsun": ("radius", u.Rsun),
    "metallicity/[M/H]": ("metallicity", u.dex),
}

_unitful_to_raw_mapping = {v[0]: (k, v[1]) for k, v in _raw_to_unitful_mapping.items()}


def _unitful_to_raw(**kwargs):
    unitless_kwargs = {}
    for k in kwargs:
        if kwargs[k] is None:
            continue
        if k not in _unitful_to_raw_mapping:
            raise ValueError(
                f"Unknown column {k}, have {list(_unitful_to_raw_mapping.keys())}"
            )
        print(f"{k=} {_unitful_to_raw_mapping[k]=}")
        name, unit = _unitful_to_raw_mapping[k]
        unitless_kwargs[name] = kwargs[k].to(unit).value
    return unitless_kwargs


def _raw_to_unitful(table):
    units = []
    names = []
    for raw_name in table.dtype.fields.keys():
        name, unit = _raw_to_unitful_mapping[raw_name]
        units.append(unit)
        names.append(name)
    sunit = u.StructuredUnit(
        units=units,
        names=names,
    )
    return table * sunit


class DiamondbackEvolution(model_grids.BaseFitsGrid):
    def __init__(self, group):
        super().__init__(
            EVOLUTION_DATA.output_filepath,
            ext=group,
            name="Sonora Diamondback (Morley 2024)",
        )

    def interpolate(
        self,
        age=None,
        mass=None,
        luminosity=None,
        temperature=None,
        surface_gravity=None,
        radius=None,
        metallicity=0.0 * u.dex,
    ):
        return self.interpolate_raw(
            **_unitful_to_raw(
                age=age,
                mass=mass,
                luminosity=luminosity,
                temperature=temperature,
                surface_gravity=surface_gravity,
                radius=radius,
                metallicity=metallicity,
            )
        )


DIAMONDBACK_EVOLUTION_HYBRID = DiamondbackEvolution("hybrid")
DIAMONDBACK_EVOLUTION_HYBRID_GRAV = DiamondbackEvolution("hybrid-grav")
DIAMONDBACK_EVOLUTION_NC = DiamondbackEvolution("nc")

SPECTRUM_FN_RE = re.compile(r".*/t(\d+)g(\d+)(f\d+|nc)_m([+-]?[\d.]+)_co([\d.]+).spec")


def parse_spectrum_filename(fn_str):
    res = SPECTRUM_FN_RE.match(fn_str)
    base_fn = fn_str.rsplit('/', 1)[1]
    if res is None:
        raise RuntimeError(f"{fn_str=} did not parse")
    temp, grav, fsed, met, co = res.groups()

    row = np.array(
        (
            float(temp),
            float(grav),
            float(fsed[1:]) if fsed != "nc" else 0.0,
            float(met),
            float(co),
        ),
        [
            ("T_eff_K", float),
            ("gravity_m_per_s2", float),
            ("f_sed", float),
            ("log_M_to_H", float),
            ("log_C_to_O", float),
        ],
    )
    return fn_str, base_fn, row


def _spectra_listing_to_params_table(spectrum_files: list[str]):
    full_fns, filenames, rows = [], [], []
    for x in spectrum_files:
        try:
            full_fn, base_fn, row = parse_spectrum_filename(x)
            full_fns.append(full_fn)
            filenames.append(base_fn)
            rows.append(row)
        except RuntimeError:
            log.info(f"Skipping filename {x} because it doesn't parse as a spectrum filename")
    return full_fns, np.asarray(filenames), np.asarray(rows)

def _parse_spectrum(archive_file):
    wls_um, fluxes_w_per_m2_per_m = [], []
    for line in archive_file:
        try:
            line = line.decode('utf8').strip()
        except ValueError:
            log.error(f"Couldn't decode {line=}")
            raise
        if not len(line):
            continue
        try:
            wl_um, flux_w_per_m2_per_m = line.split()
        except ValueError:
            continue
        wls_um.append(float(wl_um.strip()))
        fluxes_w_per_m2_per_m.append(float(flux_w_per_m2_per_m.strip()))
    
    wls_um, fluxes_w_per_m2_per_m = np.asarray(wls_um), np.asarray(fluxes_w_per_m2_per_m)
    return wls_um, fluxes_w_per_m2_per_m

def _convert_spectra(archive_filepath, output_filepath):
    arch = zipfile.ZipFile(archive_filepath)
    spectrum_files = [fn for fn in arch.namelist()]
    spectrum_files, filenames, params = _spectra_listing_to_params_table(spectrum_files)
    sorter = np.argsort(params['T_eff_K'])
    standard_wls_um = None
    all_spectra = None
    for idx, spectrum_file in enumerate(tqdm.tqdm(spectrum_files)):
        with arch.open(spectrum_file) as fh:
            these_wls_um, fluxes_w_per_m2_per_m = _parse_spectrum(fh)
        if standard_wls_um is None:
            standard_wls_um = these_wls_um
            all_spectra = np.zeros((len(spectrum_files), len(these_wls_um)))
        else:
            if not np.allclose(standard_wls_um, these_wls_um):
                raise RuntimeError(f"{spectrum_file=} has different wl sampling?")
        all_spectra[idx] = fluxes_w_per_m2_per_m
    params = params[sorter]
    filenames = filenames[sorter]
    all_spectra = all_spectra[sorter]

    hdulist = fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU(params, name='PARAMS'),
        fits.ImageHDU(standard_wls_um, name='WAVELENGTHS'),
        fits.ImageHDU(all_spectra, name='MODEL_SPECTRA'),
    ])
    hdulist['WAVELENGTHS'].header['UNIT'] = 'um'
    hdulist['MODEL_SPECTRA'].header['UNIT'] = 'W / m3'

    hdulist.writeto(output_filepath, overwrite=True)


SPECTRA_DATA = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url="https://zenodo.org/records/12735103/files/spectra.zip?download=1",
    converter_function=_convert_spectra,
    output_filename="sonora_diamondback_2024_spectra.fits",
)

class DiamondbackSpectraGrid(model_grids.ModelSpectraGrid):
    SAMPLING_UNITS = u.um


DIAMONDBACK_SPECTRA = DiamondbackSpectraGrid(SPECTRA_DATA.output_filepath) if SPECTRA_DATA.exists else None
