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
    "ELF_OWL_SPECTRA",
)

log = logging.getLogger(__name__)


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
    ELF_OWL_SPECTRA_FILENAMES = re.compile(r'')
    archive_tarfile = tarfile.open(archive_filepath)
    spectra_paths = [
        name for name in archive_tarfile.getnames() if ELF_OWL_SPECTRA_FILENAMES.match(name)
    ]
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

class ElfOwlSpectraGrid(model_grids.ModelSpectraGrid):
    SAMPLING_UNITS = u.um


DIAMONDBACK_SPECTRA = ElfOwlSpectraGrid(SPECTRA_DATA.output_filepath) if SPECTRA_DATA.exists else None
