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

import astropy.units as u
import numpy as np
from astropy.io import fits
from genericpath import exists
from numpy.lib.recfunctions import append_fields
from scipy import interpolate

from .. import math, utils
from ..modeling import spectra
from ..modeling.physics import f_nu_to_f_lambda
from . import model_grids

__all__ = ("",)

EVOLUTION_AGE_COLS = [
    "age_Gyr",
    "mass_Msun",
    "log_L_Lsun",
    "T_eff_K",
    "log_g_cm_per_s2",
    "radius_Rsun",
]


DIAMONDBACK_2024_EVOLUTION_DATA = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url="https://zenodo.org/records/12735103/files/evolution.zip?download=1",
    output_filename="diamondback_2024_evolution.zip",
)
DIAMONDBACK_2024_EVOLUTION_ARCHIVE = DIAMONDBACK_2024_EVOLUTION_DATA.output_filepath


def read_diamondback(fh, colnames, first_header_line_contains):
    cols = defaultdict(list)
    line = next(fh)
    if isinstance(line, bytes):
        decode = lambda x: x.decode("utf8")
        line = decode(line)
    else:
        decode = lambda x: x
    while first_header_line_contains not in line:
        line = decode(next(fh))
    rows = 0
    for line in fh:
        line = decode(line)
        parts = line.split()
        if len(parts) == 0:
            continue
        if len(parts) != len(colnames):
            if len(parts) == 1:
                # lines containing only an integer saying how many lines were written so far
                continue
            else:
                raise ValueError(
                    f"Line column number mismatch: got {len(parts)=} and expected {len(colnames)=}\nLine was {line=}"
                )
        for idx, col in enumerate(colnames):
            try:
                val = FLOAT_RE.findall(parts[idx])[0]
                val = float(val)
            except (ValueError, IndexError):
                val = np.nan
            cols[col].append(val)
        rows += 1
    tbl = np.zeros((rows,), dtype=[(name, "=f4") for name in colnames])
    for name in colnames:
        tbl[name] = cols[name]
    return tbl


def _load_from_resource(columns, first_header_line_contains, path_in_archive):
    archive = zipfile.open(DIAMONDBACK_2024_EVOLUTION_ARCHIVE)
    with archive.open(path_in_archive) as fh:
        tbl = read_diamondback(fh, columns, first_header_line_contains)
    return tbl


load_evolution_age = partial(_load_from_resource, EVOLUTION_AGE_COLS, "age(Gyr)")
