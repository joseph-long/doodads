from os.path import exists
import tarfile
import gzip
import numpy as np
import pytest
from ... import utils
from . import settl_cond

HAVE_BT_SETTL_ARCHIVE = exists(settl_cond.BT_SETTL_CIFIST2011C_PATH)
HAVE_AMES_COND_ARCHIVE = exists(settl_cond.AMES_COND_PATH)

## These are basic smoke tests (i.e. does smoke come out when you run it)
## since if it parses, it parses.

@pytest.mark.skipif(
    not (HAVE_BT_SETTL_ARCHIVE and HAVE_AMES_COND_ARCHIVE),
    reason='Download model spectra archives to run parser tests'
)
@pytest.mark.parametrize("archive_filepath,name_regex", [
    (settl_cond.BT_SETTL_CIFIST2011C_PATH, settl_cond.BT_SETTL_NAME_RE),
    (settl_cond.AMES_COND_PATH, settl_cond.AMES_COND_NAME_RE)
])
def test_populate_grid(archive_filepath, name_regex):
    settl_cond.make_filepath_lookup(
        tarfile.open(archive_filepath),
        name_regex
    )
    assert True

@pytest.mark.skipif(
    not HAVE_AMES_COND_ARCHIVE,
    reason='Download model spectra archives to run parser tests'
)
def test_parse_cond_rows():
    archive = tarfile.open(settl_cond.AMES_COND_PATH)
    name = 'lte01-2.5-0.0.AMES-Cond-2000.7.gz'
    compressed_file_handle = archive.extractfile(f'SPECTRA/{name}')
    wls, fluxes, bb_fluxes = settl_cond._load_one_spectrum(
        name,
        gzip.open(compressed_file_handle),
        settl_cond.parse_ames_cond_row,
        settl_cond.parse_ames_cond_stacked_format
    )

@pytest.mark.skipif(
    not HAVE_AMES_COND_ARCHIVE,
    reason='Download model spectra archives to run parser tests'
)
def test_truncated_spectrum_cond():
    archive = tarfile.open(settl_cond.AMES_COND_PATH)
    # spectrum that failed to parse because it cuts off at 2.74 um
    name = 'lte38-0.0-1.0.AMES-Cond-2000.spec.gz'
    compressed_file_handle = archive.extractfile(f'SPECTRA/{name}')
    settl_cond.parse_ames_cond_stacked_format(gzip.open(compressed_file_handle))

@pytest.mark.skipif(
    not HAVE_AMES_COND_ARCHIVE,
    reason='Download model spectra archives to run parser tests'
)
def test_inf_in_fluxes_cond():
    archive = tarfile.open(settl_cond.AMES_COND_PATH)
    # spectrum that failed to parse because it had 'Inf' in it:
    compressed_file_handle = archive.extractfile('SPECTRA/lte36-2.5-0.5.AMES-Cond-2000.spec.gz')
    settl_cond.parse_ames_cond_stacked_format(gzip.open(compressed_file_handle))
