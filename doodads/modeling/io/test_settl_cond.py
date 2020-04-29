from os.path import exists
import tarfile
import gzip
import lzma
import numpy as np
import pytest
from ... import utils
from . import settl_cond

HAVE_BT_SETTL_ARCHIVE = exists(settl_cond.BT_SETTL_CIFIST2011_2015_FITS)
HAVE_AMES_COND_ARCHIVE = exists(settl_cond.AMES_COND_PATH)

## These are basic smoke tests (i.e. does smoke come out when you run it)
## since if it parses, it parses.

@pytest.mark.skipif(
    not (HAVE_BT_SETTL_ARCHIVE and HAVE_AMES_COND_ARCHIVE),
    reason='Download model spectra archives to run parser tests'
)
@pytest.mark.parametrize("archive_filepath,name_regex", [
    (settl_cond.BT_SETTL_CIFIST2011_2015_PATH, settl_cond.BT_SETTL_NAME_RE),
    (settl_cond.AMES_COND_PATH, settl_cond.AMES_COND_NAME_RE)
])
def test_populate_grid(archive_filepath, name_regex):
    print(archive_filepath)
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
    settl_cond._load_one_spectrum(
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

@pytest.mark.skipif(
    not HAVE_AMES_COND_ARCHIVE,
    reason='Download model spectra archives to run parser tests'
)
@pytest.mark.parametrize("archive_filepath, name, decompress_function, row_parser_function, stacked_parser_function", [
    (settl_cond.BT_SETTL_CIFIST2011_2015_PATH, 'lte014.0-4.5-0.0a+0.0.BT-Settl.spec.7.xz', lzma.open, settl_cond.parse_bt_settl_row, settl_cond.parse_bt_settl_stacked_format),
    (settl_cond.AMES_COND_PATH, 'SPECTRA/lte47-4.0-0.5.AMES-Cond-2000.spec.gz', gzip.open, settl_cond.parse_ames_cond_row, settl_cond.parse_ames_cond_stacked_format)
])
def test_resampling(archive_filepath, name, decompress_function, row_parser_function, stacked_parser_function):
    '''Note that downsampling these spectra can introduce interpolation
    errors in the total flux. We test that the error is sub 0.5% over
    the whole model spectrum. (When convolved with a filter bandpass,
    the error is even smaller.)
    '''
    archive = tarfile.open(archive_filepath)
    compressed_file_handle = archive.extractfile(name)
    file_handle = decompress_function(compressed_file_handle)

    wls, fluxes, bb_fluxes = settl_cond._parse_one_spectrum(name, file_handle, row_parser_function, stacked_parser_function)
    model_wls, model_fluxes, model_bb_fluxes = settl_cond.apply_ordering_and_units(wls, fluxes, bb_fluxes)

    min_wl, max_wl = settl_cond.MODEL_WL_START.to(model_wls.unit), settl_cond.MODEL_WL_END.to(model_wls.unit)
    mask = (model_wls >= min_wl) & (model_wls <= max_wl)
    integrated_fluxes = np.trapz(model_fluxes[mask], model_wls[mask])

    resampled_fluxes = settl_cond.resample_spectrum(model_wls, model_fluxes, settl_cond.MODEL_WL)
    new_integrated_fluxes = np.trapz(resampled_fluxes, settl_cond.MODEL_WL)

    diff = (integrated_fluxes.to(new_integrated_fluxes.unit).value - new_integrated_fluxes.value) / new_integrated_fluxes.value
    assert diff < 0.0005, 'Resampling introduced flux errors of > 0.5%'
