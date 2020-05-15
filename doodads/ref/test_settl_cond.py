from os.path import exists
import tarfile
import gzip
import lzma
import numpy as np
from scipy.optimize import fmin
import pytest
import astropy.units as u
from .. import utils
from ..modeling import spectra, photometry, physics
from . import settl_cond, mko_filters, hst_calspec
from .settl_cond import BT_SETTL, AMES_COND

HAVE_BT_SETTL_ARCHIVE = exists(settl_cond.BT_SETTL_CIFIST2011_2015_ARCHIVE)
HAVE_BT_SETTL_FITS = exists(settl_cond.BT_SETTL_CIFIST2011_2015_FITS)
HAVE_AMES_COND_ARCHIVE = exists(settl_cond.AMES_COND_ARCHIVE)
HAVE_AMES_COND_FITS = exists(settl_cond.AMES_COND_FITS)

## These are basic smoke tests (i.e. does smoke come out when you run it)
## since if it parses, it parses.

@pytest.mark.skipif(
    not (HAVE_BT_SETTL_ARCHIVE and HAVE_AMES_COND_ARCHIVE),
    reason='Download model spectra archives to run parser tests'
)
@pytest.mark.parametrize("archive_filepath,name_regex", [
    (settl_cond.BT_SETTL_CIFIST2011_2015_ARCHIVE, settl_cond.BT_SETTL_NAME_RE),
    (settl_cond.AMES_COND_ARCHIVE, settl_cond.AMES_COND_NAME_RE)
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
def test_parse_cond_rows(name='lte01-2.5-0.0.AMES-Cond-2000.7.gz'):
    archive = tarfile.open(settl_cond.AMES_COND_ARCHIVE)
    compressed_file_handle = archive.extractfile(f'SPECTRA/{name}')
    return settl_cond._load_one_spectrum(
        name,
        gzip.open(compressed_file_handle),
        settl_cond.parse_ames_cond_row,
        settl_cond.parse_ames_cond_stacked_format
    )

@pytest.mark.skipif(
    not HAVE_BT_SETTL_ARCHIVE,
    reason='Download model spectra archives to run parser tests'
)
def test_parse_settl_rows(name='lte014.0-4.5-0.0a+0.0.BT-Settl.spec.7.xz'):
    archive = tarfile.open(settl_cond.BT_SETTL_CIFIST2011_2015_ARCHIVE)
    compressed_file_handle = archive.extractfile(name)
    return settl_cond._load_one_spectrum(
        name,
        lzma.open(compressed_file_handle),
        settl_cond.parse_bt_settl_row,
        settl_cond.parse_bt_settl_stacked_format
    )

@pytest.mark.skipif(
    not HAVE_AMES_COND_ARCHIVE,
    reason='Download model spectra archives to run parser tests'
)
def test_truncated_spectrum_cond():
    archive = tarfile.open(settl_cond.AMES_COND_ARCHIVE)
    # spectrum that failed to parse because it cuts off at 2.74 um
    name = 'lte38-0.0-1.0.AMES-Cond-2000.spec.gz'
    compressed_file_handle = archive.extractfile(f'SPECTRA/{name}')
    settl_cond.parse_ames_cond_stacked_format(gzip.open(compressed_file_handle))

@pytest.mark.skipif(
    not HAVE_AMES_COND_ARCHIVE,
    reason='Download model spectra archives to run parser tests'
)
def test_inf_in_fluxes_cond():
    archive = tarfile.open(settl_cond.AMES_COND_ARCHIVE)
    # spectrum that failed to parse because it had 'Inf' in it:
    compressed_file_handle = archive.extractfile('SPECTRA/lte36-2.5-0.5.AMES-Cond-2000.spec.gz')
    settl_cond.parse_ames_cond_stacked_format(gzip.open(compressed_file_handle))

@pytest.mark.skipif(
    not (HAVE_BT_SETTL_ARCHIVE and HAVE_AMES_COND_ARCHIVE),
    reason='Download model spectra archives to run parser tests'
)
@pytest.mark.parametrize("archive_filepath, name, decompress_function, row_parser_function, stacked_parser_function", [
    (settl_cond.BT_SETTL_CIFIST2011_2015_ARCHIVE, 'lte014.0-4.5-0.0a+0.0.BT-Settl.spec.7.xz', lzma.open, settl_cond.parse_bt_settl_row, settl_cond.parse_bt_settl_stacked_format),
    (settl_cond.AMES_COND_ARCHIVE, 'SPECTRA/lte47-4.0-0.5.AMES-Cond-2000.spec.gz', gzip.open, settl_cond.parse_ames_cond_row, settl_cond.parse_ames_cond_stacked_format)
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

@pytest.mark.skipif(
    not HAVE_BT_SETTL_FITS,
    reason='Download & convert BT-SETTL models to test grid lookup'
)
def test_bt_settl_grid_lookup():
    interpolated = settl_cond.BT_SETTL.get(T_eff=1200, log_g=2.5)
    # 0th row of models is for 1200 K, log g = 2.5
    assert np.allclose(settl_cond.BT_SETTL.model_spectra[0], interpolated.values.value), 'interpolator differs from FITS model spectrum data for row 0'
    interpolated_2 = settl_cond.BT_SETTL.get(T_eff=1200 + 1, log_g=2.5)
    ratio = interpolated_2.integrate() / interpolated.integrate()
    assert ratio < 1.01, 'One degree change in temp shouldn\'t be >1% change in flux values at 1200 K'


@pytest.mark.skipif(
    not HAVE_AMES_COND_FITS,
    reason='Download & convert AMES-COND models to test grid lookup'
)
def test_ames_cond_grid_lookup():
    interpolated = settl_cond.AMES_COND.get(T_eff=100, log_g=2.5, M_over_H=0)
    # 0th row of models is for 100 K, log g = 2.5, M/H = 0.0
    assert np.allclose(settl_cond.AMES_COND.model_spectra[0], interpolated.values.value), 'interpolator differs from FITS model spectrum data for row 0'
    # Use 1200 K point for test since fluxes actually change rather a lot with 1 degree around 100 K
    # (sigma T^4 and all that)
    ratio = settl_cond.AMES_COND.get(T_eff=1200 + 1, log_g=2.5, M_over_H=0).integrate() / settl_cond.AMES_COND.get(T_eff=1200, log_g=2.5, M_over_H=0).integrate()
    assert ratio < 1.01, 'One degree change in temp shouldn\'t be >1% change in flux values at 1200 K'


FILTERS_IN_MKO_ISOCHRONES = {'H', 'J', 'Ks', 'Lprime', 'Mprime'}

@pytest.mark.skipif((
    (not settl_cond.BT_SETTL_MKO_ISOCHRONES.exists) or
    (not settl_cond.BT_SETTL.exists) or
    (not hst_calspec.VEGA.exists) or
    (not mko_filters.MKO.exists)),
    reason='Testing synthetic photometry needs BT-Settl isochrones and spectra, MKO filters, and HST CALSPEC Vega'
)
def test_bt_settl_grid_magic_number(skip_by=100):
    '''Finding the magic number that produces correct absolute magnitudes for
    synthetic photometry with the BT-Settl spectra was a whole thing.

    The only real check possible is to use rows from the isochrone table
    and compare the magnitudes there with a value computed with the Vega
    spectrum, MKO filter curves, and an interpolated target spectrum at
    that T_eff and log g.

    By repeating the calculation and computing an error metric to minimize
    we can fit the best scale factor with `fmin`. Since there's a R^2
    dependence in the scaling, we also compute R in the optimizer (from the
    log g and mass).

    Evaluating every row of the MKO isochrones for BT-Settl gives
    magic number values between 4.86e-18 and 5.5e-18, mean 5.045e-18,
    median 5.035e-18, std 8.04e-20.

    By taking the median magic number (5.035e-18) as a constant and with
    the fractional error in the resulting magnitudes between -0.0194 and
    0.0115, mean -0.0014, median -0.0016, std 0.0033.
    '''
    isochrones = settl_cond.BT_SETTL_MKO_ISOCHRONES
    # Not all filters in MKO are in the isochrones
    filters_in_iso = FILTERS_IN_MKO_ISOCHRONES
    MAGIC_FACTOR = settl_cond.BT_SETTL_MAGIC_SCALE_FACTOR

    # Function to minimize with fmin
    def minimizer(params, iso_row_idx):
        '''Error is defined as max absolute fractional error in the magnitudes
        across different filters for an isochrone snapshot on row `iso_row_idx`

        Parameters
        ----------
        params : 1-tuple
            base scale factor (multiplied with radius^2 for true
            scale factor), changes with each iteration
        iso_row_idx : int
            Row in `isochrones`

        Returns
        -------
        err : float
            max absolute fractional error in the magnitudes
            across different filters
        '''
        scale_factor, = params
        iso = isochrones[iso_row_idx]
        original_mags = []
        new_mags = []

        radius = physics.mass_log_g_to_radius(iso['M_Msun'] * u.M_sun, iso['log_g'])
        model_spec = settl_cond.BT_SETTL.get(
            T_eff=iso['T_eff_K'],
            log_g=iso['log_g']
        ).multiply(scale_factor * radius.to(u.R_sun).value**2)
        for filt_name in filters_in_iso:
            filt_spec = getattr(mko_filters.MKO, filt_name)
            original_mags.append(iso[filt_name])
            new_mags.append(hst_calspec.VEGA.magnitude(model_spec, filter_spectrum=filt_spec))
        fractional_error = (np.asarray(new_mags) - np.asarray(original_mags)) / np.asarray(original_mags)
        err = np.max(np.abs(fractional_error))
        return err

    for idx in np.arange(len(isochrones.data))[::skip_by]:
        iso = isochrones[idx]
        scale_factor, = fmin(minimizer, MAGIC_FACTOR, args=(idx,))
        frac_err = (scale_factor - MAGIC_FACTOR) / MAGIC_FACTOR
        # When fitting with fmin in a notebook, scale factors were all around
        # 5.06e-18, and choice of a constant scale factor (multiplied with radius^2)
        # produced magnitudes within 1.2% even though the scale factor changed by ~9.2%
        # in the worst case (0.2% in mean case)
        assert frac_err < 0.093, "Magic factor incorrect? Fitting should produce the same value within 9.2%"
        for filt_name in filters_in_iso:
            orig_mag = iso[filt_name]
            radius = physics.mass_log_g_to_radius(iso['M_Msun'] * u.M_sun, iso['log_g'])
            real_scale_factor = scale_factor * radius.to(u.R_sun).value**2
            model_spec = settl_cond.BT_SETTL.get(T_eff=iso['T_eff_K'], log_g=iso['log_g'])
            filt_spec = getattr(mko_filters.MKO, filt_name)
            mag = hst_calspec.VEGA.magnitude(model_spec.multiply(real_scale_factor), filter_spectrum=filt_spec)
            assert (mag - orig_mag) / orig_mag < 0.01, 'Should agree with reference mag when using specially fit scale factor'
            pretty_good_scale_factor = MAGIC_FACTOR * radius.to(u.R_sun).value**2
            pretty_good_mag = hst_calspec.VEGA.magnitude(model_spec.multiply(pretty_good_scale_factor), filter_spectrum=filt_spec)
            # Numerical experiment described in docstring found max discrepancy of 1.9% in final mag when assuming constant
            # scale prefactor for pretty_good_mag. The mean and median cases are <0.2%.
            assert (pretty_good_mag - orig_mag) / orig_mag < 0.02, 'Using constant scale prefactor, agreement should be < 2%'

@pytest.mark.skipif((
    (not settl_cond.BT_SETTL_MKO_ISOCHRONES.exists) or
    (not settl_cond.BT_SETTL.exists) or
    (not hst_calspec.VEGA.exists) or
    (not mko_filters.MKO.exists)),
    reason='Testing synthetic photometry needs BT-Settl isochrones and spectra, MKO filters, and HST CALSPEC Vega'
)
def test_bt_settl_grid_mass_distance_scaling():
    # take row from isochrones
    isochrones = settl_cond.BT_SETTL_MKO_ISOCHRONES
    iso = isochrones[1]
    # convert to apparent mags at 1 pc
    dist = 1 * u.pc
    filters_in_iso = FILTERS_IN_MKO_ISOCHRONES
    absolute_mags = {name: iso[name] for name in filters_in_iso}
    print('absolute_mags', absolute_mags)
    apparent_mags_by_dist_modulus = {
        name: photometry.apparent_mag(iso[name], dist)
        for name in filters_in_iso
    }
    print('apparent_mags', apparent_mags_by_dist_modulus)
    # get interpolated grid spectrum with distance=10 pc
    model_spec_abs = settl_cond.BT_SETTL.get(
        mass=iso['M_Msun'] * u.Msun,
        T_eff=iso['T_eff_K'],
        log_g=iso['log_g']
    )
    # get interpolated grid spectrum with distance=1pc
    model_spec_1pc = settl_cond.BT_SETTL.get(
        mass=iso['M_Msun'] * u.Msun,
        distance=dist,
        T_eff=iso['T_eff_K'],
        log_g=iso['log_g']
    )
    # compute magnitudes from VEGA
    for filt_name in filters_in_iso:
        filt_spec = getattr(mko_filters.MKO, filt_name)
        abs_mag = absolute_mags[filt_name]
        my_abs_mag = hst_calspec.VEGA.magnitude(model_spec_abs, filter_spectrum=filt_spec)
        assert (my_abs_mag - abs_mag) / abs_mag < 0.01

        apparent_mag = apparent_mags_by_dist_modulus[filt_name]
        my_apparent_mag = hst_calspec.VEGA.magnitude(model_spec_1pc, filter_spectrum=filt_spec)
        # should be <3% different
        assert (my_apparent_mag - apparent_mag) / apparent_mag < 0.03
