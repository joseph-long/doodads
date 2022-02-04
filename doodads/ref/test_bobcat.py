import numpy as np
import astropy.units as u
import pytest
from . import bobcat, hst_calspec, mko_filters, model_grids
from .. import utils

@pytest.mark.skipif(not bobcat.BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA.exists,
    reason='Testing loader needs Bobcat evolution and photometry tables'
)
def test_table_loading():
    evol_tbl = bobcat.load_bobcat_evolution_age('evolution_tables/evo_tables+0.0/nc+0.0_co1.0_age')
    # age(Gyr)   M/Msun  log L/Lsun  Teff(K)  log g  R/Rsun
    # row 0:
    # 0.0010   0.0005     -5.361      631.   2.654  0.1743
    evol_row = evol_tbl[0]
    assert np.isclose(evol_row['age_Gyr'], 0.0010)
    assert np.isclose(evol_row['mass_Msun'], 0.0005)
    assert np.isclose(evol_row['log_L_Lsun'], -5.361)
    assert np.isclose(evol_row['T_eff_K'], 631)
    assert np.isclose(evol_row['log_g_cm_per_s2'], 2.654)
    assert np.isclose(evol_row['radius_Rsun'], 0.1743)
    phot_tbl = bobcat.load_bobcat_photometry('photometry_tables/mag_table+0.0')
    phot_row = phot_tbl[0]
    # just check a few of the cols
    #                                                 |                         MKO                            |         2MASS         |        Keck           |             SDSS             |             IRAC              |              WISE
    # Teff   log g    mass  R/Rsun   Y     log Kzz    Y       Z       J       H       K       L'      M'      J       H       Ks      Ks      L'      Ms      g'      r'      i'      z'    [3.6]   [4.5]   [5.7]   [7.9]     W1      W2      W3      W4
    #  200.  3.000    0.53  0.1180  0.28   -99.000  36.269  34.374  34.410  33.721  37.583  23.150  17.467  34.852  33.604  37.449  37.430  22.876  17.434  42.347  35.407  32.184  33.795  25.229  18.131  20.498  18.965  26.713  18.186  16.643  13.453
    assert np.isclose(phot_row['T_eff_K'], 200)
    assert np.isclose(phot_row['log_g_cm_per_s2'], 3.0)
    assert np.isclose(phot_row['log_Kzz'], -99)
    assert np.isclose(phot_row['mag_MKO_Lprime'], 23.150)

@pytest.mark.skipif((
    (not bobcat.BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA.exists) or
    (not bobcat.BOBCAT_SPECTRA_M0.exists)),
    reason='Testing synthetic photometry needs Bobcat isochrones and spectra, MKO filters, and HST CALSPEC Vega'
)
def test_model_grid():
    evol_tbl = bobcat.load_bobcat_evolution_age('evolution_tables/evo_tables+0.0/nc+0.0_co1.0_age')
    row = evol_tbl[10]
    ptspec = bobcat.BOBCAT_SPECTRA_M0.get(
        temperature=row['T_eff_K'] * u.K,
        surface_gravity=10**row['log_g_cm_per_s2'] * u.cm / u.s**2,
        mass=row['mass_Msun'] * u.Msun
    )
    luminosity = 10**row['log_L_Lsun']
    integrated_L_sun = (ptspec.integrate() * 4 * np.pi * (10 * u.pc)**2).to(u.L_sun).value
    assert np.abs((integrated_L_sun - luminosity) / luminosity) < 0.02, "Can't get approximate luminosity by integrating spectrum within <2%"

@pytest.mark.skipif((
    (not bobcat.BOBCAT_2021_EVOLUTION_PHOTOMETRY_DATA.exists) or
    (not bobcat.BOBCAT_SPECTRA_M0.exists) or
    (not hst_calspec.VEGA_BOHLIN_GILLILAND_2004.exists) or
    (not mko_filters.MKO.exists)),
    reason='Testing synthetic photometry needs Bobcat isochrones and spectra, MKO filters, and HST CALSPEC Vega'
)
def test_phot_agreement(downsample=100):
    phot_tbl = bobcat.load_bobcat_photometry('photometry_tables/mag_table+0.0')
    for row in phot_tbl[::downsample]:
        try:
            ptspec = bobcat.BOBCAT_SPECTRA_M0.get(
                temperature=row['T_eff_K'] * u.K,
                surface_gravity=10**(row['log_g_cm_per_s2'])*u.cm/u.s**2,
                mass=row['mass_Mjup'] * u.Mjup
            )
        except model_grids.BoundsError:
            continue
        our_lprime = hst_calspec.VEGA_BOHLIN_GILLILAND_2004.magnitude(ptspec, mko_filters.MKO.Lprime)
        their_lprime = row['mag_MKO_Lprime']
        assert np.abs((our_lprime - their_lprime) / their_lprime) < 0.01

@pytest.mark.skipif((
    (not bobcat.BOBCAT_EVOLUTION_TABLES_M0.exists) or
    (not bobcat.BOBCAT_SPECTRA_M0.exists)
), reason="Evolution tables not downloaded")
def test_evolution_interpolation(downsample=20):
    tabulated_masses = np.unique(bobcat.BOBCAT_EVOLUTION_TABLES_M0.mass['mass_Msun']) * u.Msun

    # Test return 1
    test_mass = tabulated_masses[0]
    mask = bobcat.BOBCAT_EVOLUTION_TABLES_M0.mass['mass_Msun'] == test_mass.to(u.Msun).value
    subset = bobcat.BOBCAT_EVOLUTION_TABLES_M0.mass[mask]
    T_evol, T_eff, surface_gravity = bobcat.BOBCAT_EVOLUTION_M0.mass_age_to_properties(
        tabulated_masses[0],
        subset['age_Gyr'][0] * u.Gyr,
    )
    assert utils.is_scalar(T_evol) and utils.is_scalar(T_eff) and utils.is_scalar(surface_gravity)

    # Test return many
    T_evol, T_eff, surface_gravity = bobcat.BOBCAT_EVOLUTION_M0.mass_age_to_properties(
        tabulated_masses[0],
        subset['age_Gyr'] * u.Gyr,
    )
    assert len(T_evol) == len(T_eff) == len(surface_gravity) == np.count_nonzero(mask)

    # Test agreement with table
    for test_mass in tabulated_masses[::20]:
        mask = bobcat.BOBCAT_EVOLUTION_TABLES_M0.mass['mass_Msun'] == test_mass.to(u.Msun).value
        subset = bobcat.BOBCAT_EVOLUTION_TABLES_M0.mass[mask]
        T_evol, T_eff, surface_gravity = bobcat.BOBCAT_EVOLUTION_M0.mass_age_to_properties(
            test_mass,
            subset['age_Gyr'] * u.Gyr,
        )
        assert np.allclose(subset['T_eff_K'] * u.K, T_eff)
        assert np.allclose(10**subset['log_g_cm_per_s2'] * u.cm/u.s**2, surface_gravity)
