from collections import defaultdict
import logging
import numpy as np
import astropy.units as u
from scipy import interpolate

__all__ = [
    'BOBCAT_EVOLUTION_COLS',
    'BOBCAT_PHOTOMETRY_COLS',
    'read_bobcat',
    'make_mass_age_to_temp_grav',
    'make_mass_age_to_mag',
]

log = logging.getLogger(__name__)

BOBCAT_EVOLUTION_COLS = [
    'age_Gyr', 'mass_M_sun', 'log_L_L_sun', 'T_eff_K', 'log_g_cm_s2', 'radius_R_sun']

BOBCAT_PHOTOMETRY_COLS = [
    'T_eff_K',
    'log_g_cm_s2',
    'mass_M_jup',
    'radius_R_sun',
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


def read_bobcat(fh, colnames, skip_rows=0):
    cols = defaultdict(list)
    for i in range(skip_rows):
        next(fh)
    rows = 0
    for line in fh:
        parts = line.split()
        if len(parts) != len(colnames):
            log.debug(f"Line column number mismatch: got {len(parts)=} and expected {len(colnames)=}\nLine was {line=}")
            continue
        for idx, col in enumerate(colnames):
            try:
                val = float(parts[idx])
            except ValueError:
                val = np.nan
                print(f'{parts[idx]=} -> NaN')
            cols[col].append(val)
        rows += 1
    tbl = np.zeros((rows,), dtype=[(name, '=f4') for name in colnames])
    for name in colnames:
        tbl[name] = cols[name]
    return tbl

def make_mass_age_to_temp_grav(evol_tbl, eq_temp):
    mass_age_points = np.stack([evol_tbl['mass_M_sun'], evol_tbl['age_Gyr']], axis=-1)
    modified_T_eff = np.power(evol_tbl['T_eff_K']**4 + eq_temp.to(u.K).value**4, 1/4)
    mass_age_vals = np.stack([modified_T_eff, evol_tbl['log_g_cm_s2']], axis=-1)
    interp = interpolate.LinearNDInterpolator(mass_age_points, mass_age_vals)
    def mass_age_to_temp_grav(mass, age):
        mass_vals = mass.to(u.M_sun).value
        age_val = age.to(u.Gyr).value
        if np.isscalar(age_val):
            age_val = np.repeat(age_val, len(mass_vals))
        interp_at_pts = np.stack([mass_vals, age_val], axis=-1)
        temps, gravs = interp(interp_at_pts).T
        return temps * u.K, gravs
    return mass_age_to_temp_grav

def make_temp_grav_to_mag(phot_tbl, mag_col):
    temp_grav_points = np.stack([phot_tbl['T_eff_K'], phot_tbl['log_g_cm_s2']], axis=-1)
    temp_grav_vals = phot_tbl[mag_col]
    interp = interpolate.LinearNDInterpolator(temp_grav_points, temp_grav_vals)
    def temp_grav_to_mag(temps, gravs):
        temps_K = temps.to(u.K).value
        return interp(np.stack([temps_K, gravs], axis=-1))
    return temp_grav_to_mag

def make_mass_age_to_mag(evol_tbl, phot_tbl, mag_col='mag_MKO_Lprime', eq_temp=0*u.K):
    mass_age_to_temp_grav = make_mass_age_to_temp_grav(evol_tbl, eq_temp)
    temp_grav_to_mag = make_temp_grav_to_mag(phot_tbl, mag_col)
    def interpolator(mass, age):
        '''Take mass and age as unitful quantities,
        return in-bounds masses from the original input
        and magnitudes for them
        '''
        mass_vals = mass.to(u.M_sun)
        ages = np.repeat(age, len(mass_vals))
        temps, gravs = mass_age_to_temp_grav(mass, ages)
        oob_temp_grav = np.isnan(temps)
        mags = temp_grav_to_mag(temps[~oob_temp_grav], gravs[~oob_temp_grav])
        oob_mags = np.isnan(mags)
        final_masses = mass[~oob_temp_grav][~oob_mags].to(u.M_jup)
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
    excluded_ranges : list[tuple]
        list of ``(begin, end)`` pairs of masses defining
        the ranges where the inverse relationship was double-valued
        and the smaller mass was chosen for `masses` (empty
        if `take_smaller` is False
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

def make_mag_to_mass(evol_tbl, mass_age_to_mag, age, take_smaller=True, masses=None):
    if masses is None:
        masses = np.unique(evol_tbl['mass_M_sun'] * u.M_sun)
    masses, mags = mass_age_to_mag(masses, age)
    mags, min_masses, excluded_mass_ranges = _mag_to_mass(masses, mags, take_smaller=take_smaller)
    interpolator = interpolate.interp1d(mags, min_masses, bounds_error=False)
    def mag_to_mass(mag):
        mass = interpolator(mag)
        return mass * u.M_jup
    return mag_to_mass, excluded_mass_ranges
