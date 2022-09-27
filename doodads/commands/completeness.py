import datetime
from dataclasses import dataclass
from dateutil.parser import parse
import logging
import time
import typing
from itertools import product
import xconf
from xconf.contrib import BaseRayGrid, FileConfig
import ray
import astropy.units as u
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from ..modeling.physics import equilibrium_temperature
from ..modeling.orbits import generate_random_orbit_positions
from ..modeling.spectra import Spectrum
from xpipeline.tasks import characterization, iofits
from .contrast_to_mass import HostStarConfig, BobcatModels, FilterConfig
log = logging.getLogger(__name__)

def evaluate_one_point(
    true_sep, proj_sep_x_au, proj_sep_y_au, companion_mass, model_suite,
    abs_mag_limit_interpolator, age, host_temp, host_radius, albedo, filter_spectrum
):
    eq_temp = equilibrium_temperature(
        host_temp,
        host_radius,
        true_sep,
        albedo
    )
    T_evol, T_eff, surface_grav, mag = model_suite.mass_age_to_magnitude(companion_mass, age, filter_spectrum, eq_temp)
    abs_mag_limit = abs_mag_limit_interpolator(proj_sep_x_au, proj_sep_y_au)
    # if abs mag is brighter (smaller) than limit, we keep it, evaluate to True
    return mag < abs_mag_limit

@xconf.config
class CompletenessGrid:
    n_samples : int = xconf.field(default=100, help="Number of samples (random orbits/positions) to draw at each grid point")
    companion_masses_Mjup : typing.Union[list[float],str] = xconf.field(default='infer', help="List of companion mass grid points, or 'infer' to get from model grid masses")
    semimajor_axes_AU : typing.Union[list[float],str] = xconf.field(default='infer', help="List of semimajor axis grid points, or 'infer' to get from input table separation values")

    def __post_init__(self):
        for var in (self.semimajor_axes_AU, self.companion_masses_Mjup):
            if not isinstance(var, list) and not var == 'infer':
                raise ValueError("Valid values for grid dimension are a list of floats or the string 'infer'")

@dataclass
class MassLimits:
    name : str
    contrast_r_AU : np.array
    contrast_pa_deg : typing.Optional[np.array]
    companion_abs_mags : np.array
    obs_date : datetime.datetime
    filter : Spectrum

class RadialInterpolator:
    def __init__(self, radii, values):
        TOO_FAR = -np.inf   # brighter than any mag we might generate
        self.interp = interp1d(
            radii, values,
            bounds_error=False,
            fill_value=TOO_FAR
        )
    def __call__(self, x, y):
        r = np.sqrt(x**2 + y**2)
        return self.interp(r)

def _completeness_for_mass_and_sma(
    row, model_suite, limits_by_name : dict[str,MassLimits],
    age, host_temp, host_radius, host_mass, albedo,
    n_samples
):
    start = time.time()
    out = row.copy()
    companion_mass = row['companion_mass_Mjup'] * u.Mjup
    semimajor_axis = row['semimajor_axis_AU'] * u.AU
    limits_names = list(sorted(limits_by_name.keys()))
    log.debug(f"Processing completeness for {companion_mass}, {semimajor_axis} with {limits_names}")
    fractional_year_epochs = [0 * u.year]
    for name in limits_names[1:]:
        delta = limits_by_name[name].obs_date - limits_by_name[limits_names[0]].obs_date
        delta_yr = (delta.total_seconds() * u.s).to(u.year)
        log.debug(f"Delta from {limits_by_name[limits_names[0]].obs_date} to {limits_by_name[name].obs_date}: {delta_yr}")
        fractional_year_epochs.append(delta_yr)
    true_r, proj_xy = generate_random_orbit_positions(
        n_samples, host_mass, companion_mass,
        draw_sma=False, fixed_sma=semimajor_axis,
        epochs=fractional_year_epochs,
    )
    log.debug(f"Got (epochs, random_orbits) = {true_r.shape}")
    results = np.zeros((len(limits_by_name), n_samples), dtype=bool)
    for limits_idx, limits_name in enumerate(limits_names):
        limits = limits_by_name[limits_name]
        if limits.contrast_pa_deg is not None:
            contrast_xs_AU, contrast_ys_AU = characterization.r_pa_to_x_y(
                limits.contrast_r_AU,
                limits.contrast_pa_deg,
                0,
                0
            )
            # cut out the middle
            min_r_AU = np.min(limits.contrast_r_AU) - 0.0001
            unique_pa_deg_at_min_r = np.unique(limits.contrast_pa_deg[limits.contrast_r_AU == min_r_AU])
            cutout_xs_AU, cutout_ys_AU = characterization.r_pa_to_x_y(
                min_r_AU,
                unique_pa_deg_at_min_r
            )
            contrast_xs_AU = np.concatenate([contrast_xs_AU, cutout_xs_AU])
            contrast_ys_AU = np.concatenate([contrast_ys_AU, cutout_ys_AU])
            contrast_xy_AU = np.stack([contrast_ys_AU, contrast_xs_AU], axis=-1)
            companion_abs_mags = np.concatenate([limits.companion_abs_mags, np.repeat(-np.inf, len(unique_pa_deg_at_min_r))])
            abs_mag_limit_interpolator = LinearNDInterpolator(contrast_xy_AU, companion_abs_mags)
        else:
            abs_mag_limit_interpolator = RadialInterpolator(limits.contrast_r_AU, limits.companion_abs_mags)
        for sample_idx in range(n_samples):
            results[limits_idx, sample_idx] = evaluate_one_point(
                true_r[limits_idx, sample_idx], proj_xy[limits_idx, sample_idx, 0], proj_xy[limits_idx, sample_idx, 1], companion_mass, model_suite,
                abs_mag_limit_interpolator, age, host_temp, host_radius, albedo, limits.filter
            )
        out[f'completeness_fraction_{limits_name}'] = np.count_nonzero(results[limits_idx]) / n_samples
    combined_results = np.bitwise_or.reduce(results, axis=0)
    assert len(combined_results.shape) == 1
    assert len(combined_results) == n_samples
    out['completeness_fraction'] = np.count_nonzero(combined_results) / n_samples
    out['time_total_sec'] = time.time() - start
    return out

completeness_for_mass_and_sma = ray.remote(_completeness_for_mass_and_sma)

@xconf.config
class MassLimitsData:
    fits_file : FileConfig = xconf.field(help="")
    name : str = xconf.field(default="mass_limits", help="short name used to separate out completeness stats specific to this set of mass limits in the final table")
    masses_ext : typing.Union[int, str] = xconf.field(default="mass_limits", help="FITS extension holding the mass limits table")
    obs_date : str = xconf.field(help="Date of observation (ISO UT date for consistency, but anything parseable by dateutil works)")
    companion_abs_mags_colname : str = xconf.field(default="companion_abs_mags", help="Companion absolute magnitudes table column")
    r_au_colname : str = xconf.field(default="r_au", help="Column holding separation in AU")
    pa_deg_colname : typing.Optional[str] = xconf.field(default="pa_deg", help="Column holding position angle in degrees East of North")
    radial_only : bool = xconf.field(default=False, help="Whether to skip loading PAs")
    filter : FilterConfig = xconf.field(help="Specifies filter for synthetic photometry")

    def load(self) -> MassLimits:
        with self.fits_file.open() as fh:
            hdul = iofits.load_fits(fh)
            tbl = hdul[self.masses_ext].data
        companion_abs_mags = tbl[self.companion_abs_mags_colname]
        parsed_obs_date = parse(self.obs_date)
        contrast_pa_deg = None if self.radial_only else tbl[self.pa_deg_colname]
        return MassLimits(
            name=self.name,
            contrast_r_AU=tbl[self.r_au_colname],
            contrast_pa_deg=contrast_pa_deg,
            companion_abs_mags=companion_abs_mags,
            obs_date=parsed_obs_date,
            filter=self.filter.get_spectrum(),
        )

@xconf.config
class Completeness(BaseRayGrid):
    inputs : list[MassLimitsData] = xconf.field(help="All the tables and respective column names to use")
    output_filename : str = xconf.field(default="completeness.fits", help="Output filename to write")
    host : HostStarConfig = xconf.field(help="Host star properties")
    albedo : float = xconf.field(default=0.5, help="Albedo if including irradiation with host.* properties (default: 0.5, approx. like Jupiter)")
    irradiation : bool = xconf.field(help="Include effects of irradiation from host star? (Must supply host star properties)")
    bobcat : BobcatModels = xconf.field(default_factory=BobcatModels, help="Choice of Bobcat model suite")
    grid : CompletenessGrid = xconf.field(default=CompletenessGrid(), help="Specify the grid points to be evaluated")

    def generate_grid(self):
        names = set()
        if self.grid.companion_masses_Mjup == 'infer':
            bobcat = self.bobcat.get_grid()
            companion_masses_Mjup = [float(x.to(u.Mjup).value) for x in bobcat.tabulated_masses]
        else:
            companion_masses_Mjup = [float(x) for x in self.grid.companion_masses_Mjup]
        if self.grid.semimajor_axes_AU == 'infer':
            semimajor_axes_AU = set()
        else:
            semimajor_axes_AU = [float(x) for x in self.grid.semimajor_axes_AU]

        for the_input in self.inputs:
            if the_input.name in names:
                log.error(f"Non-unique mass limits dataset name: {the_input.name}. Current input: {the_input}. Already seen: {names}")
                raise RuntimeError(f"Non-unique mass limits dataset name: {the_input.name}")
            names.add(the_input.name)
            if self.grid.semimajor_axes_AU == 'infer':
                limits_data = the_input.load()
                semimajor_axes_AU |= set([float(x) for x in limits_data.contrast_r_AU])

        n_grid_points = len(companion_masses_Mjup) * len(semimajor_axes_AU)
        dtype = [
            ('index', int),
            ('time_total_sec', float),
            ('companion_mass_Mjup', float),
            ('semimajor_axis_AU', float),
            ('completeness_fraction', float),
        ]
        for name in sorted(list(names)):
            dtype.append((f"completeness_fraction_{name}", float))
        log.info(f"{dtype=}")
        tbl = np.zeros(n_grid_points, dtype=dtype)

        for index, (mass_Mjup, sma_AU) in enumerate(product(companion_masses_Mjup, semimajor_axes_AU)):
            tbl[index]['index'] = index
            tbl[index]['companion_mass_Mjup'] = mass_Mjup
            tbl[index]['semimajor_axis_AU'] = sma_AU
        return tbl

    def compare_grid_to_checkpoint(self, checkpoint_tbl, grid_tbl):
        checkpoint_params = set((r['companion_mass_Mjup'], r['semimajor_axis_AU']) for r in checkpoint_tbl)
        grid_params = set((r['companion_mass_Mjup'], r['semimajor_axis_AU']) for r in grid_tbl)
        return checkpoint_params == grid_params

    def launch_grid(self, pending_tbl) -> list:
        refs = []
        limits_by_name = {}
        for the_input in self.inputs:
            limits = the_input.load()
            limits_by_name[limits.name] = limits
        limits_by_name_ref = ray.put(limits_by_name)
        albedo = self.albedo if self.irradiation else 1.0
        model_grid_ref = ray.put(self.bobcat.get_grid())
        for row in pending_tbl:
            ref = completeness_for_mass_and_sma.remote(
                row,
                model_grid_ref,
                limits_by_name_ref,
                self.host.age_Myr * u.Myr,
                self.host.temp_K * u.K,
                self.host.radius_Rsun * u.Rsun,
                self.host.mass_Msun * u.Msun,
                albedo,
                self.grid.n_samples,
            )
            refs.append(ref)
        return refs
