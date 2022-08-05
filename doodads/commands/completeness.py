import logging
import time
import typing
from itertools import product
import xconf
from xconf.contrib import BaseRayGrid, FileConfig
import ray
import pandas as pd
from astropy.io import fits
import astropy.units as u
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from ..modeling.physics import equilibrium_temperature
from ..ref.bobcat import BOBCAT_EVOLUTION_M0
from ..modeling.orbits import generate_random_orbit_positions

from xpipeline.tasks import characterization, improc
from xpipeline.ref import clio
from .contrast_to_mass import HostStarConfig, BobcatModels
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

def _completeness_for_mass_and_sma(
    row, model_suite, filter_spectrum, contrast_xy_AU, companion_abs_mags,
    age, host_temp, host_radius, host_mass, albedo,
    n_samples
):
    start = time.time()
    out = row.copy()
    companion_mass = row['companion_mass_Mjup'] * u.Mjup
    semimajor_axis = row['semimajor_axis_AU'] * u.AU
    abs_mag_limit_interpolator = LinearNDInterpolator(contrast_xy_AU, companion_abs_mags)
    true_r, proj_xy = generate_random_orbit_positions(
        n_samples, host_mass, companion_mass,
        draw_sma=False, fixed_sma=semimajor_axis
    )
    results = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        results[i] = evaluate_one_point(
            true_r[i], proj_xy[i, 0], proj_xy[i, 1], companion_mass, model_suite,
            abs_mag_limit_interpolator, age, host_temp, host_radius, albedo, filter_spectrum
        )
    out['completeness_fraction'] = np.count_nonzero(results) / n_samples
    out['time_total_sec'] = time.time() - start
    return out

completeness_for_mass_and_sma = ray.remote(_completeness_for_mass_and_sma)

@xconf.config
class CompletenessGrid:
    n_samples : int = xconf.field(default=100, help="Number of samples (random orbits/positions) to draw at each grid point")
    companion_masses_Mjup : list[float] = xconf.field(default_factory=lambda: [1, 10], help="List of companion mass grid points")
    semimajor_axes_AU : list[float] = xconf.field(default_factory=lambda: [1, 2], help="List of semimajor axis grid points")

@xconf.config
class Completeness(BaseRayGrid):
    input : FileConfig = xconf.field(help="FITS table file")
    output_filename : str = xconf.field(default="completeness.fits", help="Output filename to write")
    masses_ext : typing.Union[int, str] = xconf.field(default="mass_limits", help="FITS extension holding the mass limits table")
    companion_abs_mags_colname : str = xconf.field(default="companion_abs_mags", help="Companion absolute magnitudes table column")
    r_au_colname : str = xconf.field(default="r_au", help="Column holding separation in AU (projected)")
    pa_deg_colname : str = xconf.field(default="pa_deg", help="Column holding position angle in degrees East of North")
    host : HostStarConfig = xconf.field(help="Host star properties")
    albedo : float = xconf.field(default=0.5, help="Albedo if including irradiation with host.* properties (default: 0.5, approx. like Jupiter)")
    irradiation : bool = xconf.field(help="Include effects of irradiation from host star? (Must supply host star properties)")
    bobcat : BobcatModels = xconf.field(help="Choice of Bobcat model suite")
    grid : CompletenessGrid = xconf.field(default=CompletenessGrid(), help="Specify the grid points to be evaluated")

    def generate_grid(self):
        n_grid_points = len(self.grid.companion_masses_Mjup) * len(self.grid.semimajor_axes_AU)
        tbl = np.zeros(n_grid_points, dtype=[
            ('index', int),
            ('time_total_sec', float),
            ('companion_mass_Mjup', float),
            ('semimajor_axis_AU', float),
            ('completeness_fraction', float),
        ])
        for index, (mass_Mjup, sma_AU) in enumerate(product(self.grid.companion_masses_Mjup, self.grid.semimajor_axes_AU)):
            tbl[index]['index'] = index
            tbl[index]['companion_mass_Mjup'] = mass_Mjup
            tbl[index]['semimajor_axis_AU'] = sma_AU
        return tbl

    def compare_grid_to_checkpoint(self, checkpoint_tbl, grid_tbl):
        checkpoint_params = set((r['companion_mass_Mjup'], r['semimajor_axis_AU']) for r in checkpoint_tbl)
        grid_params = set((r['companion_mass_Mjup'], r['semimajor_axis_AU']) for r in grid_tbl)
        return checkpoint_params == grid_params

    def launch_grid(self, pending_tbl) -> list:
        with self.input.open(mode='rb') as fh:
            hdul = fits.open(fh, memmap=False)
            tbl = np.asarray(hdul[self.masses_ext].data).byteswap().newbyteorder()
        refs = []
        contrast_xs_AU, contrast_ys_AU = characterization.r_pa_to_x_y(
            tbl[self.r_au_colname],
            tbl[self.pa_deg_colname],
            0,
            0
        )
        contrast_xy_AU = np.stack([contrast_ys_AU, contrast_xs_AU], axis=-1)
        companion_abs_mags = tbl[self.companion_abs_mags_colname]
        albedo = self.albedo if self.irradiation else 1.0
        model_grid_ref = ray.put(self.bobcat.get_grid())
        filter_spectrum_ref = ray.put(self.bobcat.filter.get_spectrum())
        for row in pending_tbl:
            ref = completeness_for_mass_and_sma.remote(
                row,
                model_grid_ref,
                filter_spectrum_ref,
                contrast_xy_AU,
                companion_abs_mags,
                self.host.age_Myr * u.Myr,
                self.host.temp_K * u.K,
                self.host.radius_Rsun * u.Rsun,
                self.host.mass_Msun * u.Msun,
                albedo,
                self.grid.n_samples,
            )
            refs.append(ref)
        return refs
