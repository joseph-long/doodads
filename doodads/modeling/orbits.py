try:
    import projecc
except ImportError:
    class YellingProxy:
        def __init__(self, package):
            self.package = package

        def __getattr__(self, name: str):
            raise AttributeError(
                f"Package {self.package} is not installed (or failed to import)"
            )
    projecc = YellingProxy('projecc')

import astropy.units as u
import numpy as np

__all__ = [
    "generate_random_orbit_positions",
    "period",
]

def period(semi_major_axis: u.Quantity, mass: u.Quantity) -> u.Quantity:
    """Given unitful semi-major axis values/arrays and masses,
    return period in years using Kepler's 3rd law"""
    a = semi_major_axis.to(u.AU).value
    m = mass.to(u.Msun).value
    P = np.sqrt((a**3)/m) * u.year
    return P

def generate_random_orbit_positions(
    n_samples,
    primary_mass,
    companion_mass,
    sma_log_lower_bound=0,
    sma_log_upper_bound=3,
    nielsen_eccentricity_prior=True,
    draw_lon=True,
    draw_sma=True,
    solver=None,
    fixed_sma=100 * u.AU,
    epochs=None
):
    """Generate a set of n_samples simulated companions and return
    their current separation and position angle in the plane of the sky.

    Parameters
    ----------
    n_samples : int
        number of simulated companions to generate
    primary_mass : astropy.units.Quantity[mass]
        Mass of primary
    companion_mass : astropy.units.Quantity[mass]
        Mass of simulated companion
    nielsen_eccentricity_prior : bool (default: True)
        If True, draw eccentricity from a linearly descending prior given in Nielsen+ 2019,
        if False draw from uniform prior on [0,1]
    draw_lon : bool (default: True)
        If True, draw longitude of nodes from a uniform distribution
        between [0, 360] degrees.  If False all LON values will be zero.
    draw_sma : bool (default: True)
        If True, draw semi-major axis from log-linear prior. If False,
        all semi-major axes will be 100 AU.
    sma_log_lower_bound : float (default: 0.0)
    sma_log_upper_bound : float (default: 3.0)
        Draw semi-major axis from log-linear prior between these
        two bounds
    solver : callable (default: projecc.DanbySolve)
        Function to use for solving for eccentricity anomaly
    fixed_sma : astropy.units.Quantity[distance]
        If draw_sma is False, supply a value of SMA as an astropy unit object
    epochs : u.Quantity
        Observation times (fractional years relative to arbitrary epoch) at which to
        evaluate positions, or None for a single observation (the default)

    Returns
    -------
    true_r : np.ndarray
        True separation from primary (in AU)
    proj_xy : np.ndarray (n_samples, 2)
        projected separation (in AU)
    """
    if solver is None:
        solver = projecc.DanbySolve

    kep = projecc.KeplersConstant(primary_mass, companion_mass)

    sma, ecc, inc, argp, lon, meananom = projecc.DrawOrbits(
        n_samples,
        EccNielsenPrior=nielsen_eccentricity_prior,
        DrawLON=draw_lon,
        DrawSMA=draw_sma,
        SMALowerBound=sma_log_lower_bound,
        SMAUpperBound=sma_log_upper_bound,
        FixedSMA=fixed_sma,
    )

    if epochs is not None:
        all_proj_xy = None
        all_true_r = None
        pds = period(sma * u.AU, primary_mass).to(u.year).value
        # meananom is radians in [0, 2pi], 0 at time of periastron passage
        for epoch in epochs:
            # (meananom + 2pi(∆t/period)) % 2pi
            adj_mean_anomaly = (meananom + (2 * np.pi * epoch.to(u.year).value)/pds) % (2 * np.pi)
            pos, vel, acc = projecc.KeplerianToCartesian(
                sma, ecc, inc, argp, lon, adj_mean_anomaly, kep, solvefunc=solver
            )
            proj_xy = pos[:,:2]
            true_r = np.linalg.norm(pos, axis=1)
            if all_proj_xy is None:
                all_proj_xy = proj_xy[np.newaxis, :, :]
                all_true_r = true_r[np.newaxis, :]
            else:
                all_proj_xy = np.concatenate([all_proj_xy, proj_xy[np.newaxis, :, :]])
                all_true_r = np.concatenate([all_true_r, true_r[np.newaxis, :]])
        return all_true_r, all_proj_xy
    else:
        pos, vel, acc = projecc.KeplerianToCartesian(
            sma, ecc, inc, argp, lon, meananom, kep, solvefunc=solver
        )

        proj_xy = pos[:,:2]
        true_r = np.linalg.norm(pos, axis=1)
        return true_r, proj_xy
