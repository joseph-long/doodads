try:
    from projecc import DrawOrbits, DanbySolve, KeplerianToCartesian, KeplersConstant
except ImportError:
    import warnings
    warnings.warn("No projecc found, cannot generate random orbits")
import astropy.units as u
import numpy as np

__all__ = ["generate_random_orbit_positions"]

def generate_random_orbit_positions(
    n_samples,
    primary_mass,
    companion_mass,
    sma_log_lower_bound=0,
    sma_log_upper_bound=3,
    nielsen_eccentricity_prior=True,
    draw_lon=True,
    draw_sma=True,
    solver=DanbySolve,
    fixed_sma=100 * u.AU,
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

    Returns
    -------
    true_r : np.ndarray
        True separation from primary (in AU)
    proj_xy : np.ndarray (n_samples, 2)
        projected separation (in AU)
    """

    kep = KeplersConstant(primary_mass, companion_mass)

    sma, ecc, inc, argp, lon, meananom = DrawOrbits(
        n_samples,
        EccNielsenPrior=nielsen_eccentricity_prior,
        DrawLON=draw_lon,
        DrawSMA=draw_sma,
        SMALowerBound=sma_log_lower_bound,
        SMAUpperBound=sma_log_upper_bound,
        FixedSMA=fixed_sma,
    )

    pos, vel, acc = KeplerianToCartesian(
        sma, ecc, inc, argp, lon, meananom, kep, solvefunc=solver
    )
    proj_xy = pos[:,:2]
    true_r = np.linalg.norm(pos, axis=1)

    return true_r, proj_xy
