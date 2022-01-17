import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as c
from .units import FLUX_UNITS

__all__ = [
    'mass_surface_gravity_to_radius',
    'blackbody_flux',
    'wien_peak'
]

# def mass_log_g_to_radius(mass, log_g):
#     grav_accel = 10**log_g * u.cm / u.s**2
#     # g = (G M) / r^2
#     # r^2 = (G M) / g
#     # r = sqrt((G M) / g)
#     return np.sqrt((c.G * mass) / grav_accel).si

def mass_surface_gravity_to_radius(mass, surface_gravity):
    # g = (G M) / r^2
    # r^2 = (G M) / g
    # r = sqrt((G M) / g)
    return np.sqrt((c.G * mass) / surface_gravity).si

def blackbody_flux(wavelength, temperature, radius, distance):
    '''Blackbody flux at `wavelength` scaled by object radius and
    distance
    '''
    return ((
        ((2 * np.pi * c.h * c.c**2) / wavelength**5)
        /
        (np.exp((c.h*c.c) / (wavelength * c.k_B * temperature)) - 1)
    ) * (radius / distance) ** 2).to(FLUX_UNITS)

def wien_peak(T):
    return ((2898 * u.um * u.K) / T).to(u.um)

def planet_mass_to_radius_jrmales(planet_mass):
    '''From the original `mxlib documentation <https://jaredmales.github.io/mxlib-doc/group__planets.html>`_

    The goal of this function is to provide a radius given an exoplanet mass,
    for lightly-irradiated exoplanets. By lightly-irradiated we mean (roughly)
    planets at Mercury's separation or further, scaled for stellar luminosity.
    Here we make use of the transition from rocky to gas-dominated composition a
    t 1.6Re identified by Rogers [20] (see also Marcy et al. (2014) [12]).
    Below this radius we assume Earth composition and so R∝M1/3. Above this
    we scale with a power law matched to the mean radius and mass of Uranus
    and Neptune, which defines the radius between 1.63M⊕ and this
    Uranus-Neptune mean point. Above this point we use a polynomial fit
    (in log(M)) to points including the Uranus/Neptune mean, Saturn,
    Jupiter, and above Jupiter's mass the average points from the 4.5 Gyr 1 AU
    models from Fortney et al. (2007) [6]. Above 3591.1 M⊕ ( ∼11Mjup) we
    scale as M−1/8 based on the curve shown in Fortney et al. (2011) [7].
    '''
    planet_mass_M_earth = (planet_mass / c.M_earth).si.value
    if np.isscalar(planet_mass_M_earth):
        planet_mass_M_earth = np.array([planet_mass_M_earth])
        was_scalar = True
    else:
        was_scalar = False

    radius_R_earth = np.zeros_like(planet_mass_M_earth)
    mask_1 = planet_mass_M_earth < 4.1
    radius_R_earth[mask_1] = np.power(planet_mass_M_earth[mask_1], 1/3)

    mask_2 = (4.1 <= planet_mass_M_earth) & (planet_mass_M_earth < 15.84)
    radius_R_earth[mask_2] = 0.62 * np.power(planet_mass_M_earth[mask_2], 0.67)

    mask_3 = (15.84 <= planet_mass_M_earth) & (planet_mass_M_earth < 3591.1)
    radius_R_earth[mask_3] = (
        14.0211 -
        44.8414 * np.log10(planet_mass_M_earth[mask_3]) +
        53.6554 * np.power(np.log10(planet_mass_M_earth[mask_3]), 2) -
        25.3289 * np.power(np.log10(planet_mass_M_earth[mask_3]), 3) +
        5.4920 * np.power(np.log10(planet_mass_M_earth[mask_3]), 4) -
        0.4586 * np.power(np.log10(planet_mass_M_earth[mask_3]), 5)
    )
    mask_4 = planet_mass_M_earth > 3591.1
    radius_R_earth[mask_4] = 32.03 * np.power(planet_mass_M_earth[mask_4], -1/8)
    if was_scalar:
        radius_R_earth = radius_R_earth[0]
    return radius_R_earth * u.R_earth

def equilibrium_temperature(star_effective_temp, star_radius, separation, albedo):
    return (
        (1 - albedo)**(1/4) *
        (star_radius / (2 * separation))**(1/2) *
        star_effective_temp
    )

def make_planet_radius_to_mass_jrmales(M_min=0.01 * u.M_earth, M_max=10_000 * u.M_earth, num=100):
    masses = np.logspace(np.log10(M_min.to(u.M_earth).value), np.log10(M_max.to(u.M_earth).value), num=num) * u.M_earth
    radii = planet_mass_to_radius_jrmales(masses)
    interpolator = interp1d(radii, masses)
    def interp_radius_to_mass(radius):
        radius = radius.to(u.R_earth).value
        return interpolator(radius) * u.M_earth
    return interp_radius_to_mass, (np.min(radii), np.max(radii)), (M_min, M_max)


(planet_radius_to_mass_jrmales,
 planet_radius_to_mass_jrmales_range,
 planet_radius_to_mass_jrmales_domain) = make_planet_radius_to_mass_jrmales()
