import numpy as np
from astropy.io import fits
import astropy.units as u
from .spectra import Spectrum
from .units import WAVELENGTH_UNITS
from .. import utils, plotting

__all__ = [
    'FilterSet',
    'apparent_mag',
    'absolute_mag',
    'contrast_to_deltamag',
    'deltamag_to_contrast',
]

class FilterSet:
    def __init__(self, filters: dict[str, Spectrum]):
        self.filters = filters
        self.names = set(self.filters.keys())
    def __repr__(self):
        return f'<FilterSet: {self.names}>'
    def __getattr__(self, name):
        if name in self.names:
            return self.filters[name]
        raise AttributeError(f"No filter {name} in {self}")
    def __getitem__(self, key):
        if key in self.names:
            return self.filters[key]
        raise KeyError(f"No filter {key} in {self}")


    def __iter__(self):
        for filt in self.filters.values():
            yield filt

    @property
    def exists(self):
        for filtname in self.filters:
            if isinstance(self.filters[filtname], utils.LazyLoadable):
                if not self.filters[filtname].exists:
                    return False
        return True
    @utils.supply_argument(ax=plotting.gca)
    def plot_all(self, ax=None):
        min_wl, max_wl = np.inf * u.m, 0
        for filtname in self.filters:
            filt = self.filters[filtname]
            filt.display(ax=ax)
            if filt.wavelengths.min() < min_wl:
                min_wl = filt.wavelengths.min()
            if filt.wavelengths.max() > max_wl:
                max_wl = filt.wavelengths.max()
        ax.set(xlim=(min_wl, max_wl))
        ax.legend()
        return ax

def apparent_mag(absolute_mag, d):
    '''Scale an `absolute_mag` to an apparent magnitude using
    the distance modulus for `d`
    '''
    if not d.unit.is_equivalent(u.pc):
        raise ValueError(f"d must be units of distance, got {d.unit}")
    return 5 * np.log10(d / (10 * u.pc)) + absolute_mag

def absolute_mag(apparent_mag, d):
    '''Scale an `apparent_mag` at distance `d` to `d` = 10 pc
    '''
    if not d.unit.is_equivalent(u.pc):
        raise ValueError(f"d must be units of distance, got {d.unit}")
    return apparent_mag - 5 * np.log10(d / (10 * u.pc))

def contrast_to_deltamag(contrast):
    '''contrast as :math:`10^{-X}` to delta magnitude'''
    return -2.5 * np.log10(contrast)

def deltamag_to_contrast(deltamag):
    '''delta mag as an flux ratio'''
    return np.power(10, deltamag / -2.5)

def stddev_to_mag_err(value, stddev):
    # f = a log_10 (b * A)
    # sigma_f^2 = (a * sigma_A / (A * ln(10)))^2
    # sigma_f = |a * sigma_A / (A * ln(10))|
    # for magnitudes a = -2.5, b = 1
    return np.abs(-2.5 * stddev / (value * np.log(10)))
