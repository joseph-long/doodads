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
    def __init__(self, filters):
        self.filters = filters
        self.name_lookup = {filt.name: filt for filt in self.filters}
        self.names = set(self.name_lookup.keys())
    def __getattr__(self, name):
        if name in self.names:
            return self.name_lookup[name]
    @property
    def exists(self):
        for filt in self.filters:
            if isinstance(filt, utils.LazyLoadable):
                if not filt.exists:
                    return False
        return True
    @utils.supply_argument(ax=plotting.gca)
    def plot_all(self, ax=None):
        for filt in self.filters:
            filt.display(ax=ax)
        ax.legend()
        return ax

def apparent_mag(absolute_mag, d):
    if not d.unit.is_equivalent(u.pc):
        raise ValueError(f"d must be units of distance, got {d.unit}")
    return 5 * np.log10(d / (10 * u.pc)) + absolute_mag

def absolute_mag(apparent_mag, d):
    if not d.unit.is_equivalent(u.pc):
        raise ValueError(f"d must be units of distance, got {d.unit}")
    return apparent_mag - 5 * np.log10(d / (10 * u.pc))

def contrast_to_deltamag(contrast):
    '''contrast as 10^-X to delta magnitude'''
    return -2.5 * np.log10(contrast)

def deltamag_to_contrast(deltamag):
    return np.power(10, deltamag / -2.5)
