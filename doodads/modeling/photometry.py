import numpy as np
from astropy.io import fits
import astropy.units as u
from .spectra import Spectrum
from .units import WAVELENGTH_UNITS

__all__ = [
    'FilterSet',
    'apparent_mag',
    'absolute_mag',
    'contrast_to_deltamag',
    'deltamag_to_contrast',
]

class FilterSet:
    _table = None
    _names = None
    _standards = None
    def __init__(self, fits_file):
        self.fits_file = fits_file
    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.fits_file)})'
    def _lazy_load(self):
        if self._table is None:
            with open(self.fits_file, 'rb') as f:
                hdul = fits.open(f)
                self._table = hdul[1].data.copy()
                self._names = set(
                    col.name for col in self._table.columns
                    if col.name != 'wavelength'
                )
            self._standards = {}
            for name in self._names:
                spec = Spectrum(self._table['wavelength'] * WAVELENGTH_UNITS, self._table[name] * u.dimensionless_unscaled, name=name)
                setattr(self, name, spec)
    @property
    def names(self):
        self._lazy_load()
        return self._names
    def __getattr__(self, name):
        self._lazy_load()
        return super().__getattribute__(name)


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
