from astropy.io import fits
import astropy.units as u
from .spectra import Spectrum, FITSSpectrum
from .units import WAVELENGTH_UNITS
from .io import mko_filters, hst_calspec

class FilterSet:
    _table = None
    _names = None
    _standards = None
    def __init__(self, fits_file, standard_spectrum):
        self.fits_file = fits_file
        self.standard_spectrum = standard_spectrum
    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.fits_file)})'
    def _lazy_load(self):
        if self._table is None:
            with open(self.fits_file, 'rb') as f:
                hdul = fits.open(f)
                self._table = hdul[1].data.copy()
                self._names = set(col.name for col in self._table.columns if col.name != 'wavelength')
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
        if name in self.names:
            return
        return super().__getattribute__(name)


MKO = FilterSet(mko_filters.MKO_FILTERS_FITS)
VEGA = FITSSpectrum(hst_calspec.ALPHA_LYR_FITS, name='Vega')
OLD_VEGA = FITSSpectrum(hst_calspec.OLD_ALPHA_LYR_FITS, name='Vega (old)')
