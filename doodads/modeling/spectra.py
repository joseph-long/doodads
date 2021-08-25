import os.path
import logging
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import astropy.units as u
import astropy.constants as c
from astropy.convolution import convolve, Gaussian1DKernel
from . import physics
from .units import WAVELENGTH_UNITS, FLUX_UNITS, COMMON_WAVELENGTH
from ..utils import supply_argument
from .. import utils
from ..plotting import gca

__all__ = [
    'Spectrum',
    'FITSSpectrum',
    'Blackbody'
]

log = logging.getLogger(__name__)

class Spectrum:
    '''Discretized spectral distribution (of flux or unitless transmission)
    '''
    _integrated = None
    def __init__(self, wavelengths, values, name=None):
        '''
        Parameters
        ----------

        wavelengths : 1D `astropy.units.Quantity` with length units
            Wavelengths to which the values correspond
        values : 1D `numpy.ndarray` or 1D `astropy.units.Quantity`
            Values (flux or transmission) at specified wavelengths
            (If passed as ndarray, coerced to `astropy.units.dimensionless_unscaled`.)
        '''
        self.name = name
        self.wavelengths = wavelengths
        if isinstance(values, u.Quantity):
            self.values = values
        else:
            self.values = values * u.dimensionless_unscaled
            if np.any((self.values > 1) | (self.values < 0)):
                raise ValueError("Dimensionless spectra must be scaled to [0,1]")
    def __repr__(self):
        out = f'<Spectrum: '
        if self.name is not None:
            out += f'{self.name} '
        wl_min, wl_max = np.min(self.wavelengths.value), np.max(self.wavelengths.value)
        out += '{:0.3} to {:0.3} {}>'.format(wl_min, wl_max, self.wavelengths.unit)
        return out
    @supply_argument(ax=gca)
    def display(self, ax=None, wavelength_unit=None, value_unit=None, **kwargs):
        '''
        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            The axes into which the spectrum should be plotted.
            If omitted, `matplotlib.pyplot.gca` is called to get
            or create current axes.
        wavelength_unit : `astropy.units.Unit`
            Supply to plot spectra with wavelength units other
            than those stored in `self.wavelengths` (must be
            compatible)
        value_unit : `astropy.units.Unit`
            Supply to plot spectra with value units other
            than those stored in `self.values` (must be
            compatible)
        **kwargs
            Additional arguments passed through to `Axes.plot()`
        '''
        if wavelength_unit is None:
            wavelength_unit = self.wavelengths.unit
        if value_unit is None:
            value_unit = self.values.unit
        kind = 'Flux' if value_unit is not u.dimensionless_unscaled else 'Transmission'
        ax.set(
            xlabel=f'Wavelength [{wavelength_unit}]',
            ylabel=f'{kind} [{value_unit}]',
        )
        ax.plot(
            self.wavelengths.to(wavelength_unit),
            self.values.to(value_unit),
            label=self.name,
            **kwargs
        )
        return ax
    _ipython_display_ = display
    def resample(self, new_wavelengths):
        '''Return a resampled version of this spectrum on a wavelength
        grid given by `new_wavelengths`. Non-overlapping regions
        will be filled with zeros.'''
        unit = self.values.unit
        wls = self.wavelengths.to(new_wavelengths.unit).value
        new_vals = interp1d(
            wls,
            self.values.value,
            bounds_error=False,
            fill_value=0
        )(new_wavelengths.value) * unit
        return Spectrum(new_wavelengths.copy(), new_vals)

    def normalize(self, integrated_value):
        current = self.integrate()
        new_values = ((integrated_value / current) * (self.values)).to(self.values.unit)
        return Spectrum(self.wavelengths, new_values)

    def smooth(self, kernel_argument=1, kernel=Gaussian1DKernel):
        spec = Spectrum(
            self.wavelengths,
            convolve(self.values.value, kernel(kernel_argument)) * self.values.unit
        )
        return spec

    def multiply(self, other_spectrum_or_scalar):
        # n.b. Quantity objects don't behave with np.isscalar, but have .isscalar attributes
        # so we check for those first, falling back to np.isscalar
        if isinstance(other_spectrum_or_scalar, u.Quantity):
            if not other_spectrum_or_scalar.isscalar:
                raise ValueError("Can only multiply by scalar unitful Quantities")
            is_scalar = True
        elif np.isscalar(other_spectrum_or_scalar):
            is_scalar = True
        else:
            is_scalar = False

        if is_scalar:
            scale_value = other_spectrum_or_scalar
            new_values = self.values * scale_value
        else:
            other_spectrum = other_spectrum_or_scalar
            other_spectrum_interp = other_spectrum.resample(self.wavelengths)
            # We can't multiply fluxes and fluxes, only transmissions and fluxes
            if self.values.unit is not u.dimensionless_unscaled:
                if other_spectrum_interp.values.unit is not u.dimensionless_unscaled:
                    raise ValueError(f"Can't multiply {self.values.unit} (self) and {other_spectrum_interp.values.unit} (other)")
            new_values = self.values * other_spectrum_interp.values
        return Spectrum(self.wavelengths, new_values)

    def integrate(self):
        if self._integrated is None:
            self._integrated = np.trapz(self.values, self.wavelengths)
        return self._integrated

    def magnitude(self, other_spectrum, filter_spectrum=None, m_1=0.0):
        r'''Integrate `self` and `other_spectrum` and compute an
        astronomical magnitude from their flux ratios. If
        `filter_spectrum` is provided, it is applied to this instance
        and `other_spectrum` before integration.

        Magnitudes are a logarithmic unit based on the flux ratio
        of two sources. Refresh your memory with Carroll & Ostlie
        Chapter 3:

        .. math:: \frac{F_2}{F_1} = 100^{(m_1 - m_2)/5}

        (C&O 3.3)

        When calling `magnitude()`, the instance on which it is called
        serves as :math:`F_1`. :math:`m_1` is given by the argument
        `m_1`, defaulting to 0.0. (In other words, the instance on
        which you call `magnitude` defines the zero-point.
        e.g. `VEGA.multiply(mko_filters.K).magnitude(model.multiply(mko_filters.K))`)

        .. math::

            m_1 - m_2 = -2.5 \log_{10} \left(\frac{F_1}{F_2}\right)

            m_2 - m_1 = 2.5 \log_{10} \left(\frac{F_1}{F_2}\right)

            m_2 = 2.5 \log_{10} \left(\frac{F_1}{F_2}\right) + m_1

        (C&O 3.4)

        '''
        if self.values.unit is u.dimensionless_unscaled:
            raise ValueError("Attempting to compute magnitude with dimensionless spectrum")
        if not other_spectrum.values.unit.is_equivalent(self.values.unit):
            raise ValueError("Incompatible units {self.values.unit} (self) and {other_spectrum.values.unit} (other)")

        if filter_spectrum is not None:
            if filter_spectrum.values.unit is not u.dimensionless_unscaled:
                raise ValueError("Filter transmission must be dimensionless")
            ref_spectrum = self.multiply(filter_spectrum)
            other_spectrum = other_spectrum.multiply(filter_spectrum)
        else:
            ref_spectrum = self
        apparent_mag = 2.5 * np.log10(ref_spectrum.integrate() / other_spectrum.integrate()) + m_1
        return apparent_mag


class FITSSpectrum(utils.LazyLoadable, Spectrum):
    _lazy_attr_allowlist = ('name',) + utils.LazyLoadable._lazy_attr_allowlist
    def __init__(
        self, filepath, ext=1,
        wavelength_column='wavelength', value_column='flux',
        wavelength_units=WAVELENGTH_UNITS, value_units=FLUX_UNITS,
        name=None
    ):
        if name is None:
            self.name = os.path.basename(filepath)
        else:
            self.name = name
        self._ext = ext
        self._wavelength_column = wavelength_column
        self._value_column = value_column
        self._wavelength_units = wavelength_units
        self._value_units = value_units
        utils.LazyLoadable.__init__(self, filepath)
    def _lazy_load(self):
        with open(self.filepath, 'rb') as f:
            hdul = fits.open(f)
            wavelengths = hdul[self._ext].data[self._wavelength_column] * self._wavelength_units
            values = hdul[self._ext].data[self._value_column] * self._value_units
        Spectrum.__init__(self, wavelengths, values, name=self.name)

class TableSpectrum(utils.LazyLoadable, Spectrum):
    def __init__(self, filepath, wavelength_units, value_units, name=None):
        self._wavelength_units = wavelength_units
        self._value_units = value_units
        if name is not None:
            self.name = name
        else:
            self.name = os.path.basename(filepath)
        utils.LazyLoadable.__init__(self, filepath)
    def _lazy_load(self):
        wls, trans = np.genfromtxt(self.filepath, unpack=True)
        Spectrum.__init__(self, wls * self._wavelength_units, trans * self._value_units, name=self.name)

class Blackbody(Spectrum):
    '''Discretized blackbody flux on `wavelengths` grid
    (default: `COMMON_WAVELENGTH`)'''
    def __init__(self, temperature, radius, distance, wavelengths=COMMON_WAVELENGTH):
        flux = physics.blackbody_flux(wavelengths, temperature, radius, distance)
        self.temperature = temperature
        super().__init__(wavelengths, flux, name=f"B(T={temperature}, r={radius}, d={distance})")
