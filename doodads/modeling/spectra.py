import os.path
import numpy as np
from scipy.interpolate import interp1d, NearestNDInterpolator
from astropy.io import fits
import astropy.units as u
import astropy.constants as c
# from .io import settl_cond
from .units import WAVELENGTH_UNITS, FLUX_UNITS, COMMON_WAVELENGTH

class Spectrum:
    _integrated = None
    def __init__(self, wavelengths, values, name=None):
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

    def multiply(self, other_spectrum):
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


class FITSSpectrum(Spectrum):
    _loaded = False
    def __init__(self, fits_file, ext=1,
        wavelength_column='wavelength', value_column='flux',
        wavelength_units=WAVELENGTH_UNITS, value_units=FLUX_UNITS,
        name=None
    ):
        self.fits_file = fits_file
        if name is None:
            self.name = os.path.basename(fits_file)
        else:
            self.name = name
        self._ext = ext
        self._wavelength_column = wavelength_column
        self._value_column = value_column
        self._wavelength_units = wavelength_units
        self._value_units = value_units
    def _lazy_load(self):
        if self._loaded:
            return
        with open(self.fits_file, 'rb') as f:
            hdul = fits.open(f)
            wavelengths = hdul[self._ext].data[self._wavelength_column] * self._wavelength_units
            values = hdul[self._ext].data[self._value_column] * self._value_units
        super().__init__(wavelengths, values, name=self.name)
    def __getattr__(self, name):
        self._lazy_load()
        return super().__getattribute__(name)

def resample_spectrum(orig_wls, orig_fluxes, new_wls):
    unit = orig_fluxes.unit
    wls = orig_wls.to(new_wls.unit).value
    return interp1d(wls, orig_fluxes.value)(new_wls.value) * unit

def integrate(wavelengths, fluxes, filter_transmission, filter_wavelengths=None):
    if filter_wavelengths is None:
        filter_wavelengths = wavelengths
    if len(filter_transmission) != len(filter_wavelengths):
        raise ValueError(
            f"Filter transmission (shape: {filter_transmission.shape}) "
            f"mismatched with wavelengths (shape: {filter_wavelengths.shape})"
        )
    # wavelength_bins = np.diff(wavelengths)
    # if len(np.unique(wavelength_bins)) == 1:
    #     # if sampling is uniform, make one more bin on the end
    #     wavelength_bins = np.append(wavelength_bins, wavelength_bins[-1])
    # else:
    #     print('Dropping last wavelength with unknown bin width')
    #     wavelengths = wavelengths[:-1]
    #     fluxes = fluxes[:-1]

    # regrid if needed
    if filter_wavelengths is not wavelengths:
        interpolator = interp1d(
            filter_wavelengths.to(wavelengths.unit).value,
            filter_transmission,
            bounds_error=False,
            fill_value=0.0
        )
        filter_transmission = interpolator(wavelengths.value)

    # apply filter transmission at each wavelength
    fluxes = fluxes * filter_transmission  # not *= because we don't want to change in-place

    # numerically integrate
    integrated_flux = np.trapz(fluxes, wavelengths)
    return integrated_flux


def blackbody_flux(wavelength, temperature, radius, distance):
    '''Blackbody flux at `wavelength` scaled by object radius and
    distance
    '''
    return ((
        ((2 * np.pi * c.h * c.c**2) / wavelength**5)
        /
        (np.exp((c.h*c.c) / (wavelength * c.k_B * temperature)) - 1)
    ) * (radius / distance) ** 2).to(FLUX_UNITS)

class Blackbody(Spectrum):
    def __init__(self, temperature, radius, distance):
        flux = blackbody_flux(COMMON_WAVELENGTH, temperature, radius, distance)
        self.temperature = temperature
        super().__init__(COMMON_WAVELENGTH, flux, name=f"B(T={temperature})")

def wien_peak(T):
    return ((2898 * u.um * u.K) / T).to(u.um)

class ModelGrid:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot access model grid at {file_path}")
        self.name = os.path.basename(file_path)
        self.file_path = file_path
        self.hdu_list = fits.open(file_path)
        self.params = self.hdu_list['PARAMS'].data.byteswap().newbyteorder()
        self.param_names = self.params.dtype.fields.keys() - {'index'}
        self.wavelengths = self.hdu_list['WAVELENGTHS'].data.byteswap().newbyteorder()
        self.model_spectra = self.hdu_list['MODEL_SPECTRA'].data.byteswap().newbyteorder()
        self.blackbody_spectra = self.hdu_list['BLACKBODY_SPECTRA'].data.byteswap().newbyteorder()
        params_grid = np.stack([self.params[name] for name in self.param_names]).T
        self._nearest_finder = NearestNDInterpolator(
            params_grid,
            self.params['index'].astype(float),
            rescale=True
        )
    def _add_units(self, wavelengths, model_fluxes, blackbody_fluxes):
        return (
            wavelengths * WL_UNITS,
            model_fluxes * FLUX_UNITS,
            blackbody_fluxes * FLUX_UNITS
        )
    def get(self, T_eff, log_g, M_over_H):
        # TODO make this **kwargs or something so other grids
        # with other params are usable
        matching_params = self.params[
            (self.params['T_eff'] == T_eff) &
            (self.params['log_g'] == log_g) &
            (self.params['M_over_H'] == M_over_H)
        ]
        if len(matching_params) != 1:
            raise ValueError("No matching grid spectrum")
        index = matching_params['index'][0]
        return self._add_units(
            self.wavelengths,
            self.model_spectra[index],
            self.blackbody_spectra[index]
        )
    def _nearest_params(self, T_eff, log_g, M_over_H):
        # TODO make this **kwargs or something so other grids
        # with other params are usable
        index = int(self._nearest_finder(T_eff, log_g, M_over_H))
        return self.params[self.params['index'] == index][0]
    def nearest(self, T_eff, log_g, M_over_H):
        # TODO make this **kwargs or something so other grids
        # with other params are usable
        params = self._nearest_params(T_eff, log_g, M_over_H)
        index = params[index]
        return params, self._add_units(
            self.wavelengths,
            self.model_spectra[index],
            self.blackbody_spectra[index]
        )
