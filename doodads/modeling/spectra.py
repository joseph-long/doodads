import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as c
from .io import settl_cond

WAVELENGTH_UNITS = u.m
FLUX_UNITS = u.W * u.m**-3

def integrate(wavelengths, fluxes, filter_transmission, filter_wavelengths=None):
    if filter_wavelengths is None:
        filter_wavelengths = wavelengths
    if len(filter_transmission) != len(filter_wavelengths):
        raise ValueError(
            f"Filter transmission (shape: {filter_transmission.shape}) "
            f"mismatched with wavelengths (shape: {filter_wavelengths.shape})"
        )
    wavelength_bins = np.diff(wavelengths)
    if len(np.unique(wavelength_bins)) == 1:
        np.append(wavelength_bins, wavelength_bins[-1])
    else:
        print('Dropping last wavelength with unknown bin width')
        wavelengths = wavelengths[:-1]
        fluxes = fluxes[:-1]

    # regrid if needed
    if filter_wavelengths is not wavelengths:
        # force to zero at ends of input
        min_wave_idx, max_wave_idx = np.argmin(filter_wavelengths), np.argmax(filter_wavelengths)
        filter_transmission[min_wave_idx] = 0
        filter_transmission[max_wave_idx] = 0
        filter_transmission = interp1d(filter_wavelengths, filter_transmission, bounds_error=False, fill_value=0.0)(wavelengths)

    # apply filter transmission at each wavelength
    fluxes *= filter_transmission

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

AMES_COND = ModelGrid(settl_cond.AMES_COND_FITS) if os.path.exists(settl_cond.AMES_COND_FITS) else None
BT_SETTL = ModelGrid(settl_cond.BT_SETTL_CIFIST2011C_FITS) if os.path.exists(settl_cond.BT_SETTL_CIFIST2011C_FITS) else None
