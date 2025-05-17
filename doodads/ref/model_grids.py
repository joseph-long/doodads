import os.path
import typing
import logging
from astropy.io import fits
import astropy.units as u
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from ..modeling.units import WAVELENGTH_UNITS, FLUX_UNITS
from ..modeling import spectra, physics
from .. import utils

log = logging.getLogger(__name__)

__all__ = (
    'BoundsError',
    'ModelSpectraGrid',
)

class BoundsError(ValueError):
    pass

class BaseModelGrid(utils.LazyLoadable):
    SAMPLING_UNITS = u.dimensionless_unscaled,
    SPECTRUM_UNITS = u.dimensionless_unscaled
    def __init__(
        self, filepath, params_extname='PARAMS',
        wavelengths_extname='WAVELENGTHS', spectra_extname='MODEL_SPECTRA',
        name=None
    ):
        super().__init__(filepath)
        self.params_extname = params_extname
        self.wavelengths_extname = wavelengths_extname
        self.spectra_extname = spectra_extname
        self.name = os.path.basename(filepath) if name is None else name
        # populated by _lazy_load():
        self.params = None
        self.param_names = None
        self.wavelengths = None
        self.model_spectra = None

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.name}>'

    def __str__(self):
        return self.name

    def _lazy_load(self):
        with open(self.filepath, 'rb') as fh:
            hdu_list = fits.open(fh)
            params = np.asarray(hdu_list[self.params_extname].data.byteswap())
            self.params = params.view(params.dtype.newbyteorder('='))
            self.param_names = self.params.dtype.fields.keys()
            wavelengths = np.asarray(hdu_list[self.wavelengths_extname].data.byteswap())
            self.wavelengths = wavelengths.view(wavelengths.dtype.newbyteorder('=')) * self.SAMPLING_UNITS
            model_spectra = np.asarray(hdu_list[self.spectra_extname].data.byteswap())
            self.model_spectra = model_spectra.view(model_spectra.dtype.newbyteorder('='))

        # some params don't vary in all libraries, exclude those
        # so qhull doesn't refuse to interpolate
        self._real_param_names = set(self.param_names)
        for name in self.param_names:
            if len(np.unique(self.params[name])) == 1:
                self._real_param_names.remove(name)
                log.debug(f'Discarding {name} because all grid points have {name} == {np.unique(self.params[name])[0]}')
        # coerce to sequence because we can't depend on iteration order
        self._real_param_names = list(sorted(self._real_param_names))

        params_grid = np.stack([self.params[name] for name in self._real_param_names]).T
        self._interpolator = LinearNDInterpolator(
            params_grid,
            self.model_spectra,
            rescale=True
        )
    def _interpolate(self, **kwargs):
        # kwargs: all true params required, all incl. non-varying params accepted
        if (
            (not set(self.param_names).issuperset(kwargs.keys()))
            or
            (not all(name in kwargs for name in self._real_param_names))
        ):
            raise ValueError(f"Valid kwargs (from grid params) are {self.param_names}")

        interpolator_args = []
        for name in self._real_param_names:
            interpolator_args.append(kwargs[name])
        result = self._interpolator(*interpolator_args) * self.SPECTRUM_UNITS
        if np.any(np.isnan(result)):
            raise BoundsError(f"Parameters {kwargs} are out of bounds for this model grid with bounds {self.bounds}")
        return result
    @property
    def bounds(self):
        out = {}
        for name in self._real_param_names:
            out[name] = np.min(self.params[name]), np.max(self.params[name])
        return out

class ModelAtmosphereGrid(BaseModelGrid):
    SAMPLING_UNITS = WAVELENGTH_UNITS
    def get(self, airmass=0, pwv=0 * u.mm) -> spectra.Spectrum:
        kwargs = {}
        kwargs['airmass'] = airmass
        kwargs['pwv_mm'] = pwv.to(u.mm).value
        model_trans = self._interpolate(**kwargs)
        model_spec = spectra.Spectrum(
            self.wavelengths.to(WAVELENGTH_UNITS),
            model_trans,
            name=f'{self.name} sec(z)={airmass:3.1f} PWV={pwv.to(u.mm):3.1f}'
        )
        return model_spec

class ModelSpectraGrid(BaseModelGrid):
    SPECTRUM_UNITS = u.W / u.m**3
    SAMPLING_UNITS = u.m
    def __init__(self, *args, magic_scale_factor=1.0, **kwargs):
        self.magic_scale_factor = magic_scale_factor
        super().__init__(*args, **kwargs)

    def get(
        self,
        temperature: u.Quantity,
        surface_gravity: u.Quantity,
        mass: typing.Optional[u.Quantity]=None,
        distance: u.Quantity=10*u.pc,
        radius: typing.Optional[u.Quantity]=None,
        **kwargs: float
    ) -> spectra.Spectrum:
        '''Look up or interpolate a spectrum for given parameters, scaled
        appropriately for mass and distance.

        Parameters
        ----------
        temperature : units.Quantity (temperature)
        surface_gravity : units.Quantity (acceleration)
        mass : units.Quantity or None
            If mass is provided, the appropriate radius can be calculated
            for a given surface gravity and the returned `Spectrum`
            scaled correctly. Otherwise 1 Rsun is used unless `radius`
            is given explicitly.
        distance : units.Quantity or None
            Scales resulting fluxes by 1/distance^2, default 10 pc for
            absolute magnitudes.
        radius : units.Quantity or None
            Supply object radius for (radius/distance)^2 scaling,
            otherwise default to 1 R_sun
        **kwargs : dict[str,float]
            Values for grid parameters listed in the `param_names` attribute.
        '''
        title_parts = [f'T_eff={temperature:3.1f}', f"g={surface_gravity:3.1f}"]
        for name in kwargs:
            if name in self._real_param_names:
                title_parts.append(f"{name}={kwargs[name]:3.1f}")
        kwargs['T_eff_K'] = temperature.to(u.K).value
        kwargs['gravity_m_per_s2'] = surface_gravity.to(u.m / u.s**2).value

        model_fluxes = self._interpolate(**kwargs)
        model_spec = spectra.Spectrum(
            self.wavelengths.to(WAVELENGTH_UNITS),
            model_fluxes.to(FLUX_UNITS),
        )

        if radius is None:
            if mass is not None:
                radius = physics.mass_surface_gravity_to_radius(mass, surface_gravity)
            else:
                radius = 1 * u.Rsun
        scale_factor = self.magic_scale_factor * (
            (radius**2) /
            (distance**2)
        ).si
        model_spec = model_spec.multiply(scale_factor)
        model_spec.name = " ".join(title_parts) + f"\nd={distance} r={radius.to(u.Rjup):3.1f}"
        return model_spec
