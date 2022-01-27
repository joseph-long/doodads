import os.path
import typing
import logging
from astropy.io import fits
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator
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

class ModelSpectraGrid(utils.LazyLoadable):
    FLUX_UNITS = u.W / u.m**3
    WL_UNITS = u.m
    def __init__(self, filepath, magic_scale_factor=1.0):
        super().__init__(filepath)
        self.name = os.path.basename(filepath)
        self.magic_scale_factor = magic_scale_factor
        # populated by _lazy_load():
        self.hdu_list = None
        self.params = None
        self.param_names = None
        self.wavelengths = None
        self.model_spectra = None

    def _lazy_load(self):
        self.hdu_list = fits.open(self.filepath)
        self.params = np.asarray(self.hdu_list['PARAMS'].data)
        self.param_names = self.params.dtype.fields.keys() - {'index'}
        self.wavelengths = self.hdu_list['WAVELENGTHS'].data
        self.model_spectra = self.hdu_list['MODEL_SPECTRA'].data

        # some params don't vary in all libraries, exclude those
        # so qhull doesn't refuse to interpolate
        self._real_param_names = self.param_names.copy()
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
    @property
    def bounds(self):
        out = {}
        for name in self._real_param_names:
            out[name] = np.min(self.params[name]), np.max(self.params[name])
        return out

    def _args_to_params(self, temperature, surface_gravity, extra_args):
        extra_args = extra_args.copy()
        extra_args['T_eff_K'] = temperature.to(u.K).value
        extra_args['gravity_m_per_s2'] = surface_gravity.to(u.m / u.s**2)
        return extra_args

    def get(
        self,
        temperature: u.Quantity,
        surface_gravity: u.Quantity,
        mass: typing.Optional[u.Quantity]=None,
        distance: u.Quantity=10*u.pc,
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
            scaled correctly.
        distance : units.Quantity or None
            Scales resulting fluxes by 1/distance^2, default 10 pc for
            absolute magnitudes.
        **kwargs : dict[str,float]
            Values for grid parameters listed in the `param_names` attribute.
        '''
        kwargs = self._args_to_params(temperature, surface_gravity, kwargs)
        # kwargs: all true params required, all incl. non-varying params accepted
        if (
            (not self.param_names.issuperset(kwargs.keys()))
            or
            (not all(name in kwargs for name in self._real_param_names))
        ):
            raise ValueError(f"Valid kwargs (from grid params) are {self.param_names}")

        interpolator_args = []
        title_parts = []
        for name in self._real_param_names:
            interpolator_args.append(kwargs[name])
            title_parts.append(f"{name}={kwargs[name]}")

        model_fluxes = self._interpolator(*interpolator_args) * self.FLUX_UNITS
        if np.any(np.isnan(model_fluxes)):
            raise BoundsError(f"Parameters {kwargs} are out of bounds for this model grid")
        wl = self.wavelengths * self.WL_UNITS
        model_spec = spectra.Spectrum(
            wl.to(WAVELENGTH_UNITS),
            model_fluxes.to(FLUX_UNITS),
            name=" ".join(title_parts)
        )

        if mass is not None:
            radius = physics.mass_surface_gravity_to_radius(mass, surface_gravity)
        else:
            radius = 1 * u.Rsun
        scale_factor = self.magic_scale_factor * (
            (radius**2) /
            (distance**2)
        ).si

        model_spec = model_spec.multiply(scale_factor)

        return model_spec
