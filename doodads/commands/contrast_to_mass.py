import os
import typing
import logging
from pprint import pformat
import numpy as np
import astropy.units as u
import xconf
from ..modeling.photometry import contrast_to_deltamag, absolute_mag
from enum import Enum
from ..ref import mko_filters, clio, sphere, gemini_atmospheres, magellan_atmospheres
from ..ref import bobcat
log = logging.getLogger(__name__)


@xconf.config
class IrradiationConfig:
    host_temp_K : float = xconf.field(help="Host temperature (T_eff) in Kelvin")
    host_radius_R_sun : float = xconf.field(help="Host star radius (R_*) in solar radii")
    albedo : float = xconf.field(default=0.5, help="(default: 0.5, approx. like Jupiter)")

class BobcatMetallicities(Enum):
    minus0_5 = '-0.5'
    zero = '+0.0'
    plus0_5 = '+0.5'

class FilterSetChoices(Enum):
    MKO = 'MKO'
    IRDIS = 'IRDIS'
    CLIO = 'CLIO'

@xconf.config
class FilterConfig:
    set : FilterSetChoices = xconf.field(help=f"Filter set / instrument")
    name : str = xconf.field(help="Name from filter set")
    def __post_init__(self):
        if self.set is FilterSetChoices.MKO:
            self._filter_spectrum = getattr(mko_filters.MKO, self.name)
        elif self.set is FilterSetChoices.IRDIS:
            self.filter_spectrum = getattr(sphere.IRDIS, self.name)
        elif self.set is FilterSetChoices.CLIO:
            self.filter_spectrum = getattr(clio.CLIO, self.name)
        else:
            raise RuntimeError(f"Unsupported filter specification: {self.set.name}.{self.name}")

    def get_spectrum(self):
        return self._filter_spectrum

@xconf.config
class BobcatModels:
    filter : FilterConfig = xconf.field(help="Specifies filter for synthetic photometry")
    metallicity : BobcatMetallicities = xconf.field(default=BobcatMetallicities.zero, help="Metallicity, [M/H] = 0.0 for solar")

    def get_grid(self):
        if self.metallicity is not BobcatMetallicities.zero:
            raise RuntimeError("Not implemented yet")
        return bobcat.BOBCAT_EVOLUTION_M0

class AtmosphericAbsorptionModel(Enum):
    GEMINI_SOUTH = 'gemini_south'
    GEMINI_NORTH = 'gemini_north'
    MAGAOX = 'magaox'

@xconf.config
class AtmosphereModel:
    model : AtmosphericAbsorptionModel = xconf.field(default=AtmosphericAbsorptionModel.GEMINI_SOUTH)
    pwv_mm : float = xconf.field(default=None, help="Precipitable water vapor in millimeters, omit to choose minimum PWV available in model")
    airmass : float = xconf.field(default=1.0, help="Airmass (i.e. sec(z)) at which absorption is modeled")

    def __post_init__(self):
        if self.model is AtmosphericAbsorptionModel.GEMINI_SOUTH:
            self._model_grid = gemini_atmospheres.GEMINI_SOUTH_ATMOSPHERES
        elif self.model is AtmosphericAbsorptionModel.GEMINI_NORTH:
            self._model_grid = gemini_atmospheres.GEMINI_NORTH_ATMOSPHERES
        elif self.model is AtmosphericAbsorptionModel.MAGAOX:
            self._model_grid = magellan_atmospheres.MANQUI_ATMOSPHERES
        if self.pwv_mm is None:
            self.pwv_mm = self._model_grid.bounds['pwv_mm'][0]

    def get_spectrum(self, airmass, pwv):
        return self._model_grid.get(airmass, pwv)

@xconf.config
class ContrastToMass(xconf.Command):
    input : str = xconf.field(help="FITS table file")
    destination : str = xconf.field(default=".", help="Where to write output files")
    table_ext : typing.Union[int, str] = xconf.field(default="limits", help="FITS extension holding the contrast table")
    contrast_colname : str = xconf.field(default="contrast_limit_5sigma", help="Column holding contrast in (companion/host) ratio")
    r_as_colname : str = xconf.field(default="r_as", help="Column holding separation in arcseconds")
    distance_pc : float = xconf.field(help="Distance to host star in parsecs")
    irradiation : typing.Optional[IrradiationConfig] = xconf.field(help="How to calculate equilibrium temperatures and irradiation effects")
    host_apparent_mag : float = xconf.field(help="Stellar magnitude of host in bandpass used for these observations")
    host_age_Myr : float = xconf.field(help="Host star age in megayears")
    bobcat : BobcatModels = xconf.field()

    def main(self):
        from astropy.io import fits
        import astropy.units as u
        from ..modeling.astrometry import arcsec_to_au
        import pandas as pd
        hdul = fits.open(self.input)
        name = os.path.basename(self.input)
        output = os.path.join(self.destination, name.replace('.fits', '_masses.fits'))
        if os.path.exists(output):
            log.error(f"Output file {output} exists")
            return
        evolution_grid = self.bobcat.get_grid()

        df = pd.DataFrame(hdul[self.table_ext].data)
        contrasts = df[self.contrast_colname]
        separations = np.array(df[self.r_as_colname]) * u.arcsec
        distances = arcsec_to_au(separations, self.distance_pc * u.pc)
        df['r_au_in_projection'] = distances.to(u.AU).value

        if self.irradiation is not None:
            from ..modeling.physics import equilibrium_temperature
            eq_temps = equilibrium_temperature(
                self.irradiation.host_temp_K * u.K,
                self.irradiation.host_radius_R_sun * u.R_sun,
                distances,
                self.irradiation.albedo
            )
            df['eq_temp_K'] = eq_temps.to(u.K).value
        else:
            eq_temps = None
            df['eq_temp_K'] = 0

        df['companion_abs_mags'] = absolute_mag(self.host_apparent_mag + contrast_to_deltamag(contrasts), self.distance_pc * u.pc)

        masses, too_faint, too_bright, excluded_mass_ranges = evolution_grid.magnitude_age_to_mass(
            df['companion_abs_mags'],
            self.host_age_Myr * u.Myr,
            self.bobcat.filter.get_spectrum(),
            T_eq=eq_temps
        )
        df['bobcat_mass_mjup'] = masses.to(u.Mjup).value
        df['bobcat_too_bright'] = too_bright
        df['bobcat_too_faint'] = too_faint
        df['bobcat_has_exclusions'] = np.array([len(excl) > 0 for excl in excluded_mass_ranges])

        tbl = df.to_records(index=False)
        fits.BinTableHDU(tbl, name="masses").writeto(output, overwrite=True)
        log.info("Finished saving to " + output)
