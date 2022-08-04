import numpy as np
from tqdm import tqdm
import os
import typing
import logging
from pprint import pformat
import astropy.units as u
import xconf
import ray
from ..modeling.spectra import Spectrum
from ..modeling.photometry import contrast_to_deltamag, absolute_mag
from enum import Enum
from ..ref import mko_filters, clio, sphere, gemini_atmospheres, magellan_atmospheres, eso_atmospheres
from ..ref import bobcat
from .. import utils
log = logging.getLogger(__name__)

@xconf.config
class HostStarConfig:
    temp_K : float = xconf.field(help="Host temperature (T_eff) in Kelvin")
    radius_Rsun : float = xconf.field(help="Host star radius (R_*) in solar radii")
    mass_Msun : float = xconf.field(help="Host star mass (M_*) in solar masses")
    apparent_mag : float = xconf.field(help="Stellar magnitude of host in bandpass used for these observations")
    age_Myr : float = xconf.field(help="Host star age in megayears")
    distance_pc : float = xconf.field(help="Distance from observer to target in parsecs")

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
        return self.filter_spectrum

@xconf.config
class BobcatModels:
    filter : FilterConfig = xconf.field(help="Specifies filter for synthetic photometry")
    metallicity : float = xconf.field(default=0.0, help="Metallicity, [M/H] = 0.0 for solar")

    def get_grid(self):
        if self.metallicity == 0.0:
            return bobcat.BOBCAT_EVOLUTION_M0
        elif self.metallicity == -0.5:
            return bobcat.BOBCAT_EVOLUTION_Mminus0_5
        elif self.metallicity == 0.5:
            return bobcat.BOBCAT_EVOLUTION_Mplus0_5
        else:
            raise ValueError("Unsupported metallicity value, valid values are -0.5, 0.0, +0.5")

class AtmosphericAbsorptionModel(Enum):
    GEMINI_SOUTH = 'gemini south'
    GEMINI_NORTH = 'gemini north'
    MANQUI = 'manqui'
    LA_SILLA = 'la silla'

    @classmethod
    def get_model_grid(cls, val):
        if val is cls.GEMINI_NORTH:
            return gemini_atmospheres.GEMINI_NORTH_ATMOSPHERES
        elif val is cls.GEMINI_SOUTH:
            return gemini_atmospheres.GEMINI_SOUTH_ATMOSPHERES
        elif val is cls.MANQUI:
            return magellan_atmospheres.MANQUI_ATMOSPHERES
        elif val is cls.LA_SILLA:
            return eso_atmospheres.LA_SILLA_ATMOSPHERES
        else:
            raise ValueError(f"Unknown enum value {val}")

@xconf.config
class AtmosphereModel:
    model : AtmosphericAbsorptionModel = xconf.field(default=AtmosphericAbsorptionModel.LA_SILLA)
    pwv_mm : typing.Optional[float] = xconf.field(default=None, help="Precipitable water vapor in millimeters, omit to choose minimum PWV available in model")
    airmass : float = xconf.field(default=1.0, help="Airmass (i.e. sec(z)) at which absorption is modeled")

    def __post_init__(self):
        self._model_grid = AtmosphericAbsorptionModel.get_model_grid(self.model)

    def get_spectrum(self, airmass, pwv):
        if self.pwv_mm is None:  # initialized late so it doesn't trigger lazy loading
            self.pwv_mm = self._model_grid.bounds['pwv_mm'][0]
        return self._model_grid.get(airmass, pwv)

@ray.remote
def process_point(
    index: int,
    evolution_grid: bobcat.BobcatEvolutionModel,
    companion_abs_mag: float, host_age_Myr: float, filter_spectrum: Spectrum,
    eq_temp,
):
    mass, this_too_faint, this_too_bright, excluded_mass_ranges = evolution_grid.magnitude_age_to_mass(
        companion_abs_mag,
        host_age_Myr * u.Myr,
        filter_spectrum,
        T_eq=eq_temp,
    )
    log.debug(f"{index=} {mass=} {excluded_mass_ranges=}")
    this_has_exclusions = False
    if len(excluded_mass_ranges):
        for mass_range in excluded_mass_ranges:
            # mass range from min_x (Mjup) to max_x (Mjup) gets as faint as extremum_y (mag) before picking back up,
            # so if the limiting companion_abs_mag > extremum_y, it's fainter than any excluded value
            # and we're sensitive to all masses down to that corresponding to companion_abs_mag
            if companion_abs_mag < mass_range.extremum_y:
                this_has_exclusions = True
    return index, mass, this_too_bright, this_too_faint, this_has_exclusions

@xconf.config
class ContrastToMass(xconf.Command):
    input : str = xconf.field(help="FITS table file")
    destination : str = xconf.field(default=".", help="Where to write output files")
    limits_ext : typing.Union[int, str] = xconf.field(default="limits", help="FITS extension holding the contrast limits table")
    detection_ext : typing.Union[int, str] = xconf.field(default="detection", help="FITS extension holding the detection table")
    contrast_colname : str = xconf.field(default="contrast_limit_5sigma", help="Column holding contrast in (companion/host) ratio")
    signal_colname : str = xconf.field(default="signal", help="Column holding contrast in (companion/host) ratio")
    r_as_colname : str = xconf.field(default="r_as", help="Column holding separation in arcseconds")
    irradiation : bool = xconf.field(default=False, help="Include effects of irradiation from host star? (Must supply host star properties)")
    host : HostStarConfig = xconf.field(help="Host star properties")
    albedo : float = xconf.field(default=0.5, help="Albedo if including irradiation with host.* properties (default: 0.5, approx. like Jupiter)")
    bobcat : BobcatModels = xconf.field()
    atmosphere : AtmosphereModel = xconf.field(
        default=AtmosphereModel(),
        help="Atmospheric absorption model to apply in synthetic photometry"
    )

    def main(self):
        from astropy.io import fits
        import astropy.units as u
        from ..modeling.astrometry import arcsec_to_au
        import pandas as pd
        hdul = fits.open(self.input)
        name = os.path.basename(self.input)
        output = os.path.join(self.destination, name.replace('.fits', '_masses.fits'))
        if not os.path.isdir(self.destination):
            log.error("Destination is not a directory: %s", self.destination)
            return
        if os.path.exists(output):
            log.error(f"Output file {output} exists")
            return
        evolution_grid = self.bobcat.get_grid()

        limits_df = pd.DataFrame(hdul[self.limits_ext].data)
        detection_df = pd.DataFrame(hdul[self.detection_ext].data)
        extensions =[
            ("mass_limits", limits_df, self.contrast_colname),
            ("mass_detected", detection_df, self.signal_colname)
        ]
        outhdul = fits.HDUList([fits.PrimaryHDU()])
        for outextname, df, sig_colname in extensions:
            contrasts = df[sig_colname]
            separations = np.array(df[self.r_as_colname]) * u.arcsec
            distances = arcsec_to_au(separations, self.host.distance_pc * u.pc)
            df['r_au'] = distances.to(u.AU).value

            if self.irradiation:
                from ..modeling.physics import equilibrium_temperature
                eq_temps = equilibrium_temperature(
                    self.host.temp_K * u.K,
                    self.host.radius_Rsun * u.R_sun,
                    distances,
                    self.albedo
                )
                df['eq_temp_K'] = eq_temps.to(u.K).value
            else:
                eq_temps = None
                df['eq_temp_K'] = 0

            df['companion_abs_mags'] = absolute_mag(self.host.apparent_mag + contrast_to_deltamag(contrasts), self.host.distance_pc * u.pc)

            masses = np.zeros(len(df)) * u.Mjup
            too_bright = np.zeros(len(df), dtype=bool)
            too_faint = np.zeros(len(df), dtype=bool)
            has_exclusions = np.zeros(len(df), dtype=bool)
            pending = []
            for idx in range(len(df)):
                companion_abs_mag = df['companion_abs_mags'].iloc[idx]
                ref = process_point.remote(
                    idx,
                    evolution_grid_ref,
                    companion_abs_mag,
                    self.host.age_Myr,
                    filter_spectrum_ref,
                    eq_temps[idx] if eq_temps is not None else None,
                )
                pending.append(ref)

            n_completed = 0
            # Wait for results in the order they complete
            with tqdm(total=len(df), unit="points") as pbar:
                pbar.update(n_completed)
                while pending:
                    complete, pending = ray.wait(
                        pending,
                        timeout=5,
                        num_returns=min(
                            5,
                            len(pending),
                        ),
                    )
                    results_retired = 0
                    for result in ray.get(complete):
                        index, mass, this_too_bright, this_too_faint, this_has_exclusions = result
                        masses[index] = mass
                        too_bright[index] = this_too_bright
                        too_faint[index] = this_too_faint
                        has_exclusions[index] = this_has_exclusions
                        results_retired += 1
                    if len(complete):
                        pbar.update(results_retired)
                        n_completed += results_retired

            df['bobcat_mass_mjup'] = masses.to(u.Mjup).value
            df['bobcat_too_bright'] = too_bright
            df['bobcat_too_faint'] = too_faint
            df['bobcat_has_exclusions'] = has_exclusions

            tbl = df.to_records(index=False)

            outhdul.append(fits.BinTableHDU(utils.convert_obj_cols_to_str(tbl), name=outextname))
        outhdul.writeto(output, overwrite=True)
        log.info("Finished saving to " + output)
