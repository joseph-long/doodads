import os
import argparse
import typing
import logging
from pprint import pformat
from astropy.io.fits.hdu.table import BinTableHDU
import xconf
from doodads.modeling.photometry import contrast_to_deltamag, absolute_mag
from doodads.modeling.physics import equilibrium_temperature

from doodads.ref.settl_cond import AMES_COND
from .ref import hst_calspec, mko_filters
from .utils import REMOTE_RESOURCES, DIAGNOSTICS
log = logging.getLogger(__name__)

import numpy as np
import astropy.units as u
from .ref import bobcat

@xconf.config
class GetReferenceData(xconf.Command):
    exclude : list[str] = xconf.field(default_factory=list, help='Dotted module path for resources to exclude (e.g. doodads.ref.mko_filters)')
    def main(self):
        import matplotlib
        matplotlib.use('Agg')
        log.info(f"Excluded modules: {self.exclude}")
        selected_resources = REMOTE_RESOURCES.filter(exclude=self.exclude)
        log.info(f"Processing registered resources from {pformat(selected_resources.resources)}")
        selected_resources.download_and_convert()

@xconf.config
class RunDiagnostics(xconf.Command):
    def main(self):
        DIAGNOSTICS.run_all()

@xconf.config
class FitsTable(xconf.Command):
    input : str = xconf.field(help="FITS table file")
    ext : typing.Optional[str] = xconf.field(default=None, help="FITS extension to read containing a binary table (default: first one found)")
    rows : int = xconf.field(default=10)
    all : bool = xconf.field(default=False, help="Whether to ignore the rows setting and print all")
    sort_on : typing.Optional[str] = xconf.field(default=None, help="Column name on which to sort")
    reverse : bool = xconf.field(default=False, help="Reverse order (before truncating at rows)")

    def main(self):
        from astropy.io import fits
        import numpy as np
        with open(self.input, 'rb') as fh:
            hdul = fits.open(fh)
            ext = 0
            if self.ext is not None:
                hdu = hdul[self.ext]
                ext = self.ext
            else:
                for idx, hdu in enumerate(hdul):
                    if isinstance(hdu, fits.BinTableHDU):
                        break
                ext = hdu.header['extname'] if 'extname' in hdu.header else idx
            if not isinstance(hdu, fits.BinTableHDU):
                raise ValueError("Not a BinTable extension")

            tbl = hdu.data
            fields = tbl.dtype.fields
            if self.sort_on is not None and self.sort_on in fields:
                sorter = np.argsort(tbl[self.sort_on])
                tbl = tbl[sorter]
            if self.reverse:
                tbl = tbl[::-1]
            if not self.all:
                tbl = tbl[:self.rows]
            cols = []
            colfmts = []
            for fld in fields:
                width = max(len(fld), 9)
                cols.append(f"{fld:9}")
                if np.issubdtype(tbl[fld].dtype, np.number):
                    colfmts.append("{:< " + str(width) + ".3g}")
                else:
                    colfmts.append("{:<" + str(width) + "}")
            print('  '.join(cols))
            for row in tbl:
                cols = []
                for fld, colfmt in zip(fields, colfmts):
                    cols.append(colfmt.format(row[fld]))
                print('  '.join(cols))

@xconf.config
class IrradiationConfig:
    host_temp_K : float = xconf.field(help="Host temperature (T_eff) in Kelvin")
    host_radius_R_sun : float = xconf.field(help="Host star radius (R_*) in solar radii")
    albedo : float = xconf.field(default=0.5, help="(default: 0.5, approx. like Jupiter)")


@xconf.config
class BobcatModels:
    mag_col : str = xconf.field(default='mag_MKO_Lprime', help="Which column from the bobcat models to use (doodads.BOBCAT_PHOTOMETRY_COLS)")

@xconf.config
class ModelLibrary:
    bobcat : typing.Optional[BobcatModels] = xconf.field()

@xconf.config
class ContrastToMass(xconf.Command):
    input : str = xconf.field(help="FITS table file")
    destination : str = xconf.field(default=".", help="Where to write output files")
    table_ext : typing.Union[int, str] = xconf.field(default="limits", help="FITS extension holding the contrast table")
    contrast_colname : str = xconf.field(default="contrast_limit_5sigma", help="Column holding contrast in (companion/host) ratio")
    r_as_colname : str = xconf.field(default="r_as", help="Column holding separation in arcseconds")
    distance_pc : float = xconf.field(help="Distance to host star in parsecs")
    irradiation : typing.Optional[IrradiationConfig] = xconf.field(help="How to calculate equilibrium temperatures and irradiation effects")
    host_mag : float = xconf.field(help="Stellar magnitude of host in bandpass used for these observations")
    host_age_Myr : float = xconf.field(help="Host star age in megayears")
    model_library : ModelLibrary = xconf.field(
        default=ModelLibrary(bobcat=BobcatModels()),
        help="Which model library to use for contrast to mass interpretation"
    )

    def _bobcat_mag_to_mass(self, evol_tbl, phot_tbl, age, abs_mag, eq_temp=None, mag_col='mag_MKO_Lprime'):
        mass_age_to_mag = bobcat.bobcat_mass_age_to_mag(evol_tbl, phot_tbl, eq_temp=eq_temp, mag_col=mag_col)
        mag_to_mass, excluded_mass_ranges = bobcat.bobcat_mag_to_mass(evol_tbl, mass_age_to_mag, age)
        mass, too_bright, too_faint = mag_to_mass(abs_mag)
        log.debug(f"{abs_mag=:5.2f} -> {mass=:5.2f} {too_bright=} {too_faint=}")
        log.debug(pformat(excluded_mass_ranges))
        limiting_mags = np.asarray([limit_mag for mass_min, mass_max, limit_mag in excluded_mass_ranges])
        if np.all(abs_mag > limiting_mags): # abs mag is fainter than all limiting mags of excluded ranges
            excluded_mass_ranges = []
        else:
            excluded_mass_ranges = [
                (mass_min, mass_max, limit_mag)
                for mass_min, mass_max, limit_mag
                in excluded_mass_ranges
                if abs_mag < limit_mag   # if abs mag is *not* fainter than this range's limiting mag
            ]
        return mass, too_bright, too_faint, excluded_mass_ranges

    def mags_to_mass(self, companion_abs_mags, model_library : ModelLibrary, eq_temps):
        evol_tbl = bobcat.load_bobcat_evolution_age('evolution_tables/evo_tables+0.0/nc+0.0_co1.0_age')
        phot_tbl = bobcat.load_bobcat_photometry('photometry_tables/mag_table+0.0')
        if eq_temps is None:
            eq_temps = np.zeros_like(companion_abs_mags) * u.K
        masses = np.zeros_like(companion_abs_mags) * u.Mjup
        was_too_bright = np.zeros(len(companion_abs_mags), dtype=bool)
        was_too_faint = np.zeros(len(companion_abs_mags), dtype=bool)
        excluded_mass_ranges = []
        for idx in range(len(masses)):
            mag, eq_temp = companion_abs_mags[idx], eq_temps[idx]
            _mass, _too_bright, _too_faint, _excluded_ranges = self._bobcat_mag_to_mass(
                evol_tbl, phot_tbl, self.host_age_Myr * u.Myr, mag, eq_temp,
                mag_col=self.model_library.bobcat.mag_col
            )
            masses[idx] = _mass
            was_too_bright[idx] = _too_bright
            was_too_faint[idx] = _too_faint
            excluded_mass_ranges.append(_excluded_ranges)
        return masses, was_too_bright, was_too_faint, excluded_mass_ranges

    def main(self):
        from astropy.io import fits
        import astropy.units as u
        from .modeling.astrometry import arcsec_to_au
        import pandas as pd
        hdul = fits.open(self.input)
        name = os.path.basename(self.input)
        output = os.path.join(self.destination, name.replace('.fits', '_masses.fits'))
        if os.path.exists(output):
            log.error(f"Output file {output} exists")
            return
        df = pd.DataFrame(hdul[self.table_ext].data)
        contrasts = df[self.contrast_colname]
        separations = np.array(df[self.r_as_colname]) * u.arcsec
        distances = arcsec_to_au(separations, self.distance_pc * u.pc)
        df['r_au_in_projection'] = distances.to(u.AU).value

        if self.irradiation is not None:
            from .modeling.physics import equilibrium_temperature
            eq_temps = equilibrium_temperature(
                self.irradiation.host_temp_K * u.K,
                self.irradiation.host_radius_R_sun * u.R_sun,
                distances,
                self.irradiation.albedo
            )
            df['eq_temp_K'] = eq_temps.value
        else:
            eq_temps = None
            df['eq_temp_K'] = 0

        df['companion_abs_mags'] = absolute_mag(self.host_mag + contrast_to_deltamag(contrasts), self.distance_pc * u.pc)

        masses, too_bright, too_faint, excluded_mass_ranges = self.mags_to_mass(df['companion_abs_mags'], self.model_library, eq_temps)
        df['bobcat_mass_mjup'] = masses.to(u.Mjup).value
        df['bobcat_too_bright'] = too_bright
        df['bobcat_too_faint'] = too_faint
        df['bobcat_has_exclusions'] = np.array([len(excl) > 0 for excl in excluded_mass_ranges])

        tbl = df.to_records(index=False)
        fits.BinTableHDU(tbl, name="masses").writeto(output, overwrite=True)
        log.info("Finished saving to " + output)


DISPATCHER = xconf.Dispatcher([
    GetReferenceData,
    RunDiagnostics,
    FitsTable,
    ContrastToMass,
])
