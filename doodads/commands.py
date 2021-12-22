import argparse
import typing
import logging
from pprint import pformat
from astropy.io.fits.hdu.table import BinTableHDU
import xconf
from doodads.modeling.physics import equilibrium_temperature

from doodads.ref.settl_cond import AMES_COND
from .ref import hst_calspec, mko_filters
from .utils import REMOTE_RESOURCES, DIAGNOSTICS
log = logging.getLogger(__name__)

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
            if self.ext is not None:
                hdu = hdul[self.ext]
            else:
                for idx, hdu in enumerate(hdul):
                    if isinstance(hdu, fits.BinTableHDU):
                        break
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

import enum

class ModelLibrary(enum.Enum):
    # bt_settl = "BT-Settl"
    bobcat = "Sonora Bobcat"

@xconf.config
class ContrastToMass(xconf.Command):
    input : str = xconf.field(help="FITS table file")
    table_ext : typing.Union[int, str] = xconf.field(help="FITS extension holding the contrast table")
    contrast_colname : str = xconf.field(help="Column holding contrast in (companion/host) ratio")
    r_as_colname : str = xconf.field(help="Column holding separation in arcseconds")
    distance_pc : float = xconf.field(help="Distance to host star in parsecs")
    irradiation : typing.Optional[IrradiationConfig] = xconf.field(help="How to calculate equilibrium temperatures and irradiation effects")
    host_mag : float = xconf.field(help="Stellar magnitude of host in bandpass used for these observations")
    host_age_Myr : float = xconf.field(help="Host star age in megayears")
    model_library : ModelLibrary = xconf.field(help="Which model library to use for contrast to mass interpretation")

    def bobcat_mag_to_mass():
    def mags_to_mass(self, companion_abs_mags, model_library : ModelLibrary, eq_temps):
        import numpy as np
        import astropy.units as u
        from .ref import bobcat
        evol_tbl = bobcat.load_bobcat_evolution_age('evolution_tables/evo_tables+0.0/nc+0.0_co1.0_age')
        phot_tbl = bobcat.load_bobcat_photometry('photometry_tables/mag_table+0.0')
        evol_masses = np.unique(evol_tbl['mass_M_sun']) * u.M_sun
        finer_masses = np.linspace(np.min(evol_masses), np.max(evol_masses), num=10 * len(evol_masses))
        if eq_temps is None:
            mass_age_to_mag = bobcat.bobcat_make_mass_age_to_mag(evol_tbl, phot_tbl)
            mag_to_mass, excluded_mass_ranges = bobcat.bobcat_make_mag_to_mass(evol_tbl, mass_age_to_mag, self.host_age_Myr * u.Myr)
            masses = mag_to_mass(companion_abs_mags)
        else:
            masses = np.zeros_like(companion_abs_mags)



    def main(self):
        from astropy.io import fits
        import astropy.units as u
        from .modeling.astrometry import arcsec_to_au
        import pandas as pd
        hdul = fits.open(self.input)
        df = pd.DataFrame(hdul[self.table_ext].data)
        contrasts = df[self.contrast_colname]
        separations = df[self.r_as_colname] * u.arcsec
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
            df['eq_temp_K'] = pd.NA

        masses = self.contrast_to_mass(contrasts, self.model_library, eq_temps)

        tbl = df.to_records()


DISPATCHER = xconf.Dispatcher([
    GetReferenceData,
    RunDiagnostics,
    FitsTable,
    ContrastToMass,
])
