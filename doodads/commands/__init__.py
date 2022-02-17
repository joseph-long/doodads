import os
import argparse
import coloredlogs
import typing
import logging
from pprint import pformat
from astropy.io.fits.hdu.table import BinTableHDU
import xconf
from ..modeling.photometry import contrast_to_deltamag, absolute_mag
from ..modeling.physics import equilibrium_temperature
from enum import Enum
from ..ref.settl_cond import AMES_COND
from ..ref import hst_calspec, mko_filters, clio, sphere, gemini_atmospheres, magellan_atmospheres
from ..utils import REMOTE_RESOURCES, DIAGNOSTICS
log = logging.getLogger(__name__)

import numpy as np
import astropy.units as u
from ..ref import bobcat
from .contrast_to_mass import ContrastToMass

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
        import matplotlib
        matplotlib.use('Agg')
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

class DoodadsDispatcher(xconf.Dispatcher):
    def configure_logging(self, level):
        logger = logging.getLogger('doodads')
        coloredlogs.install(level='DEBUG', logger=logger)

DISPATCHER = DoodadsDispatcher([
    GetReferenceData,
    RunDiagnostics,
    FitsTable,
    ContrastToMass,
])
