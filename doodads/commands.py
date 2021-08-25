import argparse
import logging
from pprint import pformat
from .ref import hst_calspec, mko_filters
from .utils import REMOTE_RESOURCES, DIAGNOSTICS
log = logging.getLogger(__name__)

def get_reference_data():
    import matplotlib
    matplotlib.use('Agg')
    logging.basicConfig(level='INFO')
    log.info("Starting up")
    parser = argparse.ArgumentParser('dd-get-reference-data', description='Fetch some or all remote resources')
    parser.add_argument('-x', '--exclude', nargs='*', help='Dotted module path for resources to exclude (e.g. doodads.ref.mko_filters)')
    args = parser.parse_args()
    log.info(f"Excluded modules: {args.exclude}")
    selected_resources = REMOTE_RESOURCES.filter(exclude=args.exclude)
    log.info(f"Processing registered resources from {pformat(selected_resources.resources)}")
    selected_resources.download_and_convert()

def run_diagnostics():
    DIAGNOSTICS.run_all()

def fits_table():
    from astropy.io import fits
    logging.basicConfig(level='INFO')
    parser = argparse.ArgumentParser('dd-fits-table', description='Inspect a FITS bintable')
    parser.add_argument('FILE', help="Path to FITS file")
    parser.add_argument("EXT", nargs='?', default=0, help="Extension within FITS file")
    parser.add_argument("--describe", help="Column to summarize")
    parser.add_argument("-n", type=int, default=10, help="Number of rows")
    args = parser.parse_args()
    with open(args.FILE, 'rb') as fh:
        hdul = fits.open(fh)
        hdu = hdul[args.EXT]
        if not isinstance(hdu, fits.BinTableHDU):
            raise ValueError("Not a BinTable extension")
        tbl = hdu.data[:args.n]
        fields = tbl.dtype.fields
        cols = []
        for fld in fields:
            cols.append(fld)
        print('\t'.join(cols))
        for row in tbl:
            cols = []
            for fld in fields:
                cols.append(str(row[fld]))
            print('\t'.join(cols))
