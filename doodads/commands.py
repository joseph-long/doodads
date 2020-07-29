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
