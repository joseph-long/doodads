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
    # log.info("Downloading and converting HST standard spectra")
    # hst_calspec.download_and_convert_hst_standards()
    # log.info("Downloading and converting MKO filters")
    # mko_filters.download_and_convert_mko_filters()
    log.info(f"Processing registered resources from {pformat(REMOTE_RESOURCES.resources)}")
    REMOTE_RESOURCES.download_and_convert_all()

def run_diagnostics():
    DIAGNOSTICS.run_all()
