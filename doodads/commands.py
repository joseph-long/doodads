import logging
from .ref import settl_cond, hst_calspec, mko_filters
log = logging.getLogger(__name__)

def get_reference_data():
    import matplotlib
    matplotlib.use('Agg')
    logging.basicConfig(level='INFO')
    log.info("Starting up")
    log.info("Downloading and converting Allard models")
    settl_cond.download_and_convert_settl_cond()
    log.info("Downloading and converting HST standard spectra")
    hst_calspec.download_and_convert_hst_standards()
    log.info("Downloading and converting MKO filters")
    mko_filters.download_and_convert_mko_filters()
