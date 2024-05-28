from astropy.io import fits
import astropy.units as u
from .. import utils
from ..modeling.units import WAVELENGTH_UNITS
from ..modeling import photometry, spectra
import logging
log = logging.getLogger(__name__)


def filter_from_fits(filepath, name):
    return spectra.FITSSpectrum(
        filepath,
        wavelength_column='wavelength',
        wavelength_units=WAVELENGTH_UNITS,
        value_column='transmission',
        value_units=u.dimensionless_unscaled,
        name=name
    )

def generate_filter_set_diagnostic_plot(filter_set : photometry.FilterSet, name : str, ymin=0, ymax=1.0):
    '''Helper task to generate diagnostic plot of all filters'''
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    filter_set.plot_all(ax=ax)
    ax.set(title=name, ylim=(ymin, ymax))
    ax.legend(loc=(0, 1.1), ncol=3)
    filename_safe_name = ''.join(filter(lambda x: x not in '/\\$,.', name.replace(' ', '_')))
    savepath = utils.generated_path(f'{filename_safe_name}_filters.png')
    # fig.tight_layout()
    ax.figure.savefig(savepath)
    log.info(f"Saved {name} filters plot to {savepath}")
