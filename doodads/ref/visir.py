from functools import partial
import numpy as np

import astropy.units as u

from ..modeling import photometry, spectra
from .. import utils
from .helpers import generate_filter_set_diagnostic_plot

NEAR = photometry.FilterSet({
    'N': spectra.Spectrum([10.0 - 0.001, 10.0, 12.5, 12.5 + 0.001] * u.um, np.array([0.0, 1.0, 1.0, 0.0]))
})

utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, NEAR, 'NEAR'))
