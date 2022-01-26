import numpy as np
import astropy.units as u

WAVELENGTH_UNITS = u.m
FLUX_UNITS = u.W * u.m**-3
FLUX_PER_FREQUENCY_UNITS = u.W * u.m**-2 * u.Hz**-1

# Encompass shortest MagAO-X filters and longest Clio2/MagAO filters
COMMON_WAVELENGTH_START, COMMON_WAVELENGTH_END = (0.4 * u.um).to(WAVELENGTH_UNITS), (6 * u.um).to(WAVELENGTH_UNITS)
COMMON_WAVELENGTH_STEP = (0.0005 * u.um).to(WAVELENGTH_UNITS)
COMMON_WAVELENGTH = np.arange(COMMON_WAVELENGTH_START.value, COMMON_WAVELENGTH_END.value, COMMON_WAVELENGTH_STEP.value) * WAVELENGTH_UNITS
COMMON_WAVELENGTH.flags.writeable = False
