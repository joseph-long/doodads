import logging
from functools import partial
import numpy as np
from scipy.interpolate import interp1d

from astropy.io import fits
import astropy.units as u

from ..modeling import photometry
from ..modeling.units import WAVELENGTH_UNITS
from .. import utils
from .helpers import filter_from_fits, generate_filter_set_diagnostic_plot

__all__ = (
    'IRDIS',
)

log = logging.getLogger(__name__)
_irdis_filter_urls = {
    'B_Y': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_B_Y.dat',
    'B_J': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_B_J.dat',
    'B_H': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_B_H.dat',
    'B_Ks': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_B_Ks.dat',
    'N_HeI': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_HeI.dat',
    'N_CntJ': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_CntJ.dat',
    'N_PaB': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_PaB.dat',
    'N_CntH': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_CntH.dat',
    'N_FeII': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_FeII.dat',
    'N_CntK1': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_CntK1.dat',
    'N_H2': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_H2.dat',
    'N_BrG': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_BrG.dat',
    'N_CntK2': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_CntK2.dat',
    'N_CO': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_N_CO.dat',
}
_irdis_differential_filter_urls = {
    'D_Y23': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_D_Y23.dat',
    'D_J23': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_D_J23.dat',
    'D_H23': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_D_H23.dat',
    'D_ND-H23': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_D_ND-H23.dat',
    'D_H34': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_D_H34.dat',
    'D_K12': 'https://www.eso.org/sci/facilities/paranal/instruments/sphere/inst/filters/SPHERE_IRDIS_D_K12.dat',
}

_irdis_filters = []

def _convert_sphere_filter(download_filepath, output_filepath):
    table = np.genfromtxt(download_filepath, names=['wavelength', 'transmission'])
    wl = (table['wavelength'] * u.nm).to(WAVELENGTH_UNITS)
    trans = table['transmission']
    columns = [
        fits.Column(name='wavelength', format='E', array=wl),
        fits.Column(name='transmission', format='E', array=trans),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(output_filepath, overwrite=True)


def _convert_sphere_differential_filter(download_filepath, output_filepath, which=1):
    cols = ['wavelength', 'transmission1', 'transmission2']
    table = np.genfromtxt(download_filepath, names=cols)
    wl = (table['wavelength'] * u.nm).to(WAVELENGTH_UNITS)
    if which == 'both':
        trans = table['transmission1'] + table['transmission2']
    else:
        trans = table[cols[which]]
    columns = [
        fits.Column(name='wavelength', format='E', array=wl),
        fits.Column(name='transmission', format='E', array=trans),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(output_filepath, overwrite=True)


for name in _irdis_filter_urls:
    res = utils.REMOTE_RESOURCES.add(
        module=__name__,
        url=_irdis_filter_urls[name],
        converter_function=_convert_sphere_filter,
        output_filename=f'IRDIS_{name}.fits',
    )
    _irdis_filters.append(filter_from_fits(res.output_filepath, name))

for name in _irdis_differential_filter_urls:
    res = utils.REMOTE_RESOURCES.add(
        module=__name__,
        url=_irdis_differential_filter_urls[name],
        converter_function=partial(_convert_sphere_differential_filter, which='both'),
        output_filename=f'IRDIS_{name}.fits',
    )
    _irdis_filters.append(filter_from_fits(res.output_filepath, f"{name}"))
    res = utils.REMOTE_RESOURCES.add(
        module=__name__,
        url=_irdis_differential_filter_urls[name],
        converter_function=partial(_convert_sphere_differential_filter, which=1),
        output_filename=f'IRDIS_{name}_1.fits',
    )
    _irdis_filters.append(filter_from_fits(res.output_filepath, f"{name}_1"))
    res = utils.REMOTE_RESOURCES.add(
        module=__name__,
        url=_irdis_differential_filter_urls[name],
        converter_function=partial(_convert_sphere_differential_filter, which=2),
        output_filename=f'IRDIS_{name}_2.fits',
    )
    _irdis_filters.append(filter_from_fits(res.output_filepath, f"{name}_2"))

IRDIS = photometry.FilterSet(_irdis_filters)

utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, IRDIS, 'SPHERE_IRDIS'))
