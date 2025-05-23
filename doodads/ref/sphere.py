import pathlib
import logging
from functools import partial
import numpy as np
from scipy.interpolate import interp1d

from astropy.io import fits
import astropy.units as u

from ..modeling import photometry, spectra
from ..modeling.units import WAVELENGTH_UNITS
from .. import utils
from .helpers import filter_from_fits, generate_filter_set_diagnostic_plot

__all__ = (
    'IRDIS',
    'SPHERE_IFS',
    'SphereVigan2015Contrasts',
    'SphereBeuzit2019RawContrasts',
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

_irdis_filters = {}

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


for shortname in _irdis_filter_urls:
    res = utils.REMOTE_RESOURCES.add_from_url(
        module=__name__,
        url=_irdis_filter_urls[shortname],
        converter_function=_convert_sphere_filter,
        output_filename=f'IRDIS_{shortname}.fits',
    )
    _irdis_filters[shortname] = filter_from_fits(res.output_filepath, f"IRDIS {shortname}")

for shortname in _irdis_differential_filter_urls:
    res = utils.REMOTE_RESOURCES.add_from_url(
        module=__name__,
        url=_irdis_differential_filter_urls[shortname],
        converter_function=partial(_convert_sphere_differential_filter, which='both'),
        output_filename=f'IRDIS_{shortname}.fits',
    )
    _irdis_filters[shortname] = filter_from_fits(res.output_filepath, f"IRDIS {shortname}")
    res = utils.REMOTE_RESOURCES.add_from_url(
        module=__name__,
        url=_irdis_differential_filter_urls[shortname],
        converter_function=partial(_convert_sphere_differential_filter, which=1),
        output_filename=f'IRDIS_{shortname}_1.fits',
    )
    _irdis_filters[f"{shortname}_1"] = filter_from_fits(res.output_filepath, f"IRDIS {shortname}_1")
    res = utils.REMOTE_RESOURCES.add_from_url(
        module=__name__,
        url=_irdis_differential_filter_urls[shortname],
        converter_function=partial(_convert_sphere_differential_filter, which=2),
        output_filename=f'IRDIS_{shortname}_2.fits',
    )
    _irdis_filters[f"{shortname}_2"] = filter_from_fits(res.output_filepath, f"IRDIS {shortname}_2")

IRDIS = photometry.FilterSet(_irdis_filters)
SPHERE_IFS = photometry.FilterSet({
    'YJH': spectra.Spectrum([0.9499, 0.95, 1.65, 1.6501] * u.um, np.array([0.0, 1.0, 1.0, 0.0]))
})

utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, IRDIS, 'SPHERE_IRDIS'))
utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, SPHERE_IFS, 'SPHERE_IFS'))

class _SphereContrasts(utils.LazyLoadable):
    modes : set[str]

    def __init__(self, fname):
        super().__init__(filepath=pathlib.Path(__file__).parent / fname)

    def __iter__(self):
        return iter(self._table)

    def _lazy_load(self):
        self._table = utils.read_webplotdigitizer(open(self.filepath), x_label='separation', y_label='contrast', x_unit=u.arcsec)
        for k in self._table:
            print(repr(k))

    @property
    def modes(self):
        return set(self._table.keys())

    def __getitem__(self, name):
        if name not in self._table:
            raise KeyError(f"No such mode {repr(name)}, have these modes: {self.modes}")
        return self._table[name]

    def __repr__(self):
        return f"<contrasts for {self.modes}>"

SphereBeuzit2019RawContrasts = _SphereContrasts('sphere_beuzit_2019_fig9.csv')
SphereVigan2015Contrasts = _SphereContrasts('Vigan2015_Fig2_contrast.csv')
