from collections import defaultdict
import os.path
from functools import partial
import astropy.units as u
from ..modeling import spectra
from ..modeling import photometry
from .helpers import filter_from_fits, generate_filter_set_diagnostic_plot
from .. import utils

__all__ = [
    'CAMSCI1_FILTERS',
    'CAMSCI2_FILTERS',
    'H_ALPHA',
]

_base_url = "https://magao-x.org/docs/handbook/_static/ref/filters/"

CAMSCI1_FILTERS = defaultdict(lambda: defaultdict(dict))
CAMSCI2_FILTERS = defaultdict(lambda: defaultdict(dict))


_wfs_bs_url_part = {
    'open': 'open',
    '65/35': '65-35',
    r'H-$\alpha$/IR': 'halpha-ir',
}

_sci_bs_url_part = {
    '50/50': '5050',
    'open': 'open',
    r'H-$\alpha$/Cont.': 'halpha',
}

_filter_url_part = {
    "$g'$": "gp",
    "$r'$": "rp",
    r"H-$\alpha$": "halpha",
    r"H-$\alpha$ Cont.": "halpha-cont",
    "$i'$": "ip",
    r"$[\text{CH}_4]$": "ch4",
    "$z'$": "zp",
    r"$[\text{CH}_4]$ Cont.": "ch4-cont",
}

for (wfs_bs, sci_bs, filter_set) in [
    (r'H-$\alpha$/IR', r'H-$\alpha$/Cont.', [r"H-$\alpha$ Cont."]),
    ('65/35', '50/50', ["$r'$", "$i'$", r"$[\text{CH}_4]$", "$z'$"]),
    ('65/35', 'open', ["$r'$", "$i'$", "$z'$"]),
]:
    filters = {}
    for filter_name in filter_set:
        filter_dat = utils.REMOTE_RESOURCES.add_from_url(
            module=__name__,
            url=_base_url + f"magaox_sci1-{_filter_url_part[filter_name]}_bs-{_wfs_bs_url_part[wfs_bs]}_scibs-{_sci_bs_url_part[sci_bs]}.dat",
        )
        filters[filter_name] = spectra.TableSpectrum(
            filter_dat.output_filepath,
            u.m,
            u.dimensionless_unscaled,
            name=f"{filter_name} (MagAO-X sci1, {wfs_bs} WFS, {sci_bs} BS)"
        )
    CAMSCI1_FILTERS[wfs_bs][sci_bs] = photometry.FilterSet(filters)
    utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, CAMSCI1_FILTERS[wfs_bs][sci_bs], f'MagAO-X sci1, {wfs_bs} WFS, {sci_bs} BS'))


for (wfs_bs, sci_bs, filter_set) in [
    (r'H-$\alpha$/IR', r'H-$\alpha$/Cont.', [r"H-$\alpha$"]),
    ('65/35', '50/50', ["$g'$", "$r'$", "$i'$", "$z'$", r"$[\text{CH}_4]$ Cont."]),
]:
    filters = {}
    for filter_name in filter_set:
        filter_dat = utils.REMOTE_RESOURCES.add_from_url(
            module=__name__,
            url=_base_url + f"magaox_sci2-{_filter_url_part[filter_name]}_bs-{_wfs_bs_url_part[wfs_bs]}_scibs-{_sci_bs_url_part[sci_bs]}.dat",
        )
        filters[filter_name] = spectra.TableSpectrum(
            filter_dat.output_filepath,
            u.m,
            u.dimensionless_unscaled,
            name=f"{filter_name} (MagAO-X sci1, {wfs_bs} WFS, {sci_bs} BS)"
        )
    CAMSCI2_FILTERS[wfs_bs][sci_bs] = photometry.FilterSet(filters)
    utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, CAMSCI2_FILTERS[wfs_bs][sci_bs], f'MagAO-X sci2, {wfs_bs} WFS, {sci_bs} BS'))


H_ALPHA_DAT = os.path.join(os.path.dirname(__file__), 'Alluxa_656.3-1_OD4_Ultra_Narrow_Bandpass_Filter_7019_T.dat')
H_ALPHA = spectra.TableSpectrum(H_ALPHA_DAT, u.nm, u.dimensionless_unscaled, name=r'[H$\alpha$] filter only')

MAGAOX_MISC = photometry.FilterSet({
    'H_ALPHA': H_ALPHA,
})

utils.DIAGNOSTICS.add(partial(generate_filter_set_diagnostic_plot, MAGAOX_MISC, 'MagAO-X misc.'))
