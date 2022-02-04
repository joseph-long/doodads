from dataclasses import dataclass
import logging
import numpy as np
import astropy.units as u
from . import utils
from .utils import ArrayOrQuantity, DIAGNOSTICS
from .plotting import supply_argument, gca

log = logging.getLogger(__name__)

__all__ = [
    'modified_gram_schmidt',
    'make_monotonic_increasing',
    'make_monotonic_decreasing',
]

def modified_gram_schmidt(matrix_a, in_place=False):
    """Orthonormalize the columns of `matrix_a` by the
    modified Gram-Schmidt algorithm.

    Parameters
    ----------
    matrix_a : array, shape (n, m)
        Matrix whose column vectors are to be orthonormalized
        (must be real, and float or double if not in-place)
    in_place : boolean, default: False
        Whether to modify `matrix_a` in-place or copy it

    Returns
    -------
    matrix_a : array, shape (n, m)
        Orthonormalized copy of `matrix_a`, unless `in_place` is set
    """
    if not in_place:
        matrix_a = matrix_a.astype(np.float64)
    n = matrix_a.shape[1]
    for k in range(n):
        norm = np.linalg.norm(matrix_a[:, k])
        if norm != 0:
            matrix_a[:, k] /= norm
        for j in range(k+1, n):
            matrix_a[:, j] -= np.vdot(matrix_a[:, j],
                                      matrix_a[:, k]) * matrix_a[:, k]
    return matrix_a

@dataclass
class ExcludedRange:
    min_x : float
    max_x : float
    extremum_y : float

def make_monotonic_increasing(
    xs : ArrayOrQuantity,
    ys : ArrayOrQuantity,
    display : bool=False,
    eps : float=1e-8,
    ax=None,
    ignore_nan: bool=True,
) -> tuple[ArrayOrQuantity,ArrayOrQuantity,list[ExcludedRange]]:
    '''Given tabulated values of some function y = f(x),
    return values such that y is a strictly increasing
    function of x. This is accomplished by iterating through
    the samples and discarding any sample y[i] < y[i - 1].

    Parameters
    ----------
    xs : array
    ys : array
    display : bool
    eps : float
    ax : Optional[matplotlib.axes.Axes]
    ignore_nan : bool

    Returns
    -------
    new_xs : array
    new_ys : array
    '''
    if ignore_nan:
        mask_isnan = np.isnan(xs) | np.isnan(ys)
        xs = xs[~mask_isnan]
        ys = ys[~mask_isnan]
    sorter = np.argsort(xs)
    sorted_xs = xs[sorter]
    sorted_ys = ys[sorter]

    new_xs, new_ys = [sorted_xs[0]], [sorted_ys[0]]
    excluded = []
    excluded_ranges = []
    exclude_from = sorted_xs[0]
    for x, y in zip(sorted_xs[1:], sorted_ys[1:]):
        if y > new_ys[-1]:
            if len(excluded):
                excluded_ranges.append(ExcludedRange(exclude_from, x, np.min(excluded)))
            new_xs.append(x)
            new_ys.append(y)
            exclude_from = x
            excluded = []
        else:
            excluded.append(y)
    if len(excluded):
        # ended with open exclusion range
        excluded_ranges.append(ExcludedRange(exclude_from, x, np.min(excluded)))
        new_xs.append(x)
        new_ys.append(new_ys[-1])

    if isinstance(xs, u.Quantity):
        new_xs_2 = np.zeros(len(new_xs)) * xs.unit
        new_xs_2[:] = new_xs
        new_xs = new_xs_2
    else:
        new_xs = np.asarray(new_xs)

    if isinstance(ys, u.Quantity):
        new_ys_2 = np.zeros(len(new_xs)) * ys.unit
        new_ys_2[:] = new_ys
        new_ys = new_ys_2
    else:
        new_ys = np.asarray(new_ys)

    if display:
        if ax is None:
            ax = gca()
        ax.plot(xs, ys, '.-')
        ax.plot(new_xs, new_ys, '.-')
        for er in excluded_ranges:
            ax.axvspan(er.min_x, er.max_x, alpha=0.2)
            ax.axhline(er.extremum_y, ls=':')
    return new_xs, new_ys, excluded_ranges

def make_monotonic_decreasing(
    xs : ArrayOrQuantity,
    ys : ArrayOrQuantity,
    **kwargs
) -> tuple[ArrayOrQuantity,ArrayOrQuantity,list[ExcludedRange]]:
    '''See `make_monotonic_increasing`'''

    display = kwargs.get('display', False)
    kwargs['display'] = False
    ax = kwargs.get('ax', None)
    if 'ax' in kwargs:
        del kwargs['ax']
    new_xs, new_ys, excluded_ranges_neg = make_monotonic_increasing(xs, -ys, **kwargs)
    new_ys = -new_ys
    excluded_ranges = []
    for er in excluded_ranges_neg:
        excluded_ranges.append(ExcludedRange(
            er.min_x,
            er.max_x,
            -1 * er.extremum_y
        ))
    if display:
        if ax is None:
            ax = gca()
        ax.plot(xs, ys, '.-')
        ax.plot(new_xs, new_ys, '.-')
        for er in excluded_ranges:
            ax.axvspan(er.min_x, er.max_x, alpha=0.2)
            ax.axhline(er.extremum_y, ls=':')
    return new_xs, new_ys, excluded_ranges

def _visualize_make_monotonic(num=10):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.set_title('Make sequence monotonic')

    xs = np.linspace(0, 3, num=num)
    ys = np.sin(xs*4) + xs
    ax.plot(xs, ys, '.-', label="original samples")
    new_xs, new_ys, excluded_ranges = make_monotonic_increasing(xs, ys)
    ax.plot(new_xs, new_ys, '+-')
    ax.axvspan(excluded_ranges[0].min_x, excluded_ranges[0].max_x, alpha=0.2, label='excluded')
    ax.axhline(excluded_ranges[0].extremum_y, ls=':', label='min excluded')

    xs = np.linspace(1, 4, num=num)
    ys = -1 * (np.sin(xs*4) + xs)
    ax.plot(xs, ys, '.-', label="original samples")
    new_xs, new_ys, excluded_ranges = make_monotonic_decreasing(xs, ys)
    ax.plot(new_xs, new_ys, '+-')
    ax.axvspan(excluded_ranges[0].min_x, excluded_ranges[0].max_x, fc='C1', alpha=0.2, label='excluded')
    ax.axhline(excluded_ranges[0].extremum_y, ls=':', c='C1', label='max excluded')


    ax.legend()
    savepath = utils.generated_path('make_monotonic.png')
    ax.figure.savefig(savepath)
    log.info(f"Saved monotonic functions plot to {savepath}")

utils.DIAGNOSTICS.add(_visualize_make_monotonic)
