from dataclasses import dataclass
from typing import Union
import numpy as np
from scipy import interpolate
import astropy.units as u

from .utils import ArrayOrQuantity
from .plotting import supply_argument, gca

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

@supply_argument(ax=gca)
def make_monotonic_increasing(
    xs : ArrayOrQuantity,
    ys : ArrayOrQuantity,
    display : bool=False,
    eps : float=1e-8,
    ax=None,
) -> tuple[ArrayOrQuantity,ArrayOrQuantity,list[ExcludedRange]]:
    '''Given tabulated values of some function y = f(x),
    return values such that y is a strictly increasing
    function of x. This is accomplished by iterating through
    the samples and discarding any sample y[i] < y[i - 1].
    To avoid unjustified linear interpolation across omitted
    samples, after finding the next sample y[i] > y[i - N]
    (after skipping N samples) a new point with x = x[i] - `eps`
    and y = y[i - 1] is inserted.

    Parameters
    ----------
    xs : array
    ys : array

    Returns
    -------
    new_xs : array
    new_ys : array
    '''
    sorter = np.argsort(xs)
    sorted_xs = xs[sorter]
    sorted_ys = ys[sorter]

    # strip units (if any)
    if isinstance(xs, u.Quantity):
        sorted_xs = sorted_xs.value
    if isinstance(ys, u.Quantity):
        sorted_ys = sorted_ys.value

    new_xs, new_ys = [sorted_xs[0]], [sorted_ys[0]]
    excluded = []
    excluded_ranges = []
    exclude_from = sorted_xs[0]
    for x, y in zip(sorted_xs[1:], sorted_ys[1:]):
        if y > new_ys[-1]:
            if len(excluded):
                excluded_ranges.append(ExcludedRange(exclude_from, x, np.min(excluded)))
                x_minus_epsilon = x - eps
                new_xs.append(x_minus_epsilon)
                new_ys.append(new_ys[-1])
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

    new_xs, new_ys = np.array(new_xs), np.array(new_ys)
    # reapply units
    if isinstance(xs, u.Quantity):
        new_xs = new_xs * xs.unit
    if isinstance(ys, u.Quantity):
        new_ys = new_ys * ys.unit
    if display:
        ax.plot(xs, ys, '.-')
        ax.plot(new_xs, new_ys, '.-')
        for from_x, to_x, min_y in excluded_ranges:
            ax.axvspan(from_x, to_x, alpha=0.2)
            ax.axhline(min_y, ls=':')
    return new_xs, new_ys, excluded_ranges

def make_monotonic_decreasing(
    xs : ArrayOrQuantity,
    ys : ArrayOrQuantity,
    **kwargs
) -> tuple[ArrayOrQuantity,ArrayOrQuantity,list[ExcludedRange]]:
    '''See `make_monotonic_increasing`'''

    new_xs, new_ys, excluded_ranges_neg = make_monotonic_increasing(xs, -ys, **kwargs)
    excluded_ranges = []
    for er in excluded_ranges_neg:
        excluded_ranges.append(ExcludedRange(
            er.min_x,
            er.max_x,
            -1 * er.extremum_y
        ))
    return new_xs, -new_ys, excluded_ranges
