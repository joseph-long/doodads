import numpy as np
from scipy import interpolate
import astropy.units as u

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

def make_monotonic_increasing(xs : np.ndarray, ys : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Given tabulated values of some function y = f(x),
    return values such that y is a strictly increasing
    function of x, using cubic splines to find
    the ends of troughs in the interpolated function

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

    # strip units (if any) before spline stuff
    if isinstance(xs, u.Quantity):
        sorted_xs = sorted_xs.value
    if isinstance(ys, u.Quantity):
        sorted_ys = sorted_ys.value

    mask = np.ones(len(sorted_xs), dtype=bool)
    extra_xs, extra_ys = [], []
    spl = interpolate.CubicSpline(sorted_xs, sorted_ys)
    dspl = spl.derivative()
    ddspl = dspl.derivative()
    roots = dspl.roots()
    for idx, inflection_x in enumerate(roots):
        print(f"{idx=} {inflection_x=} {ddspl(inflection_x)=}")
        if ddspl(inflection_x) > 0:
            if idx == 0:
                # function was initially decreasing, so
                # first point is max_y and we proceed
                # as if we had an inflection point there
                inflection_x = sorted_xs[0]
                max_y = sorted_ys[0]
            else:
                # reached a minimum and continuing upwards
                # so we skip
                continue
        else:
            max_y = spl(inflection_x)
            if idx == len(roots) - 1:
                # second derivative says we're decreasing,
                # and this is the last inflection point, so
                # points with x > inflection_x will have to be
                # smaller f(x) values and should be excluded,
                # but we want to keep the domain
                excluded = sorted_xs > inflection_x
                max_x = max(sorted_xs)
                mask &= ~excluded
                extra_xs.extend([inflection_x, max_x])
                extra_ys.extend([max_y, max_y])
                continue

        solns = spl.solve(max_y)
        # excluding the current inflection point,
        # find the next place where the y value
        # reaches spl(inflection_x)
        soln = solns[solns > inflection_x][0]
        excluded = (sorted_xs > inflection_x) & (sorted_xs < soln)
        mask &= ~excluded
        extra_xs.extend([inflection_x, soln])
        extra_ys.extend([max_y, max_y])
    final_xs = np.concatenate([sorted_xs[mask], extra_xs])
    final_sorter = np.argsort(final_xs)
    final_xs = final_xs[final_sorter]
    final_ys = np.concatenate([sorted_ys[mask], extra_ys])[final_sorter]

    # reapply units
    if isinstance(xs, u.Quantity):
        final_xs = final_xs * xs.unit
    if isinstance(ys, u.Quantity):
        final_ys = final_ys * ys.unit

    return final_xs, final_ys


def make_monotonic_decreasing(xs : np.ndarray, ys : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Given tabulated values of some function y = f(x),
    return values such that y is a strictly decreasing
    function of x, using cubic splines to find
    the ends of peaks in the interpolated function

    Parameters
    ----------
    xs : array
    ys : array

    Returns
    -------
    new_xs : array
    new_ys : array
    '''
    new_xs, new_ys = make_monotonic_increasing(xs, -ys)
    return new_xs, -new_ys
