import numpy as np
import astropy.units as u
from .math import modified_gram_schmidt, make_monotonic_decreasing, make_monotonic_increasing

def test_modified_gram_schmidt():
    init_x = np.asarray([
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ])
    result = modified_gram_schmidt(init_x)

    ref = np.asarray([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    assert np.allclose(result, ref), "Last column should be zeroed by MGS"



def goofy_function(x):
    # construct a function with inflection points
    # at x = 1 and x = 2

    # first we want d/dx(f(x)) to have zeros at
    # those coordinates:
    # d/dx(f(x)) = (x - 1) * (x - 2)
    #            = x**2 - 3 * x + 2
    # then we get f(x) by integrating, choosing
    # a constant of 0
    return x**3 / 3 - 3 * x**2 / 2 + 2 * x

def test_make_monotonic():
    xs = np.array([0, 0.5, 2/3, 1, 1.5, 2, 2.5, 3])
    new_xs, new_ys, excluded_ranges = make_monotonic_increasing(xs, goofy_function(xs))
    assert np.isclose(excluded_ranges[0].extremum_y, 2/3)
    assert not np.any(new_ys[new_xs > 1] < goofy_function(1))
    new_xs, new_ys, excluded_ranges = make_monotonic_decreasing(xs, -goofy_function(xs))
    assert np.isclose(excluded_ranges[0].extremum_y, -2/3)
    assert not np.any(new_ys[new_xs > 1] > goofy_function(1))

def test_make_monotonic_units():
    xs = np.linspace(0, 3) * u.m
    ys = xs**2
    new_xs, new_ys, excluded_ranges = make_monotonic_increasing(xs, ys)
    assert new_xs.unit == xs.unit
    assert new_ys.unit == ys.unit
