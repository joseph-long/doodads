import numpy as np
from .analysis import (
    gaussian_2d,
    gaussian_2d_symmetric,
    quick_fit_point_source_gaussian,
    Gaussian2DSymmetricWithBackgroundFit,
    moffat_xy,
    quick_fit_point_source_moffat,
    MoffatWithBackgroundFit,
)


def test_gaussian_2d():
    from astropy.modeling.functional_models import Gaussian2D

    xx, yy = np.meshgrid(
        np.linspace(-100, 100, num=1000), np.linspace(-100, 100, num=1000)
    )
    astropy_version = Gaussian2D().evaluate(xx, yy, 1, 0, 0, 1, 1, 0)
    model = gaussian_2d(xx, yy)
    assert np.allclose(
        astropy_version, model
    ), "2d gaussian discrepancy in evaluation between astropy and doodads"

def test_gaussian_quick_fit():
    npix = 128
    xs = np.arange(0, npix) + 0.5
    xx, yy = np.meshgrid(xs, xs)
    xc, yc = 60, 90
    distrib = gaussian_2d_symmetric(xx, yy, sigma=20, x_center=xc, y_center=yc)
    res : Gaussian2DSymmetricWithBackgroundFit = quick_fit_point_source_gaussian(distrib, (npix - 1)/2, (npix - 1)/2)
    assert (res.x_center - xc) < 0.01
    assert (res.y_center - yc) < 0.01


def test_moffat_norm():
    from astropy.modeling.functional_models import Moffat2D

    xs = np.linspace(-100, 100, num=1000)
    xx, yy = np.meshgrid(xs, xs)

    amplitude = 1.0
    x_center = 0.0
    y_center = 0.0
    power = 1.5
    width = 1.0
    norm_const = (power - 1) / (np.pi * width**2)
    ref = Moffat2D().evaluate(xx, yy, amplitude=norm_const*amplitude, x_0=x_center, y_0=y_center, gamma=width, alpha=power)
    distrib = moffat_xy(xx, yy)
    assert np.allclose(ref, distrib), 'discrepancy between Astropy Moffat (with extra norm factor) and ours'

    dx = xs[1] - xs[0]

    integral = np.trapz(np.trapz(distrib, dx=dx, axis=-1), dx=dx, axis=-1)
    assert np.abs(integral - 1) < 0.01, 'normalization to unit integrated flux is off more than 1% on finite grid'

def test_moffat_quick_fit():
    npix = 128
    xs = np.arange(0, npix) + 0.5
    xx, yy = np.meshgrid(xs, xs)
    xc, yc = 60, 90
    distrib = moffat_xy(xx, yy, core_width=20, x_center=xc, y_center=yc)
    res : MoffatWithBackgroundFit = quick_fit_point_source_moffat(distrib, (npix - 1)/2, (npix - 1)/2)
    assert (res.x_center - xc) < 0.01
    assert (res.y_center - yc) < 0.01
