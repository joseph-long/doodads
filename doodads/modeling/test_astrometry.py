import numpy as np
from .astrometry import x_y_to_r_pa, r_pa_to_x_y

def test_r_pa_conversions():
    r_px, pa_deg = x_y_to_r_pa(10, 0, 0, 0)
    assert np.isclose(r_px, 10)
    assert np.isclose(pa_deg, 270)

    xc, yc = 10, 5
    r_px, pa_deg = x_y_to_r_pa(10 + xc, 0 + yc, xc, yc)
    assert np.isclose(r_px, 10)
    assert np.isclose(pa_deg, 270)

    x, y = r_pa_to_x_y(10, 0, 0, 0)
    assert np.isclose(x, 0)
    assert np.isclose(y, 10)

    x, y = r_pa_to_x_y(10, 0, xc, yc)
    assert np.isclose(x, xc)
    assert np.isclose(y, 10 + yc)

    x, y = r_pa_to_x_y(10, -90, xc, yc)
    assert np.isclose(x, 10 + xc)
    assert np.isclose(y, yc)

    x, y = r_pa_to_x_y(10, 270, xc, yc)
    assert np.isclose(x, 10 + xc)
    assert np.isclose(y, yc)
