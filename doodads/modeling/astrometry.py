import numpy as np
import astropy.units as u

__all__ = ('arcsec_to_au', 'au_to_arcsec', 'r_pa_to_x_y', 'x_y_to_r_pa')

def arcsec_to_au(sep, d):
    return (np.tan(sep.to(u.rad)) * d).to(u.AU)

def au_to_arcsec(semimaj, d):
    return (np.arctan(((semimaj.to(u.AU)) / d.to(u.pc)).si.value) * u.rad).to(u.arcsec)


def r_pa_to_x_y(r_px, pa_deg, xcenter, ycenter):
    return (
       r_px * np.cos(np.deg2rad(90 + pa_deg)) + xcenter,
       r_px * np.sin(np.deg2rad(90 + pa_deg)) + ycenter
    )

def x_y_to_r_pa(x, y, xcenter, ycenter):
    dx = x - xcenter
    dy = y - ycenter
    pa_deg = np.rad2deg(np.arctan2(dy, dx)) - 90
    r_px = np.sqrt(dx**2 + dy**2)
    if np.any(pa_deg < 0):
        pa_deg = 360 + pa_deg
    return r_px, pa_deg
