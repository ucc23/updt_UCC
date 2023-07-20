
import numpy as np


def run(x, y, pmRA, pmDE, plx, i, j, Nmax=2):
    """
    Identify a cluster as a duplicate following an arbitrary definition
    that depends on the parallax

    Nmax: maximum number of times allowed for the two objects to be apart
    in any of the dimensions. If this happens for any of the dimensions,
    return a probability of zero
    """

    # Define reference parallax
    if np.isnan(plx[i]) and np.isnan(plx[j]):
        plx_ref = np.nan
    elif np.isnan(plx[i]):
        plx_ref = plx[j]
    elif np.isnan(plx[j]):
        plx_ref = plx[i]
    else:
        plx_ref = (plx[i] + plx[j]) * .5

    # Arbitrary 'duplicate regions' for different parallax brackets
    if np.isnan(plx_ref):
        rad, plx_r, pm_r = 2.5, np.nan, 0.2
    elif plx_ref >= 4:
        rad, plx_r, pm_r = 20, 0.5, 1
    elif 3 <= plx_ref < 4:
        rad, plx_r, pm_r = 15, 0.25, 0.75
    elif 2 <= plx_ref < 3:
        rad, plx_r, pm_r = 10, 0.2, 0.5
    elif 1.5 <= plx_ref < 2:
        rad, plx_r, pm_r = 7.5, 0.15, 0.35
    elif 1 <= plx_ref < 1.5:
        rad, plx_r, pm_r = 5, 0.1, 0.25
    elif .5 <= plx_ref < 1:
        rad, plx_r, pm_r = 2.5, 0.075, 0.2
    elif plx_ref < .5:
        rad, plx_r, pm_r = 2, 0.05, 0.15
    elif plx_ref < .25:
        rad, plx_r, pm_r = 1.5, 0.025, 0.1

    # Angular distance in arcmin
    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) * 60
    # PMs distance
    pm_d = np.sqrt((pmRA[i]-pmRA[j])**2 + (pmDE[i]-pmDE[j])**2)
    # Parallax distance
    plx_d = abs(plx[i] - plx[j])

    # If *any* distance is *very* far away, return no duplicate disregarding
    # the rest with 0 probability
    if d > Nmax * rad or pm_d > Nmax * pm_r or plx_d > Nmax * plx_r:
        return 0

    d_prob = lin_relation(d, rad)
    pms_prob = lin_relation(pm_d, pm_r)
    plx_prob = lin_relation(plx_d, plx_r)
    # prob = round((d_prob+pms_prob+plx_prob)/3., 2)
    prob = np.nanmean((d_prob, pms_prob, plx_prob))

    return round(prob, 2)


def lin_relation(dist, d_max):
    """
    d_min=0 is fixed
    Linear relation for: (0, d_max), (1, d_min)
    """
    # m, h = (d_min - d_max), d_max
    # prob = (dist - h) / m
    # m, h = -d_max, d_max
    p = (dist - d_max) / -d_max
    if p < 0:  # np.isnan(p) or
        return 0
    return p


def max_coords_rad(plx_i):
    """
    Parallax dependent maximum radius in arcmin
    """
    if np.isnan(plx_i):
        rad = 5
    elif plx_i >= 4:
        rad = 20
    elif 3 <= plx_i < 4:
        rad = 15
    elif 2 <= plx_i < 3:
        rad = 10
    elif 1.5 <= plx_i < 2:
        rad = 7.5
    elif 1 <= plx_i < 1.5:
        rad = 5
    elif .5 <= plx_i < 1:
        rad = 2.5
    elif plx_i < .5:
        rad = 1.5
    rad = rad / 60  # To degrees
    return rad


# if __name__ == '__main__':
#     arr = np.array([
#         [112.7202,   0.5,   0.1,   -3.998, -3.058],
#         [112.6979,   0.4,  0.75,   -3.998, -3.058],
#     ])
#     x, y, plx, pmRA, pmDE = arr.T
#     i, j = 0, 1
#     print(run(x, y, pmRA, pmDE, plx, i, j))
#     breakpoint()
