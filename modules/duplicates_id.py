
import numpy as np
from scipy.spatial.distance import cdist


def run(df, N_dups=20, prob_cut=0.25):
    """
    Assign a 'duplicate probability' for each cluster in 'df' compared to the
    rest of the listed clusters
    """
    x, y = df['GLON_m'], df['GLAT_m']
    pmRA, pmDE, plx = df['pmRA_m'], df['pmDE_m'], df['plx_m']

    x, y = df['GLON'], df['GLAT']
    pmRA, pmDE, plx = df['pmRA'], df['pmDE'], df['plx']

    coords = np.array([x, y]).T
    # Find the distances to all clusters, for all clusters
    dist = cdist(coords, coords)
    # Change distance to itself from 0 to inf
    msk = dist == 0.
    dist[msk] = np.inf

    dups_fnames, dups_probs = [], []
    for i, cl in enumerate(dist):
        # Only look for duplicates among the closest 'N_dups' clusters
        idx = np.argsort(cl)[:N_dups]

        dups_fname, dups_prob = [], []
        for j in idx:
            dup_prob = duplicate_probs(x, y, pmRA, pmDE, plx, i, j)
            if dup_prob >= prob_cut:
                # Store just the first fname
                dups_fname.append(df['fnames'][j].split(';')[0])
                dups_prob.append(str(dup_prob))

        if dups_fname:
            dups_fname = ";".join(dups_fname)
            dups_prob = ";".join(dups_prob)
        else:
            dups_fname, dups_prob = 'nan', 'nan'

        dups_fnames.append(dups_fname)
        dups_probs.append(dups_prob)

    return dups_fnames, dups_probs


def duplicate_probs(x, y, pmRA, pmDE, plx, i, j, Nmax=2):
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
        plx_ref = (plx[i] + plx[i]) * .5

    # Arbitrary 'duplicate regions' for different parallax brackets
    if np.isnan(plx_ref):
        rad, plx_r, pm_r = 2.5, np.nan, 0.2
    elif plx_ref >= 4:
        rad, plx_r, pm_r = 20, 0.5, 1
    elif 3 <= plx_ref and plx_ref < 4:
        rad, plx_r, pm_r = 15, 0.25, 0.75
    elif 2 <= plx_ref and plx_ref < 3:
        rad, plx_r, pm_r = 10, 0.2, 0.5
    elif 1.5 <= plx_ref and plx_ref < 2:
        rad, plx_r, pm_r = 7.5, 0.15, 0.35
    elif 1 <= plx_ref and plx_ref < 1.5:
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

    # # # If the coordinates distance is larger than the defined radius,
    # # # mark as no duplicate disregarding PMs and Plx, but return the
    # # # probability value
    # # if d > rad:
    # #     return False, prob

    # # PMs+plx not nans
    # if not np.isnan(plx_d) and not np.isnan(pm_d):
    #     if pm_d < pm_r and plx_d < plx_r:
    #         return True, prob

    # # plx not nan & PMs nan
    # if not np.isnan(plx_d) and np.isnan(pm_d):
    #     if plx_d < plx_r:
    #         return True, prob

    # # PMs not nan & plx nan
    # if np.isnan(plx_d) and not np.isnan(pm_d):
    #     if pm_d < pm_r:
    #         return True, prob

    # # PMs+plx both nans
    # if np.isnan(plx_d) and np.isnan(pm_d):
    #     # If the coordinates distance is within the duplicates range and
    #     # neither PMs or Plx distances could be obtained, also mark as
    #     # possible duplicate
    #     return True, prob

    # return False, prob


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


if __name__ == '__main__':
    arr = np.array([
        [112.6979,   0.9084,  np.nan, np.nan, np.nan],
        [112.7202,   0.875,   0.589,   -3.998, -3.058],
    ])
    x, y, plx, pmRA, pmDE = arr.T
    i, j = 0, 1
    print(duplicate_probs(x, y, pmRA, pmDE, plx, i, j))
    breakpoint()

