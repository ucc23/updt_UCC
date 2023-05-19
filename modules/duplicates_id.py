
import numpy as np
from scipy.spatial.distance import cdist


def run(df, N_dups=20):
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
            dup_prob = duplicate_find(x, y, pmRA, pmDE, plx, i, j)
            if dup_prob >= 0.5:
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

    df['dups_fnames'] = dups_fnames
    df['dups_probs'] = dups_probs
    df.to_csv("/home/gabriel/Descargas/temp.csv", index=False)
    breakpoint()

    return dups_fnames, dups_probs


def duplicate_find(x, y, pmRA, pmDE, plx, i, j):
    """
    Identify a cluster as a duplicate following an arbitrary definition
    that depends on the parallax
    """

    # Arbitrary 'duplicate regions' for different parallax brackets
    if np.isnan(plx[i]):
        plx_r, rad, pm_r = np.nan, 5, 0.5
    elif plx[i] >= 4:
        rad, plx_r, pm_r = 15, 0.5, 1
    elif 3 <= plx[i] and plx[i] < 4:
        rad, plx_r, pm_r = 10, 0.25, 0.5
    elif 2 <= plx[i] and plx[i] < 3:
        rad, plx_r, pm_r = 5, 0.15, 0.25
    elif 1 <= plx[i] and plx[i] < 2:
        rad, plx_r, pm_r = 2.5, 0.1, 0.15
    elif plx[i] < 1:
        rad, plx_r, pm_r = 2, 0.75, 0.1
    elif plx[i] < .5:
        rad, plx_r, pm_r = 1, 0.05, 0.75

    # Angular distance in arcmin
    d_prob = 0.
    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) * 60
    if d < rad:
        d_prob = lin_relation(d, rad)

    # PMs distance
    pms_prob = 0.
    pm_d = np.sqrt((pmRA[i]-pmRA[j])**2 + (pmDE[i]-pmDE[j])**2)
    if pm_d < pm_r:
        pms_prob = lin_relation(pm_d, pm_r)

    # Parallax distance
    plx_prob = 0.
    plx_d = abs(plx[i] - plx[j])
    if plx_d < plx_r:
        plx_prob = lin_relation(plx_d, plx_r)

    return round((d_prob+pms_prob+plx_prob)/3., 2)


def lin_relation(dist, d_max):
    """
    d_min=0 is fixed
    Linear relation for: (0, d_max), (1, d_min)
    """
    # m, h = (d_min - d_max), d_max
    # prob = (dist - h) / m
    # m, h = -d_max, d_max
    return (dist - d_max) / -d_max


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("/home/gabriel/Github/UCC/add_New_DB/UCC_cat_20230519.csv")
    run(df)
