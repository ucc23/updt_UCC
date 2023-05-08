
import numpy as np
import pandas as pd
from astropy.coordinates import angular_separation
from scipy.spatial.distance import cdist


def run(df_comb, new_db_dict, idx_rm_comb_db):
    """
    """
    # Remove clusters in the new DB that were already in the old combined DB
    df_comb_no_new = df_comb.drop(df_comb.index[idx_rm_comb_db])
    df_comb_no_new = df_comb_no_new.drop(columns=[
        "dups_fnames", "Class", "Class_v", "RV", "N_membs"])
    df_comb_no_new.reset_index(drop=True, inplace=True)

    df_all = pd.concat([df_comb_no_new, pd.DataFrame(new_db_dict)],
                       ignore_index=True)
    # breakpoint()

    df_all['dups_fnames'] = find_dups(df_all)

    return df_all


def find_dups(df, N_dups=10):
    """
    Find the closest clusters to all clusters
    """
    coords = np.array([df['GLON'], df['GLAT']]).T
    # Find the distances to all clusters, for all clusters
    dist = cdist(coords, coords)
    # Change distance to itself from 0 to inf
    msk = dist == 0.
    dist[msk] = np.inf

    dups_fnames = []
    for i, cl in enumerate(dist):
        idx = np.argsort(cl)[:N_dups]

        dups_fname = []
        for j in idx:
            # Angular distance in arcmin (rounded)
            d = round(angular_separation(
                df['GLON'][i], df['GLAT'][i], df['GLON'][j],
                df['GLAT'][j]) * 60, 2)
            # PMs distance
            pm_d = np.sqrt(
                (df['pmRA'][i]-df['pmRA'][j])**2
                + (df['pmDE'][i]-df['pmDE'][j])**2)
            # Parallax distance
            plx_d = abs(df['plx'][i] - df['plx'][j])

            dup_flag = duplicate_find(d, pm_d, plx_d, df['plx'][i])

            if dup_flag:
                dups_fname.append(df['fname'][j])

            if df['fname'][i] == 'chamaleoni':
                print(dup_flag, i, j, df['fname'][j], d, pm_d, plx_d)

        if df['fname'][i] == 'chamaleoni':
            breakpoint()

        if dups_fname:
            print(i, df['fname'][i], len(dups_fname), dups_fname)
            dups_fname = ";".join(dups_fname)
        else:
            dups_fname = 'nan'

        dups_fnames.append(dups_fname)

    return dups_fnames


def duplicate_find(d, pm_d, plx_d, plx):
    """
    Identify a cluster as a duplicate following an arbitrary definition
    that depends on the parallax
    """
    if plx >= 4:
        rad, plx_r, pm_r = 15, 0.5, 1
    elif 3 <= plx and plx < 4:
        rad, plx_r, pm_r = 10, 0.25, 0.5
    elif 2 <= plx and plx < 3:
        rad, plx_r, pm_r = 5, 0.15, 0.25
    elif 1 <= plx and plx < 2:
        rad, plx_r, pm_r = 2.5, 0.1, 0.15
    else:
        rad, plx_r, pm_r = 1, 0.05, 0.1

    if pm_d < pm_r and plx_d < plx_r and d < rad:
        return True

    return False
