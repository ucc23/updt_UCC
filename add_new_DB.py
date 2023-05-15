
import datetime
import json
import csv
import pandas as pd
# from astropy.coordinates import angular_separation
from modules import fastMP_process, DBs_combine


# Name of the DB to add
new_DB = "PERREN22"

# Date of the latest version of the UCC catalogue
UCC_cat_date_old = "20230508"

# Paths to required files for the fastMP call
GAIADR3_path = '/media/gabriel/backup/gabriel/GaiaDR3/'
frames_path = GAIADR3_path + 'datafiles_G20/'
frames_ranges = GAIADR3_path + 'files_G20/frame_ranges.txt'
UCC_cat = "UCC_cat_20230511.csv"
GCs_cat = "./databases/globulars.csv"
out_path = "../../"
# Load local version of fastMP
# insert at 1, 0 is the script path (or '' in REPL)
import sys
sys.path.insert(1, '/home/gabriel/Github/fastmp/')  # Path to fastMP
from fastmp import fastMP


def main(
    dbs_folder='databases/', DBs_json='all_dbs.json', sep=',', N_dups=10
):
    """
    """

    # Load column data for the new catalogue
    with open(dbs_folder + DBs_json) as f:
        dbs_used = json.load(f)
    json_pars = dbs_used[new_DB]

    # Load the latest version of the combined catalogue: 'UCC_cat_20XXYYZZ.csv'
    df_comb = pd.read_csv("UCC_cat_" + UCC_cat_date_old + ".csv")
    print(f"N={len(df_comb)} clusters in combined DB")

    # Load the new DB
    df_new = pd.read_csv(dbs_folder + new_DB + '.csv')
    print(f"N={len(df_new)} clusters in new DB")

    new_DB_fnames = DBs_combine.get_fnames_new_DB(df_new, json_pars, sep)

    db_matches = DBs_combine.get_matches_new_DB(df_comb, new_DB_fnames)

    new_db_dict, idx_rm_comb_db = DBs_combine.combine_new_DB(
        new_DB, df_comb, df_new, json_pars, new_DB_fnames, db_matches, sep)
    print(f"N={len(df_new) - len(idx_rm_comb_db)} new clusters in new DB")

    ucc_ids_old = list(df_comb['UCC_ID'].values)
    for i, UCC_ID in enumerate(new_db_dict['UCC_ID']):
        if str(UCC_ID) != 'nan':
            # This cluster already has a UCC_ID assigned
            continue
        lon_i, lat_i = new_db_dict['GLON'][i], new_db_dict['GLAT'][i]
        new_db_dict['UCC_ID'][i] = DBs_combine.assign_UCC_ids(
            lon_i, lat_i, ucc_ids_old)
        ucc_ids_old += [new_db_dict['UCC_ID'][i]]

    # Remove clusters in the new DB that were already in the old combined DB
    df_comb_no_new = df_comb.drop(df_comb.index[idx_rm_comb_db])
    df_comb_no_new.reset_index(drop=True, inplace=True)
    df_all = pd.concat([df_comb_no_new, pd.DataFrame(new_db_dict)],
                       ignore_index=True)

    print(f"Finding possible duplicates (max={N_dups})...")
    df_all['dups_fnames'] = DBs_combine.dups_identify(df_all, N_dups)

    # Save new version of the UCC catalogue to file
    d = datetime.datetime.now()
    date = d.strftime('%Y%m%d')
    UCC_cat = 'UCC_cat_' + date + '.csv'
    df_all.to_csv(
        UCC_cat, na_rep='nan', index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"File {UCC_cat} updated")

    # Update cluster's JSON file (used by 'ucc.ar' seach)
    df = pd.DataFrame(df_all[[
        'ID', 'fnames', 'UCC_ID', 'RA_ICRS', 'DE_ICRS', 'GLON', 'GLAT']])
    df['ID'] = [_.split(';')[0] for _ in df['ID']]
    df.to_json('../ucc/_clusters/clusters.json', orient="records", indent=1)
    print("File 'clusters.json' updated")

    # Process each cluster in the new DB with fastMP and store the result in
    # the output folder
    # fastMP_process.run(new_DB, UCC_cat)
    fastMP_process.run(
        fastMP, new_DB, frames_path, frames_ranges, UCC_cat, GCs_cat, out_path)


# def dups_identify(df, N_dups):
#     """
#     Find the closest clusters to all clusters
#     """
#     x, y = df['GLON'], df['GLAT']
#     pmRA, pmDE, plx = df['pmRA'], df['pmDE'], df['plx']
#     coords = np.array([x, y]).T
#     # Find the distances to all clusters, for all clusters
#     dist = cdist(coords, coords)
#     # Change distance to itself from 0 to inf
#     msk = dist == 0.
#     dist[msk] = np.inf

#     dups_fnames = []
#     for i, cl in enumerate(dist):
#         idx = np.argsort(cl)[:N_dups]

#         dups_fname = []
#         for j in idx:
#             # Angular distance in arcmin (rounded)
#             d = round(angular_separation(x[i], y[i], x[j], y[j]) * 60, 2)
#             # PMs distance
#             pm_d = np.sqrt((pmRA[i]-pmRA[j])**2 + (pmDE[i]-pmDE[j])**2)
#             # Parallax distance
#             plx_d = abs(plx[i] - plx[j])

#             dup_flag = duplicate_find(d, pm_d, plx_d, plx[i])

#             if dup_flag:
#                 fname = df['fnames'][j].split(';')[0]
#                 dups_fname.append(fname)

#         if dups_fname:
#             # print(i, df['fnames'][i], len(dups_fname), dups_fname)
#             dups_fname = ";".join(dups_fname)
#         else:
#             dups_fname = 'nan'

#         dups_fnames.append(dups_fname)

#     return dups_fnames


# def duplicate_find(d, pm_d, plx_d, plx):
#     """
#     Identify a cluster as a duplicate following an arbitrary definition
#     that depends on the parallax
#     """
#     if plx >= 4:
#         rad, plx_r, pm_r = 15, 0.5, 1
#     elif 3 <= plx and plx < 4:
#         rad, plx_r, pm_r = 10, 0.25, 0.5
#     elif 2 <= plx and plx < 3:
#         rad, plx_r, pm_r = 5, 0.15, 0.25
#     elif 1 <= plx and plx < 2:
#         rad, plx_r, pm_r = 2.5, 0.1, 0.15
#     else:
#         rad, plx_r, pm_r = 1, 0.05, 0.1

#     if not np.isnan(plx_d) and not np.isnan(pm_d):
#         if pm_d < pm_r and plx_d < plx_r and d < rad:
#             return True
#     elif not np.isnan(plx_d) and np.isnan(pm_d):
#         if plx_d < plx_r and d < rad:
#             return True
#     elif np.isnan(plx_d):
#         rad, pm_r = 5, 0.5
#         if not np.isnan(pm_d):
#             if pm_d < pm_r and d < rad:
#                 return True
#         else:
#             if d < rad:
#                 return True

#     return False


if __name__ == '__main__':
    main()
