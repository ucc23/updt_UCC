
import datetime
import json
import csv
import pandas as pd
from modules import fastMP_process, DBs_combine, duplicates_id

#
# EDIT THIS TWO VARIABLES AS REQUIRED
# Name of the DB to add
new_DB = "PERREN22"
# Date of the latest version of the UCC catalogue
UCC_cat_date_old = "20230517"

#
# Paths to required files for the fastMP call
GAIADR3_path = '/media/gabriel/backup/gabriel/GaiaDR3/'
frames_path = GAIADR3_path + 'datafiles_G20/'
frames_ranges = GAIADR3_path + 'files_G20/frame_ranges.txt'
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

    # Add UCC_IDs and quadrant for new clusters
    ucc_ids_old = list(df_comb['UCC_ID'].values)
    for i, UCC_ID in enumerate(new_db_dict['UCC_ID']):
        if str(UCC_ID) != 'nan':
            # This cluster already has a UCC_ID assigned
            continue
        lon_i, lat_i = new_db_dict['GLON'][i], new_db_dict['GLAT'][i]
        new_db_dict['UCC_ID'][i] = DBs_combine.assign_UCC_ids(
            lon_i, lat_i, ucc_ids_old)
        new_db_dict['quad'][i] = DBs_combine.QXY_fold(new_db_dict['UCC_ID'][i])
        ucc_ids_old += [new_db_dict['UCC_ID'][i]]

    # Remove clusters in the new DB that were already in the old combined DB
    df_comb_no_new = df_comb.drop(df_comb.index[idx_rm_comb_db])
    df_comb_no_new.reset_index(drop=True, inplace=True)
    df_all = pd.concat([df_comb_no_new, pd.DataFrame(new_db_dict)],
                       ignore_index=True)

    # These duplicates are different from the final ones that are stored in
    # the final version of the catalogue. These are used to remove close
    # clusters from the field so that fastMP won't get confused
    print(f"Finding possible duplicates (max={N_dups})...")
    df_all['dups_fnames'] = DBs_combine.dups_identify(df_all, N_dups)

    d = datetime.datetime.now()
    date_new = d.strftime('%Y%m%d')
    if date_new == UCC_cat_date_old:
        date_new = date_new + '_2'
    UCC_cat = 'UCC_cat_' + date_new + '.csv'
    # Save new version of the UCC catalogue to file before processing with
    # fastMP
    df_all.to_csv(
        UCC_cat, na_rep='nan', index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"File {UCC_cat} updated")

    # Process each cluster in the new DB with fastMP and store the result in
    # the output folder. This function will also update the UCC cat file
    # 'df_UCC' with values for the columns that are still marked with 'nan'
    df_UCC = fastMP_process.run(
        fastMP, new_DB, frames_path, frames_ranges, UCC_cat, GCs_cat, out_path)

    # Finally identify possible duplicates (and assign a probability) using
    # the positions estimated with the most likely members.
    print("Finding final duplicates and their probabilities...")
    dups_fnames, dups_probs = duplicates_id.run(df_UCC)
    df_UCC['dups_fnames'] = dups_fnames  # This column is rewritten here
    df_UCC['dups_probs'] = dups_probs
    df_UCC.to_csv(
        UCC_cat, na_rep='nan', index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"File {UCC_cat} updated")

    # Update cluster's JSON file (used by 'ucc.ar' seach)
    df = pd.DataFrame(df_UCC[[
        'ID', 'fnames', 'UCC_ID', 'RA_ICRS', 'DE_ICRS', 'GLON', 'GLAT']])
    df['ID'] = [_.split(';')[0] for _ in df['ID']]
    df.to_json('../ucc/_clusters/clusters.json', orient="records", indent=1)
    print("File 'clusters.json' updated")


if __name__ == '__main__':
    main()
