import os
import json
import pandas as pd
from . import read_ini_file
from . import DBs_combine


def main(logging, dbs_folder, all_DBs_json, UCC_folder):
    """ """
    pars_dict = read_ini_file.main()
    new_DB = pars_dict["new_DB"]

    # Load column data for the new catalogue
    with open(dbs_folder + all_DBs_json) as f:
        dbs_used = json.load(f)
    json_pars = dbs_used[new_DB]

    # Load the latest version of the UCC
    df_UCC = latest_cat_detect(logging, UCC_folder)[0]

    # Load the new DB
    df_new = pd.read_csv(dbs_folder + new_DB + ".csv")
    logging.info(f"New DB {new_DB} loaded (N={len(df_new)})")

    logging.info(f"Standardize names in {new_DB}")
    new_DB_fnames = DBs_combine.get_fnames_new_DB(df_new, json_pars)

    db_matches = DBs_combine.get_matches_new_DB(df_UCC, new_DB_fnames)
    N_matches = sum(_ is not None for _ in db_matches)
    logging.info(f"Found {N_matches} matches in {new_DB}")
    N_new = len(df_new) - N_matches
    logging.info(f"Found {N_new} new OCs in {new_DB}")
    if N_new < 100:
        for i, oc_new_db in enumerate(df_new[pars_dict["ID"]].values):
            if db_matches[i] is None:
                print(f"  {i}: {oc_new_db.strip()}")

    return df_UCC, df_new, json_pars, new_DB_fnames, db_matches


def latest_cat_detect(logging, UCC_folder):
    """
    Load the *latest* version of the catalogue.
    """
    all_versions = os.listdir(UCC_folder)
    vers_init = 0
    for file in all_versions:
        version = file.split(".")[0].split("_")[-1]
        last_version = max(int(vers_init), int(version))
        vers_init = last_version

    # Load the latest version of the combined catalogue
    UCC_cat = UCC_folder + "UCC_cat_" + str(last_version) + ".csv"
    df_UCC = pd.read_csv(UCC_cat)
    if logging is not None:
        logging.info(f"UCC version {last_version} loaded (N={len(df_UCC)})")

    return df_UCC, UCC_cat


if __name__ == "__main__":
    main()
