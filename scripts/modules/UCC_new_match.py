import json
import os

import pandas as pd

from . import DBs_combine, read_ini_file


def load_data(
    logging, dbs_folder, all_DBs_json, UCC_folder
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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

    return df_UCC, df_new, json_pars


def standardize_and_match(
    logging,
    df_UCC: pd.DataFrame,
    df_new: pd.DataFrame,
    json_pars: dict,
    pars_dict: dict,
    new_DB: str,
    show_entries: bool = False,
) -> tuple[list[str], list[str | None]]:
    """
    Standardizes names in a new database and matches them against an existing UCC
    database.

    This function processes entries in a new database, standardizes their names,
    and attempts to match them with entries in an existing UCC database. It also
    provides logging information about the number of matches found and optionally
    displays new OCs that weren't matched.

    Args:
        logging: logger object
        df_UCC (pd.DataFrame): DataFrame containing the existing UCC database entries
        df_new (pd.DataFrame): DataFrame containing the new database entries
        json_pars (dict): Parameters for database combination operations
        pars_dict (dict): Dictionary containing parameter mappings including "ID" field
        new_DB (str): Name of the new database being processed
        show_entries (bool, optional): If True, prints unmatched OCs. Defaults to False

    Returns:
        Tuple[List[str], List[Optional[str]]]: A tuple containing:
            - List of standardized filenames from the new database
            - List of matching entries where None indicates no match found

    Logs:
        - Info about standardization process
        - Number of matches found
        - Number of new OCs found
    """
    logging.info(f"\nStandardize names in {new_DB}")
    new_DB_fnames = DBs_combine.get_fnames_new_DB(df_new, json_pars)
    db_matches = DBs_combine.get_matches_new_DB(df_UCC, new_DB_fnames)
    N_matches = sum(match is not None for match in db_matches)
    logging.info(f"Found {N_matches} matches in {new_DB}")
    N_new = len(df_new) - N_matches
    logging.info(f"Found {N_new} new OCs in {new_DB}")

    if show_entries:
        for i, oc_new_db in enumerate(df_new[pars_dict["ID"]].values):
            if db_matches[i] is None:
                print(f"  {i}: {oc_new_db.strip()}")

    return new_DB_fnames, db_matches


def latest_cat_detect(logging, UCC_folder):
    """
    Load the *latest* version of the catalogue.
    """
    all_versions = os.listdir(UCC_folder)
    last_version, vers_init = "", 0
    for file in all_versions:
        if file == "README.txt":
            continue
        version = file.split(".")[0].split("_")[-1]
        last_version = max(int(vers_init), int(version))
        vers_init = last_version

    if last_version == "":
        raise ValueError("Could not read version number")

    # Load the latest version of the combined catalogue
    UCC_cat = UCC_folder + "UCC_cat_" + str(last_version) + ".csv"
    df_UCC = pd.read_csv(UCC_cat)
    if logging is not None:
        logging.info(f"UCC version {last_version} loaded (N={len(df_UCC)})")

    return df_UCC, UCC_cat
