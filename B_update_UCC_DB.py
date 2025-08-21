import json
import os
import re
import sys

import numpy as np
import pandas as pd

from modules.HARDCODED import (
    GCs_cat,
    UCC_folder,
    dbs_folder,
    manual_pars_file,
    name_DBs_json,
    temp_fold,
)
from modules.update_database.add_new_DB_funcs import (
    QXY_fold,
    assign_UCC_ids,
    combine_UCC_new_DB,
    duplicates_check,
    duplicates_fnames_check,
)
from modules.update_database.check_new_DB_funcs import (
    GCs_check,
    close_OC_inner_check,
    close_OC_UCC_check,
    dups_check_newDB_UCC,
    dups_fnames_inner_check,
    fnames_check_UCC_new_DB,
    positions_check,
    prep_newDB,
    vdberg_check,
)
from modules.update_database.standardize_and_match_funcs import (
    get_fnames_new_DB,
    get_matches_new_DB,
)
from modules.utils import (
    diff_between_dfs,
    get_last_version_UCC,
    logger,
    radec2lonlat,
    save_df_UCC,
)

# =========================================================================
# Select the mode used to run the script
updt_DB_name = ""

# # Add a new DB
# run_mode = "new_DB"

# Update the UCC for selected entries listed in the 'manual_pars' file
run_mode = "manual"

# # Update a database already included in the UCC
# run_mode = "updt_DB"
# updt_DB_name = "CHI2023_1"
# =========================================================================


def main():
    """
    First function to update the UCC (Unified Cluster Catalogue) with a new database.
    """
    logging = logger()

    if run_mode not in ("new_DB", "manual", "updt_DB"):
        raise ValueError(
            f"Invalid run_mode '{run_mode}'. "
            + "Choose from: 'new_DB', 'manual' or 'updt_DB'"
        )

    logging.info(f"=== Running B script in '{run_mode}' mode ===\n")

    # Generate paths and check for required folders and files
    (
        temp_database_folder,
        ucc_file,
        temp_JSON_file,
    ) = get_paths_check_paths(logging)

    (
        df_UCC_old,
        current_JSON,
        df_GCs,
        manual_pars,
        df_new,
        newDB_json,
        new_DB,
    ) = load_data(logging, ucc_file, temp_JSON_file, temp_database_folder)

    if run_mode == "new_DB" or run_mode == "updt_DB":
        # Check for required columns in the new DB
        check_new_DB_cols(logging, run_mode, current_JSON, new_DB, df_new, newDB_json)

        # Standardize and match the new DB with the UCC
        new_DB_fnames, db_matches = standardize_and_match(
            logging, new_DB, df_UCC_old, df_new, newDB_json
        )

        # Check the entries in the new DB
        check_new_DB(
            logging,
            df_GCs,
            new_DB,
            df_UCC_old,
            df_new,
            newDB_json,
            new_DB_fnames,
            db_matches,
        )

        # Generate new UCC file with the new DB incorporated
        df_UCC_new = add_new_DB(
            logging,
            run_mode,
            new_DB,
            newDB_json,
            df_UCC_old,
            df_new,
            new_DB_fnames,
            db_matches,
        )
        diff_between_dfs(logging, df_UCC_old, df_UCC_new)
    else:  # run_mode == "manual"
        fname_UCC = df_UCC_old["fnames"].str.split(";").str[0]
        # Find matches between df1['name'] and df2['first_fnames']
        matches = fname_UCC.isin(manual_pars["fname"])
        # Update `N_50` for matching rows
        df_UCC_old.loc[matches, "N_50"] = np.nan
        #
        df_UCC_new = df_UCC_old

    N_new = np.isnan(df_UCC_new["N_50"]).sum()
    logging.info(f"Entries marked for re-processing: {N_new}")

    if input("\nMove files to their final paths? (y/n): ").lower() != "y":
        sys.exit()
    move_files(
        logging,
        run_mode,
        temp_JSON_file,
        new_DB,
        df_UCC_new,
        temp_database_folder,
    )

    logging.info("\nProceed with the next script")


def get_paths_check_paths(
    logging,
) -> tuple[
    str,
    str,
    str,
]:
    """ """
    # If file exists, read and return it
    if os.path.isfile(temp_fold + "df_UCC_B_updt.csv"):
        logging.warning(
            "WARNING: file 'df_UCC_B_updt.csv' exists. Moving on will re-write it"
        )
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Temporary databases/ folder
    temp_database_folder = ""
    if run_mode == "new_DB":
        temp_database_folder = temp_fold + dbs_folder
        # Create if required
        if not os.path.exists(temp_database_folder):
            os.makedirs(temp_database_folder)

    last_version = get_last_version_UCC(UCC_folder)
    # Path to the current UCC csv file
    ucc_file = UCC_folder + last_version

    # Path to the new (temp) JSON file
    temp_JSON_file = temp_fold + name_DBs_json

    return (
        temp_database_folder,
        ucc_file,
        temp_JSON_file,
    )


def load_data(
    logging,
    ucc_file: str,
    temp_JSON_file: str,
    temp_database_folder: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    str,
]:
    """ """
    # Load current UCC version
    df_UCC = pd.read_csv(ucc_file)
    logging.info(f"\nUCC version {ucc_file} loaded (N={len(df_UCC)})")

    # Load current JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    # Attempt to load actual data
    if run_mode == "new_DB" or run_mode == "updt_DB":
        # Extract new DB's name
        if run_mode == "new_DB":
            # Load new temp JSON file
            with open(temp_JSON_file) as f:
                temp_JSON = json.load(f)

            new_DB = list(set(temp_JSON.keys()) - set(current_JSON.keys()))[0]
            # Load column data for the new catalogue
            newDB_json = temp_JSON[new_DB]
            read_dbs_folder = temp_database_folder
        else:
            new_DB = updt_DB_name
            # Load column data for the catalogue
            newDB_json = current_JSON[new_DB]
            read_dbs_folder = dbs_folder

        # Load the DB
        new_DB_file = new_DB + ".csv"
        df_new = pd.read_csv(read_dbs_folder + new_DB_file)
        logging.info(f"New DB {new_DB} loaded (N={len(df_new)})")
        # Dummy data
        manual_pars = pd.DataFrame()

    else:  # run_mode == "manual"
        # Read OCs manual parameters
        manual_pars = pd.read_csv(manual_pars_file)
        # Find repeated entries in 'fname' column
        msk = manual_pars["fname"].duplicated()
        if msk.sum() > 0:
            rep_fnames = ", ".join([_ for _ in manual_pars[msk]["fname"]])
            raise ValueError(
                f"Repeated entries in {manual_pars_file} file:\n{rep_fnames}"
            )
        # Dummy data
        new_DB, new_DB_file, df_new, newDB_json = "", "", pd.DataFrame(), {}

    return (
        df_UCC,
        current_JSON,
        df_GCs,
        manual_pars,
        df_new,
        newDB_json,
        new_DB,
    )


def check_new_DB_cols(
    logging, run_mode, current_JSON, new_DB, df_new, newDB_json, show_entries=True
) -> None:
    """ """
    # Check that the new DB is not already present in the 'old' JSON file
    if new_DB in current_JSON.keys() and run_mode != "updt_DB":
        raise ValueError(f"The DB '{new_DB}' is in the current JSON file")

    # Check for required columns
    cols = [newDB_json["names"]]
    for entry in ("pos", "pars", "e_pars"):
        for _, v in newDB_json[entry].items():
            cols.append(v)
    # Access columns stored in JSON file
    for col in cols:
        df_new[col]
    logging.info(f"\nAll columns present in {new_DB}")

    # Check for bad characters in name column
    logging.info("\nPossible bad characters in names '(, ), ;, *'")
    all_bad_names = []
    for new_cl in df_new[newDB_json["names"]]:
        if bool(re.search(r"[();*]", new_cl)):
            all_bad_names.append(new_cl)
    if len(all_bad_names) == 0:
        logging.info("No bad-chars found in name(s) column")
    else:
        logging.info(
            f"{len(all_bad_names)} entries with bad-chars found in name(s) column"
        )
        if show_entries:
            for new_cl in all_bad_names:
                logging.info(f"{new_cl}: bad char found")
        raise ValueError("Resolve the above issues before moving on.")


def standardize_and_match(
    logging,
    new_DB: str,
    df_UCC: pd.DataFrame,
    df_new: pd.DataFrame,
    newDB_json: dict,
    show_entries: bool = True,
) -> tuple[list[list[str]], list[int | None]]:
    """
    Standardizes names in a new database and matches them against an existing UCC
    database.

    This function processes entries in a new database, standardizes their names,
    and attempts to match them with entries in an existing UCC database. It also
    provides logging information about the number of matches found and optionally
    displays new OCs that weren't matched.

    Args:
        logging: logger object
        new_DB (str): Name of the new DB
        df_UCC (pd.DataFrame): DataFrame containing the existing UCC database entries
        df_new (pd.DataFrame): DataFrame containing the new database entries
        newDB_json (dict): Parameters for database combination operations
        show_entries (bool, optional): If True, prints unmatched OCs. Defaults to False

    Returns:
        tuple[list[list[str]], list[int | None]]: A tuple containing:
            - List of list of standardized filenames from the new database
            - List of matching entries where None indicates no match found
    """
    logging.info(f"\nStandardize names in {new_DB}")
    new_DB_fnames = get_fnames_new_DB(df_new, newDB_json)
    db_matches = get_matches_new_DB(df_UCC, new_DB_fnames)

    N_matches = sum(match is not None for match in db_matches)
    logging.info(f"Found {N_matches} matches in {new_DB}")
    N_new = len(df_new) - N_matches
    logging.info(f"Found {N_new} new OCs in {new_DB}")

    name_col = newDB_json["names"]
    if show_entries:
        for i, oc_new_db in enumerate(df_new[name_col].values):
            if db_matches[i] is None:
                logging.info(f"  {i}: {oc_new_db.strip()}")

    return new_DB_fnames, db_matches


def check_new_DB(
    logging,
    df_GCs: pd.DataFrame,
    new_DB: str,
    df_UCC,
    df_new,
    newDB_json,
    new_DB_fnames,
    db_matches,
    rad_dup=10,
) -> None:
    """
    1. Checks for duplicate entries between the new database and the UCC.
    2. Checks for nearby GCs.
    3. Checks for OCs very close to each other within the new database.
    4. Checks for OCs very close to each other between the new database and the UCC.
    5. Checks for instances of 'vdBergh-Hagen' and 'vdBergh'.
    6. Checks positions and flags for attention if required.
    """
    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    logging.info("\nPossible vdBergh-Hagen/vdBergh check")
    if vdberg_check(logging, newDB_json, df_new):
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    ra_new, dec_new, lon_new, lat_new = None, None, None, None
    if "RA" in newDB_json["pos"].keys() and "DEC" in newDB_json["pos"].keys():
        ra_new = df_new[newDB_json["pos"]["RA"]].values
        dec_new = df_new[newDB_json["pos"]["DEC"]].values
        lon_new, lat_new = radec2lonlat(ra_new, dec_new)

    if lon_new is not None:
        # Check for GCs
        logging.info("\nClose GC check")
        if GCs_check(logging, df_GCs, newDB_json, df_new, lon_new, lat_new):
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit()

    # Check all fnames in the new DB against all fnames in the UCC
    if fnames_check_UCC_new_DB(logging, df_UCC, new_DB_fnames):
        raise ValueError("\nResolve the above issues before moving on")

    # Check the first fname for all entries in the new DB
    if dups_fnames_inner_check(logging, new_DB, newDB_json, df_new, new_DB_fnames):
        raise ValueError("\nResolve the above issues before moving on")

    # Check for duplicate entries in the new DB that also exist in the UCC
    if dups_check_newDB_UCC(logging, new_DB, df_UCC, new_DB_fnames, db_matches):
        raise ValueError("\nResolve the above issues before moving on")

    if ra_new is not None:
        # Check for OCs very close to each other in the new DB
        logging.info("\nProbable inner duplicates check")
        if close_OC_inner_check(logging, newDB_json, df_new, ra_new, dec_new, rad_dup):
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit()

        # Check for OCs very close to other OCs in the UCC
        logging.info("\nProbable UCC duplicates check")
        if close_OC_UCC_check(
            logging, df_UCC, new_DB_fnames, db_matches, lon_new, lat_new, rad_dup
        ):
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit()

    # Prepare information from a new database matched with the UCC
    new_db_info = prep_newDB(newDB_json, df_new, new_DB_fnames, db_matches)

    # Check positions and flag for attention if required
    if positions_check(logging, df_UCC, new_db_info, rad_dup):
        if input("\nMove on? (y/n): ").lower() != "y":
            sys.exit()


def add_new_DB(
    logging,
    run_mode: str,
    new_DB: str,
    newDB_json: dict,
    df_UCC_old,
    df_new,
    new_DB_fnames,
    db_matches,
) -> pd.DataFrame:
    """
    Adds a new database to the Unified Cluster Catalogue (UCC).
    This function performs the following steps:

    1. Combines the UCC and the new database.
    2. Assigns UCC IDs and quadrants for new clusters.
    3. Drops clusters from the UCC that are present in the new database.
    4. Performs a final duplicate check.

    Args:
        logging (logging.Logger): Logger instance for logging messages.
        run_mode (str): Running mode of the script
        pars_dict (dict): Dictionary containing parameters for the new database.
        df_UCC (pd.DataFrame): DataFrame containing the current UCC.
        df_new (pd.DataFrame): DataFrame containing the new database.
        json_pars (dict): Dictionary containing JSON parameters for the new database.
        new_DB_fnames (list): List of lists, each containing the filenames of the
        entries in the new database.
        db_matches (list): List of indexes into the UCC pointing to each entry in the
        new database.

    Returns:
        pd.DataFrame: Updated UCC DataFrame with the new database incorporated.

    Raises:
        ValueError: If duplicated entries are found in the 'ID', 'UCC_ID', or 'fnames'
        columns.
    """

    logging.info(f"Adding DB: {new_DB}")
    new_db_dict = combine_UCC_new_DB(
        logging,
        run_mode,
        new_DB,
        newDB_json,
        df_UCC_old,
        df_new,
        new_DB_fnames,
        db_matches,
    )

    # Add UCC_IDs and quadrants for new clusters
    ucc_ids_old = list(df_UCC_old["UCC_ID"].values)
    for i, UCC_ID in enumerate(new_db_dict["UCC_ID"]):
        # Only process new OCs
        if str(UCC_ID) != "nan":
            continue
        new_db_dict["UCC_ID"][i] = assign_UCC_ids(
            logging, new_db_dict["GLON"][i], new_db_dict["GLAT"][i], ucc_ids_old
        )
        new_db_dict["quad"][i] = QXY_fold(new_db_dict["UCC_ID"][i])
        ucc_ids_old += [new_db_dict["UCC_ID"][i]]

    # Drop OCs from the UCC that are present in the new DB
    # Remove 'None' entries first from the indexes list
    idx_rm_comb_db = [_ for _ in db_matches if _ is not None]
    df_UCC_no_new = df_UCC_old.drop(list(df_UCC_old.index[idx_rm_comb_db]))
    df_UCC_no_new.reset_index(drop=True, inplace=True)
    df_UCC_new = pd.concat(
        [df_UCC_no_new, pd.DataFrame(new_db_dict)], ignore_index=True
    )

    # NaN as "nan" in "N_50" column (important to identify new OCs to apply fastMP)
    df_UCC_new["N_50"] = df_UCC_new["N_50"].fillna(np.nan)

    # Final duplicate check
    if duplicates_check(logging, df_UCC_new):
        raise ValueError(
            "Duplicated entries found in either 'ID, UCC_ID, fnames' column"
        )

    # Check every individual fname for duplicates
    if duplicates_fnames_check(logging, df_UCC_new):
        raise ValueError("Duplicated entries found in 'fnames' column")

    return df_UCC_new


def move_files(
    logging,
    run_mode: str,
    temp_JSON_file: str,
    new_DB: str,
    df_UCC_new: pd.DataFrame,
    temp_database_folder: str,
) -> None:
    """ """
    logging.info("\nUpdate files...")

    # Update JSON file with all the DBs and store the new DB in place
    if run_mode == "new_DB":
        # Move JSON file
        json_stored = name_DBs_json
        json_temp = temp_JSON_file
        os.rename(json_temp, json_stored)
        logging.info(json_temp + " --> " + json_stored)

        # Move new DB file
        new_DB_file = new_DB + ".csv"
        db_stored = dbs_folder + new_DB_file
        db_temp = temp_database_folder + new_DB_file
        os.rename(db_temp, db_stored)
        logging.info(db_temp + " --> " + db_stored)

    # Save df_UCC_new to temp file
    file_path = temp_fold + "df_UCC_B_updt.csv"
    save_df_UCC(logging, df_UCC_new, file_path)


if __name__ == "__main__":
    main()
