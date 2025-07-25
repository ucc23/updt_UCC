import csv
import datetime
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from modules.HARDCODED import (
    GCs_cat,
    UCC_archive,
    UCC_folder,
    UCC_members_file,
    dbs_folder,
    manual_pars_file,
    members_folder,
    name_DBs_json,
    parquet_dates,
    path_gaia_frames,
    path_gaia_frames_ranges,
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
from modules.update_database.member_files_updt_funcs import get_fastMP_membs
from modules.update_database.standardize_and_match_funcs import (
    get_fnames_new_DB,
    get_matches_new_DB,
)
from modules.utils import (
    get_last_version_UCC,
    logger,
    radec2lonlat,
)

# =========================================================================
# Select the mode used to run the script

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
    Main function to update the UCC (Unified Cluster Catalogue) with a new database.
    """
    logging = logger()

    logging.info(f"=== Running in {run_mode} mode ===\n")

    # Generate paths and check for required folders and files
    (
        temp_database_folder,
        ucc_file,
        temp_zenodo_fold,
        new_ucc_file,
        temp_JSON_file,
        archived_UCC_file,
    ) = get_paths_check_paths(logging)

    (
        gaia_frames_data,
        df_GCs,
        manual_pars,
        df_UCC_old,
        current_JSON,
        df_new,
        newDB_json,
        new_DB_file,
        new_DB,
    ) = load_data(logging, ucc_file, temp_JSON_file, temp_database_folder)

    if run_mode == "new_DB" or run_mode == "updt_DB":
        # 1. Check for required columns in the new DB
        check_new_DB_cols(logging, run_mode, current_JSON, new_DB, df_new, newDB_json)

        # 2. Standardize and match the new DB with the UCC
        new_DB_fnames, db_matches = standardize_and_match(
            logging, new_DB, df_UCC_old, df_new, newDB_json
        )

        # 3. Check the entries in the new DB
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

        # 4. Generate new UCC file with the new DB incorporated
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
        df_UCC_new2 = diff_between_dfs(logging, df_UCC_old, df_UCC_new)
        if input("UCC updated. Move on to fastMP? (y/n): ").lower() != "y":
            sys.exit()
    else:  # run_mode == "manual"
        fname_UCC = df_UCC_old["fnames"].str.split(";").str[0]
        # Find matches between df1['name'] and df2['first_fnames']
        matches = fname_UCC.isin(manual_pars["fname"])
        # Update `N_50` for matching rows
        df_UCC_old.loc[matches, "N_50"] = np.nan
        #
        df_UCC_new2 = df_UCC_old

    # 5. Entries with no 'N_50' value are identified as new and processed with fastMP
    N_new = np.isnan(df_UCC_new2["N_50"]).sum()

    if N_new > 0:
        logging.info(f"\nProcessing {N_new} new OCs in '{new_DB}' with fastMP")

        # # Load file if it already exists and the .parquet files were generated
        # df_UCC_updt = pd.read_csv(temp_fold + "df_UCC_updt.csv")

        # Generate member files for new OCs and obtain their data
        df_UCC_updt = member_files_updt(
            logging, run_mode, df_UCC_new2, gaia_frames_data, df_GCs, manual_pars
        )

        # Update the UCC with the new OCs member's data. Obtain here the duplicate
        # probabilities using the members data
        df_UCC_new3 = update_UCC_membs_data(df_UCC_new2, df_UCC_updt)
        df_UCC_new4 = diff_between_dfs(logging, df_UCC_new2, df_UCC_new3)
    else:
        logging.info("No new OCs to process")
        df_UCC_new4 = df_UCC_new2

    # 6. Save updated UCC to CSV file
    save_final_UCC(logging, temp_zenodo_fold, new_ucc_file, df_UCC_new4)

    # 7. Move temporary files to their final destination
    if input("\nMove files to their final destination? (y/n): ").lower() != "y":
        sys.exit()
    move_files(
        logging,
        run_mode,
        temp_JSON_file,
        new_DB_file,
        ucc_file,
        new_ucc_file,
        temp_zenodo_fold,
        temp_database_folder,
        archived_UCC_file,
    )

    # # 8. Final check of new UCC
    # N_UCC = len(df_UCC_new4)
    # file_checker(logging, N_UCC, root_UCC_path)

    # if input("\nRemove temporary files and folders? (y/n): ").lower() == "y":
    #     # shutil.rmtree(temp_fold)
    #     logging.info(f"Folder removed: {temp_fold}")

    logging.info("\nAll done! Proceed with the next script")


def get_paths_check_paths(
    logging,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
]:
    """ """
    txt = ""
    # Check for Gaia files
    if not os.path.isdir(path_gaia_frames):
        # raise FileNotFoundError(f"Folder {path_gaia_frames} is not present")
        txt += f"Folder {path_gaia_frames} is not present\n"
    if not os.path.isfile(path_gaia_frames_ranges):
        # raise FileNotFoundError(f"File {path_gaia_frames_ranges} is not present")
        txt += f"File {path_gaia_frames_ranges} is not present"
    if txt != "":
        logging.info(txt)
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # If file exists, read and return it
    if os.path.isfile(temp_fold + "df_UCC_updt.csv"):
        logging.warning(
            "WARNING: file 'df_UCC_updt.csv' exists. Moving on will re-write it"
        )
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Temporary databases/ folder
    temp_database_folder = temp_fold + dbs_folder
    # Create if required
    if not os.path.exists(temp_database_folder):
        os.makedirs(temp_database_folder)

    # Create folder to store parquet member files
    out_path = temp_fold + members_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        if len(os.listdir(out_path)) > 0:
            logging.warning(
                f"WARNING: There are .parquet files in '{out_path}'"
                + "\nIf left there, they will be used when the script ends"
            )
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit()

    last_version = get_last_version_UCC(UCC_folder)
    # Path to the current UCC csv file
    ucc_file = UCC_folder + last_version

    # Temporary zenodo/ folder
    temp_zenodo_fold = temp_fold + UCC_folder
    # Create if required
    if not os.path.exists(temp_zenodo_fold):
        os.makedirs(temp_zenodo_fold)

    # Path to the new (temp) version of the UCC database
    new_version = datetime.datetime.now().strftime("%Y%m%d%H")[2:]
    new_ucc_file = "UCC_cat_" + new_version + ".csv"
    # Check if file already exists
    if os.path.exists(temp_zenodo_fold + new_ucc_file):
        logging.info(
            f"File {temp_zenodo_fold + new_ucc_file} already exists. Moving on will re-write it"
        )
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Path to the new (temp) JSON file
    temp_JSON_file = temp_fold + name_DBs_json

    # Path to archive the current UCC csv file
    archived_UCC_file = UCC_archive + last_version.replace(".csv", ".csv.gz")

    return (
        temp_database_folder,
        ucc_file,
        temp_zenodo_fold,
        new_ucc_file,
        temp_JSON_file,
        archived_UCC_file,
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
    pd.DataFrame,
    dict,
    str,
    str,
]:
    """ """
    # Load file with Gaia frames ranges
    gaia_frames_data = pd.DataFrame([])
    if os.path.isfile(path_gaia_frames_ranges):
        gaia_frames_data = pd.read_csv(path_gaia_frames_ranges)
    else:
        warnings.warn(f"File {path_gaia_frames_ranges} not found")

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    df_UCC = pd.read_csv(ucc_file)
    logging.info(f"\nUCC version {ucc_file} loaded (N={len(df_UCC)})")

    # Load current JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    # Dummy
    manual_pars = pd.DataFrame()
    df_new = pd.DataFrame()
    newDB_json = {}
    new_DB_file = ""
    new_DB = "manual_pars"
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

    else:  # run_mode == "manual"
        # Read OCs manual parameters
        manual_pars = pd.read_csv(manual_pars_file)

    return (
        gaia_frames_data,
        df_GCs,
        manual_pars,
        df_UCC,
        current_JSON,
        df_new,
        newDB_json,
        new_DB_file,
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

    # Final duplicate check
    if duplicates_check(logging, df_UCC_new):
        raise ValueError(
            "Duplicated entries found in either 'ID, UCC_ID, fnames' column"
        )

    # Check every individual fname for duplicates
    if duplicates_fnames_check(logging, df_UCC_new):
        raise ValueError("Duplicated entries found in 'fnames' column")

    return df_UCC_new


def member_files_updt(
    logging, run_mode, df_UCC, gaia_frames_data, df_GCs, manual_pars
) -> pd.DataFrame:
    """
    Updates the Unified Cluster Catalogue (UCC) with new open clusters (OCs).
    """
    # For each new OC
    df_UCC_updt = {
        "UCC_idx": [],
        "N_clust": [],
        "N_clust_max": [],
        "C3": [],
        "GLON_m": [],
        "GLAT_m": [],
        "RA_ICRS_m": [],
        "DE_ICRS_m": [],
        "Plx_m": [],
        "pmRA_m": [],
        "pmDE_m": [],
        "Rv_m": [],
        "N_Rv": [],
        "N_50": [],
        "r_50": [],
    }
    N_cl = 0
    for UCC_idx, new_cl in df_UCC.iterrows():
        # Check if this is a new OC that should be processed
        if not np.isnan(new_cl["N_50"]):
            continue

        fnames, ra_c, dec_c, glon_c, glat_c, pmra_c, pmde_c, plx_c = (
            new_cl["fnames"],
            float(new_cl["RA_ICRS"]),
            float(new_cl["DE_ICRS"]),
            float(new_cl["GLON"]),
            float(new_cl["GLAT"]),
            float(new_cl["pmRA"]),  # This can be nan
            float(new_cl["pmDE"]),  # This can be nan
            float(new_cl["Plx"]),  # This can be nan
        )
        fname0 = str(fnames).split(";")[0]
        logging.info(f"\n{N_cl} Processing {fname0} (idx={UCC_idx})")

        df_UCC_updt = get_fastMP_membs(
            logging,
            run_mode,
            manual_pars,
            df_UCC,
            df_GCs,
            gaia_frames_data,
            UCC_idx,
            ra_c,
            dec_c,
            glon_c,
            glat_c,
            pmra_c,
            pmde_c,
            plx_c,
            fname0,
            df_UCC_updt,
        )

        # Update file with information. Do this for each iteration to avoid
        # losing data if something goes wrong with any cluster
        df = pd.DataFrame(df_UCC_updt)
        df.to_csv(temp_fold + "df_UCC_updt.csv", index=False, na_rep="nan")

        N_cl += 1

    # This dataframe (and file) contains the data extracted from all the new entries,
    # used to update the UCC
    df_UCC_updt = pd.DataFrame(df_UCC_updt)
    logging.info("\nTemp file df_UCC_updt saved")

    return df_UCC_updt


def update_UCC_membs_data(df_UCC, df_UCC_updt) -> pd.DataFrame:
    """
    Update the UCC database using the data extracted from the processed OCs'
    members.
    """
    # Generate copy to not disturb the dataframe given to this function which is
    # later used to generate the diffs files
    df_inner = df_UCC.copy()

    # Update 'df_inner' using 'df_UCC_updt' data
    for _, row in df_UCC_updt.iterrows():
        UCC_idx = row["UCC_idx"]
        for key, val in row.items():
            if key == "UCC_idx":
                continue
            df_inner.at[UCC_idx, key] = val

    return df_inner


def diff_between_dfs(
    logging,
    df_old: pd.DataFrame,
    df_new_in: pd.DataFrame,
    cols_exclude=None,
) -> pd.DataFrame:
    """
    Order by (lon, lat) and change NaN as "nan".

    Compare two DataFrames, find non-matching rows while preserving order, and
    output these rows in two files.

    Args:
        df_old (pd.DataFrame): First DataFrame to compare.
        df_new (pd.DataFrame): Second DataFrame to compare.
        cols_exclude (list | None): List of columns to exclude from the diff
    """
    # Order by (lon, lat)
    df_new = df_new_in.copy()
    df_new = df_new.sort_values(["GLON", "GLAT"])
    df_new = df_new.reset_index(drop=True)

    # NaN as "nan" in "N_50" column (important to identify OCs to apply fastMP)
    df_new["N_50"] = df_new["N_50"].fillna(np.nan)

    if cols_exclude is not None:
        logging.info(f"\n{cols_exclude} columns excluded")
        for col in cols_exclude:
            if col in df_old.keys():
                df_old = df_old.drop(columns=(col))
            if col in df_new.keys():
                df_new = df_new.drop(columns=(col))
    else:
        logging.info("\nNo columns excluded")
    df1 = df_old
    df2 = df_new

    # Convert DataFrames to lists of tuples (rows) for comparison
    rows1 = [[str(_) for _ in row] for row in df1.values]
    rows2 = [[str(_) for _ in row] for row in df2.values]

    # Convert lists to sets for quick comparison
    set1, set2 = set(map(tuple, rows1)), set(map(tuple, rows2))

    # Get non-matching rows in original order
    non_matching1 = [row for row in rows1 if tuple(row) not in set2]
    non_matching2 = [row for row in rows2 if tuple(row) not in set1]

    if len(non_matching1) == 0 and len(non_matching2) == 0:
        logging.info("No differences found\n")
        return df_new

    if len(non_matching1) > 0:
        # Write intertwined lines to the output file
        with open(temp_fold + "UCC_diff_old.csv", "w", newline="") as out:
            writer = csv.writer(out)
            for row in non_matching1:
                writer.writerow(row)
    if len(non_matching2) > 0:
        with open(temp_fold + "UCC_diff_new.csv", "w", newline="") as out:
            writer = csv.writer(out)
            for row in non_matching2:
                writer.writerow(row)

    logging.info("Files 'UCC_diff_xxx.csv' saved\n")
    return df_new


def save_final_UCC(
    logging, temp_zenodo_fold: str, new_ucc_file: str, df_UCC: pd.DataFrame
) -> None:
    """ """

    # Order by (lon, lat) first
    df_UCC = df_UCC.sort_values(["GLON", "GLAT"])
    df_UCC = df_UCC.reset_index(drop=True)
    # Save UCC to CSV file
    df_UCC.to_csv(
        temp_zenodo_fold + new_ucc_file,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    logging.info(f"UCC updated: {temp_zenodo_fold + new_ucc_file} (N={len(df_UCC)})")


def fnames_checker(df_UCC: pd.DataFrame) -> None:
    """
    Ensure that filenames in the DataFrame are unique.

    Parameters
    ----------
    df_UCC : pd.DataFrame
        DataFrame containing UCC data.

    Returns
    -------
    None
    """
    fname0_UCC = [_.split(";")[0] for _ in df_UCC["fnames"]]
    NT = len(fname0_UCC)
    N_unique = len(list(set(fname0_UCC)))
    if NT != N_unique:
        raise ValueError("fnames are not unique")


def move_files(
    logging,
    run_mode: str,
    temp_JSON_file: str,
    new_DB_file: str,
    ucc_file: str,
    new_ucc_file: str,
    temp_zenodo_fold: str,
    temp_database_folder: str,
    archived_UCC_file: str,
) -> None:
    """ """
    logging.info("\nUpdate files:")

    if run_mode == "new_DB":
        # Move JSON file
        json_stored = name_DBs_json
        json_temp = temp_JSON_file
        os.rename(json_temp, json_stored)
        logging.info(json_temp + " --> " + json_stored)

        # Move new DB file
        db_stored = dbs_folder + new_DB_file
        db_temp = temp_database_folder + new_DB_file
        os.rename(db_temp, db_stored)
        logging.info(db_temp + " --> " + db_stored)

    # Generate '.gz' compressed file for the old UCC and archive it
    df = pd.read_csv(ucc_file)
    df.to_csv(
        archived_UCC_file,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    # Remove old UCC csv file
    os.remove(ucc_file)
    logging.info(ucc_file + " --> " + archived_UCC_file)
    # Move new UCC file into place
    ucc_temp = temp_zenodo_fold + new_ucc_file
    ucc_stored = UCC_folder + new_ucc_file
    os.rename(ucc_temp, ucc_stored)
    logging.info(ucc_temp + " --> " + ucc_stored)

    # Combine individual parquet files into a single one
    path = temp_fold + members_folder
    member_files = os.listdir(path)
    if len(member_files) > 0:
        process_member_files(logging, path, member_files)
    else:
        logging.info("No member files found in " + path)


def process_member_files(logging, path, member_files):
    """ """
    # Load JSON file with last updated dates
    fname_json = UCC_folder + parquet_dates
    with open(fname_json, "r") as file:
        json_data = json.load(file)

    # Combine individual parquet files into a single one
    tmp = []
    for file in member_files:
        df = pd.read_parquet(path + file)

        # Round before storing
        df[["RA_ICRS", "DE_ICRS", "GLON", "GLAT"]] = df[
            ["RA_ICRS", "DE_ICRS", "GLON", "GLAT"]
        ].round(6)
        df[
            [
                "Plx",
                "e_Plx",
                "pmRA",
                "e_pmRA",
                "pmDE",
                "e_pmDE",
                "RV",
                "e_RV",
                "Gmag",
                "BP-RP",
                "e_Gmag",
                "e_BP-RP",
                "probs",
            ]
        ] = df[
            [
                "Plx",
                "e_Plx",
                "pmRA",
                "e_pmRA",
                "pmDE",
                "e_pmDE",
                "RV",
                "e_RV",
                "Gmag",
                "BP-RP",
                "e_Gmag",
                "e_BP-RP",
                "probs",
            ]
        ].round(4)

        fname = file.replace(".parquet", "")
        df.insert(loc=0, column="name", value=fname)
        tmp.append(df)

        # Update entry for this file
        json_data[fname] = f"updated ({datetime.datetime.now().strftime('%y%m%d%H')})"

    # Update JSON file
    json_data = dict(sorted(json_data.items()))  # Sort
    with open(fname_json, "w") as file:
        json.dump(json_data, file, indent=2)
    logging.info("JSON file with dates updated --> " + fname_json)

    # Concatenate all temporary DataFrames into one
    df_comb = pd.concat(tmp, ignore_index=True)

    # Move the final combined parquet file to the 'zenodo/' folder
    zenodo_members_file = UCC_folder + UCC_members_file + ".temp"
    df_comb.to_parquet(zenodo_members_file, index=False)
    logging.info("Temp file with members stored --> " + zenodo_members_file)

    # Delete all individual parquet files?


if __name__ == "__main__":
    main()
