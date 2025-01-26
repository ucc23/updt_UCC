import csv
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from modules.HARDCODED import (
    GCs_cat,
    UCC_archive,
    UCC_folder,
    dbs_folder,
    manual_pars_file,
    members_folder,
    name_DBs_json,
    temp_fold,
)
from modules.update_database.add_new_DB_funcs import (
    QXY_fold,
    assign_UCC_ids,
    combine_UCC_new_DB,
    duplicates_check,
)
from modules.update_database.check_new_DB_funcs import (
    GCs_check,
    close_OC_check,
    close_OC_UCC_check,
    dups_check_newDB_UCC,
    positions_check,
    prep_newDB,
    vdberg_check,
)
from modules.update_database.member_files_updt_funcs import (
    extract_cl_data,
    process_new_OC,
    save_cl_datafile,
    split_membs_field,
)
from modules.update_database.possible_duplicates_funcs import duplicate_probs
from modules.update_database.standardize_and_match_funcs import (
    get_fnames_new_DB,
    get_matches_new_DB,
)
from modules.utils import logger

# Paths to the Gaia DR3 files
root = "/media/gabriel/backup/gabriel/GaiaDR3/"
path_gaia_frames = root + "datafiles_G20/"
# Paths to the file that informs the sky area covered by each file
path_gaia_frames_ranges = root + "files_G20/frame_ranges.txt"
# Maximum magnitude to retrieve
gaia_max_mag = 20


def main():
    """
    Main function to update the UCC (Unified Cluster Catalogue) with a new database.

    This function performs the following steps:
    3. Checks the accessibility of required files and folders, generate required paths.
    4. Prepares the new database format.
    6. Loads the current UCC, the new database, and its JSON values.
    7. Standardizes and matches the new database with the UCC.
    8. Checks the entries in the new database.
    9. Generates a new UCC file with the new database incorporated.
    10. Updates membership probabilities if there are new OCs to process.
    11. Compares the old and new versions of the UCC.
    12. Updates (move + rename + remove) files

    Raises:
        ValueError: If required Gaia data files are not accessible.
    """
    logging = logger()

    # Generate paths and check for required folders and files
    (
        root_folder,
        root_UCC_folder,
        current_JSON,
        temp_JSON,
        gaia_frames_data,
        df_GCs,
        manual_pars,
        temp_database_folder,
        GCs_path,
        new_DB_file,
        ucc_file,
        temp_ucc_file,
        archived_UCC_file,
        new_DB,
        JSON_file,
        temp_JSON_file,
        df_UCC,
        df_new,
    ) = get_paths_check_paths(logging)

    # Load column data for the new catalogue
    newDB_json = temp_JSON[new_DB]

    # Check for required columns in the new DB
    check_new_DB_cols(logging, current_JSON, new_DB, df_new, newDB_json)
    df_UCC_old = df_UCC.copy()

    # Standardize and match the new DB with the UCC
    new_DB_fnames, db_matches = standardize_and_match(
        logging, new_DB, df_UCC, df_new, newDB_json
    )

    # Check the entries in the new DB
    check_new_DB(
        logging, GCs_path, new_DB, df_UCC, df_new, newDB_json, new_DB_fnames, db_matches
    )

    # Generate new UCC file with the new DB incorporated
    df_UCC = add_new_DB(
        logging, new_DB, newDB_json, df_UCC, df_new, new_DB_fnames, db_matches
    )
    df_UCC = possible_duplicates(logging, df_UCC, "literature")

    df_UCC = save_and_reload(logging, temp_ucc_file, df_UCC)
    df_UCC_old2 = df_UCC.copy()
    diff_between_dfs(logging, df_UCC_old, df_UCC, cols_exclude=None)
    if input("Move on? (y/n): ").lower() != "y":
        sys.exit()

    N_new = db_matches.count(None)
    if N_new > 0:
        logging.info(f"\nProcessing {N_new} new OCs in {new_DB} with fastMP...\n")
        df_UCC = member_files_updt(
            logging, df_UCC, gaia_frames_data, df_GCs, manual_pars
        )
        # Update membership probabilities
        df_UCC = possible_duplicates(logging, df_UCC, "UCC_members")
        df_UCC = save_and_reload(logging, temp_ucc_file, df_UCC)
        diff_between_dfs(logging, df_UCC_old2, df_UCC, cols_exclude=None)
    else:
        logging.info("No new OCs to process")

    # logging.info(
    #     f"\nCheck last version (N={len(df_UCC_old)}) vs new version (N={len(df_UCC)})"
    # )
    # check_UCC_versions(logging, df_UCC_old, df_UCC, new_DB_fnames, db_matches)

    if input("\nMove files to their final destination? (y/n): ").lower() != "y":
        sys.exit()
    move_files(
        logging,
        JSON_file,
        root_folder,
        root_UCC_folder,
        temp_JSON_file,
        ucc_file,
        temp_ucc_file,
        archived_UCC_file,
    )

    # Check number of files
    file_checker(logging, root_UCC_folder)

    logging.info("\nAll done! Proceed with the next script")


def get_paths_check_paths(
    logging,
) -> tuple[
    str,
    str,
    dict,
    dict,
    pd.DataFrame | None,
    pd.DataFrame,
    pd.DataFrame,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    pd.DataFrame,
    pd.DataFrame,
]:
    """ """
    # Root UCC folder: take root to this folder and remove this folder
    root_folder = os.getcwd()
    root_UCC_folder = "/".join(os.getcwd().split("/")[:-1]) + "/"

    # Check for Gaia files
    if not os.path.isdir(path_gaia_frames):
        logging.info(f"Folder {path_gaia_frames} is not present")

    if not os.path.isfile(path_gaia_frames_ranges):
        logging.info(f"File {path_gaia_frames_ranges} is not present")
        gaia_frames_data = None
    else:
        gaia_frames_data = pd.read_csv(path_gaia_frames_ranges)

    GCs_path = dbs_folder + GCs_cat
    # Load GCs data
    df_GCs = pd.read_csv(GCs_path)
    # Read OCs manual parameters
    manual_pars = pd.read_csv(manual_pars_file)

    # Generate required temp folders
    # Temporary zenodo/ folder
    temp_zenodo_fold = temp_fold + UCC_folder
    # Create new temp zenodo folder if required
    if not os.path.exists(temp_zenodo_fold):
        os.makedirs(temp_zenodo_fold)
    # Temporary databases/ folder
    temp_database_folder = temp_fold + dbs_folder
    # Create new temp databases folder if required
    if not os.path.exists(temp_database_folder):
        os.makedirs(temp_database_folder)

    # Create quadrant folders
    for Nquad in range(1, 5):
        for lat in ("P", "N"):
            quad = "Q" + str(Nquad) + lat + "/"
            out_path = temp_fold + quad + members_folder
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    # Path to the latest version of the UCC catalogue
    last_version = None
    for file in os.listdir(UCC_folder):
        if file.endswith("csv"):
            last_version = file
            break
    if last_version is None:
        raise ValueError(f"UCC file not found in {UCC_folder}")
    # Path to the current UCC csv file
    ucc_file = UCC_folder + last_version
    df_UCC = pd.read_csv(ucc_file)
    logging.info(f"\nUCC version {ucc_file} loaded (N={len(df_UCC)})")

    # Path to the new (temp) version of the UCC database
    new_version = datetime.datetime.now().strftime("%Y%m%d%H")[2:]
    temp_ucc_file = temp_zenodo_fold + "UCC_cat_" + new_version + ".csv"
    # Check if file already exists
    if os.path.exists(temp_ucc_file):
        logging.info(f"File {temp_ucc_file} already exists. Moving on will re-write it")
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Path to the current JSON file
    JSON_file = dbs_folder + name_DBs_json
    # Path to the new (temp) JSON file
    temp_JSON_file = temp_database_folder + name_DBs_json

    # Load current JSON file
    with open(JSON_file) as f:
        current_JSON = json.load(f)
    # Load temp JSON file
    with open(temp_JSON_file) as f:
        temp_JSON = json.load(f)
    logging.info(f"JSON file {temp_JSON_file} loaded")
    # Extract new DB's name
    new_DB = list(set(temp_JSON.keys()) - set(current_JSON.keys()))[0]

    # Path to the new DB
    new_DB_file = temp_database_folder + new_DB + ".csv"
    # Load the new DB
    df_new = pd.read_csv(new_DB_file)
    logging.info(f"New DB {new_DB} loaded (N={len(df_new)})")

    # Path to the archived current UCC csv file
    archived_UCC_file = (
        UCC_folder + UCC_archive + last_version.replace(".csv", ".csv.gz")
    )

    return (
        root_folder,
        root_UCC_folder,
        current_JSON,
        temp_JSON,
        gaia_frames_data,
        df_GCs,
        manual_pars,
        temp_database_folder,
        GCs_path,
        new_DB_file,
        ucc_file,
        temp_ucc_file,
        archived_UCC_file,
        new_DB,
        JSON_file,
        temp_JSON_file,
        df_UCC,
        df_new,
    )


def check_new_DB_cols(
    logging, current_JSON, new_DB, df_new, newDB_json, show_entries=True
) -> None:
    """ """
    # Check that the new DB is not already present in the 'old' JSON file
    if new_DB in current_JSON.keys():
        raise ValueError(f"The DB '{new_DB}' is in the current JSON file")

    # Check for required columns
    cols = [newDB_json["names"]]
    for entry in ("pos", "pars", "e_pars"):
        for k, v in newDB_json[entry].items():
            cols.append(v)
    # Access columns stored in JSON file
    for col in cols:
        df_new[col]
    logging.info(f"\nAll columns present in {new_DB}")

    # Check for semi-colon and underscore present in name column
    logging.info("\nPossible bad characters in names (';', '_')")
    all_bad_names = []
    for new_cl in df_new[newDB_json["names"]]:
        if ";" in new_cl or "_" in new_cl:
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
    GCs_path: str,
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

    # Duplicate check between entries in the new DB and the UCC
    logging.info(f"\nChecking for entries in {new_DB} that must be combined")
    dup_flag = dups_check_newDB_UCC(logging, new_DB, df_UCC, new_DB_fnames, db_matches)
    if dup_flag:
        raise ValueError("Resolve the above issues before moving on.")
    else:
        logging.info("No issues found")

    # Check for GCs
    logging.info("\nClose GC check")
    glon, glat, gc_flag = GCs_check(logging, GCs_path, newDB_json, df_new)
    if gc_flag:
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Check for OCs very close to each other (possible duplicates)
    logging.info("\nProbable inner duplicates check")
    inner_flag = close_OC_check(logging, newDB_json, df_new, rad_dup)
    if inner_flag:
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Check for OCs very close to each other (possible duplicates)
    logging.info("\nProbable UCC duplicates check")
    dups_flag = close_OC_UCC_check(
        logging, df_UCC, new_DB_fnames, db_matches, glon, glat, rad_dup
    )
    if dups_flag:
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    logging.info("\nPossible vdBergh-Hagen/vdBergh check")
    vdb_flag = vdberg_check(logging, newDB_json, df_new)
    if vdb_flag:
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Prepare information from a new database matched with the UCC
    new_db_info = prep_newDB(newDB_json, df_new, new_DB_fnames, db_matches)

    # Check positions and flag for attention if required
    attention_flag = positions_check(logging, df_UCC, new_db_info, rad_dup)
    if attention_flag is True:
        if input("\nMove on? (y/n): ").lower() != "y":
            sys.exit()


def add_new_DB(
    logging, new_DB: str, newDB_json: dict, df_UCC, df_new, new_DB_fnames, db_matches
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

    logging.info(f"Adding new DB: {new_DB}")

    logging.info("")
    new_db_dict = combine_UCC_new_DB(
        logging,
        new_DB,
        newDB_json,
        df_UCC,
        df_new,
        new_DB_fnames,
        db_matches,
    )
    N_new = len(df_new) - sum(_ is not None for _ in db_matches)
    logging.info(f"\nN={N_new} new clusters in {new_DB}")
    logging.info("")

    # Add UCC_IDs and quadrants for new clusters
    ucc_ids_old = list(df_UCC["UCC_ID"].values)
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
    df_UCC_no_new = df_UCC.drop(list(df_UCC.index[idx_rm_comb_db]))
    df_UCC_no_new.reset_index(drop=True, inplace=True)
    df_all = pd.concat([df_UCC_no_new, pd.DataFrame(new_db_dict)], ignore_index=True)

    # Final duplicate check
    dup_flag = duplicates_check(logging, df_all)
    if dup_flag:
        raise ValueError(
            "Duplicated entries found in either 'ID, UCC_ID, fnames' column"
        )

    return df_all


def possible_duplicates(logging, df_UCC: pd.DataFrame, data_orig: str) -> pd.DataFrame:
    """
    Assign a 'duplicate probability' for each cluster in the UCC, based either on
    positions from literature values or from its estimated members.

    Parameters:
        logging: Logger instance for logging messages.
        df_UCC: Dictionary containing cluster data, including key 'fnames',
        and positional data
        data_orig (str): String that informs if the data to be used is literature
        data or UCC members data.

    Returns:
        Updated df_UCC dictionary with added keys 'dups_fnames_m' and 'dups_probs_m',
        if duplicates are found.
    """
    logging.info(f"Finding {data_orig} duplicates and their probabilities...")

    # prob_cut: Float representing the probability cutoff for identifying duplicates.
    if data_orig == "literature":
        d_id = ""
        prob_cut = 0.5
    elif data_orig == "UCC_members":
        d_id = "_m"
        prob_cut = 0.25
    else:
        raise ValueError(f"Incorrect 'data_orig' value: {data_orig}")

    cols = ("GLON", "GLAT", "Plx", "pmRA", "pmDE")
    x, y, plx, pmRA, pmDE = [np.array(df_UCC[col + d_id]) for col in cols]

    # Use members data
    dups_fnames, dups_probs = duplicate_probs(
        list(df_UCC["fnames"]),
        x,
        y,
        plx,
        pmRA,
        pmDE,
        prob_cut,
    )

    if data_orig == "literature":
        df_UCC["dups_fnames"], df_UCC["dups_probs"] = dups_fnames, dups_probs
    else:
        df_UCC["dups_fnames_m"], df_UCC["dups_probs_m"] = dups_fnames, dups_probs
    logging.info("Duplicates (using members data) added to UCC\n")

    return df_UCC


def member_files_updt(logging, df_UCC, gaia_frames_data, df_GCs, manual_pars):
    """
    Updates the Unified Cluster Catalogue (UCC) with new open clusters (OCs).
    This function performs the following steps:

    2. Constructs a KD-tree for efficient spatial queries on the UCC.
    3. Processes each new OC

    Args:
        logging (logging.Logger): Logger instance for logging messages.
        pars_dict (dict): Dictionary containing parameters for the new database.
        df_UCC (pd.DataFrame): DataFrame containing the current UCC.

    Returns:
        pd.DataFrame: Updated UCC DataFrame with new OCs processed.
    """

    # Parameters used to search for close-by clusters
    xys = np.array([df_UCC["GLON"].values, df_UCC["GLAT"].values]).T
    tree = KDTree(xys)

    # For each new OC
    df_UCC_updt = {
        "fname": [],
        "UCC_idx": [],
        "C1": [],
        "C2": [],
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
    for UCC_idx, new_cl in df_UCC.iterrows():
        # Check if this is a new OC that should be processed
        if str(new_cl["C3"]) != "nan":
            continue

        logging.info(f"\nProcessing {new_cl['fnames']} (idx={UCC_idx}) with fastMP")
        gaia_frame, probs_all = process_new_OC(
            logging,
            df_UCC,
            path_gaia_frames,
            gaia_max_mag,
            gaia_frames_data,
            df_GCs,
            manual_pars,
            tree,
            UCC_idx,
            new_cl,
        )

        # Split into members and field stars according to the probability values
        # assigned
        df_membs, df_field = split_membs_field(gaia_frame, probs_all)

        # Extract data to update the UCC
        dict_UCC_updt = extract_cl_data(df_membs, df_field)

        # Update UCC (and temporary csv file)
        fname0 = new_cl["fnames"].split(";")[0]
        df_UCC_updt["fname"].append(fname0)
        df_UCC_updt["UCC_idx"].append(UCC_idx)
        for key, val in dict_UCC_updt.items():
            df_UCC.at[UCC_idx, key] = val
            df_UCC_updt[key].append(val)

        # Write selected member stars to file
        save_cl_datafile(logging, temp_fold, members_folder, new_cl, df_membs)

    df_UCC_updt = pd.DataFrame(df_UCC_updt)
    df_UCC_updt.to_csv(temp_fold + "df_UCC_updt.csv", index=False)
    logging.info("\nTemp file df_UCC_updt saved")

    return df_UCC


def save_and_reload(logging, temp_ucc_file, df_UCC):
    """ """
    # Order by (lon, lat) first
    df_UCC = df_UCC.sort_values(["GLON", "GLAT"])
    df_UCC = df_UCC.reset_index(drop=True)
    # Save UCC to CSV file
    df_UCC.to_csv(
        temp_ucc_file, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC
    )
    # Load new UCC
    df_UCC = pd.read_csv(temp_ucc_file)

    logging.info(f"UCC updated (N={len(df_UCC)})")

    return df_UCC


def diff_between_dfs(
    logging,
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    cols_exclude=None,
):
    """
    Compare two DataFrames, find non-matching rows while preserving order, and
    output these rows in two files.

    Args:
        df_old (pd.DataFrame): First DataFrame to compare.
        df_new (pd.DataFrame): Second DataFrame to compare.
        cols_exclude (list | None): List of columns to exclude from the diff
    """
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
        return

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


# def check_UCC_versions(
#     logging,
#     UCC_old: pd.DataFrame,
#     UCC_new: pd.DataFrame,
#     new_DB_fnames: list,
#     db_matches: list,
# ) -> None:
#     """Run checks on old and new UCC files to ensure consistency and identify possible
#     issues.

#     Parameters:
#     - logging: Logger instance for recording messages.
#     - UCC_old: DataFrame containing the old UCC data.
#     - UCC_new: DataFrame containing the new UCC data.
#     - new_DB_fnames: List of lists with fnames of the new DB
#     - db_matches: List of indexes of OCs in the new DB into the UCC ('None' for new OCs)

#     Returns:
#     - None
#     """

#     fnames_old_all = list(UCC_old["fnames"])
#     fnames_new_all = list(UCC_new["fnames"])

#     fname_old_all_lst = []
#     for fnames_old in fnames_old_all:
#         # temp = []
#         # for fname_old in fnames_old.split(";"):
#         #     temp.append(fname_old)
#         fname_old_all_lst.append(fnames_old.split(";"))

#     i_fname_new_in_old = []
#     for fnames_new in fnames_new_all:
#         try:
#             i_fname_new_in_old.append(fnames_old_all.index(fnames_new))
#         except ValueError:
#             # 'fnames_new' not in old UCC
#             i_fname_new_in_old.append(None)

#     # Extract new fnames
#     new_fnames = []
#     for i, j in enumerate(db_matches):
#         if j is None:
#             new_fnames.append(";".join(new_DB_fnames[i]))

#     # Check new entries
#     logging.info("\nOCs with fnames that changed or are new:")
#     for i_new, i_old in enumerate(i_fname_new_in_old):
#         if i_old is None:
#             fnames_new = fnames_new_all[i_new]
#             if fnames_new in new_fnames:
#                 logging.info(f"new     : {fnames_new}")
#             else:
#                 logging.info(f"changed : {fnames_new}")

#             # Check new entries in the UCC that are not present in the old UCC
#             idxs_old_match = []
#             for fname_new in fnames_new.split(";"):
#                 for i_old, fnames_old in enumerate(fname_old_all_lst):
#                     if fname_new in fnames_old:
#                         idxs_old_match.append(i_old)
#             idxs_old_match = list(set(idxs_old_match))
#             if len(idxs_old_match) > 1:
#                 raise ValueError(f"Duplicate fname, new:{i_new}, old:{idxs_old_match}")

#     # Check existing entries
#     logging.info("\nOCs with fnames that did not change:")
#     logging.info("fnames     --> column name: old value | new value; ...")
#     for i_new, i_old in enumerate(i_fname_new_in_old):
#         if i_old is None:
#             continue

#         # If 'fnames_new' was found in the old UCC, compare both entries
#         # Extract rows
#         row_new = UCC_new.iloc[i_new]
#         row_old = UCC_old.iloc[i_old]

#         # Compare rows using pandas method
#         row_compared = row_new.compare(row_old)

#         # If rows are not equal
#         if row_compared.empty is False:
#             # Extract column names with differences
#             row_dict = row_compared.to_dict()
#             diff_cols = list(row_dict["self"].keys())

#             # If the only diffs are in the ID columns, skip check
#             if (
#                 diff_cols == ["DB", "DB_i"]
#                 or diff_cols == ["DB"]
#                 or diff_cols == ["DB_i"]
#             ):
#                 continue

#             fnames = str(UCC_old["fnames"][i_old])
#             fs.check_rows(logging, fnames, diff_cols, row_old, row_new)


def move_files(
    logging,
    JSON_file: str,
    root_folder: str,
    root_UCC_folder: str,
    temp_JSON_file: str,
    ucc_file: str,
    temp_ucc_file: str,
    archived_UCC_file: str,
) -> None:
    """ """

    # Move JSON file
    # os.remove(current_JSON)
    print("remove: ", root_folder + "/" + JSON_file)
    # os.rename(temp_JSON, current_JSON)
    print(
        "rename: ",
        root_folder + "/" + temp_JSON_file,
        " to: ",
        root_folder + "/" + JSON_file,
    )
    logging.info("JSON file updated")

    # Generate '.gz' compressed file for the old UCC and archive it
    df = pd.read_csv(ucc_file)
    df.to_csv(
        archived_UCC_file,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    print("Create: ", root_folder + "/" + archived_UCC_file)
    # Remove old csv file
    # os.remove(ucc_file)
    print("remove: ", root_folder + "/" + ucc_file)
    # Move new UCC file
    new_ucc_path = "/".join(temp_ucc_file.split("/")[1:])
    # os.rename(temp_ucc_file, new_ucc_path)
    print(
        "rename: ",
        root_folder + "/" + temp_ucc_file,
        " to: ",
        root_folder + "/" + new_ucc_path,
    )
    logging.info("UCC file updated")

    # Move all .parquet member files
    for qN in range(1, 5):
        for lat in ("P", "N"):
            qfold = "Q" + str(qN) + lat + "/"
            # Check if folder exists
            qmembs_fold = temp_fold + qfold + members_folder
            if os.path.exists(qmembs_fold):
                # For every file in this folder
                for file in os.listdir(qmembs_fold):
                    print(
                        "rename: ",
                        root_UCC_folder + qmembs_fold + "/" + file,
                        " to: ",
                        root_UCC_folder + qfold + members_folder + "/" + file,
                    )

    # # Remove folder
    # shutil.rmtree(temp_fold)
    # logging.info("temp/ folder removed")


def file_checker(logging, root_UCC_fold: str) -> None:
    """Check the number and types of files in directories for consistency.

    Parameters:
    - logging: Logger instance for recording messages.
    - UCC_new: DataFrame containing the new UCC data.

    Returns:
    - None
    """
    logging.info("\nChecking number of files")
    logging.info("    parquet webp  aladin  extra")

    flag_error = False
    NT_parquet, NT_webp, NT_webp_aladin, NT_extra = 0, 0, 0, 0
    for qnum in range(1, 5):
        for lat in ("P", "N"):
            N_parquet, N_webp, N_webp_aladin, N_extra = 0, 0, 0, 0
            for ffolder in ("datafiles", "plots"):
                qfold = root_UCC_fold + "Q" + str(qnum) + lat + f"/{ffolder}/"
                # Read all files in Q folder
                for file in os.listdir(qfold):
                    if "HUNT23" in file or "CANTAT20" in file:
                        pass
                    elif "aladin" in file:
                        N_webp_aladin += 1
                        NT_webp_aladin += 1
                    elif "parquet" in file:
                        N_parquet += 1
                        NT_parquet += 1
                    elif "webp" in file:
                        N_webp += 1
                        NT_webp += 1
                    else:
                        N_extra += 1
                        NT_extra += 1

            mark = "V" if (N_parquet == N_webp == N_webp_aladin) else "X"
            if N_extra > 0:
                mark = "X"
            logging.info(
                f"{str(qnum) + lat}:   {N_parquet}  {N_webp}  {N_webp_aladin}    {N_extra} <-- {mark}"
            )
            if mark == "X":
                flag_error = True
    logging.info(
        f"Total parquet/webp/aladin/extra: {NT_parquet}, {NT_webp}, {NT_webp_aladin}, {NT_extra}"
    )
    if not (NT_parquet == NT_webp == NT_webp_aladin) or NT_extra > 0:
        flag_error = True
    if flag_error:
        raise ValueError("The file check was unsuccessful")


if __name__ == "__main__":
    main()
