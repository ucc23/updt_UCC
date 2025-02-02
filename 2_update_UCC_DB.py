import csv
import datetime
import json
import os
import sys
import warnings

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
    get_close_cls,
    get_frame_limits,
    process_new_OC,
    save_cl_datafile,
    split_membs_field,
)
from modules.update_database.possible_duplicates_funcs import duplicate_probs
from modules.update_database.standardize_and_match_funcs import (
    get_fnames_new_DB,
    get_matches_new_DB,
)
from modules.utils import file_checker, get_last_version_UCC, logger

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
    """
    logging = logger()

    # Check for Gaia files
    if not os.path.isdir(path_gaia_frames):
        raise FileNotFoundError(f"Folder {path_gaia_frames} is not present")
    if not os.path.isfile(path_gaia_frames_ranges):
        raise FileNotFoundError(f"File {path_gaia_frames_ranges} is not present")

    # Generate paths and check for required folders and files
    (
        root_current_folder,
        root_UCC_folder,
        GCs_path,
        temp_database_folder,
        ucc_file,
        temp_zenodo_fold,
        new_ucc_file,
        temp_JSON_file,
        JSON_file,
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
    ) = load_data(
        logging, GCs_path, ucc_file, JSON_file, temp_JSON_file, temp_database_folder
    )

    # Check for required columns in the new DB
    check_new_DB_cols(logging, current_JSON, new_DB, df_new, newDB_json)

    # Standardize and match the new DB with the UCC
    new_DB_fnames, db_matches = standardize_and_match(
        logging, new_DB, df_UCC_old, df_new, newDB_json
    )

    # Check the entries in the new DB
    check_new_DB(
        logging,
        GCs_path,
        new_DB,
        df_UCC_old,
        df_new,
        newDB_json,
        new_DB_fnames,
        db_matches,
    )

    # Generate new UCC file with the new DB incorporated
    df_UCC_new = add_new_DB(
        logging, new_DB, newDB_json, df_UCC_old, df_new, new_DB_fnames, db_matches
    )
    df_UCC_new2 = diff_between_dfs(logging, df_UCC_old, df_UCC_new)
    if input("Move on? (y/n): ").lower() != "y":
        sys.exit()

    # Check the entries with no C3 value are identified as new and processed with fastMP
    N_new = (df_UCC_new2["C3"] == "nan").sum()
    if N_new > 0:
        logging.info(f"\nProcessing {N_new} new OCs in {new_DB} with fastMP...")

        # Generate member files for new OCs and obtain their data
        df_UCC_updt = member_files_updt(
            logging, df_UCC_new2, gaia_frames_data, df_GCs, manual_pars
        )

        # Update the UCC with the new OCs member's data
        df_UCC_new3 = update_UCC_membs_data(logging, df_UCC_new2, df_UCC_updt)
        df_UCC_new4 = diff_between_dfs(logging, df_UCC_new2, df_UCC_new3)
    else:
        logging.info("No new OCs to process")
        df_UCC_new4 = df_UCC_new2

    # Save updated UCC to CSV file
    save_final_UCC(logging, temp_zenodo_fold, new_ucc_file, df_UCC_new4)

    if input("\nMove files to their final destination? (y/n): ").lower() != "y":
        sys.exit()
    move_files(
        logging,
        root_current_folder,
        root_UCC_folder,
        JSON_file,
        temp_JSON_file,
        new_DB_file,
        ucc_file,
        new_ucc_file,
        temp_zenodo_fold,
        temp_database_folder,
        archived_UCC_file,
    )

    # Check number of files
    N_UCC = len(df_UCC_new4)
    file_checker(logging, N_UCC, root_UCC_folder)

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
    str,
    str,
    str,
    str,
]:
    """ """
    # Current root + main UCC folder
    root_current_folder = os.getcwd()
    # Go up one level
    root_UCC_folder = os.path.dirname(root_current_folder)

    # Path to file with GCs data
    GCs_path = dbs_folder + GCs_cat

    # Temporary databases/ folder
    temp_database_folder = temp_fold + dbs_folder
    # Create if required
    if not os.path.exists(temp_database_folder):
        os.makedirs(temp_database_folder)

    # Create quadrant folders
    for Nquad in range(1, 5):
        for lat in ("P", "N"):
            quad = "Q" + str(Nquad) + lat + "/"
            out_path = temp_fold + quad + members_folder
            if not os.path.exists(out_path):
                os.makedirs(out_path)

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
        logging.info(f"File {new_ucc_file} already exists. Moving on will re-write it")
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Path to the current JSON file
    JSON_file = dbs_folder + name_DBs_json

    # Path to the new (temp) JSON file
    temp_JSON_file = temp_database_folder + name_DBs_json

    # Path to the archived current UCC csv file
    archived_UCC_file = (
        UCC_folder + UCC_archive + last_version.replace(".csv", ".csv.gz")
    )

    return (
        root_current_folder,
        root_UCC_folder,
        GCs_path,
        temp_database_folder,
        ucc_file,
        temp_zenodo_fold,
        new_ucc_file,
        temp_JSON_file,
        JSON_file,
        archived_UCC_file,
    )


def load_data(
    logging,
    GCs_path: str,
    ucc_file: str,
    JSON_file: str,
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
    df_GCs = pd.read_csv(GCs_path)

    # Read OCs manual parameters
    manual_pars = pd.read_csv(manual_pars_file)

    df_UCC = pd.read_csv(ucc_file)
    logging.info(f"\nUCC version {ucc_file} loaded (N={len(df_UCC)})")

    # Load current JSON file
    with open(JSON_file) as f:
        current_JSON = json.load(f)

    # Load temp JSON file
    with open(temp_JSON_file) as f:
        temp_JSON = json.load(f)

    # Extract new DB's name
    new_DB = list(set(temp_JSON.keys()) - set(current_JSON.keys()))[0]

    # Load column data for the new catalogue
    newDB_json = temp_JSON[new_DB]

    # Load the new DB
    new_DB_file = new_DB + ".csv"
    df_new = pd.read_csv(temp_database_folder + new_DB_file)
    logging.info(f"New DB {new_DB} loaded (N={len(df_new)})")

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
    logging,
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
    new_db_dict = combine_UCC_new_DB(
        logging,
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
    dup_flag = duplicates_check(logging, df_UCC_new)
    if dup_flag:
        raise ValueError(
            "Duplicated entries found in either 'ID, UCC_ID, fnames' column"
        )

    return df_UCC_new


def member_files_updt(
    logging, df_UCC, gaia_frames_data, df_GCs, manual_pars
) -> pd.DataFrame:
    """
    Updates the Unified Cluster Catalogue (UCC) with new open clusters (OCs).
    This function performs the following steps:
    """

    # If file exists, read and return it
    if os.path.isfile(temp_fold + "df_UCC_updt.csv"):
        df_UCC_updt = pd.read_csv(temp_fold + "df_UCC_updt.csv")
        logging.info("\nTemp file df_UCC_updt loaded")
        return df_UCC_updt

    # Parameters used to search for close-by clusters
    xys = np.array([df_UCC["GLON"].values, df_UCC["GLAT"].values]).T
    tree = KDTree(xys)

    # For each new OC
    df_UCC_updt = {
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

        # Generate frame
        box_s, plx_min = get_frame_limits(new_cl)

        # Check for close clusters
        get_close_cls(
            logging,
            df_UCC,
            UCC_idx,
            new_cl,
            tree,
            box_s,
            df_GCs,
        )

        logging.info(f"\nProcessing {new_cl['fnames']} (idx={UCC_idx}) with fastMP")
        gaia_frame, probs_all = process_new_OC(
            logging,
            box_s,
            plx_min,
            path_gaia_frames,
            gaia_max_mag,
            gaia_frames_data,
            manual_pars,
            new_cl,
        )

        # Split into members and field stars according to the probability values
        # assigned
        df_membs, df_field = split_membs_field(gaia_frame, probs_all)

        # Write selected member stars to file
        save_cl_datafile(logging, temp_fold, members_folder, new_cl, df_membs)

        # Extract data to update the UCC
        dict_UCC_updt = extract_cl_data(df_membs, df_field)

        #
        df_UCC_updt["UCC_idx"].append(UCC_idx)
        for key, val in dict_UCC_updt.items():
            df_UCC_updt[key].append(val)

    df_UCC_updt = pd.DataFrame(df_UCC_updt)
    df_UCC_updt.to_csv(temp_fold + "df_UCC_updt.csv", index=False)
    logging.info("\nTemp file df_UCC_updt saved")

    return df_UCC_updt


def update_UCC_membs_data(logging, df_UCC, df_UCC_updt, prob_cut: float = 0.25):
    """
    prob_cut: Probability cutoff for identifying duplicates.
    """
    # Generate copy to not disturb the dataframe given to this function which is
    # later used to generate the diffs files
    df_inner = df_UCC.copy()

    #
    for _, row in df_UCC_updt.iterrows():
        UCC_idx = row["UCC_idx"]
        for key, val in row.items():
            if key == "UCC_idx":
                continue
            df_inner.at[UCC_idx, key] = val

    logging.info("Finding duplicates and their probabilities...")
    df_inner["dups_fnames_m"], df_inner["dups_probs_m"] = duplicate_probs(
        list(df_inner["fnames"]),
        np.array(df_inner["GLON_m"], dtype=float),
        np.array(df_inner["GLAT_m"], dtype=float),
        np.array(df_inner["Plx_m"], dtype=float),
        np.array(df_inner["pmRA_m"], dtype=float),
        np.array(df_inner["pmDE_m"], dtype=float),
        prob_cut,
    )

    return df_inner


def diff_between_dfs(
    logging,
    df_old: pd.DataFrame,
    df_new_in: pd.DataFrame,
    cols_exclude=None,
) -> pd.DataFrame:
    """
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
    # NaN as "nan"
    df_new = df_new.infer_objects(copy=False).fillna("nan")

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
):
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
    logging.info(f"UCC updated (N={len(df_UCC)})")


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
    root_current_folder: str,
    root_UCC_folder: str,
    JSON_file: str,
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

    # Move JSON file
    json_stored = root_current_folder + "/" + JSON_file
    json_temp = root_current_folder + "/" + temp_JSON_file
    os.rename(json_temp, json_stored)
    logging.info(json_temp + " --> " + json_stored)
    logging.info("JSON file updated")

    # Move new DB file
    db_stored = root_current_folder + "/" + dbs_folder + new_DB_file
    db_temp = root_current_folder + "/" + temp_database_folder + new_DB_file
    os.rename(db_temp, db_stored)
    logging.info(db_temp + " --> " + db_stored)
    logging.info("New DB stored")

    # Generate '.gz' compressed file for the old UCC and archive it
    df = pd.read_csv(ucc_file)
    gz_UCC_stored = root_current_folder + "/" + archived_UCC_file
    df.to_csv(
        gz_UCC_stored,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    logging.info("Create: " + gz_UCC_stored)
    # Remove old csv file
    old_ucc = root_current_folder + "/" + ucc_file
    os.remove(old_ucc)
    logging.info("Remove: " + old_ucc)
    # Move new UCC file
    ucc_temp = root_current_folder + "/" + temp_zenodo_fold + new_ucc_file
    ucc_stored = root_current_folder + "/" + UCC_folder + new_ucc_file
    os.rename(ucc_temp, ucc_stored)
    logging.info(ucc_temp + " --> " + ucc_stored)
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
                    parquet_temp = root_current_folder + "/" + qmembs_fold + "/" + file
                    parquet_stored = (
                        root_UCC_folder + "/" + qfold + members_folder + "/" + file
                    )
                    os.rename(parquet_temp, parquet_stored)
                    logging.info(parquet_temp + " --> " + parquet_stored)
    logging.info("Cluster member files stored")


if __name__ == "__main__":
    main()
