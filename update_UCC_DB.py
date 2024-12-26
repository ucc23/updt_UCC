import configparser
import csv
import datetime
import json
import os
import shutil
import sys

import pandas as pd

from modules import (
    add_new_DB,
    aux,
    check_new_DB,
    check_UCC_versions,
    duplicate_probs,
    member_files_updt_UCC,
    prepare_new_DB,
    standardize_and_match,
)
from modules.HARDCODED import (
    UCC_archive,
    UCC_folder,
    dbs_folder,
    name_DBs_json,
    temp_fold,
)


def main():
    """
    Main function to update the UCC (Unified Cluster Catalogue) with a new database.

    This function performs the following steps:
    1. Sets up logging.
    2. Reads parameters from the `params.ini` configuration file.
    3. Checks the accessibility of required Gaia data files.
    4. Loads the latest and new versions of the UCC.
    5. Prepares the new database format.
    6. Adds the new database to the JSON file.
    7. Loads the current UCC, the new database, and its JSON values.
    8. Standardizes and matches the new database with the UCC.
    9. Checks the entries in the new database.
    10. Generates a new UCC file with the new database incorporated.
    11. Updates membership probabilities if there are new OCs to process.
    12. Compares the old and new versions of the UCC.
    13. Updates (move + rename + remove) files

    Raises:
        ValueError: If required Gaia data files are not accessible.
    """
    logging = aux.logger()
    pars_dict = read_ini_file()

    # Check if the required Gaia datafiles are accessible
    check_Gaia_files(pars_dict)

    # Load the temporary path to the new version of the UCC
    temp_ucc_path = get_new_version_path(logging)

    # New DB format check
    all_dbs_json, db_year = prepare_new_DB.run(logging, pars_dict)

    # Adds the new database to the JSON file.
    add_DB_to_JSON(logging, pars_dict, all_dbs_json, db_year)

    # Load the current UCC, the new DB, and its JSON values
    last_version, df_UCC, df_new, json_pars = load_data(logging, pars_dict)
    df_UCC_old = df_UCC.copy()

    # Standardize and match the new DB with the UCC
    new_DB_fnames, db_matches, N_new = standardize_and_match.run(
        logging, df_UCC, df_new, json_pars, pars_dict
    )

    # Check the entries in the new DB
    check_new_DB.run(
        logging, pars_dict, df_UCC, df_new, json_pars, new_DB_fnames, db_matches
    )

    # Generate new UCC file with the new DB incorporated
    df_UCC = add_new_DB.run(
        logging, pars_dict, df_UCC, df_new, json_pars, new_DB_fnames, db_matches
    )
    df_UCC = possible_duplicates(logging, df_UCC, "literature")

    df_UCC = save_and_reload(logging, temp_ucc_path, df_UCC)
    df_UCC_old2 = df_UCC.copy()
    diff_between_dfs(logging, df_UCC_old, df_UCC, cols_exclude=None)
    if input("Move on? (y/n): ").lower() != "y":
        sys.exit()

    if N_new > 0:
        logging.info(
            f"\nProcessing {N_new} new OCs in {pars_dict['new_DB']} with fastMP...\n"
        )
        df_UCC = member_files_updt_UCC.run(logging, pars_dict, df_UCC)
        # Update membership probabilities
        df_UCC = possible_duplicates(logging, df_UCC, "UCC_members")
        df_UCC = save_and_reload(logging, temp_ucc_path, df_UCC)
        diff_between_dfs(logging, df_UCC_old2, df_UCC, cols_exclude=None)
    else:
        logging.info("No new OCs to process")

    logging.info(
        f"Check last version (N={len(df_UCC_old)}) vs new version (N={len(df_UCC)})"
    )
    check_UCC_versions.run(logging, df_UCC_old, df_UCC)

    move_files(logging, last_version, temp_ucc_path)

    logging.info("\nAll done! Proceed with the next script")


def read_ini_file():
    """
    Load .ini config file
    """
    in_params = configparser.ConfigParser()
    in_params.read("params.ini")
    print("Loaded params.ini")

    pars_dict = {}

    pars = in_params["New DB data"]
    for col in (
        "new_DB",
        "DB_name",
        "DB_ref",
        "ID",
        "RA",
        "DEC",
        "Plx",
        "pmRA",
        "pmDE",
        "Rv",
        "Av/E_bv",
        "dm/dist",
        "Age/logt",
        "Z/FeH",
        "Mass",
        "binar_frac",
        "blue_stragglers",
        "e_Av/E_bv",
        "e_dm/dist",
        "e_Age/logt",
        "e_Z/FeH",
        "e_Mass",
        "e_binar_frac",
        "e_blue_stragglers",
    ):
        pars_dict[col] = pars.get(col)

    pars = in_params["New DB check"]
    pars_dict["search_rad"] = pars.getfloat("search_rad")
    pars_dict["leven_rad"] = pars.getfloat("leven_rad")
    pars_dict["rad_dup"] = pars.getfloat("rad_dup")

    pars = in_params["Run fastMP / Updt UCC"]
    pars_dict["frames_path"] = pars.get("frames_path")
    pars_dict["frames_ranges"] = pars.get("frames_ranges")
    pars_dict["max_mag"] = pars.getfloat("max_mag")
    pars_dict["manual_pars_f"] = pars.get("manual_pars_f")
    pars_dict["verbose"] = pars.getint("verbose")

    return pars_dict


def check_Gaia_files(pars_dict):
    """ """
    if not os.path.isdir(pars_dict["frames_path"]):
        raise ValueError(f"Folder {pars_dict['frames_path']} is not present")

    if not os.path.isfile(pars_dict["frames_ranges"]):
        raise ValueError(f"File {pars_dict['frames_ranges']} is not present")


def get_new_version_path(logging) -> str:
    """
    Generate path to the new (temporary) version.
    """
    # Create new temp folder if required
    out_path = temp_fold + UCC_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Get new version of UCC database
    new_version = datetime.datetime.now().strftime("%Y%m%d%H")[2:]
    temp_ucc_path = out_path + "UCC_cat_" + new_version + ".csv"

    # Check if file exists
    if os.path.exists(temp_ucc_path):
        logging.info(f"File {temp_ucc_path} already exists. Moving on will re-write it")
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    return temp_ucc_path


def add_DB_to_JSON(logging, pars_dict, all_dbs_json, db_year):
    """ """
    db_year = int(db_year)
    # Extract years
    years = []
    for db in all_dbs_json.keys():
        years.append(int(db.split("_")[0][-4:]))

    # Index into a sorted list of integers, maintaining the sorted order
    index = 0
    while index < len(years) and years[index] < db_year:
        index += 1
    if index < 0:
        index = 0  # Prepend if index is negative
    elif index > len(all_dbs_json):
        index = len(all_dbs_json)  # Append if index is beyond the end

    #
    new_db_json = {}
    new_db_json["ref"] = f"[{pars_dict['DB_name']}]({pars_dict['DB_ref']})"
    new_db_json["names"] = f"{pars_dict['ID']}"
    #
    pos_entry = ""
    for col in ("RA", "DEC", "Plx", "pmRA", "pmDE", "Rv"):
        if pars_dict[col] != "None":
            pos_entry += pars_dict[col] + ","
        else:
            pos_entry += "None,"
    pos_entry = pos_entry[:-1]
    new_db_json["pos"] = f"{pos_entry}"

    pars_entry, e_pars_entry = [], []
    for col in (
        "Av/E_bv",
        "dm/dist",
        "Age/logt",
        "Z/FeH",
        "Mass",
        "binar_frac",
        "blue_stragglers",
    ):
        if pars_dict[col] != "None":
            pars_entry.append(pars_dict[col].strip())
            if pars_dict["e_" + col] != "None":
                e_pars_entry.append(pars_dict["e_" + col])
            else:
                e_pars_entry.append("None")
    if pars_entry:
        pars_entry = ",".join(pars_entry)
    else:
        pars_entry = ""
    new_db_json["pars"] = pars_entry
    #
    if e_pars_entry:
        e_pars_entry = ",".join(e_pars_entry)
    else:
        e_pars_entry = ""
    new_db_json["pars"] = pars_entry
    new_db_json["e_pars"] = e_pars_entry

    # Adds an element to a JSON array within a file at a specific index.
    dbs_keys = list(all_dbs_json.keys())
    dbs_keys.insert(index, pars_dict["new_DB"])
    new_json_dict = {}
    for key in dbs_keys:
        if key != pars_dict["new_DB"]:
            new_json_dict[key] = all_dbs_json[key]
        else:
            new_json_dict[pars_dict["new_DB"]] = new_db_json

    out_path = temp_fold + dbs_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(out_path + name_DBs_json, "w") as f:
        json.dump(new_json_dict, f, indent=2)  # Use indent for readability

    logging.info("\nJSON file updated")


def load_data(logging, pars_dict: dict) -> tuple[str, pd.DataFrame, pd.DataFrame, dict]:
    """ """
    # Load column data for the new catalogue
    with open(temp_fold + dbs_folder + name_DBs_json) as f:
        dbs_used = json.load(f)
    json_pars = dbs_used[pars_dict["new_DB"]]
    logging.info(f"JSON file {name_DBs_json} loaded")

    # Load the latest version of the combined catalogue
    last_version = None
    for file in os.listdir(UCC_folder):
        if file.endswith("csv"):
            last_version = file
            break
    if last_version is None:
        raise ValueError(f"UCC file not found in {UCC_folder}")
    df_UCC = pd.read_csv(UCC_folder + last_version)
    logging.info(f"UCC version {last_version} loaded (N={len(df_UCC)})")

    # Load the new DB
    df_new = pd.read_csv(dbs_folder + pars_dict["new_DB"] + ".csv")
    logging.info(f"New DB {pars_dict['new_DB']} loaded (N={len(df_new)})")

    return last_version, df_UCC, df_new, json_pars


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
        x, y, plx, pmRA, pmDE = (
            df_UCC["GLON"],
            df_UCC["GLAT"],
            df_UCC["Plx"],
            df_UCC["pmRA"],
            df_UCC["pmDE"],
        )
        prob_cut = 0.5
    elif data_orig == "UCC_members":
        x, y, plx, pmRA, pmDE = (
            df_UCC["GLON_m"],
            df_UCC["GLAT_m"],
            df_UCC["Plx_m"],
            df_UCC["pmRA_m"],
            df_UCC["pmDE_m"],
        )
        prob_cut = 0.25
    else:
        raise ValueError(f"Incorrect 'data_orig' value: {data_orig}")

    # Use members data
    dups_fnames, dups_probs = duplicate_probs.run(
        df_UCC["fnames"],
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


def save_and_reload(logging, temp_ucc_path, df_UCC):
    """ """
    # Order by (lon, lat) first
    df_UCC = df_UCC.sort_values(["GLON", "GLAT"])
    df_UCC = df_UCC.reset_index(drop=True)
    df_UCC.to_csv(
        temp_ucc_path, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC
    )

    # Load new UCC file
    df_UCC = pd.read_csv(temp_ucc_path)

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


def move_files(logging, last_version, temp_ucc_path):
    """ """
    # Move JSON file
    JSON_file_path = dbs_folder + name_DBs_json
    temp_JSON_file_path = temp_fold + dbs_folder + name_DBs_json
    os.remove(JSON_file_path)
    os.rename(temp_JSON_file_path, JSON_file_path)
    logging.info("\nJSON file updated")

    # Path to last (old) UCC database
    old_UCC = UCC_folder + last_version
    # Generate '.gz' compressed file for the old UCC and archive it
    df = pd.read_csv(old_UCC)
    df.to_csv(
        UCC_folder + UCC_archive + last_version.replace(".csv", ".gz"),
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    # Remove old csv file
    os.remove(old_UCC)
    # Move new UCC file
    new_ucc_path = "/".join(temp_ucc_path.split("/")[1:])
    os.rename(temp_ucc_path, new_ucc_path)
    logging.info("UCC file updated")

    # Move all .parquet member files

    # Remove folder
    shutil.rmtree(temp_fold)
    logging.info("temp/ folder removed")


if __name__ == "__main__":
    main()
