import csv
import json
import re
import sys

import numpy as np
import pandas as pd

from .HARDCODED import dbs_folder, name_DBs_json


def run(logging, pars_dict, show_entries=True):
    """
    Prepares a new database for inclusion in the Unified Cluster Catalogue (UCC).

    This function performs the following steps:
    1. Checks if the new database is already in the JSON file.
    2. Validates the new database's name.
    3. Validates the new database's year.
    4. Checks for required and non-required columns.
    5. Checks for special characters in the name column.
    6. Replaces empty positions with NaNs.

    Args:
        logging (logging.Logger): Logger instance for logging messages.
        pars_dict (dict): Dictionary containing parameters for the new database.
        show_entries (bool): Flag to show entries with bad characters in names.

    Raises:
        ValueError: If the new database name contains special characters, does not
        contain a valid year, or if there are issues with columns.
    """
    # Check that the new DB is not already present in the 'old' JSON file
    with open(dbs_folder + name_DBs_json) as f:
        all_dbs_json = json.load(f)
    if pars_dict["new_DB"] in all_dbs_json.keys():
        logging.info(f"The DB {pars_dict['new_DB']} is already in the JSON file")
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Check DB's name
    if contains_special_char(pars_dict["new_DB"]):
        raise ValueError(
            f"The DB name {pars_dict['new_DB']} contains special characters"
        )
    db_root = pars_dict["new_DB"].split("_")[0]
    db_year = db_root[-4:]
    if is_convertible_to_int(db_year) is False:
        raise ValueError(
            f"The DB {pars_dict['new_DB']} does not contain a valid year: {db_year}"
        )
    if int(db_year) < 1950 or int(db_year) > 2050:
        raise ValueError(f"The DB year {db_year} looks suspicious")

    # Load CSV
    df_new = pd.read_csv(dbs_folder + pars_dict["new_DB"] + ".csv")
    logging.info(f"\nName & url: {pars_dict['DB_name']}, {pars_dict['DB_ref']}")

    # Check for required columns
    cols = ""
    for col in ("ID", "RA", "DEC"):
        df_new[pars_dict[col]]
        cols += col + " "

    # Check non-required cols
    for col in ("Plx", "pmRA", "pmDE", "Rv"):
        if pars_dict[col] != "None":
            df_new[pars_dict[col]]
            cols += col + " "

    # Check 'pars'
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
            par = pars_dict[col].strip()
            df_new[par]
            cols += par + " "
        if pars_dict["e_" + col] != "None":
            par = "e_" + pars_dict[col].strip()
            df_new[par]

    logging.info(f"Columns ({cols.strip()}) are present")

    # Check for semi-colon and underscore present in name column
    logging.info("\nPossible bad characters in names (';', '_')")
    bad_name_flag = name_chars_check(logging, df_new, pars_dict, show_entries)
    if bad_name_flag:
        raise ValueError("Resolve the above issues before moving on.")

    # Replace empty positions with 'nans'
    empty_nan_replace(pars_dict["new_DB"], df_new)
    logging.info(f"\nEmpty entries replace finished in {pars_dict['new_DB']}")

    return all_dbs_json, db_year


def contains_special_char(text):
    """
    Checks if a string contains any character other than letters, numbers, or underscores.

    Args:
      text: The string to check.

    Returns:
      True if the string contains a special character, False otherwise.
    """
    pattern = (
        r"[^a-zA-Z0-9_]"  # Regular expression to match any character NOT in the set
    )
    match = re.search(pattern, text)
    return bool(match)


def is_convertible_to_int(string):
    """
    Checks if a string can be converted to an integer.

    Args:
      string: The string to check.

    Returns:
      True if the string can be converted to an integer, False otherwise.
    """
    try:
        int(string)
        return True
    except ValueError:
        return False


def name_chars_check(logging, df_new, pars_dict, show_entries):
    """ """
    ID = pars_dict["ID"]
    all_bad_names = []
    for new_cl in df_new[ID]:
        if ";" in new_cl or "_" in new_cl:
            all_bad_names.append(new_cl)

    if len(all_bad_names) == 0:
        bad_name_flag = False
        logging.info("No bad-chars found in name(s) column")
    else:
        bad_name_flag = True
        logging.info(
            f"{len(all_bad_names)} entries with bad-chars found in name(s) column"
        )

    if show_entries:
        for new_cl in all_bad_names:
            logging.info(f"{new_cl}: bad char found")

    return bad_name_flag


def empty_nan_replace(new_DB, df_new):
    """
    Replace possible empty entries in columns
    """
    # Remove leading and trailing spaces
    df_new = df_new.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Replace empty strings or whitespace-only strings with NaN
    df_new = df_new.replace(r"^\s*$", np.nan, regex=True)

    df_new.to_csv(
        dbs_folder + new_DB + ".csv",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
