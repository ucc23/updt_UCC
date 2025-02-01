import csv
import json
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
import requests
from astroquery.vizier import Vizier

from modules.HARDCODED import (
    dbs_folder,
    name_DBs_json,
    temp_fold,
)

from modules.utils import logger

# NASA/ADS bibcode for the new DB
ADS_bibcode = "2011A%26A...532A.131B"


def main():
    """
    Main function to download and process a database using NASA/ADS and Vizier data.

    Steps:
    1. Load the current JSON database file.
    2. Check if the URL is already listed in the current database.
    3. Fetch publication authors and year from NASA/ADS
    4. Generate a new database name based on extracted metadata.
    5. Handle temporary database files and check for existing data.
    6. Fetch Vizier data or allow manual input for Vizier IDs.
    7. Match new database columns with current JSON structure.
    8. Update the JSON file and save the database as CSV.

    Raises:
        ValueError: If URL data fetching or metadata extraction fails.
    """
    logging = logger()

    # Load current JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    # Check if url is already listed in the current JSON file
    ADS_url = "https://ui.adsabs.harvard.edu/abs/" + ADS_bibcode
    for db, vals in current_JSON.items():
        if ADS_url == vals["ADS_url"]:
            raise ValueError(
                f"The URL {ADS_url}\nis already in the JSON file under: {db}"
            )
            # logging.info(f"The URL {ADS_url}\nis already in the JSON file under: {db}")

    logging.info("Fetching NASA/ADS data...")
    authors, year = get_ADS_data()
    logging.info(f"Extracted author ({authors}) and year ({year})")

    DB_name = get_DB_name(current_JSON, authors, year)
    logging.info(f"New DB name obtained: {DB_name}")

    # Temporary databases/ folder
    temp_database_folder = temp_fold + dbs_folder
    # Create folder if it does not exist
    Path(temp_database_folder).mkdir(parents=True, exist_ok=True)
    # Path to the new (temp) JSON file
    temp_JSON_file = temp_fold + name_DBs_json
    # Path to the new (temp) DB file
    temp_CSV_file = temp_database_folder + DB_name + ".csv"

    quest = "n"
    if Path(temp_CSV_file).is_file():
        quest = input("Load Vizier database from file (else download)? (y/n): ").lower()
    if quest == "y":
        df_all = [pd.read_csv(temp_CSV_file)]
        logging.info("Vizier CSV file loaded from file")
    else:
        df_all = get_CDS_table(logging)
        if df_all is not None:
            # Save the database(s) to a CSV file(s)
            save_DB_CSV(temp_CSV_file, df_all)
            logging.info(f"New DB csv file(s) stored {temp_CSV_file}\n")

    # Extract the names, positions, parameters, and uncertainties column names from
    # the current JSON
    names_dict, pos_dict, pars_dict, e_pars_dict = current_JSON_vals(current_JSON)

    # Extract the matches for each column in the new DB
    df_col_id = None
    if df_all is not None:
        df_col_id = new_DB_columns_match(
            logging, names_dict, pos_dict, pars_dict, e_pars_dict, df_all
        )
        logging.info("Column names for temp JSON file extracted")

    names, pos_dict, pars_dict, e_pars_dict = proper_json_struct(df_col_id)

    # Create new temporary JSON
    add_DB_to_JSON(
        ADS_url,
        current_JSON,
        temp_JSON_file,
        DB_name,
        authors,
        year,
        names,
        pos_dict,
        pars_dict,
        e_pars_dict,
    )
    logging.info("Temp JSON file generated")

    logging.info("\n********************************************************")
    logging.info("Check the JSON and CSV files carefully before moving on!")
    logging.info("********************************************************")


def get_ADS_data():
    """ """
    # ADS API endpoint for searching
    api_url = "https://api.adsabs.harvard.edu/v1/search/query"
    # Read token from file
    with open("NASA_API_TOKEN", "r") as file:
        NASA_API_TOKEN = file.read().strip()
    headers = {"Authorization": f"Bearer {NASA_API_TOKEN}"}

    # Replace code with character
    bibcode_int = ADS_bibcode.replace("%26", "&")

    # Define the query parameters
    params = {"q": f"bibcode:{bibcode_int}", "fl": "author,year", "rows": 1}
    # Make the request to the ADS API
    response = requests.get(api_url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        if data["response"]["numFound"] > 0:
            article = data["response"]["docs"][0]

            authors_lst = article.get("author")
            if len(authors_lst) == 1:
                authors = authors_lst[0].split(",")[0]
            elif len(authors_lst) == 2:
                authors = (
                    authors_lst[0].split(",")[0] + " & " + authors_lst[1].split(",")[0]
                )
            else:
                authors = authors_lst[0].split(",")[0] + " et al."

            year = article.get("year")
        else:
            raise ValueError(f"No article found with the given bibcode: {bibcode_int}")
    else:
        raise ValueError(f"Failed to fetch data: {response.status_code}")

    return authors, year


def get_DB_name(current_JSON: dict, authors: str, year: str) -> str:
    """
    Generate a unique database name based on the author's name and year.

    Parameters:
    current_JSON (dict): A dictionary containing existing database names.
    authors (str): A string with the authors name properly formatted.
    year (str): The year to be included in the database name.

    Returns:
    str: A unique database name.
    """
    # Create a new database name
    DB_name = authors.split(" ")[0].strip().split("-")[0].upper() + year

    DBs_in_JSON = list(current_JSON.keys())

    if DB_name in DBs_in_JSON:
        DB_name += "_1"
        i = 2
        while True:
            if DB_name in DBs_in_JSON:
                DB_name = DB_name.replace(f"_{i - 1}", f"_{i}")
            else:
                break
            i += 1

    return DB_name


def get_CDS_table(logging) -> list | None:
    """ """
    # Obtain the available tables. Use row_limit=1 to avoid downloading the entire
    # tables
    viz = Vizier(row_limit=1)
    cat = viz.get_catalogs(ADS_bibcode)  # pyright: ignore
    if len(cat) == 0:
        raise ValueError(f"Could not extract data from {ADS_bibcode}")

    # Print info to screen
    logging.info(
        f"\nFull url: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={ADS_bibcode}\n"
    )
    rows = cat.__str__().split("\n")
    logging.info(rows[0].strip())
    for i, row in enumerate(rows[1:]):
        logging.info(
            f"{i}: " + cat[i].meta["description"] + " : " + row.split("and")[0].strip()
        )

    # Select table(s) to download
    while True:
        tab_idx = input("\nInput index(es) of table(s) to store (-1 for none): ")
        if tab_idx.strip() == "-1":
            table_url = []
            break
        tab_idx = tab_idx.strip().split(" ")
        try:
            tab_idx = np.array([int(_) for _ in tab_idx])
            if (tab_idx >= 0).all() and (tab_idx < len(cat)).all():
                table_url = [cat.keys()[_] for _ in tab_idx]
                break
            else:
                logging.info("Invalid input")
        except ValueError:
            logging.info("Invalid input")

    # Download selected table(s), if any
    df_all = None
    if len(table_url) > 0:
        # No limit to number of rows or columns
        viz = Vizier(row_limit=-1, columns=["all"])
        df_all = []
        for turl in table_url:
            # Download table
            try:
                cat = viz.get_catalogs(turl)  # pyright: ignore
                # Convert to pandas before storing
                df_all.append(cat.values()[0].to_pandas())
            except Exception as e:
                raise ValueError(f"Could not extract the data from {turl}\n{str(e)}")

    return df_all


def get_DB_from_Vizier(table_url: list) -> list:
    """
    Retrieve a database from Vizier and convert it to a pandas DataFrame.

    Parameters:
    table_url (list): The Vizier catalog's table identifier.

    Returns:
    list of pd.DataFrame: List of DataFrame containing the Vizier database.
    """
    # No limit to number of rows or columns
    viz = Vizier(row_limit=-1, columns=["all"])

    df_all = []
    for turl in table_url:
        # Download table
        try:
            cat = viz.get_catalogs(turl)  # pyright: ignore
            # Convert to pandas before storing
            df_all.append(cat.values()[0].to_pandas())
        except Exception as e:
            raise ValueError(f"Could not extract the data from {turl}\n{str(e)}")

    return df_all


def save_DB_CSV(temp_CSV_file, df_all):
    """
    Replace possible empty entries in columns
    """
    for idx, df in enumerate(df_all):
        name_CSV_file = str(temp_CSV_file)
        if idx > 0:
            name_CSV_file = temp_CSV_file.replace(".csv", f"_{idx}.csv")

        # Save and reload to bypass issue:
        # https://github.com/astropy/astropy/issues/17601
        df.to_csv(
            name_CSV_file,
            index=False,
        )
        df = pd.read_csv(name_CSV_file)

        # Remove leading and trailing spaces
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Replace empty strings or whitespace-only strings with NaN
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # Remove columns that are not useful
        for col in ("recno", "Simbad", "SimbadName"):
            if col in df.columns:
                df = df.drop(columns=[col])

        df.to_csv(
            name_CSV_file,
            na_rep="nan",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )


def current_JSON_vals(current_JSON):
    """ """
    # Extract unique names from the 'current_JSON' file
    names_lst = []
    for key in current_JSON.keys():
        names_lst.append(current_JSON[key]["names"])
    names_lst = list(set(names_lst))
    names_dict = {"names": names_lst}

    # Extract unique column names for the position, parameters and their uncertainties
    # from the 'current_JSON' file
    all_dicts = []
    for _id in ("pos", "pars", "e_pars"):
        # Extract all the entries (dicts) associated to this _id in the current JSON
        # file
        dicts_id = []
        for obj in current_JSON.values():
            dicts_id.append(obj[_id])

        # Extract all the entries listed in the model JSON object
        pdict = dict(JSON_struct["SMITH2500"][_id])

        # For each key in this entry of the model JSON object
        for key in pdict.keys():
            # For each dict associated to this _id in the current JSON file
            for obj_id in dicts_id:
                # If this object from the current JSON file contains the key, extract
                # its value
                if key in obj_id.keys():
                    # If this is a list, add
                    if isinstance(obj_id[key], list):
                        pdict[key] += obj_id[key]
                    else:
                        # If it is not a list, append
                        pdict[key].append(obj_id[key])
            pdict[key] = list(set(pdict[key]))
        all_dicts.append(pdict)

    return names_dict, *all_dicts


def new_DB_columns_match(
    logging, names_dict, pos_dict, pars_dict, e_pars_dict, df_all, ratio_min_fix=0.5
):
    """ """
    # Combine into a single dictionary
    merged_dict = {**names_dict, **pos_dict, **pars_dict, **e_pars_dict}

    df = df_all[0]
    if len(df_all) > 1:
        logging.info(f"Tables downloaded: {len(df_all)}")
        while True:
            tab_idx = input("Index of table used to obtain column names: ")
            try:
                tab_idx = int(tab_idx)
                if tab_idx >= 0 and tab_idx < len(df_all):
                    break
                else:
                    logging.info("Invalid input")
            except ValueError:
                logging.info("Invalid input")
        df = df_all[tab_idx]

    df_col_id = {}
    for col in df.keys():
        id_match, ratio_min = None, ratio_min_fix

        for key, vals in merged_dict.items():
            vals_ratios = []
            for val in vals:
                ld1 = Levenshtein.ratio(col, val)
                ld2 = Levenshtein.ratio(col.lower(), val.lower())
                vals_ratios.append(max(ld1, ld2))
            if vals_ratios:
                ratio_max = max(vals_ratios)
                if ratio_max > ratio_min:
                    id_match = key
                    ratio_min = ratio_max

        # Keep the match with the largest value for this 'key'
        if id_match is not None:
            try:
                ratio_old = df_col_id[id_match][0]
                if ratio_min > ratio_old:
                    df_col_id[id_match] = [ratio_min, col]
            except KeyError:
                df_col_id[id_match] = [ratio_min, col]

    # Only values are used
    df_col_id = {k: v[1] for k, v in df_col_id.items()}

    no_match_cols = []
    for key in df.keys():
        if key not in df_col_id.values():
            no_match_cols.append(key)
    logging.info(f"Columns in new DB with no match:\n  {no_match_cols}")

    return df_col_id


def proper_json_struct(df_col_id):
    """ """
    # Generate the proper structure for storing in the JSON file
    names = "None"
    if df_col_id is not None:
        if "names" in df_col_id.keys():
            names = df_col_id["names"]

    all_dicts = []
    for _id in ("pos", "pars", "e_pars"):
        pdict = {}
        for val in JSON_struct["SMITH2500"][_id].keys():
            try:
                if df_col_id is not None:
                    pdict[val] = df_col_id[val]
                else:
                    pdict[val] = "None"
            except KeyError:
                pass
        all_dicts.append(pdict)

    return names, *all_dicts


def add_DB_to_JSON(
    ADS_url,
    current_JSON,
    temp_JSON_file,
    DB_name,
    authors,
    year,
    names,
    pos_dict,
    pars_dict,
    e_pars_dict,
) -> None:
    """ """
    # Extract years in current JSON file
    years = []
    for db in current_JSON.keys():
        years.append(int(current_JSON[db]["year"]))

    # Index into a sorted list of integers, maintaining the sorted order
    index = 0
    while index < len(years) and years[index] < int(year):
        index += 1
    if index < 0:
        index = 0  # Prepend if index is negative
    elif index > len(current_JSON):
        index = len(current_JSON)  # Append if index is beyond the end

    # Create 'new_db_json' dictionary with the new DB's params
    new_db_json = {}
    new_db_json["ADS_url"] = ADS_url
    new_db_json["authors"] = authors
    new_db_json["year"] = year
    new_db_json["names"] = names
    new_db_json["pos"] = pos_dict
    new_db_json["pars"] = pars_dict
    new_db_json["e_pars"] = e_pars_dict

    # Adds an object to a JSON dictionary at a specific index.
    dbs_keys = list(current_JSON.keys())
    dbs_keys.insert(index, DB_name)
    new_json_dict = {}
    for key in dbs_keys:
        if key != DB_name:
            new_json_dict[key] = current_JSON[key]
        else:
            new_json_dict[DB_name] = new_db_json

    # Save to (temp) JSON file
    with open(temp_JSON_file, "w") as f:
        json.dump(new_json_dict, f, indent=2)  # Use indent for readability


if __name__ == "__main__":
    # This is the structure for each database in the JSON file
    JSON_struct = {
        "SMITH2500": {
            "ADS_url": "https://ui.adsabs.harvard.edu/abs/xxxx",
            "authors": "Smith et al.",
            "year": "2050",
            "names": "Name",
            "pos": {
                "RA": [],
                "DEC": [],
                "plx": [],
                "pmra": [],
                "pmde": [],
                "Rv": [],
            },
            "pars": {
                "ext": [],
                "diff_ext": [],
                "dist": [],
                "age": [],
                "met": [],
                "mass": [],
                "bi_frac": [],
                "bs_frac": [],
            },
            "e_pars": {
                "e_ext": [],
                "e_diff_ext": [],
                "e_dist": [],
                "e_age": [],
                "e_met": [],
                "e_mass": [],
                "e_bi_frac": [],
                "e_bs_frac": [],
            },
        }
    }

    main()
