import csv
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astroquery.vizier import Vizier
from rapidfuzz import fuzz

from .utils import logger
from .variables import (
    JSON_struct,
    NASA_API_TOKEN_file,
    dbs_folder,
    name_DBs_json,
    temp_folder,
)


def main():
    """
    Main function to download and process a database using NASA/ADS and Vizier data.

    ADS_bibcode: NASA/ADS bibcode for the new DB, e.g.: 2018MNRAS.481.3902B

    Steps:
    - Load the current JSON database file.
    - Check if the URL is already listed in the current database.
    - Fetch publication authors and year from NASA/ADS
    - Generate a new database name based on extracted metadata.
    - Handle temporary database files and check for existing data.
    - Fetch Vizier data or allow manual input for Vizier IDs.
    - Match new database columns with current JSON structure.
    - Update the JSON file and save the database as CSV.

    Raises:
        ValueError: If URL data fetching or metadata extraction fails.
    """
    logging = logger()

    # Handle temporary database files and check for existing data.
    # Temporary databases/ folder
    temp_database_folder = temp_folder + dbs_folder
    # Create folder if it does not exist
    Path(temp_database_folder).mkdir(parents=True, exist_ok=True)
    # Path to the new (temp) JSON file
    temp_JSON_file = temp_folder + name_DBs_json

    # Load current JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    # Check that no key in current_JSON share "received" values
    recieved_seen, citations_date = {}, {}
    for key, vals in current_JSON.items():
        value = vals["received"]
        if value in recieved_seen:
            first_key = recieved_seen[value]
            raise ValueError(
                f"Duplicate 'received={value}' found for keys: {first_key}, {key}"
            )
        recieved_seen[value] = key
        citations_date[key] = [vals["citations_count"]["date"], vals["SCIX_url"]]

    # Define dates
    date_now = datetime.datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.datetime.now().year
    six_months_ago = datetime.datetime.now() - datetime.timedelta(days=180)

    # If any date value in 'citations_date' (format: YYY-MM-DD) is older than 6 months,
    # offer to update them before moving on
    outdated_keys = {}
    for key, vals in citations_date.items():
        date_str, scix_url = vals
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        if date < six_months_ago:
            outdated_keys[key] = scix_url.split("/")[-1]
    if outdated_keys:
        logging.info(
            f"The following databases have 'citations_count' data older than 6 months:\n  {', '.join(outdated_keys)}"
        )
        if input("Update these data before moving on? (y/n): ").lower() == "y":
            for key, ADS_bibcode in outdated_keys.items():
                authors, year, title, citations = get_ADS_data(ADS_bibcode)
                citations_year = get_citations_year(current_year, year, citations)
                current_JSON["citations_count"] = {
                    "date": date_now,
                    "count": citations,
                    "citations_year": citations_year,
                }
            with open(name_DBs_json, "w") as f:
                json.dump(temp_JSON_file, f, indent=2)
            logging.info(
                "Updated 'citations_count'. Check the JSON file before moving on."
            )
            sys.exit(0)

    # Request user for bibcode
    ADS_bibcode = get_ads_bibcode()

    # Print info to screen
    pars_vals_all = dict(JSON_struct["SMITH2500"]["pars"])
    for _, v in current_JSON.items():
        for k, par in v["pars"].items():
            pars_vals_all[k] += list(set(list(par.keys())))
    logging.info("Available parameter keys in current JSON file:")
    for k, v in pars_vals_all.items():
        logging.info(f"{k:<10} : {', '.join(list(set(v)))}")
    logging.info("")

    # Check if url is already listed in the current JSON file
    ADS_bibcode = ADS_bibcode.replace("/", "")
    SCIX_url = "https://scixplorer.org/abs/" + ADS_bibcode
    for db, vals in current_JSON.items():
        if SCIX_url == vals["SCIX_url"]:
            logging.info(f"The URL {SCIX_url}\nis already in the JSON file under: {db}")
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit(0)

    # Fetch publication authors and year from NASA/ADS
    logging.info("Fetching NASA/ADS data...")
    authors, year, title, citations = get_ADS_data(ADS_bibcode)
    logging.info(
        f"Extracted author ({authors}), year ({year}), citations ({citations}), and title:"
    )
    logging.info(f"{title}")

    # Generate a new database name based on extracted metadata.
    DB_name = get_DB_name(current_JSON, authors, year)
    logging.info(f"New DB name obtained: {DB_name}")
    # Path to the new (temp) DB file
    temp_CSV_file = temp_database_folder + DB_name + ".csv"

    # Fetch Vizier data or allow manual input for Vizier IDs.
    vizier_url, quest = "N/A", "n"
    if Path(temp_CSV_file).is_file():
        quest = input("Load Vizier database from file (else download)? (y/n): ").lower()
    if quest == "y":
        df_all = [pd.read_csv(temp_CSV_file)]
        logging.info("Vizier CSV file loaded from file")
    else:
        table_url = get_CDS_table(logging, ADS_bibcode)
        df_all = get_DB_from_Vizier(logging, table_url)
        if df_all is not None:
            # Save the database(s) to a CSV file(s)
            save_DB_CSV(temp_CSV_file, df_all)
            logging.info(f"New DB csv file(s) stored {temp_CSV_file}\n")
            vizier_url = (
                f"https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={ADS_bibcode}"
            )

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

    # Update the JSON file and save the database as CSV.
    add_DB_to_JSON(
        JSON_struct,
        SCIX_url,
        citations,
        date_now,
        current_year,
        vizier_url,
        current_JSON,
        temp_JSON_file,
        DB_name,
        authors,
        title,
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


def get_citations_year(current_year, year, citations):
    """ """
    # Add 'citation_count/year'
    cyear_gap = current_year - int(year)
    if cyear_gap == 0:
        cyear_gap = 0.5
    citations_year = round(int(citations) / cyear_gap, 1)

    return citations_year


def get_ads_bibcode() -> str:
    """Get ADS bibcode from user input with validation."""
    print("\n📚 NASA/ADS Bibcode Required")
    print("Examples:")
    print("  • 2018MNRAS.481.3902B")
    print("  • 2021A&A...652A.102C")
    print("  • 2020ApJ...904...15K")

    while True:
        bibcode = input("\n📝 Enter the NASA/ADS bibcode (or 'c' to abort): ").strip()

        if not bibcode:
            print("❌ Bibcode cannot be empty. Please try again.")
            continue

        # Basic validation - should have year and journal format
        if len(bibcode) < 10:
            print("❌ Bibcode seems too short. Please check and try again.")
            continue

        return bibcode


def get_ADS_data(ADS_bibcode: str) -> tuple[str, str, str, str]:
    """ """
    # ADS API endpoint for searching
    api_url = "https://api.adsabs.harvard.edu/v1/search/query"
    # Read token from file
    with open(NASA_API_TOKEN_file, "r") as file:
        NASA_API_TOKEN = file.read().strip()
    headers = {"Authorization": f"Bearer {NASA_API_TOKEN}"}

    # Replace code with character
    bibcode_int = ADS_bibcode.replace("%26", "&")

    # Define the query parameters
    params = {
        "q": f"bibcode:{bibcode_int}",
        "fl": "author,year,title,citation_count",
        "rows": 1,
    }
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
            title = article.get("title")[0]
            # Remove invalid characters from title
            title = re.sub(r'[<>:"/\\|?*#&]', "", title)
            citations = str(article.get("citation_count", 0))
        else:
            raise ValueError(f"No article found with the given bibcode: {bibcode_int}")
    else:
        raise ValueError(f"Failed to fetch data: {response.status_code}")

    return authors, year, title, citations


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


def get_CDS_table(logging, ADS_bibcode: str) -> list:
    """Obtain the available tables from 'ADS_bibcode'"""
    # Use row_limit=1 to avoid downloading the entire tables
    viz = Vizier(row_limit=1)
    cat = viz.get_catalogs(ADS_bibcode)  # pyright: ignore
    if len(cat) == 0:
        logging.info(f"Could not extract data from {ADS_bibcode}")
        if input("Supply manual Vizier ID(s) instead? ") == "y":
            vizier_ID = input("Input Vizier ID (e.g.: J/PAZh/38/571): ").strip()
            cat = viz.get_catalogs(vizier_ID)  # pyright: ignore
            if len(cat) == 0:
                logging.info(f"Could not extract data from {vizier_ID}")
                return []
        else:
            return []

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

    return table_url


def get_DB_from_Vizier(logging, table_url: list) -> list | None:
    """
    Retrieve a database from Vizier and convert it to a pandas DataFrame.

    Parameters:
    table_url (list): The Vizier catalog's table identifier.

    Returns:
    list of pd.DataFrame: List of DataFrame containing the Vizier database.
    """
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
                logging.info(f"Could not extract the data from {turl}\n{str(e)}")

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
                    if isinstance(obj_id[key], str):  # _id='pos'
                        pdict[key].append(obj_id[key])
                    elif isinstance(obj_id[key], dict):  # _id='pars','e_pars'
                        for _, v in obj_id[key].items():
                            if isinstance(v, list):
                                pdict[key] += v
                            else:
                                pdict[key].append(v)
                    else:
                        raise ValueError(
                            f"Unknown type {type(obj_id[key])} for key {key}"
                        )

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
                ld1 = fuzz.ratio(col, val)
                ld2 = fuzz.ratio(col.lower(), val.lower())
                vals_ratios.append(max(ld1, ld2) / 100)
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
    """Generate the proper structure for storing in the JSON file"""
    names = "None"
    all_dicts = [{}, {}, {}]
    if df_col_id is None:
        return names, *all_dicts

    if "names" in df_col_id:
        names = df_col_id["names"]

    # Positions
    posdict = {}
    for val in JSON_struct["SMITH2500"]["pos"].keys():
        if val in df_col_id:
            posdict[val] = df_col_id[val]
    all_dicts = [posdict]

    # Parameters and their uncertainties. We use default values here, these need
    # manual checking once the JSON file is updated
    def_pars = {
        "av": "Av",
        "diff_ext": "dAv",
        "dist": "dm",
        "age": "loga",
        "met": "feh",
        "mass": "mass",
        "bi_frac": "bf",
        "blue_str": "bs",
    }
    for _id in ("pars", "e_pars"):
        pdict = {}
        for val in JSON_struct["SMITH2500"][_id].keys():
            if val in df_col_id:
                if _id == "pars":
                    pdict[val] = {def_pars[val]: df_col_id[val]}
                elif _id == "e_pars":
                    pdict[val] = {"e" + def_pars[val[2:]]: df_col_id[val]}
        all_dicts.append(pdict)

    return names, *all_dicts


def add_DB_to_JSON(
    JSON_struct: dict,
    SCIX_url: str,
    citations: str,
    date_now: str,
    current_year: int,
    vizier_url: str,
    current_JSON: dict,
    temp_JSON_file: str,
    DB_name: str,
    authors: str,
    title: str,
    year: str,
    names: str,
    pos_dict: dict,
    pars_dict: dict,
    e_pars_dict: dict,
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

    citations_year = get_citations_year(current_year, year, citations)

    # Create 'new_db_json' dictionary with the new DB's params
    new_db_json = {}
    new_db_json["SCIX_url"] = SCIX_url
    new_db_json["data_cmmts"] = JSON_struct["SMITH2500"]["data_cmmts"]
    new_db_json["citations_count"] = {
        "date": date_now,
        "count": citations,
        "citations_year": citations_year,
    }
    new_db_json["data_url"] = vizier_url
    new_db_json["authors"] = authors
    new_db_json["title"] = title
    new_db_json["year"] = year
    new_db_json["received"] = JSON_struct["SMITH2500"]["received"]
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
