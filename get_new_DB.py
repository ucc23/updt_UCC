import csv
import json
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
import requests
from astroquery.vizier import Vizier
from bs4 import BeautifulSoup

from modules import (
    aux,
)
from modules.HARDCODED import (
    dbs_folder,
    name_DBs_json,
    temp_fold,
)

# NASA/ADS url to the new DB
ADS_url0 = "https://ui.adsabs.harvard.edu/abs/xxxx"

# This is the structure for each database in the JSON fle
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


def main():
    """ """
    logging = aux.logger()

    # Proper url format
    ADS_url = ADS_url0
    if ADS_url.endswith("/"):
        ADS_url = ADS_url[:-1]
    if ADS_url.endswith("/abstract"):
        ADS_url = ADS_url.replace("/abstract", "")

    # Load current JSON file
    JSON_file = dbs_folder + name_DBs_json
    with open(JSON_file) as f:
        current_JSON = json.load(f)

    # Check if url is already listed in the current JSON file
    for db, vals in current_JSON.items():
        if ADS_url == vals["ADS_url"]:
            # raise ValueError(
            #     f"The URL {ADS_url}\nis already in the JSON file under: {db}"
            # )
            logging.info(f"The URL {ADS_url}\nis already in the JSON file under: {db}")

    logging.info("Fetching NASA/ADS url...")
    try:
        ADS_soup = get_ADS_soup(ADS_url)
    except Exception as e:
        raise ValueError(f"Could not fetch the URL {ADS_url}\n{str(e)}")
    logging.info("NASA/ADS data downloaded")
    # with open("temp.html", "w") as f:
    #     f.write(str(ADS_soup))
    # with open("temp.html", "rb") as f:
    #     ADS_soup = BeautifulSoup(f.read(), "html.parser")

    authors, year = get_autors_year(ADS_soup)
    if year is None:
        raise ValueError("Could not extract the publication year")
    if authors is None:
        raise ValueError("Could not create the authors list")
    logging.info(f"Extracted authors ({authors}) and year ({year})")

    DB_name = get_DB_name(current_JSON, authors, year)
    logging.info(f"New DB name obtained: {DB_name}")

    # Temporary databases/ folder
    temp_database_folder = temp_fold + dbs_folder
    # Create folder if it does not exist
    Path(temp_database_folder).mkdir(parents=True, exist_ok=True)
    # Path to the new (temp) JSON file
    temp_JSON_file = temp_database_folder + name_DBs_json
    # Path to the new (temp) DB file
    temp_CSV_file = temp_database_folder + DB_name + ".csv"

    quest = "n"
    if Path(temp_CSV_file).is_file():
        quest = input("Load Vizier database from file (else download)? (y/n): ").lower()
    if quest == "y":
        df_all = [pd.read_csv(temp_CSV_file)]
        logging.info("Vizier CSV file loaded from file")
    else:
        cds_url = get_CDS_url(logging, ADS_soup)
        if cds_url is None:
            logging.info("Could not extract the CDS url")
            if (
                input(
                    "Input Vizier id manually (format '2010AA..36..75G')? (y/n): "
                ).lower()
                == "y"
            ):
                cds_url = str(input("Vizier id (format '2010AA..36..75G'): "))

        df_all = None
        if cds_url is not None:
            df_all = get_DB_from_Vizier(logging, cds_url)

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

    logging.info("\nCheck carefully the JSON and CSV file before moving on!")


def get_ADS_soup(ADS_url):
    """Fetch the webpage content"""
    response = requests.get(ADS_url)
    response.raise_for_status()
    ADS_soup = BeautifulSoup(response.text, "html.parser")

    return ADS_soup


def get_autors_year(ADS_soup) -> tuple[str | None, str | None]:
    """ """
    # Extract year
    year = None
    try:
        year_element = ADS_soup.find(
            "meta", attrs={"property": "article:published_time"}
        )
        if year_element is not None:
            year = year_element.get("content")
            year = str(year).split("/")[1]
    except Exception:
        pass

    # Extract authors
    authors = None
    try:
        authors_lst = [author.text for author in ADS_soup.select(".author")]
        if len(authors_lst) == 1:
            authors = authors_lst[0].split(",")[0]
        elif len(authors_lst) == 2:
            authors = (
                authors_lst[0].split(",")[0] + " & " + authors_lst[1].split(",")[0]
            )
        else:
            authors = authors_lst[0].split(",")[0] + " et al."
    except Exception:
        pass

    return authors, year


def get_DB_name(current_JSON: dict, authors: str, year: str) -> str:
    """
    Generate a unique database name based on the author's name and year.

    Parameters:
    current_JSON (dict): A dictionary containing existing database names.
    authors (str): A string of author names.
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
                DB_name = DB_name.replace(f"_{i-1}", f"_{i}")
            else:
                break
            i += 1

    return DB_name


def get_CDS_url(logging, ADS_soup):
    """ """
    cds_url = None
    try:
        # Extract CDS URL if available
        cds_link = ADS_soup.find("a", href=True, string=lambda _: _ and "CDS" in _)
        base_url = "https://ui.adsabs.harvard.edu"
        if cds_link:
            cds_url = base_url + cds_link["href"]
            response = requests.get(cds_url)
            soup = BeautifulSoup(response.text, "html.parser")
            vizier_element = soup.find("a", {"uib-popover": "Query VizieR table(s)"})
            if vizier_element:
                cds_url = vizier_element.get("href")
    except Exception as e:
        logging.info(f"\n{e}\n")
        pass

    if cds_url is not None:
        cds_url = str(cds_url)
        # Extract string required for the Vizier query
        if cds_url.split("/")[-1] == "CDS":
            cds_url = cds_url.split("/")[-2]
        elif "VizieR?-source=" in cds_url:
            cds_url = cds_url.split("VizieR?-source=")[-1]

        logging.info(f"\nCDS url obtained: {cds_url}")

    return cds_url


def get_DB_from_Vizier(logging, cds_url: str) -> list | None:
    """
    Retrieve a database from Vizier and convert it to a pandas DataFrame.

    Parameters:
    cds_url (str): The Vizier catalog identifier.

    Returns:
    pd.DataFrame: DataFrame containing the Vizier database.
    """
    # No limit to number of rows or columns
    viz = Vizier(row_limit=-1, columns=["all"])
    # Find catalog
    vdict = viz.find_catalogs(cds_url)

    # Extract catalog's name
    cat_name = list(vdict.keys())[0]
    if cat_name is None:
        raise ValueError(f"Could not extract the catalog's name from {cds_url}")
    logging.info(
        f"Full url: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={cat_name}\n"
    )
    for k, v in vdict.items():
        logging.info(str(k) + ":" + str(v.description))

    try:
        # Download full database
        cat = viz.get_catalogs(vdict)
        if len(cat) > 1:
            rows = cat.__str__().split("\n")
            logging.info(rows[0].strip())
            for i, row in enumerate(rows[1:]):
                logging.info(f"{i}: " + cat[i].meta["description"])
                logging.info("   " + row.strip())
            while True:
                tab_idx = input("Input index(es) of table(s) to store (-1 for none): ")
                if tab_idx.strip() == "-1":
                    return None
                tab_idx = tab_idx.strip().split(" ")
                try:
                    tab_idx = np.array([int(_) for _ in tab_idx])
                    if (tab_idx >= 0).all() and (tab_idx < len(cat)).all():
                        break
                    else:
                        logging.info("Invalid input")
                except ValueError:
                    logging.info("Invalid input")
        else:
            tab_idx = [0]

        # Convert to pandas before storing
        df_all = []
        for idx in tab_idx:
            df_all.append(cat[int(idx)].to_pandas())
    except Exception as e:
        raise ValueError(f"Could not extract the data from {cat_name}\n{str(e)}")

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
    main()
