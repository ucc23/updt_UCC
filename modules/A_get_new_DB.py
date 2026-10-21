import csv
import datetime
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astroquery.vizier import Vizier
from rapidfuzz import fuzz

from .utils import comments_check
from .variables import (
    JSON_struct,
    NASA_API_TOKEN_file,
    SCIX_root,
    abund_col_re,
    citation_count_days_check,
    dbs_folder,
    ec_val_re,
    err_col_re,
    err_percentiles,
    err_prefix,
    err_suffix,
    fmt_name_hints,
    fmt_ranges,
    fmt_to_UCC,
    meta_col_re,
    name_DBs_json,
    pars_formats,
    temp_folder,
)


def main():
    """
    Main function to download and process a database using NASA/ADS and Vizier data.

    ADS_bibcode: NASA/ADS bibcode for the new DB, e.g.: 2018MNRAS.481.3902B
    """

    temp_database_folder, temp_JSON_file, date_now, current_year, six_months_ago = (
        get_paths_dates()
    )

    current_JSON, citations_date = load_data()

    # Check if any citation counts are older than 6 months and offer to update them
    update_citation_count(
        date_now,
        current_year,
        six_months_ago,
        current_JSON,
        citations_date,
    )

    # Check that all parameters in the current JSON file are valid according to the
    # model JSON structure
    check_pars_format(current_JSON)

    # Request user for bibcode
    ADS_bibcode = get_ads_bibcode(current_JSON)

    # Fetch publication authors and year from NASA/ADS
    authors, year, title, citations = get_ADS_data(ADS_bibcode)

    # Generate a new database name based on extracted metadata.
    DB_name = get_DB_name(current_JSON, authors, year)

    # Fetch Vizier data or allow manual input for Vizier IDs.
    df_all, vizier_url = fetch_vizier_data(temp_database_folder, ADS_bibcode, DB_name)

    # Extract the names, positions, parameters, uncertainties and types column names
    # from the current JSON
    known_cols, types_info = current_JSON_vals(current_JSON)

    # Extract the matches for each column in the new DB
    col_match, types = new_DB_columns_match(known_cols, types_info, df_all, title)

    names, pos_dict, pars_dict, e_pars_dict = proper_json_struct(col_match)

    # Update the JSON file and save the database as CSV.
    new_json_dict = add_DB_to_JSON(
        temp_JSON_file,
        date_now,
        current_year,
        ADS_bibcode,
        citations,
        vizier_url,
        current_JSON,
        DB_name,
        authors,
        title,
        year,
        names,
        types,
        pos_dict,
        pars_dict,
        e_pars_dict,
    )
    print("Temp JSON file generated")

    print("\n********************************************************")
    print("Check the JSON and CSV files carefully before moving on!")
    print("********************************************************")
    comments_check(new_json_dict)


def get_paths_dates():
    """
    Get paths and dates for temporary files and citation checks.
    """
    # Temporary databases/ folder
    temp_database_folder = temp_folder + dbs_folder
    # Create folder if it does not exist
    Path(temp_database_folder).mkdir(parents=True, exist_ok=True)

    # Path to the new (temp) JSON file
    temp_JSON_file = temp_folder + name_DBs_json

    # Define dates
    date_now = datetime.datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.datetime.now().year
    six_months_ago = datetime.datetime.now() - datetime.timedelta(
        days=citation_count_days_check
    )

    return temp_database_folder, temp_JSON_file, date_now, current_year, six_months_ago


def load_data():
    """
    Load and validate the current JSON file, ensuring no duplicate 'received' values
    """
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

    return current_JSON, citations_date


def update_citation_count(
    date_now,
    current_year,
    six_months_ago,
    current_JSON,
    citations_date,
):
    """
    If any date value in 'citations_date' (format: YYY-MM-DD) is older than 6 months,
    offer to update them before moving on
    """
    outdated_keys = {}
    for key, vals in citations_date.items():
        date_str, scix_url = vals
        # Remove trailing '/'
        scix_url = scix_url.rstrip("/")
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        if date < six_months_ago:
            outdated_keys[key] = scix_url.split("/")[-1]
    # Only ask about half of the times to avoid prompting on every run
    if outdated_keys and random.random() < 0.5:
        print(
            "The following databases have 'citations_count' data older "
            f"than 6 months:\n  {', '.join(outdated_keys)}"
        )
        if input("Update these data before moving on? (y/n): ").lower() == "y":
            N_updt, failed = 0, {}
            for key, ADS_bibcode in outdated_keys.items():
                try:
                    _, year, _, citations = get_ADS_data(ADS_bibcode)
                except (ValueError, requests.RequestException) as e:
                    failed[key] = str(e)
                    continue
                citations_year = get_citations_year(current_year, year, citations)
                current_JSON[key]["citations_count"] = {
                    "date": date_now,
                    "count": citations,
                    "citations_year": citations_year,
                }
                N_updt += 1

            if failed:
                print("\nCould not update 'citations_count' for:")
                for key, err in failed.items():
                    print(f"  {key}: {err}")
                print(
                    "  (arXiv bibcodes may have been replaced by the published "
                    "one, check their 'SCIX_url')"
                )
            if N_updt == 0:
                print("No 'citations_count' updated, moving on")
                return

            with open(name_DBs_json, "w") as f:
                json.dump(current_JSON, f, indent=2)
            print(
                f"\nUpdated 'citations_count' for {N_updt} DBs. "
                "Check the JSON file before moving on."
            )
            sys.exit(1)


def get_citations_year(current_year, year, citations):
    """
    Calculate the citations per year for a given publication.
    """
    # Add 'citation_count/year'
    cyear_gap = current_year - int(year)
    if cyear_gap == 0:
        cyear_gap = 0.5
    citations_year = round(int(citations) / cyear_gap, 1)

    return citations_year


def check_pars_format(current_JSON):
    """
    Check that all parameters in the current JSON file are valid according to the
    model JSON structure
    """
    # Extract all parameter values from the model JSON structure
    pars_general_all = dict(JSON_struct["SMITH2500"]["pars"])

    # Internal consistency check, If this fails, there's something wrong in the
    # 'variables' file
    if pars_general_all.keys() != pars_formats.keys():
        raise ValueError(
            "Mismatch between 'pars_formats' keys and 'JSON_struct[pars]' keys."
            "Check the 'variables.py' file."
        )
    # Every format must have a converter to the UCC units (used by B)
    no_conv = [f for v in pars_formats.values() for f in v if f not in fmt_to_UCC]
    if no_conv:
        raise ValueError(
            f"Formats in 'pars_formats' without a converter in 'fmt_to_UCC': "
            f"{no_conv}. Check the 'variables.py' file."
        )

    # Print info to screen
    print("\nAvailable parameter formats:")
    for k, v in pars_formats.items():
        print(f"{k:<10} : {', '.join(list(set(v)))}")

    e_pars_formats = {}
    for k, v in pars_formats.items():
        if len(v) > 1:
            e_pars_formats["e_" + k] = ["e" + v for v in v]
        else:
            e_pars_formats["e_" + k] = ["e" + v[0]]
    pars_epars = {"pars": pars_formats, "e_pars": e_pars_formats}

    error_flag = False
    # Check each DB
    for db_name, db_vals in current_JSON.items():
        for col, pars_epars_formats in pars_epars.items():
            for db_key_par_general, db_dict_par_general in db_vals[col].items():
                if db_key_par_general not in pars_epars_formats:
                    print(
                        f"Parameter {db_key_par_general} in {db_name} is not "
                        "in the model JSON structure"
                    )
                    error_flag = True
                for db_par_format in db_dict_par_general:
                    if db_par_format not in pars_epars_formats[db_key_par_general]:
                        print(
                            f"Parameter {db_par_format} in {db_name} is not valid --> "
                            f"{col}"
                            + "{"
                            + f"{db_key_par_general}: {pars_epars_formats[db_key_par_general]}"
                            + "}"
                        )
                        error_flag = True
    if error_flag:
        raise ValueError(
            "\nOne or more parameters in the current JSON file are not "
            "valid according to the model JSON structure. "
            "Please check the logs for details."
        )
    print()


def get_ads_bibcode(current_JSON) -> str:
    """Get ADS bibcode from user input with validation."""
    print("\n📚 NASA/ADS Bibcode Required")
    print("Examples:")
    print("  • 2018MNRAS.481.3902B")
    print("  • 2021A&A...652A.102C")
    print("  • 2020ApJ...904...15K")

    while True:
        ADS_bibcode = input(
            "\n📝 Enter the NASA/ADS bibcode (or 'c' to abort): "
        ).strip()

        if not ADS_bibcode:
            print("❌ Bibcode cannot be empty. Please try again.")
            continue

        # Basic validation - should have year and journal format
        if len(ADS_bibcode) < 10:
            print("❌ Bibcode seems too short. Please check and try again.")
            continue

        # return bibcode
        break

    # Check if url is already listed in the current JSON file
    ADS_bibcode = ADS_bibcode.replace("/", "")
    SCIX_url = SCIX_root + ADS_bibcode
    for db, vals in current_JSON.items():
        if SCIX_url == vals["SCIX_url"]:
            print(f"The URL {SCIX_url}\nis already in the JSON file under: {db}")
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit(0)

    return ADS_bibcode


def get_ADS_data(ADS_bibcode: str) -> tuple[str, str, str, str]:
    """
    Fetch publication metadata from NASA/ADS using the provided bibcode.
    """
    print("Fetching NASA/ADS data...")

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

    print(
        f"Extracted author ({authors}), year ({year}), citations ({citations}), and title:"
    )
    print(f"{title}")

    return authors, year, title, citations


def get_DB_name(current_JSON: dict, authors: str, year: str) -> str:
    """
    Generate a unique database name based on the author's name and year.
    """
    # Create a new database name
    DB_name = authors.split(" ")[0].strip().split("-")[0]
    # Replace letters that do not decompose into base letter + accent
    special_chars = {
        "ø": "o", "Ø": "O", "ß": "ss", "ẞ": "SS", "æ": "ae", "Æ": "AE",
        "œ": "oe", "Œ": "OE", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
        "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH", "ı": "i", "ħ": "h", "Ħ": "H",
    }  # fmt: skip
    DB_name = DB_name.translate(str.maketrans(special_chars))
    # Remove accents (e.g.: 'É' -> 'E')
    DB_name = "".join(
        c for c in unicodedata.normalize("NFKD", DB_name) if not unicodedata.combining(c)
    )
    # Drop any remaining non-ASCII character
    DB_name = DB_name.encode("ascii", "ignore").decode()
    DB_name = DB_name.upper() + year

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

    print(f"New DB name obtained: {DB_name}")

    return DB_name


def get_CDS_table(ADS_bibcode: str) -> list:
    """Obtain the available tables from 'ADS_bibcode'"""
    # Use row_limit=1 to avoid downloading the entire tables
    viz = Vizier(row_limit=1)
    cat = viz.get_catalogs(ADS_bibcode)  # pyright: ignore
    if len(cat) == 0:
        print(f"Could not extract data from {ADS_bibcode}")
        if input("Supply manual Vizier ID(s) instead? (y/n): ") == "y":
            vizier_ID = input("Input Vizier ID (e.g.: J/PAZh/38/571): ").strip()
            vizier_ID = vizier_ID.replace(
                "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=", ""
            )
            cat = viz.get_catalogs(vizier_ID)  # pyright: ignore
            if len(cat) == 0:
                print(f"Could not extract data from {vizier_ID}")
                return []
        else:
            return []

    # Print info to screen
    print(
        f"\nFull url: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={ADS_bibcode}\n"
    )
    rows = cat.__str__().split("\n")
    print(rows[0].strip())
    for i, row in enumerate(rows[1:]):
        print(
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
                print("Invalid input")
        except ValueError:
            print("Invalid input")

    return table_url


def get_DB_from_Vizier(table_url: list) -> list | None:
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
            except Exception as e:  # noqa: BLE001
                print(f"Could not extract the data from {turl}\n{e!s}")

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


def fetch_vizier_data(temp_database_folder, ADS_bibcode, DB_name):
    """
    Fetch Vizier data for the given ADS bibcode. If a temporary CSV file exists,
    prompt the user to load it or download new data. Save the database(s) to CSV
    files if downloaded.
    """

    # Path to the new (temp) DB file
    temp_CSV_file = temp_database_folder + DB_name + ".csv"

    vizier_url, quest = "N/A", "n"

    if Path(temp_CSV_file).is_file():
        quest = input("Load Vizier database from file (else download)? (y/n): ").lower()

    if quest == "y":
        df_all = [pd.read_csv(temp_CSV_file)]
        print("Vizier CSV file loaded from file")
    else:
        table_url = get_CDS_table(ADS_bibcode)
        df_all = get_DB_from_Vizier(table_url)
        if df_all is not None:
            # Save the database(s) to a CSV file(s)
            save_DB_CSV(temp_CSV_file, df_all)
            print(f"New DB csv file(s) stored {temp_CSV_file}\n")
            vizier_url = (
                f"https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={ADS_bibcode}"
            )

    return df_all, vizier_url


def current_JSON_vals(current_JSON):
    """
    Extract the column names used in the current JSON file for the names, positions,
    parameters, uncertainties, and embedded clusters types.

    Returns:
    - known_cols: {target_key: {col_name: fmt}}, where 'target_key' is "names", a
      'pos' key (e.g. "RA"), a 'pars' key (e.g. "dist") or an 'e_pars' key (e.g.
      "e_dist"), and 'fmt' is the format key most often used in the JSON file
      with that column for 'pars' and 'e_pars' (e.g. "dpc", "edpc"), or None for
      names and positions
    - types_info: (set of known 'types' column names, set of known EC ids)
    """
    known_cols = {"names": {}}
    for _id in ("pos", "pars", "e_pars"):
        for key in JSON_struct["SMITH2500"][_id]:
            known_cols[key] = {}
    # Count of the formats used for each (key, column)
    fmt_counts = {}

    type_cols, ec_ids = set(), set()
    for obj in current_JSON.values():
        known_cols["names"][obj["names"]] = None

        for key, col in obj["pos"].items():
            known_cols.setdefault(key, {})[col] = None

        for _id in ("pars", "e_pars"):
            for key, fmt_dict in obj[_id].items():
                if not isinstance(fmt_dict, dict):
                    raise TypeError(f"Unknown type {type(fmt_dict)} for key {key}")
                for fmt, cols in fmt_dict.items():
                    cols = cols if isinstance(cols, list) else [cols]
                    for col in cols:
                        cnt = fmt_counts.setdefault((key, col), {})
                        cnt[fmt] = cnt.get(fmt, 0) + 1

        types = obj.get("types", {})
        if types.get("col_name", "N/A") not in ("N/A", "ALL"):
            type_cols.add(types["col_name"])
            ec_ids.update(types["ec_id"])

    for (key, col), cnt in fmt_counts.items():
        known_cols.setdefault(key, {})[col] = max(cnt, key=cnt.get)

    return known_cols, (type_cols, ec_ids)


def _norm_col(col: str) -> str:
    """Lower case and remove non alpha-numeric characters"""
    return re.sub(r"[^a-z0-9]", "", col.lower())


def _col_score(col: str, val: str) -> float:
    """Similarity in [0, 1] between a new column name and a known column name"""
    if col == val:
        return 1.0
    ncol, nval = _norm_col(col), _norm_col(val)
    if ncol and ncol == nval:
        return 0.99
    # Fuzzy matching is meaningless for very short names (e.g. 'D' vs 'R')
    if min(len(ncol), len(nval)) < 3:
        return 0.0
    return (
        max(
            fuzz.ratio(col, val),
            fuzz.ratio(col.lower(), val.lower()),
            fuzz.ratio(ncol, nval),
        )
        / 100
    )


def _col_stats(df, col):
    """
    1st, 50th and 99th percentiles of a numeric column, or None if the column is
    not numeric
    """
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) == 0 or len(vals) < 0.5 * df[col].notna().sum():
        return None
    return np.percentile(vals, (1, 50, 99))


def _guess_fmt(df, col, par, fmt):
    """
    Check that the values of 'col' are compatible with parameter 'par' and return
    the format to use, or None if no format is compatible (spurious match) or the
    column is not numeric.

    For 'age' and 'dist' the format is inferred from the column name (e.g. 'Age_Myr',
    'd_kpc', 'logt', 'MOD') or else from its values, since the same column name is
    used with different formats in different DBs (e.g. 'Age'). For the remaining
    parameters 'fmt' (from the closest matching known column) is kept if its range
    is compatible, else the first compatible format is used.
    """
    stats = _col_stats(df, col)
    if stats is None:
        return None
    p01, med, p99 = stats

    if par != "met" and abund_col_re.search(col):
        return None

    if par in fmt_name_hints:
        for regex, hint_fmt in fmt_name_hints[par]:
            if re.search(regex, col, flags=re.IGNORECASE):
                return hint_fmt
        if par == "age":
            if p01 >= 5 and p99 <= 10.3:
                return "loga"
            return "agyr" if p99 <= 14 else "amyr"
        if par == "dist":
            if p01 >= 2 and p99 <= 22:
                return "dm"
            return "dpc" if med >= 60 else "dkpc"

    ranges = fmt_ranges.get(par, {})
    if fmt not in ranges:
        return fmt
    lo, hi = ranges[fmt]
    if lo <= med <= hi:
        return fmt
    for f, (lo, hi) in ranges.items():
        if lo <= med <= hi:
            return f
    return None


def _find_err_cols(col, cols_lower):
    """
    Find the uncertainty column(s) associated to 'col': a single column (e.g.
    'e_X', 'X_sigma') or a pair of percentiles (e.g. 'X_16', 'X_84'; if 'col' is
    a median such as 'X_50' the base name 'X' is used).
    """
    for pre in err_prefix:
        if (pre + col).lower() in cols_lower:
            return cols_lower[(pre + col).lower()]
    for suf in err_suffix:
        if (col + suf).lower() in cols_lower:
            return cols_lower[(col + suf).lower()]
    base = re.sub(r"_?(50|p50)$", "", col)
    for lo, hi in err_percentiles:
        c_lo, c_hi = (base + lo).lower(), (base + hi).lower()
        if c_lo in cols_lower and c_hi in cols_lower:
            return [cols_lower[c_lo], cols_lower[c_hi]]
    return None


def _types_match(df, types_info, title, max_unique=20):
    """
    Auto-detect the column (if any) that identifies embedded clusters.

    A column is a candidate if it has a small number of unique values and some of
    them match EC ids used in other DBs, or look like an EC id ('EC*', 'IRC*').
    Columns whose name matches a known 'types'
    column but contain no known EC ids are only reported, for manual checking.
    """
    type_cols, ec_ids = types_info
    types = dict(JSON_struct["SMITH2500"]["types"])
    types["ec_id"] = []

    best, best_score = None, (0, 0.0)
    for col in df:
        if df[col].dtype.kind not in "OSU":  # Only string columns
            continue
        uniq = set(df[col].dropna().astype(str).str.strip().unique())
        if not uniq or len(uniq) > max_unique:
            continue
        name_score = max([_col_score(str(col), _) for _ in type_cols], default=0.0)
        hits = {_ for _ in uniq if _ in ec_ids or ec_val_re.match(_)}
        if hits:
            score = (len(hits), name_score)
            if score > best_score:
                best, best_score = (str(col), sorted(hits)), score
        elif name_score >= 0.99:
            print(
                f"Column '{col}' may identify embedded clusters, but contains no "
                + f"known EC ids. Unique values: {sorted(uniq)}"
            )

    if best is not None:
        types["col_name"], types["ec_id"] = best
        print(f"Embedded clusters column detected: {types}")
    elif "embedded" in title.lower():
        print(
            "No embedded clusters column found but the title mentions 'embedded'. "
            + 'If all entries are ECs use: "types": {"col_name": "ALL", ...}'
        )

    return types


def new_DB_columns_match(known_cols, types_info, df_all, title="", ratio_min=0.8):
    """
    Match columns in the new database with the current JSON structure.

    1. The column that identifies embedded clusters (if any) is detected and
       removed from the pool of columns to match.
    2. Every (column, key) pair for the names, positions and parameters is scored
       against all the column names already used in the JSON file for that key.
       Uncertainty columns (e.g. 'e_Dist', 'AV16') are excluded here. Names must
       be string columns, positions and parameters must be numeric, and parameters
       must have values in a plausible range (which is also used to correct the
       format, e.g. 'loga' vs 'amyr').
    3. Pairs are assigned greedily by decreasing score so that each column and each
       key are used at most once.
    4. Uncertainties are assigned from the matched parameters: if parameter 'X' is
       matched, columns such as 'e_X', 'X_sigma' or the percentiles 'X_16', 'X_84'
       are searched for and stored with the same format as the parameter.

    Returns:
    - col_match: {key: (column(s), fmt)}
    - types: dict for the 'types' entry of the JSON file
    """
    types = dict(JSON_struct["SMITH2500"]["types"])
    if df_all is None:
        return None, types

    df = df_all[0]
    if len(df_all) > 1:
        print(f"Tables downloaded: {len(df_all)}")
        while True:
            tab_idx = input("Index of table used to obtain column names: ")
            try:
                tab_idx = int(tab_idx)
                if tab_idx >= 0 and tab_idx < len(df_all):
                    break
                else:
                    print("Invalid input")
            except ValueError:
                print("Invalid input")
        df = df_all[tab_idx]
    df = df.rename(columns=str)

    # 1. Embedded clusters column
    types = _types_match(df, types_info, title)

    cols = list(df.columns)
    match_cols = [
        _
        for _ in cols
        if _ != types["col_name"]
        and not err_col_re.search(_)
        and not meta_col_re.match(_)
    ]
    pars_keys = JSON_struct["SMITH2500"]["pars"]

    # 2. Score all pairs
    pairs = []
    for col in match_cols:
        is_str = _col_stats(df, col) is None
        for key, vals in known_cols.items():
            if key.startswith("e_") or is_str != (key == "names"):
                continue
            best_score, best_fmt = 0.0, None
            for val, fmt in vals.items():
                if val == "None":
                    continue
                score = _col_score(col, val)
                if score > best_score:
                    best_score, best_fmt = score, fmt
            if best_score < ratio_min:
                continue
            if key in pars_keys:
                best_fmt = _guess_fmt(df, col, key, best_fmt)
                if best_fmt is None:
                    continue
            pairs.append((best_score, col, key, best_fmt))

    # 3. Greedy one-to-one assignment
    col_match, scores, used_cols = {}, {}, set()
    for score, col, key, fmt in sorted(pairs, key=lambda _: -_[0]):
        if key in col_match or col in used_cols:
            continue
        col_match[key] = (col, fmt)
        scores[key] = score
        used_cols.add(col)

    # 4. Uncertainties derived from the matched parameters
    cols_lower = {_.lower(): _ for _ in cols}
    for key in pars_keys:
        if key not in col_match:
            continue
        col, fmt = col_match[key]
        e_cols = _find_err_cols(col, cols_lower)
        if e_cols is None:
            continue
        col_match["e_" + key] = (e_cols, "e" + fmt)
        used_cols.update(e_cols if isinstance(e_cols, list) else [e_cols])

    print("Columns matched (score):")
    for key, (col, fmt) in col_match.items():
        fmt_str = f" [{fmt}]" if fmt else ""
        print(f"  {key:<12} -> {col}{fmt_str} ({scores.get(key, 1.0):.2f})")

    no_match_cols = [_ for _ in cols if _ not in used_cols and _ != types["col_name"]]
    print(f"Columns in new DB with no match:\n  {no_match_cols}")
    print("Column names for temp JSON file extracted")

    return col_match, types


def proper_json_struct(col_match):
    """Generate the proper structure for storing in the JSON file"""
    names = "None"
    all_dicts = [{}, {}, {}]

    if col_match is None:
        return names, *all_dicts

    if "names" in col_match:
        names = col_match["names"][0]

    # Positions
    posdict = {}
    for val in JSON_struct["SMITH2500"]["pos"]:
        if val in col_match:
            posdict[val] = col_match[val][0]
    all_dicts = [posdict]

    # Parameters and their uncertainties. The format is the one used by the closest
    # matching column in the JSON file, it needs manual checking once the JSON file
    # is updated
    for _id in ("pars", "e_pars"):
        pdict = {}
        for val in JSON_struct["SMITH2500"][_id]:
            if val in col_match:
                col, fmt = col_match[val]
                if fmt is None:
                    par = val[2:] if _id == "e_pars" else val
                    fmt = ("e" if _id == "e_pars" else "") + pars_formats[par][0]
                pdict[val] = {fmt: col}
        all_dicts.append(pdict)

    return names, *all_dicts


def add_DB_to_JSON(
    temp_JSON_file: str,
    date_now: str,
    current_year: int,
    ADS_bibcode: str,
    citations: str,
    vizier_url: str,
    current_JSON: dict,
    DB_name: str,
    authors: str,
    title: str,
    year: str,
    names: str,
    types: dict,
    pos_dict: dict,
    pars_dict: dict,
    e_pars_dict: dict,
) -> dict:
    """
    Add a new database entry to the current JSON structure and save it to a temporary
    JSON file.
    """
    # Extract years in current JSON file
    years = []
    for db in current_JSON:
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
    SCIX_url = SCIX_root + ADS_bibcode
    new_db_json = {}
    new_db_json["SCIX_url"] = SCIX_url
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
    new_db_json["types"] = types
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

    return new_json_dict
