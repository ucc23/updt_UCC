from string import ascii_lowercase

import numpy as np
import pandas as pd

from ..utils import date_order_DBs, radec2lonlat, rename_standard


def combine_UCC_new_DB(
    logging,
    new_DB: str,
    newDB_json: dict,
    df_UCC: pd.DataFrame,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
    sep: str = ",",
) -> dict:
    """
    Combines a new database with the UCC, handling new and existing entries.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    new_DB : str
        Name of the new database.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    df_new : pd.DataFrame
        DataFrame of the new database.
    new_DB_fnames : list
        List of lists, where each inner list contains the
        standardized names for each cluster in the new catalogue.
    db_matches : list
        List of indexes into the UCC pointing to each entry in the
        new DB.
    sep : str, optional
        Separator used for splitting names in the new database. Default is ",".

    Returns
    -------
    dict
        Dictionary representing the updated database with new and modified entries.
    """
    N_new, N_updt = 0, 0
    new_db_dict = {_: [] for _ in df_UCC.keys()}
    # For each entry in the new DB
    for i_new_cl, new_cl in enumerate(new_DB_fnames):
        row_n = dict(df_new.iloc[i_new_cl])

        # For each comma separated name for this OC in the new DB
        oc_names = row_n[newDB_json["names"]].split(sep)
        # Rename certain entries (ESO, FSR, etc)
        new_names_rename = []
        for _ in oc_names:
            name = rename_standard(_.strip())
            new_names_rename.append(name)
        oc_names = ";".join(new_names_rename)

        # Index of the match for this new cluster in the UCC (if any)
        if db_matches[i_new_cl] is None:
            # The cluster is not present in the UCC, add to the new Db dictionary
            new_db_dict = new_OC_not_in_UCC(
                new_DB, new_db_dict, i_new_cl, new_cl, row_n, oc_names, newDB_json
            )
            N_new += 1
        else:
            # The cluster is already present in the UCC
            # Row in UCC where this match is located
            row = dict(df_UCC.iloc[db_matches[i_new_cl]])
            # Add to the new Db dictionary
            new_db_dict = OC_in_UCC(
                new_DB, new_db_dict, i_new_cl, new_cl, oc_names, row
            )
            N_updt += 1

    logging.info(f"New entries: {N_new}, Updated entries: {N_updt}")

    return new_db_dict


def new_OC_not_in_UCC(
    new_DB: str,
    new_db_dict: dict,
    i_new_cl: int,
    fnames_new_cl: list[str],
    row_n: dict,
    oc_names: str,
    newDB_json: dict,
) -> dict:
    """
    Adds a new OC, not present in the UCC, to the new database dictionary.

    Parameters
    ----------
    new_DB : str
        Name of the new database.
    new_db_dict : dict
        Dictionary representing the updated database.
    i_new_cl : int
        Index of the new cluster in the new database.
    fnames_new_cl : list
        List of standardized names for the new cluster.
    row_n : dict
        Dictionary representing the row of the new cluster in the new database.
    oc_names : str
        Semicolon-separated string of names for the new cluster.
    newDB_json : dict
        Dictionary with column name mappings for the new database.

    Returns
    -------
    dict
        Updated dictionary representing the database with the new OC added.
    """
    # Remove duplicates from names and fnames
    ID = oc_names
    if ";" in oc_names:
        ID = rm_name_dups(oc_names)
    fnames = ";".join(fnames_new_cl)
    if ";" in fnames:
        fnames = rm_name_dups(fnames)

    ra_n, dec_n = row_n[newDB_json["pos"]["RA"]], row_n[newDB_json["pos"]["DEC"]]
    # Galactic coordinates
    lon_n, lat_n = radec2lonlat(ra_n, dec_n)
    #
    plx_n = np.nan
    if "plx" in newDB_json["pos"]:
        plx_n = row_n[newDB_json["pos"]["plx"]]
    pmra_n = np.nan
    if "pmra" in newDB_json["pos"]:
        pmra_n = row_n[newDB_json["pos"]["pmra"]]
    pmde_n = np.nan
    if "pmde" in newDB_json["pos"]:
        pmde_n = row_n[newDB_json["pos"]["pmde"]]

    new_vals = {
        "DB": new_DB,
        "DB_i": str(i_new_cl),
        "ID": ID,
        "RA_ICRS": round(ra_n, 4),
        "DE_ICRS": round(dec_n, 4),
        "GLON": round(lon_n, 4),
        "GLAT": round(lat_n, 4),
        "Plx": round(plx_n, 4),
        "pmRA": round(pmra_n, 4),
        "pmDE": round(pmde_n, 4),
        "fnames": fnames,
    }
    new_db_dict = updt_new_db_dict(new_db_dict, new_vals)

    return new_db_dict


def OC_in_UCC(
    new_DB: str,
    new_db_dict: dict,
    i_new_cl: int,
    fnames_new_cl: list[str],
    oc_names: str,
    row: dict,
) -> dict:
    """
    Updates an existing OC in the UCC with information from the new database.

    Parameters
    ----------
    new_DB : str
        Name of the new database.
    new_db_dict : dict
        Dictionary representing the updated database.
    i_new_cl : int
        Index of the cluster in the new database.
    fnames_new_cl : list
        List of standardized names for the cluster.
    oc_names : str
        Semicolon-separated string of names for the cluster.
    row : dict
        Dictionary representing the row of the cluster in the UCC.

    Returns
    -------
    dict
        Updated dictionary representing the database with the modified OC.
    """
    DB_ID = row["DB"] + ";" + new_DB
    DB_i = row["DB_i"] + ";" + str(i_new_cl)
    # Order by years before storing
    DB_ID, DB_i = date_order_DBs(DB_ID, DB_i)

    # Attach name(s) and fname(s) present in new DB to the UCC, removing duplicates
    ID = row["ID"] + ";" + oc_names
    ID = rm_name_dups(ID)
    # The first fname is the most important one as all files for this OC use this
    # naming. The 'rm_name_dups' function will always keep this name first in line
    fnames = row["fnames"] + ";" + ";".join(fnames_new_cl)
    fnames = rm_name_dups(fnames)

    # Galactic coordinates
    lon_n, lat_n = radec2lonlat(row["RA_ICRS"], row["DE_ICRS"])

    new_vals = {
        "DB": DB_ID,
        "DB_i": DB_i,
        "ID": ID,
        "RA_ICRS": round(row["RA_ICRS"], 4),
        "DE_ICRS": round(row["DE_ICRS"], 4),
        "GLON": round(lon_n, 4),
        "GLAT": round(lat_n, 4),
        "Plx": round(row["Plx"], 4),
        "pmRA": round(row["pmRA"], 4),
        "pmDE": round(row["pmDE"], 4),
        "fnames": fnames,
        # Copy the remaining values from the UCC row
    }
    new_db_dict = updt_new_db_dict(new_db_dict, new_vals, row)

    return new_db_dict


def rm_name_dups(names: str) -> str:
    """
    Removes duplicate names from a semicolon-separated string, considering variations
    with or without spaces and underscores, and retains the first occurrence.

    Removes duplicates of the kind:

        Berkeley 102, Berkeley102, Berkeley_102

    keeping only the name with the space.

    Parameters
    ----------
    names : str
        A semicolon-separated string of names.

    Returns
    -------
    str
        A semicolon-separated string of unique names, with duplicates removed.
    """
    names_l = names.split(";")
    for n in names_l:
        n2 = n.replace(" ", "")
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n
        n2 = n.replace(" ", "_")
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n

    names = ";".join(list(dict.fromkeys(names_l)))

    return names


def updt_new_db_dict(
    new_db_dict: dict, new_vals: dict, row: dict | None = None
) -> dict:
    """
    Updates the new database dictionary with new values and, optionally, existing
    values from a row.

    Parameters
    ----------
    new_db_dict : dict
        Dictionary representing the updated database.
    new_vals : dict
        Dictionary of new values to add to the database.
    row : dict, optional
        Optional; Dictionary representing an existing row in the UCC to copy
        values from.

    Returns
    -------
    dict
        Updated dictionary representing the database with new and existing values.
    """
    for col in (
        "DB",
        "DB_i",
        "ID",
        "RA_ICRS",
        "DE_ICRS",
        "GLON",
        "GLAT",
        "Plx",
        "pmRA",
        "pmDE",
        "fnames",
    ):
        new_db_dict[col].append(new_vals[col])

    for col in (
        "UCC_ID",
        "quad",
        "r_50",
        "N_50",
        "C1",
        "C2",
        "C3",
        "GLON_m",
        "GLAT_m",
        "RA_ICRS_m",
        "DE_ICRS_m",
        "Plx_m",
        "pmRA_m",
        "pmDE_m",
        "Rv_m",
        "N_Rv",
    ):
        if row is None:
            new_db_dict[col].append(np.nan)
        else:
            new_db_dict[col].append(row[col])

    return new_db_dict


def assign_UCC_ids(logging, glon: float, glat: float, ucc_ids_old: list[str]) -> str:
    """
    Assigns a new UCC ID based on galactic coordinates, avoiding duplicates.

    Format: UCC GXXX.X+YY.Y

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    glon : float
        Galactic longitude.
    glat : float
        Galactic latitude.
    ucc_ids_old : list
        List of existing UCC IDs.

    Returns
    -------
    str
        A new, unique UCC ID.
    """

    def trunc(values, decs=1):
        return np.trunc(values * 10**decs) / (10**decs)

    ll = trunc(np.array([glon, glat]).T)
    lon, lat = str(ll[0]), str(ll[1])

    if ll[0] < 10:
        lon = "00" + lon
    elif ll[0] < 100:
        lon = "0" + lon

    if ll[1] >= 10:
        lat = "+" + lat
    elif ll[1] < 10 and ll[1] > 0:
        lat = "+0" + lat
    elif ll[1] == 0:
        lat = "+0" + lat.replace("-", "")
    elif ll[1] < 0 and ll[1] >= -10:
        lat = "-0" + lat[1:]
    elif ll[1] < -10:
        pass

    ucc_id = "UCC G" + lon + lat

    i = 0
    while True:
        if i > 25:
            ucc_id += "ERROR"
            logging.info(f"ERROR NAMING: {glon}, {glat} --> {ucc_id}")
            break
        if ucc_id in ucc_ids_old:
            if i == 0:
                # Add a letter to the end
                ucc_id += ascii_lowercase[i]
            else:
                # Replace last letter
                ucc_id = ucc_id[:-1] + ascii_lowercase[i]
            i += 1
        else:
            break

    return ucc_id


def QXY_fold(UCC_ID: str) -> str:
    """
    Determines the quadrant and north/south designation of a cluster based on its UCC ID.

    Parameters
    ----------
    UCC_ID : str
        The UCC ID of the cluster.

    Returns
    -------
    str
        A string representing the quadrant and north/south designation (e.g., "Q1P",
        "Q3N").
    """
    # UCC_ID = cl['UCC_ID']
    lonlat = UCC_ID.split("G")[1]
    lon = float(lonlat[:5])
    try:
        lat = float(lonlat[5:])
    except ValueError:
        lat = float(lonlat[5:-1])

    Qfold = "Q"
    if lon >= 0 and lon < 90:
        Qfold += "1"
    elif lon >= 90 and lon < 180:
        Qfold += "2"
    elif lon >= 180 and lon < 270:
        Qfold += "3"
    elif lon >= 270 and lon < 3600:
        Qfold += "4"
    if lat >= 0:
        Qfold += "P"
    else:
        Qfold += "N"

    return Qfold


def duplicates_check(logging, df_all: pd.DataFrame) -> bool:
    """
    Checks for duplicate entries in specified columns of a DataFrame.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_all : pd.DataFrame
        DataFrame to check for duplicates.

    Returns
    -------
    bool
        True if duplicates are found in any of the specified columns, False otherwise.
    """

    def list_duplicates(seq: list) -> list:
        """
        Identifies duplicate elements in a list.

        Args:
            seq: The input list.

        Returns:
            A list of duplicate elements.
        """
        seen = set()
        seen_add = seen.add
        # adds all elements it doesn't know yet to seen and all other to seen_twice
        seen_twice = set(x for x in seq if x in seen or seen_add(x))
        # turn the set into a list (as requested)
        return list(seen_twice)

    def dup_check(df_all: pd.DataFrame, col: str) -> bool:
        """
        Checks for duplicates in a specific column of a DataFrame.

        Args:
            df_all: DataFrame to check.
            col: Name of the column to check.

        Returns:
            True if duplicates are found, False otherwise.
        """
        dups = list_duplicates(list(df_all[col]))
        if len(dups) > 0:
            logging.info(f"\nWARNING! N={len(dups)} duplicates found in '{col}':")
            for dup in dups:
                print(dup)
            logging.info("UCC was not updated")
            return True
        else:
            return False

    for col in ("ID", "UCC_ID", "fnames"):
        dup_flag = dup_check(df_all, col)
        if dup_flag:
            return True

    return False
