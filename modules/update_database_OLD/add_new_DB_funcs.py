import numpy as np
import pandas as pd

from ..HARDCODED import naming_order
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
) -> pd.DataFrame:
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
    pd.DataFrame
        Dataframe representing the updated database with new and modified entries.
    """
    N_new, N_updt = 0, 0
    new_db_dict = {_: [] for _ in df_UCC.keys()}
    # For each entry in the new DB
    for i_new_cl, fnames_new_cl in enumerate(new_DB_fnames):
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
            # The cluster is not present in the UCC
            new_db_dict = new_OC_not_in_UCC(
                new_DB,
                new_db_dict,
                i_new_cl,
                fnames_new_cl,
                oc_names,
                row_n,
                newDB_json,
            )
            N_new += 1
        else:
            # The cluster is already present in the UCC
            # Row in UCC where this match is located
            row = dict(df_UCC.iloc[db_matches[i_new_cl]])
            new_db_dict = OC_in_UCC(
                new_DB, new_db_dict, i_new_cl, fnames_new_cl, oc_names, row
            )
            N_updt += 1

    # Replace lists of values with nanmedians
    df_new_db = pd.DataFrame(new_db_dict)
    for col in ["RA_ICRS", "DE_ICRS", "GLON", "GLAT", "Plx", "pmRA", "pmDE"]:
        df_new_db[col] = df_new_db[col].apply(
            lambda x: np.nanmedian(x) if np.any(~np.isnan(x)) else np.nan
        )

    logging.info(f"\nNew entries: {N_new}, Updated entries: {N_updt}")

    return df_new_db


def new_OC_not_in_UCC(
    new_DB: str,
    new_db_dict: dict,
    i_new_cl: int,
    fnames_new_cl: list[str],
    oc_names: str,
    row_n: dict,
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
    # # Remove duplicates from names and fnames
    # names = oc_names
    # if ";" in oc_names:
    #     names = rm_name_dups(oc_names)
    fnames = ";".join(fnames_new_cl)
    # if ";" in fnames:
    #     fnames = rm_name_dups(fnames)

    ra_n, dec_n, lon_n, lat_n = [np.nan] * 4
    # Don't append KHARCHENKO2012 for coordinates
    if new_DB != "KHARCHENKO2012":
        ra_n, dec_n = row_n[newDB_json["pos"]["RA"]], row_n[newDB_json["pos"]["DEC"]]
        # Galactic coordinates
        lon_n, lat_n = radec2lonlat(ra_n, dec_n)

    #
    plx_n, pmra_n, pmde_n = [np.nan] * 3
    # Don't append LOKTIN2017 Pms, Plx values
    if new_DB != "LOKTIN2017":
        if "plx" in newDB_json["pos"]:
            plx_n = row_n[newDB_json["pos"]["plx"]]
        if "pmra" in newDB_json["pos"]:
            pmra_n = row_n[newDB_json["pos"]["pmra"]]
        if "pmde" in newDB_json["pos"]:
            pmde_n = row_n[newDB_json["pos"]["pmde"]]

    new_vals = {
        "DB": new_DB,
        "DB_i": str(i_new_cl),
        "Names": oc_names,
        "fnames": fnames,
        "RA_ICRS": round(ra_n, 4),
        "DE_ICRS": round(dec_n, 4),
        "GLON": round(float(lon_n), 4),
        "GLAT": round(float(lat_n), 4),
        "Plx": round(plx_n, 4),
        "pmRA": round(pmra_n, 4),
        "pmDE": round(pmde_n, 4),
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
    run_mode : str
        Mode of the run
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
    # DB_ID = row["DB"]
    # DB_i = row["DB_i"]

    DB_ID = row["DB"] + ";" + new_DB
    DB_i = row["DB_i"] + ";" + str(i_new_cl)

    # Order by years before storing
    DB_ID, DB_i = date_order_DBs(DB_ID, DB_i)

    # Attach name(s) and fname(s) present in new DB to the UCC, removing duplicates
    names = row["Names"] + ";" + oc_names
    # ID = rm_name_dups(ID)
    # The first fname is the most important one as all files for this OC use this
    # naming. The 'rm_name_dups' function will always keep this name first in line
    fnames = row["fnames"] + ";" + ";".join(fnames_new_cl)
    # fnames = rm_name_dups(fnames)

    # Galactic coordinates
    lon_n, lat_n = radec2lonlat(row["RA_ICRS"], row["DE_ICRS"])

    new_vals = {
        "DB": DB_ID,
        "DB_i": DB_i,
        "Names": names,
        "fnames": fnames,
        "RA_ICRS": round(row["RA_ICRS"], 4),
        "DE_ICRS": round(row["DE_ICRS"], 4),
        "GLON": round(float(lon_n), 4),
        "GLAT": round(float(lat_n), 4),
        "Plx": round(row["Plx"], 4),
        "pmRA": round(row["pmRA"], 4),
        "pmDE": round(row["pmDE"], 4),
    }
    new_db_dict = updt_new_db_dict(new_db_dict, new_vals)

    return new_db_dict


def rm_name_dups_order(df_new_db: pd.DataFrame) -> pd.DataFrame:
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

    def rm_dups(strings: list[str]) -> tuple[list[str], list[int]]:
        """
        Remove duplicates from a list of strings and return unique values with their
        original indexes.

        Parameters
        ----------
        strings : list[str]
            List of strings, possibly containing duplicates.

        Returns
        -------
        unique_strings : list[str]
            List of unique strings, preserving the first occurrence order.
        indexes : list[int]
            Indexes of the unique strings in the original list.
        """
        seen = {}
        unique_strings = []
        rm_idx = []
        for i, s in enumerate(strings):
            if s not in seen:
                seen[s] = True
                unique_strings.append(s)
            else:
                rm_idx.append(i)

        return unique_strings, rm_idx

    def order_by_name(items: list[str]) -> list[int]:
        """
        Return indexes that reorder a list, moving to the front elements that start
        with any of the given prefixes. The prefixes earlier in the tuple have higher
        priority.

        Parameters
        ----------
        items : list of str
            List of strings to reorder.

        Returns
        -------
        List[int]
            Indexes that reorder the list accordingly.
        """

        def priority(item: str) -> tuple[int, int]:
            for j, p in enumerate(naming_order):
                if item.startswith(p):
                    return (0, j)  # found: higher priority, sorted by prefix order
            return (1, -1)  # not found: lowest priority

        indexed_items = list(enumerate(items))
        indexed_items.sort(key=lambda x: priority(x[1]))
        return [i for i, _ in indexed_items]

    #
    for i, fnames in enumerate(df_new_db["fnames"]):
        unq_fnames, rm_idx = rm_dups(fnames.split(";"))
        unq_names = str(df_new_db["Names"][i]).split(";")
        if rm_idx:
            unq_names = [_ for i, _ in enumerate(unq_names) if i not in rm_idx]

        # Name ordering
        idx_order = order_by_name(unq_fnames)
        df_new_db["Names"].iloc[i] = ";".join([unq_names[i] for i in idx_order])
        df_new_db["fnames"].iloc[i] = ";".join([unq_fnames[i] for i in idx_order])

    return df_new_db


def updt_new_db_dict(new_db_dict: dict, new_vals: dict) -> dict:
    """
    Updates the new database dictionary with new values and, optionally, existing
    values from a row.

    Parameters
    ----------
    new_db_dict : dict
        Dictionary representing the updated database.
    new_vals : dict
        Dictionary of new values to add to the database.

    Returns
    -------
    dict
        Updated dictionary representing the database with new and existing values.
    """
    for col in (
        "DB",
        "DB_i",
        "Names",
        "fnames",
        "RA_ICRS",
        "DE_ICRS",
        "GLON",
        "GLAT",
        "Plx",
        "pmRA",
        "pmDE",
    ):
        new_db_dict[col].append(new_vals[col])

    return new_db_dict
