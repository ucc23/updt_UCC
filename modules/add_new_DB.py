from string import ascii_lowercase

import numpy as np
import pandas as pd

from . import aux
from .standardize_and_match import rename_standard


def run(logging, pars_dict, df_UCC, df_new, json_pars, new_DB_fnames, db_matches):
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

    logging.info(f"Adding new DB: {pars_dict['new_DB']}")

    logging.info("")
    new_db_dict = combine_UCC_new_DB(
        logging,
        pars_dict["new_DB"],
        df_UCC,
        df_new,
        json_pars,
        new_DB_fnames,
        db_matches,
    )
    N_new = len(df_new) - sum(_ is not None for _ in db_matches)
    logging.info(f"\nN={N_new} new clusters in {pars_dict['new_DB']}")
    logging.info("")

    # Add UCC_IDs and quadrants for new clusters
    ucc_ids_old = list(df_UCC["UCC_ID"].values)
    for i, UCC_ID in enumerate(new_db_dict["UCC_ID"]):
        # Only process new OCs
        if str(UCC_ID) != "nan":
            continue
        new_db_dict["UCC_ID"][i] = assign_UCC_ids(
            new_db_dict["GLON"][i], new_db_dict["GLAT"][i], ucc_ids_old
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


def combine_UCC_new_DB(
    logging,
    new_DB: str,
    df_UCC: pd.DataFrame,
    df_new: pd.DataFrame,
    json_pars: dict,
    new_DB_fnames: list,
    db_matches: list,
    sep=",",
) -> dict:
    """
    Args:
        new_DB (str): Name of the new DB
        new_DB_fnames (list): List of lists, each one containing the fnames of the
        entry in the new DB
        db_matches (list): List of indexes into the UCC pointing to each entry in the
        new DB
    """
    new_db_dict = {_: [] for _ in df_UCC.keys()}
    # For each entry in the new DB
    for i_new_cl, new_cl in enumerate(new_DB_fnames):
        row_n = dict(df_new.iloc[i_new_cl])

        # For each comma separated name for this OC in the new DB
        oc_names = row_n[json_pars["names"]].split(sep)
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
                new_DB, new_db_dict, i_new_cl, new_cl, row_n, oc_names, json_pars
            )
            logging.info(f"{i_new_cl} {','.join(new_cl)} is a new OC")
        else:
            # The cluster is already present in the UCC
            # Row in UCC where this match is located
            row = dict(df_UCC.iloc[db_matches[i_new_cl]])
            # Add to the new Db dictionary
            new_db_dict = OC_in_UCC(
                new_DB, new_db_dict, i_new_cl, new_cl, oc_names, row
            )
            logging.info(
                f"{i_new_cl} {','.join(new_cl)} is in the UCC: update DB indexes"
            )

    return new_db_dict


def new_OC_not_in_UCC(
    new_DB: str,
    new_db_dict: dict,
    i_new_cl: int,
    fnames_new_cl: list,
    row_n: dict,
    oc_names: str,
    json_pars: dict,
):
    """
    Args:
        new_DB (str): Name of the new DB
        new_db_dict (dict): Dictionary with the same keys as the UCC, with the OCs
            in the new DB
        i_new_cl (int): Index of this OC in the new DB
        fnames_new_cl (list): List of comma separated fnames for this OC
        row_n (dict): Row in the new DB that corresponds to this OC
        oc_names (str): Names associated to this OC in a ';' separated string
        json_pars (dict): Dictionary with the column names for this new DB
    """
    # Remove duplicates from names and fnames
    ID = oc_names
    if ";" in oc_names:
        ID = rm_name_dups(oc_names)
    fnames = ";".join(fnames_new_cl)
    if ";" in fnames:
        fnames = rm_name_dups(fnames)

    # Extract names of (ra, dec, plx, pmRA, pmDE) columns for this new DB
    ra_c, dec_c, plx_c, pmra_c, pmde_c, _ = json_pars["pos"].split(",")
    ra_n, dec_n = row_n[ra_c], row_n[dec_c]
    # Galactic coordinates
    lon_n, lat_n = aux.radec2lonlat(ra_n, dec_n)
    #
    plx_n = row_n[plx_c] if plx_c != "None" else np.nan
    pmra_n = row_n[pmra_c] if plx_c != "None" else np.nan
    pmde_n = row_n[pmde_c] if plx_c != "None" else np.nan

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
    fnames_new_cl: list,
    oc_names: str,
    row: dict,
):
    """
    Add an OC that is already present in the UCC

    Args:
        new_DB (str): Name of the new DB
        new_db_dict (dict): Dictionary with the same keys as the UCC, with the OCs
            in the new DB
        i_new_cl (int): Index of this OC in the new DB
        fnames_new_cl (list): List of comma separated fnames for this OC
        oc_names (str): Names associated to this OC in a ';' separated string
        row (dict): Row in the UCC where this match is located
    """

    # # Check if this new DB is already in the UCC list.
    # if new_DB in row["DB"]:
    #     # Remove new DB and its OC index from the UCC
    #     DB_ID_lst = row["DB"].split(";")
    #     DB_i_lst = row["DB_i"].split(";")
    #     # Index of new DB in the UCC
    #     i_rm = DB_ID_lst.index(new_DB)
    #     del DB_ID_lst[i_rm]
    #     del DB_i_lst[i_rm]
    #     # Re-add new DB and its OC index
    #     if len(DB_ID_lst) > 0:
    #         DB_ID = ";".join(DB_ID_lst) + ";" + new_DB
    #         DB_i = ";".join(DB_i_lst) + ";" + str(i_new_cl)
    #     else:
    #         DB_ID = new_DB
    #         DB_i = str(i_new_cl)
    # else:
    DB_ID = row["DB"] + ";" + new_DB
    DB_i = row["DB_i"] + ";" + str(i_new_cl)
    # Order by years before storing
    DB_ID, DB_i = aux.date_order_DBs(DB_ID, DB_i)

    # Attach name(s) and fname(s) present in new DB to the UCC, removing duplicates
    ID = row["ID"] + ";" + oc_names
    ID = rm_name_dups(ID)
    # The first fname is the most important one as all files for this OC use this
    # naming. The 'rm_name_dups' function will always keep this name first in line
    fnames = row["fnames"] + ";" + ";".join(fnames_new_cl)
    fnames = rm_name_dups(fnames)

    # Galactic coordinates
    lon_n, lat_n = aux.radec2lonlat(row["RA_ICRS"], row["DE_ICRS"])

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


def rm_name_dups(names):
    """
    Removes duplicate names from a semicolon-separated string, considering variations
    with or without spaces and underscores, and retains the first occurrence.

    Removes duplicates of the kind:

        Berkeley 102, Berkeley102, Berkeley_102

    keeping only the name with the space.


    Args:
        names (str): A semicolon-separated string of names.

    Returns:
        str: A semicolon-separated string of unique names, with duplicates removed.
    """
    names_l = names.split(";")
    for i, n in enumerate(names_l):
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


def updt_new_db_dict(new_db_dict: dict, new_vals: dict, row: None | dict = None):
    """ """
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
        "dups_fnames",
        "dups_probs",
        "r_50",
        "N_50",
        "N_fixed",
        "fixed_cent",
        "cent_flags",
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
        "dups_fnames_m",
        "dups_probs_m",
    ):
        if row is None:
            new_db_dict[col].append(np.nan)
        else:
            new_db_dict[col].append(row[col])

    return new_db_dict


def assign_UCC_ids(glon, glat, ucc_ids_old):
    """
    Format: UCC GXXX.X+YY.Y
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
            print("ERROR NAMING")
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


def QXY_fold(UCC_ID):
    """ """
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


def duplicates_check(logging, df_all):
    """ """

    def list_duplicates(seq):
        seen = set()
        seen_add = seen.add
        # adds all elements it doesn't know yet to seen and all other to seen_twice
        seen_twice = set(x for x in seq if x in seen or seen_add(x))
        # turn the set into a list (as requested)
        return list(seen_twice)

    def dup_check(df_all, col):
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
