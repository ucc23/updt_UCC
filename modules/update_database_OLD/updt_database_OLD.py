import json
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

import Levenshtein
import numpy as np
import pandas as pd
from astropy.coordinates import angular_separation
from scipy.spatial.distance import cdist

from .HARDCODED import (
    GCs_cat,
    UCC_folder,
    dbs_folder,
    name_DBs_json,
    naming_order,
    temp_fold,
)
from .utils import (
    check_centers,
    date_order_DBs,
    get_last_version_UCC,
    radec2lonlat,
    rename_standard,
    save_df_UCC,
)


def get_paths_check_paths(
    logging,
) -> tuple[str, str, str]:
    """ """
    # If file exists, read and return it
    if os.path.isfile(temp_fold + "df_UCC_B_updt.csv"):
        logging.warning(
            "WARNING: file 'df_UCC_B_updt.csv' exists. Moving on will re-write it"
        )
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Temporary databases/ folder
    # temp_database_folder = ""
    # if run_mode == "new_DB":
    temp_database_folder = temp_fold + dbs_folder
    # Create if required
    if not os.path.exists(temp_database_folder):
        os.makedirs(temp_database_folder)

    last_version = get_last_version_UCC(UCC_folder)
    # Path to the current UCC csv file
    ucc_file = UCC_folder + last_version

    # Path to the new (temp) JSON file
    temp_JSON_file = temp_fold + name_DBs_json

    return (
        temp_database_folder,
        ucc_file,
        temp_JSON_file,
    )


def load_data(
    logging,
    ucc_file: str,
    temp_JSON_file: str,
    temp_database_folder: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
    pd.DataFrame,
    bool,
]:
    """ """
    # Load current UCC version
    df_UCC = pd.read_csv(ucc_file)
    logging.info(f"\nUCC version {ucc_file} loaded (N={len(df_UCC)})")

    # Load current JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    all_dbs_data = {}
    if os.path.isfile(temp_JSON_file):
        logging.info("\n=== Adding new DBs to the UCC ===\n")
        flag_interactive = True

        # Load new temp JSON file
        with open(temp_JSON_file) as f:
            temp_JSON = json.load(f)

        # Extract new DB's name(s)
        new_DBs = list(set(temp_JSON.keys()) - set(current_JSON.keys()))

        for new_DB in new_DBs:
            # Sanity check
            if new_DB in current_JSON.keys():
                raise ValueError(
                    f"The DB '{new_DB}' is already in the current JSON file"
                )

            # Load the DB
            df_new = pd.read_csv(temp_database_folder + new_DB + ".csv")
            # Load column data for the new catalogue
            newDB_json = temp_JSON[new_DB]
            # all_dbs_data.append([new_DB, df_new, newDB_json])
            all_dbs_data[new_DB] = [df_new, newDB_json]
            logging.info(f"New DB {new_DB} loaded (N={len(df_new)})")

        df_UCC_new = df_UCC.copy()

    else:
        logging.info("\n=== Rebuilding the UCC completely===\n")
        flag_interactive = False

        # All DBs (assumes sorted by year)
        all_dbs = list(current_JSON.keys())
        # BICA2019 should be the first entry
        all_dbs.remove("BICA2019")
        all_dbs = ["BICA2019"] + all_dbs

        # Load all DBs
        for new_DB in all_dbs:
            df_new = pd.read_csv(dbs_folder + new_DB + ".csv")
            logging.info(f"{new_DB} loaded (N={len(df_new)})")
            newDB_json = current_JSON[new_DB]
            all_dbs_data[new_DB] = [df_new, newDB_json]

        # Empty dataframe
        df_UCC_new = pd.DataFrame(
            {
                "DB": pd.Series(dtype="string"),
                "DB_i": pd.Series(dtype="int64"),
                "Names": pd.Series(dtype="string"),
                "fnames": pd.Series(dtype="string"),
                "RA_ICRS": pd.Series(dtype="float64"),
                "DE_ICRS": pd.Series(dtype="float64"),
                "GLON": pd.Series(dtype="float64"),
                "GLAT": pd.Series(dtype="float64"),
                "Plx": pd.Series(dtype="float64"),
                "pmRA": pd.Series(dtype="float64"),
                "pmDE": pd.Series(dtype="float64"),
            }
        )

    return (df_UCC, df_GCs, all_dbs_data, df_UCC_new, flag_interactive)


def check_new_DB(
    logging,
    df_GCs: pd.DataFrame,
    new_DB: str,
    df_new: pd.DataFrame,
    newDB_json: dict,
    flag_interactive: bool,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Check new DB for required columns, bad characters in names, wrong VDBH/BH naming,
    and close GCs.
    """
    # Check for required columns
    cols = [newDB_json["names"]]
    for entry in ("pos", "pars", "e_pars"):
        for _, v in newDB_json[entry].items():
            if isinstance(v, list):
                cols += v
            else:
                cols.append(v)
    missing_cols = [col for col in cols if col not in df_new.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns in {new_DB}: {missing_cols}")

    # Check for bad characters in name column
    all_bad_names = []
    for new_cl in df_new[newDB_json["names"]]:
        if bool(re.search(r"[();*]", new_cl)):
            all_bad_names.append(new_cl)
    N_bad_names = len(all_bad_names)
    if N_bad_names > 0:
        logging.info(
            f"\nFound {N_bad_names} entries with bad characters in names '(, ), ;, *'"
        )
        if flag_interactive:
            for new_cl in all_bad_names:
                logging.info(f"  {new_cl}")  # bad char found
        raise ValueError("Resolve the above issues before moving on.")

    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    # logging.info("\nPossible vdBergh-Hagen/vdBergh check")
    if vdberg_check(logging, newDB_json, df_new):
        if flag_interactive:
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit()

    lon_new, lat_new = None, None
    if "RA" in newDB_json["pos"].keys():
        ra_new = df_new[newDB_json["pos"]["RA"]].to_numpy()
        dec_new = df_new[newDB_json["pos"]["DEC"]].to_numpy()
        lon_new, lat_new = radec2lonlat(ra_new, dec_new)

        # Check for OCs very close to each other in the new DB
        # if ra_new is not None:
        # logging.info("\nProbable inner duplicates check")
        if close_OC_inner_check(logging, newDB_json, df_new, ra_new, dec_new):
            if flag_interactive:
                if input("Move on? (y/n): ").lower() != "y":
                    sys.exit()

        # Check for close GCs
        # if lon_new is not None:
        # logging.info("\nClose GC check")
        if GCs_check(logging, df_GCs, newDB_json, df_new, lon_new, lat_new):
            if flag_interactive:
                if input("Move on? (y/n): ").lower() != "y":
                    sys.exit()

    return lon_new, lat_new


def vdberg_check(logging, newDB_json: dict, df_new: pd.DataFrame) -> bool:
    """
    Check for instances of 'vdBergh-Hagen' and 'vdBergh'

    Per CDS recommendation:

    * BH, VDBergh-Hagen --> VDBH
    * VDBergh           --> VDB

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_new : pd.DataFrame
        DataFrame of the new database.

    Returns
    -------
    bool
        Boolean flag indicating if probable vdBergh-Hagen/vdBergh OCs were found.
    """
    names_lst = [
        "vdBergh-Hagen",
        "vdBergh",
        "van den Bergh–Hagen",
        "van den Bergh",
        "BH",
    ]
    names_lst = [_.lower().replace("-", "").replace(" ", "") for _ in names_lst]
    proper_names = ["VDBH", "VDB"]

    vds_found = []
    for i, new_cl in enumerate(df_new[newDB_json["names"]]):
        new_cl_r = re.sub(r"\d", "", new_cl)  # Remove all numbers

        # If the root of the name is a proper name, skip check
        if new_cl_r.split("_")[0].split(" ")[0] in proper_names:
            continue

        new_cl_r = (
            new_cl_r.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
        )
        for name_check in names_lst:
            sm_ratio = SequenceMatcher(None, new_cl_r, name_check).ratio()
            if sm_ratio > 0.5:
                vds_found.append([i, new_cl, name_check, round(sm_ratio, 2)])
                break

    if len(vds_found) == 0:
        vdb_flag = False
        # logging.info("No vdBergh-Hagen/vdBergh found that need renaming")
    else:
        vdb_flag = True
        logging.info(f"\nFound {len(vds_found)} entries that could need name editing")
        logging.info("* BH ; vdBergh-Hagen --> VDBH")
        logging.info("* vdBergh            --> VDB")
        for i, new_cl, name_check, sm_ratio in vds_found:
            logging.info(f"{i}, {new_cl} --> {name_check} (P={sm_ratio})")

    return vdb_flag


def close_OC_inner_check(
    logging,
    newDB_json: dict,
    df_new: pd.DataFrame,
    ra,
    dec,
    rad_dup: float = 30.0,
    leven_rad: float = 0.85,
    sep: str = ",",
) -> bool:
    """
    Looks for OCs in the new DB that are close to other OCs in the new DB (RA, DEC)
    whose names are somewhat similar (Levenshtein distance).

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_new : pd.DataFrame
        DataFrame of the new database.
    rad_dup : float
        Search radius in arcmin.
    leven_rad : float, optional
        Levenshtein ratio threshold. Default is 0.85.

    Returns
    -------
    bool
        Boolean flag indicating if probable inner duplicates were found.
    """
    coords = np.array([ra, dec]).T
    # Find the distances to all clusters, for all clusters (in arcmin)
    dist = cdist(coords, coords) * 60
    # Change distance to itself from 0 to inf
    msk = dist == 0.0
    dist[msk] = np.inf

    col_1 = df_new[newDB_json["names"]]
    col_2 = None
    db_matches = None
    ID_call = "inner"

    return close_OC_check(
        logging, dist, db_matches, col_1, col_2, ID_call, rad_dup, leven_rad, sep
    )


def close_OC_UCC_check(
    logging,
    df_UCC: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
    glon,
    glat,
    rad_dup: float = 30.0,
    leven_rad: float = 0.5,
    sep: str = ";",
) -> bool:
    """
    Looks for OCs in the new DB that are close to OCs in the UCC (GLON, GLAT) but
    with different names.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_DB_fnames : list
        List of lists, where each inner list contains the
        standardized names for each cluster in the new catalogue.
    db_matches : list
        List of indexes into the UCC pointing to each entry in the
        new DB.
    glon : np.ndarray
        Array of galactic longitudes of the clusters in the new database.
    glat : np.ndarray
        Array of galactic latitudes of the clusters in the new database.
    rad_dup : float
        Search radius in arcmin.

    Returns
    -------
    bool
        Boolean flag indicating if probable UCC duplicates were found.
    """
    coords_new = np.array([glon, glat]).T
    coords_UCC = np.array([df_UCC["GLON"], df_UCC["GLAT"]]).T
    # Find the distances to all clusters, for all clusters (in arcmin)
    dist = cdist(coords_new, coords_UCC) * 60

    col_1 = df_UCC["fnames"]
    col_2 = new_DB_fnames
    ID_call = "UCC"

    return close_OC_check(
        logging, dist, db_matches, col_1, col_2, ID_call, rad_dup, leven_rad, sep
    )


def close_OC_check(
    logging,
    dist,
    db_matches,
    col_1,
    col_2,
    ID_call: str,
    rad_dup: float,
    leven_rad: float,
    sep: str,
):
    """ """
    idxs = np.arange(0, len(col_1))
    all_dups, dups_list = [], []
    for i, cl_d in enumerate(dist):
        # If no OC is within rad_dup for this OC, continue with next
        msk = cl_d < rad_dup
        if msk.sum() == 0:
            continue

        # Already in dups_list
        if ID_call == "inner":
            cl_name = str(col_1[i])
        else:
            cl_name = sep.join(col_2[i])
        if cl_name in dups_list:
            continue

        if ID_call == "UCC":
            # If cl_name is present in the UCC (df_UCC['fnames'][db_matches[i]])
            if db_matches[i] is not None:
                continue

        # For each OC within the rad_dup region
        N_inner_dups, dups, dist, L_ratios = 0, [], [], []
        for j in idxs[msk]:
            dup_names = str(col_1[j]).split(sep)

            L_ratio = 0
            for dup_name in dup_names:
                L_ratio = max(L_ratio, Levenshtein.ratio(cl_name, dup_name.strip()))

            if L_ratio > leven_rad:
                N_inner_dups += 1
                dups_list.append(sep.join(dup_names))
                dups.append(sep.join(dup_names))
                dist.append(str(round(cl_d[j], 1)))
                L_ratios.append(str(round(L_ratio, 2)))
        if dups:
            all_dups.append([i, cl_name, N_inner_dups, dups, dist, L_ratios])

    dups_flag = False
    dups_found = len(all_dups)
    if dups_found > 0:
        dups_flag = True

        # Extract indexes that sort by distance
        all_dists = []
        for dup in all_dups:
            all_dists.append(min([float(_) for _ in dup[-2]]))
        i_sort = np.argsort(all_dists)

        logging.info(f"\nFound {dups_found} probable {ID_call} duplicates")
        for idx in i_sort:
            i, cl_name, N_inner_dups, dups, dist, L_ratios = all_dups[idx]
            logging.info(
                f"{i:<6} {cl_name:<15} (N={N_inner_dups}) --> "
                + f"{';'.join(dups):<15} | d={';'.join(dist)}, L={';'.join(L_ratios)}"
            )
    # else:
    #     logging.info(f"No {ID_call} duplicates found")

    return dups_flag


def GCs_check(
    logging,
    df_GCs: pd.DataFrame,
    newDB_json: dict,
    df_new: pd.DataFrame,
    glon,
    glat,
    search_rad: float = 15,
) -> bool:
    """
    Check for nearby GCs for a new database

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_GCs : pd.DataFrame
        GCs database.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_new : pd.DataFrame
        DataFrame of the new database.
    search_rad : float, optional
        Search radius in arcmin. Default is 15.

    Returns
    -------
    bool
        Flag indicating if probable GCs were found.
    """
    glon_glat = np.array([glon, glat]).T

    # Read GCs DB
    l_gc, b_gc = df_GCs["GLON"].values, df_GCs["GLAT"].values  # pyright: ignore

    gc_all, GCs_found = [], 0
    for idx, (glon_i, glat_i) in enumerate(glon_glat):
        d_arcmin = np.rad2deg(angular_separation(glon_i, glat_i, l_gc, b_gc)) * 60
        j1 = np.argmin(d_arcmin)

        if d_arcmin[j1] < search_rad:
            GCs_found += 1
            gc_all.append(
                [
                    idx,
                    df_new.iloc[idx][newDB_json["names"]],
                    j1,
                    df_GCs["Name"][j1],
                    d_arcmin[j1],
                ]
            )

    gc_flag = False
    if GCs_found > 0:
        gc_all = np.array(gc_all).T
        i_sort = np.argsort(np.array(gc_all[-1], dtype=float))
        gc_all = gc_all[:, i_sort].T

        gc_flag = True
        logging.info(f"\nFound {GCs_found} probable GCs:")
        for gc in gc_all:
            idx, row_id, idx_gc, df_gcs_name, d_arcmin = gc
            row_id = row_id.strip()
            logging.info(
                f"{idx:<6} {row_id:<15} --> {idx_gc:<6} {df_gcs_name.strip():<15}"
                + f"d={round(float(d_arcmin), 2)}"
            )
    # else:
    #     logging.info("No probable GCs found")

    return gc_flag


def get_fnames_new_DB(
    logging, df_new: pd.DataFrame, newDB_json: dict, sep: str = ","
) -> list[list[str]]:
    """
    Extract and standardize all names in the new catalogue

    Parameters
    ----------
    df_new : pd.DataFrame
        DataFrame of the new catalogue.
    newDB_json : dict
        Dictionary with the parameters of the new catalogue.
    sep : str, optional
        Separator used to split the names in the new catalogue. Default is ",".

    Returns
    -------
    list
        List of lists, where each inner list contains the standardized names for
        each cluster in the new catalogue.
    """
    names_all = df_new[newDB_json["names"]]
    new_DB_fnames = []
    for names in names_all:
        names_l = []
        names_s = str(names).split(sep)
        for name in names_s:
            name = name.strip()
            name = rename_standard(name)
            names_l.append(rm_chars_from_name(name))
        new_DB_fnames.append(names_l)

    # Check that no fname is repeated within entries
    seen, duplicates = {}, {}
    for i, sublist in enumerate(new_DB_fnames):
        # use set(sublist) so duplicates inside the same sublist are ignored
        for item in set(sublist):
            if item in seen and seen[item] != i:
                duplicates.setdefault(item, {seen[item]}).add(i)
            else:
                seen[item] = i
    if duplicates:
        for name, idxs in duplicates.items():
            logging.info(f"Duplicate '{name}' found in entries: {sorted(idxs)}")
        raise ValueError("\nDuplicate fnames found in new DB entries")

    return new_DB_fnames


def rm_chars_from_name(name: str) -> str:
    """
    Removes special characters from a cluster name and converts it to lowercase.

    Parameters
    ----------
    name : str
        The cluster name.

    Returns
    -------
    str
        The cluster name with special characters removed and converted to lowercase.
    """
    # We replace '+' with 'p' to avoid duplicating names for clusters
    # like 'Juchert J0644.8-0925' and 'Juchert_J0644.8+0925'
    name = (
        name.lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("'", "")
        .replace("+", "p")
    )
    return name


def get_matches_new_DB(
    df_UCC: pd.DataFrame, new_DB_fnames: list[list[str]]
) -> list[int | None]:
    """
    Get cluster matches for the new DB being added to the UCC

    Parameters
    ----------
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_DB_fnames : list
        List of lists, where each inner list contains the
        standardized names for each cluster in the new catalogue.

    Returns
    -------
    list
        List of indexes into the UCC pointing to each entry in the new DB.
        If an entry in the new DB is not present in the UCC, the corresponding
        index in the list will be None.
    """
    db_matches = []
    # For each fname(s) for each entry in the new DB
    for new_cl_fnames in new_DB_fnames:
        found = None
        # For each fname in this new DB entry
        for new_cl_fname in new_cl_fnames:
            # For each fname(s) for each entry in the UCC
            for j, ucc_cl_fnames in enumerate(df_UCC["fnames"]):
                # Check if the fname in the new DB is included in the UCC
                if new_cl_fname in ucc_cl_fnames.split(";"):
                    # If it is, return the UCC index
                    found = j
                    break
            if found is not None:
                break
        db_matches.append(found)

    return db_matches


def check_new_DB_vs_UCC(
    logging,
    df_UCC,
    df_new,
    newDB_json,
    new_DB_fnames,
    db_matches,
    lon_new,
    lat_new,
    flag_interactive: bool,
) -> None:
    """
    - Checks for duplicate entries between the new database and the UCC.
    - Checks for OCs very close to each other between the new database and the UCC.
    - Checks positions and flags for attention if required.
    """
    # Check all fnames in the new DB against all fnames in the UCC
    # logging.info("\nChecking uniqueness of fnames")
    if fnames_check_UCC_new_DB(logging, df_UCC, new_DB_fnames):
        raise ValueError("\nResolve the above issues before moving on")

    # # Check the first fname for all entries in the new DB
    # logging.info("\nChecking for entries that must be combined")
    # if dups_fnames_inner_check(logging, new_DB, newDB_json, df_new, new_DB_fnames):
    #     raise ValueError("\nResolve the above issues before moving on")

    # # Check for duplicate entries in the new DB that also exist in the UCC
    # if dups_check_newDB_UCC(logging, new_DB, df_UCC, new_DB_fnames, db_matches):
    #     raise ValueError("\nResolve the above issues before moving on")

    if lon_new is not None:
        # Check for OCs very close to other OCs in the UCC
        # logging.info("\nProbable UCC duplicates check")
        if close_OC_UCC_check(
            logging, df_UCC, new_DB_fnames, db_matches, lon_new, lat_new
        ):
            if flag_interactive:
                if input("Move on? (y/n): ").lower() != "y":
                    sys.exit()

    # Check positions and flag for attention if required
    if positions_check(
        logging,
        df_UCC,
        newDB_json,
        df_new,
        new_DB_fnames,
        db_matches,
        flag_interactive,
    ):
        if flag_interactive:
            if input("\nMove on? (y/n): ").lower() != "y":
                sys.exit()


def positions_check(
    logging,
    df_UCC: pd.DataFrame,
    newDB_json,
    df_new,
    new_DB_fnames,
    db_matches,
    flag_interactive: bool,
    rad_dup: float = 30.0,
) -> bool:
    """
    Checks the positions of clusters in the new database against those in the UCC.

    The logic for flagging for attention is handled as follows:

    Is the OC already present in the UCC?
        |         |
        v         |--> No --> do nothing
       Yes
        |
        v
    Is the difference between the old vs new centers values large?
        |         |
        v         |--> No --> do nothing
       Yes
        |
        v
    Request attention

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_db_info : dict
        Dictionary with the information of the new database.
    rad_dup : float
        Search radius in arcmin.

    Returns
    -------
    bool
        Boolean flag indicating if OCs were flagged for attention.
    """
    # Prepare information from a new database matched with the UCC
    new_db_info = prep_newDB(newDB_json, df_new, new_DB_fnames, db_matches)

    ocs_attention = []
    for i, fnames in enumerate(new_db_info["fnames"]):
        j = new_db_info["UCC_idx"][i]
        # Check centers if the OC is already present in the UCC
        if j is not None:
            bad_center, d_arcmin, pmra_p, pmde_p, plx_p = check_centers(
                # (df_UCC["GLON_m"].iloc[j], df_UCC["GLAT_m"].iloc[j]),
                # (df_UCC["pmRA_m"].iloc[j], df_UCC["pmDE_m"].iloc[j]),
                # df_UCC["Plx_m"].iloc[j],
                (df_UCC["GLON"].iloc[j], df_UCC["GLAT"].iloc[j]),
                (df_UCC["pmRA"].iloc[j], df_UCC["pmDE"].iloc[j]),
                df_UCC["Plx"].iloc[j],
                (new_db_info["GLON"][i], new_db_info["GLAT"][i]),
                (new_db_info["pmRA"][i], new_db_info["pmDE"][i]),
                new_db_info["Plx"][i],
                rad_dup,
            )
            # Is the difference between the old vs new center values large?
            if bad_center == "nnn":
                continue
            # Store information on the OCs that require attention
            ocs_attention.append([fnames, bad_center, d_arcmin, pmra_p, pmde_p, plx_p])

    attention_flag = False
    if len(ocs_attention) > 0:
        attention_flag = True
        if flag_interactive:
            logging.info(f"\nFlagged {len(ocs_attention)} OCs for attention:")
            logging.info(
                "{:<25} {:<5} {}".format(
                    "name", "bad_cent", "[arcmin] [pmRA %] [pmDE %] [plx %]"
                )
            )
            for oc in ocs_attention:
                fnames, bad_center, d_arcmin, pmra_p, pmde_p, plx_p = oc
                flag_log(logging, bad_center, d_arcmin, pmra_p, pmde_p, plx_p, fnames)

    return attention_flag


def prep_newDB(
    newDB_json: dict,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
) -> dict:
    """
    Prepare information from a new database matched with the UCC

    Parameters
    ----------
    newDB_json : dict
        Dictionary containing parameters specifying column names for
        position (RA, Dec, etc.).
    df_new : pd.DataFrame
        DataFrame containing information about the new database.
    new_DB_fnames : list
        List of lists, where each inner list contains file names
        for each cluster in the new database.
    db_matches : list
        List of indices representing matches in the UCC, or None if
        no match exists.

    Returns
    -------
    dict
        A dictionary containing:
            - "fnames" (list): List of concatenated filenames for each cluster.
            - "RA_ICRS" (list): List of right ascension (RA) values.
            - "DE_ICRS" (list): List of declination (Dec) values.
            - "pmRA" (list): List of proper motion in RA.
            - "pmDE" (list): List of proper motion in Dec.
            - "Plx" (list): List of parallax values.
            - "UCC_idx" (list): List of indices of matched clusters in the UCC, or None
              for new clusters.
            - "GLON" (list): List of Galactic longitude values.
            - "GLAT" (list): List of Galactic latitude values.
    """
    new_db_info = {
        "fnames": [],
        "RA_ICRS": [],
        "DE_ICRS": [],
        "pmRA": [],
        "pmDE": [],
        "Plx": [],
        "UCC_idx": [],
    }

    for i, fnames_lst in enumerate(new_DB_fnames):
        # Use semi-colon here to math the UCC format
        fnames = ";".join(fnames_lst)
        row_n = df_new.iloc[i]

        # Coordinates for this cluster in the new DB
        ra_n, dec_n, plx_n, pmra_n, pmde_n = np.nan, np.nan, np.nan, np.nan, np.nan
        if "RA" in newDB_json["pos"]:
            ra_n = row_n[newDB_json["pos"]["RA"]]
            dec_n = row_n[newDB_json["pos"]["DEC"]]
        if "plx" in newDB_json["pos"]:
            plx_n = row_n[newDB_json["pos"]["plx"]]
        if "pmra" in newDB_json["pos"]:
            pmra_n = row_n[newDB_json["pos"]["pmra"]]
        if "pmde" in newDB_json["pos"]:
            pmde_n = row_n[newDB_json["pos"]["pmde"]]

        # Index of the match for this new cluster in the old DB (if any)
        db_match_j = db_matches[i]

        # If the cluster is already present in the UCC
        if db_match_j is not None:
            new_db_info["fnames"].append(fnames)
            new_db_info["RA_ICRS"].append(ra_n)
            new_db_info["DE_ICRS"].append(dec_n)
            new_db_info["pmRA"].append(pmra_n)
            new_db_info["pmDE"].append(pmde_n)
            new_db_info["Plx"].append(plx_n)
            new_db_info["UCC_idx"].append(db_match_j)
        else:
            # This is a new cluster
            new_db_info["fnames"].append(fnames)
            new_db_info["RA_ICRS"].append(np.nan)
            new_db_info["DE_ICRS"].append(np.nan)
            new_db_info["pmRA"].append(np.nan)
            new_db_info["pmDE"].append(np.nan)
            new_db_info["Plx"].append(plx_n)
            new_db_info["UCC_idx"].append(None)

    lon_n, lat_n = radec2lonlat(
        new_db_info["RA_ICRS"],
        new_db_info["DE_ICRS"],
    )
    new_db_info["GLON"] = list(np.round(lon_n, 4))
    new_db_info["GLAT"] = list(np.round(lat_n, 4))

    return new_db_info


def flag_log(
    logging,
    bad_center: str,
    d_arcmin: float,
    pmra_p: float,
    pmde_p: float,
    plx_p: float,
    fnames: str,
) -> None:
    """
    Logs details about OCs flagged for attention based on center comparison.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    bad_center : str
        String indicating the quality of the center comparison.
    fnames : str
        String of fnames for the cluster.
    """
    txt = ""
    if bad_center[0] == "y":
        txt += "{:.1f} ".format(d_arcmin)
    else:
        txt += "-- "

    if bad_center[1] == "y":
        txt += "{:.1f} {:.1f} ".format(pmra_p, pmde_p)
    else:
        txt += "-- -- "

    if bad_center[2] == "y":
        txt += "{:.1f}".format(plx_p)
    else:
        txt += "--"

    txt = txt.split()
    logging.info(
        "{:<25} {:<5} {:>12} {:>8} {:>8} {:>7}".format(fnames[:24], bad_center, *txt)
    )


def fnames_check_UCC_new_DB(
    logging,
    df_UCC: pd.DataFrame,
    new_DB_fnames: list[list[str]],
) -> bool:
    """
    Check that no fname associated to each entry in the new DB is listed in more than
    one entry in the UCC.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame representing the UCC.
    new_DB_fnames : list
        List of lists, each containing standardized names for a cluster in the new DB.

    Returns
    -------
    bool
        True if duplicate entries are found, False otherwise.
    """
    # Create a dictionary to map filenames to their corresponding row indices
    # Using defaultdict eliminates the explicit check for key existence
    filename_map = defaultdict(list)
    for i, fnames in enumerate(df_UCC["fnames"]):
        for fname in fnames.split(";"):
            filename_map[fname].append(i)

    # Find matches between UCC fnames and new_DB_fnames
    fnames_ucc_idxs = {}
    for k, fnames in enumerate(new_DB_fnames):
        fnames_ucc_idxs[k] = []
        for fname in fnames:
            if fname in filename_map:  # Check if the filename exists in df_fnames
                # 'filename_map[fname]' will always contain a single element
                fnames_ucc_idxs[k].append(filename_map[fname][0])

    # Check if any new entry has more than one entry in the UCC associated to it
    dup_flag = False
    for k, v in fnames_ucc_idxs.items():
        if len(list(set(v))) > 1:
            dup_flag = True
            logging.info(f"{new_DB_fnames[k]} --> {v}")

    return dup_flag


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
        df_new_db.loc[i, "Names"] = ";".join([unq_names[i] for i in idx_order])
        df_new_db.loc[i, "fnames"] = ";".join([unq_fnames[i] for i in idx_order])

    return df_new_db


def add_new_DB(
    df_UCC: pd.DataFrame,
    db_matches: list,
    df_new_db: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds a new database to the Unified Cluster Catalogue (UCC).
    """
    # Drop OCs from the UCC that are present in the new DB
    # Remove 'None' entries first from the indexes list
    idx_rm_comb_db = [_ for _ in db_matches if _ is not None]
    df_UCC_no_new = df_UCC.drop(list(df_UCC.index[idx_rm_comb_db]))
    df_UCC_no_new.reset_index(drop=True, inplace=True)

    # Final UCC with the new DB incorporated
    df_UCC_new = pd.concat([df_UCC_no_new, df_new_db], ignore_index=True)

    return df_UCC_new


def duplicates_fnames_check(logging, df_UCC_new: pd.DataFrame) -> bool:
    """
    Check for duplicate filenames across rows in 'df_UCC_new' DataFrame.

    The function scans the 'fnames' column of the DataFrame, where each entry
    may contain one or more filenames separated by semicolons. It identifies
    any filenames that appear in more than one row. For each duplicate, the
    row indices and the filename are logged, ensuring that duplicate pairs
    are reported only once.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC_new : pd.DataFrame
        DataFrame to check for duplicates.

    Returns
    -------
    bool
        True if duplicates are found, False otherwise.
    """
    # Create a dictionary to map filenames to their corresponding row indices
    filename_map = {}
    # Populate the dictionary
    for i, fnames in enumerate(df_UCC_new["fnames"]):
        for fname in fnames.split(";"):
            if fname not in filename_map:
                filename_map[fname] = []
            filename_map[fname].append(i)

    # Track printed pairs
    printed_pairs = set()

    dup_flag = False
    # Find and print matches
    for fname, indices in filename_map.items():
        if len(indices) > 1:  # Check if a filename appears in more than one row
            for i in indices:
                for j in indices:
                    if i != j:
                        dup_flag = True
                        # Ensure consistent order for pairs
                        pair = tuple(sorted((i, j)))
                        if pair not in printed_pairs:
                            logging.info(f"{i}, {j}, {fname}")
                            printed_pairs.add(pair)

    return dup_flag


def move_files(
    logging,
    temp_JSON_file: str,
    all_dbs_data: dict,
    df_UCC_new: pd.DataFrame,
    temp_database_folder: str,
) -> None:
    """ """
    logging.info("\nUpdate files...")

    # Update JSON file with all the DBs and store the new DB in place
    if os.path.isfile(temp_JSON_file):
        # Move JSON file from temp folder to final folder
        os.rename(temp_JSON_file, name_DBs_json)
        logging.info(temp_JSON_file + " --> " + name_DBs_json)

    for new_DB in all_dbs_data.keys():
        # Move new DB file
        new_DB_file = new_DB + ".csv"
        db_temp = temp_database_folder + new_DB_file
        if os.path.isfile(db_temp):
            db_stored = dbs_folder + new_DB_file
            os.rename(db_temp, db_stored)
            logging.info(db_temp + " --> " + db_stored)

    # Save df_UCC_new to temp file
    file_path = temp_fold + "df_UCC_B_updt.csv"
    save_df_UCC(logging, df_UCC_new, file_path)
