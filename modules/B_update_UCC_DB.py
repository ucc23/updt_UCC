import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from astropy.coordinates import angular_separation
from rapidfuzz import fuzz, process
from scipy.spatial.distance import cdist

from .utils import diff_between_dfs, logger, radec2lonlat, save_df_UCC
from .variables import (
    GCs_cat,
    data_folder,
    dbs_folder,
    merged_dbs_file,
    name_DBs_json,
    naming_order,
    selected_center_coords,
    temp_folder,
)


def main():
    """
    First function to update the UCC (Unified Cluster Catalogue) with a new database.
    """
    logging = logger()

    # Generate paths and check for required folders and files
    temp_database_folder, dbs_merged_current_file, temp_JSON_file = (
        get_paths_check_paths(logging)
    )

    dbs_merged_current, df_GCs, all_dbs_data, dbs_merged_new, flag_interactive = (
        load_data(
            logging, dbs_merged_current_file, temp_JSON_file, temp_database_folder
        )
    )

    for new_DB, (df_new, newDB_json) in all_dbs_data.items():
        logging.info("\n" + "-" * 40)
        logging.info(f"Adding {new_DB} to the UCC")

        # Add inner columns with galactic coordinates if possible
        if "RA" in newDB_json["pos"].keys():
            df_new["GLON_"], df_new["GLAT_"] = radec2lonlat(
                df_new[newDB_json["pos"]["RA"]].to_numpy(),
                df_new[newDB_json["pos"]["DEC"]].to_numpy(),
            )

        # Check the new DB for basic requirements
        check_new_DB(
            logging,
            df_GCs,
            new_DB,
            df_new,
            newDB_json,
            flag_interactive,
        )

        # Standardize names in the new DB
        new_DB_fnames = get_fnames_new_DB(logging, df_new, newDB_json)

        # Match the new DB with the UCC
        db_matches = get_matches_new_DB(dbs_merged_new, new_DB_fnames)

        # Check the entries in the new DB vs the entries in the UCC
        check_new_DB_vs_UCC(
            logging,
            dbs_merged_new,
            df_new,
            new_DB_fnames,
            db_matches,
            flag_interactive,
        )

        # Combine the new DB with the UCC
        df_new_db = combine_UCC_new_DB(
            logging,
            new_DB,
            newDB_json,
            dbs_merged_new,
            df_new,
            new_DB_fnames,
            db_matches,
        )

        # Generate new UCC file with the new DB incorporated
        dbs_merged_new = add_new_DB(dbs_merged_new, db_matches, df_new_db)

    # Remove duplicate names and order by naming priority
    dbs_merged_new = rm_name_dups_order(dbs_merged_new)

    # Order DBs by years
    dbs_merged_new = date_order_DBs(dbs_merged_new)

    # Last sanity check. Check every individual fname for duplicates
    if duplicates_fnames_check(logging, dbs_merged_new):
        raise ValueError("Duplicated entries found in 'fnames' column")

    #
    diff_between_dfs(logging, dbs_merged_current, dbs_merged_new)

    file_path = dbs_merged_current_file.replace(data_folder, temp_folder)
    save_df_UCC(logging, dbs_merged_new, file_path, order_col="fnames")

    if input("\nMove files to their final paths? (y/n): ").lower() == "y":
        move_files(
            logging,
            dbs_merged_current_file,
            temp_JSON_file,
            all_dbs_data,
            dbs_merged_new,
            temp_database_folder,
        )

    logging.info("\nProceed with the next script")


def get_paths_check_paths(
    logging,
) -> tuple[str, str, str]:
    """ """
    # Path to the current DBs merged file
    dbs_merged_current_file = data_folder + merged_dbs_file

    # If file exists, read and return it
    # last_version = get_last_version_UCC(data_folder)
    temp_merged_f = temp_folder + merged_dbs_file
    if os.path.isfile(temp_merged_f):
        logging.warning(
            f"WARNING: file {temp_merged_f} exists. Moving on will re-write it"
        )
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Temporary databases/ folder
    temp_database_folder = temp_folder + dbs_folder
    # Create if required
    if not os.path.exists(temp_database_folder):
        os.makedirs(temp_database_folder)

    # Path to the new (temp) JSON file
    temp_JSON_file = temp_folder + name_DBs_json

    return (
        temp_database_folder,
        dbs_merged_current_file,
        temp_JSON_file,
    )


def load_data(
    logging,
    dbs_merged_current_file: str,
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
    dbs_merged_current = pd.read_csv(dbs_merged_current_file)
    logging.info(
        f"\nUCC version {dbs_merged_current_file} loaded (N={len(dbs_merged_current)})"
    )

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

        dbs_merged_new = dbs_merged_current.copy()

    else:
        logging.info("\n=== Rebuilding the UCC completely===\n")
        flag_interactive = False

        # All DBs (assumes sorted by year)
        all_dbs = list(current_JSON.keys())
        # First entries
        all_dbs = [_ for _ in all_dbs if _ not in ("DIAS2002", "BICA2019")]
        all_dbs = ["DIAS2002", "BICA2019"] + all_dbs

        # Load all DBs
        for new_DB in all_dbs:
            df_new = pd.read_csv(dbs_folder + new_DB + ".csv")
            logging.info(f"{new_DB} loaded (N={len(df_new)})")
            newDB_json = current_JSON[new_DB]
            all_dbs_data[new_DB] = [df_new, newDB_json]

        # Empty dataframe
        dbs_merged_new = pd.DataFrame(dbs_merged_current[0:0])

    return dbs_merged_current, df_GCs, all_dbs_data, dbs_merged_new, flag_interactive


def check_new_DB(
    logging,
    df_GCs: pd.DataFrame,
    new_DB: str,
    df_new: pd.DataFrame,
    newDB_json: dict,
    flag_interactive: bool,
) -> None:
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
    if vdberg_check(logging, newDB_json, df_new):
        if flag_interactive:
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit()

    if "RA" in newDB_json["pos"].keys():
        # Check for OCs very close to each other in the new DB
        if close_OC_inner_check(logging, newDB_json, df_new):
            if flag_interactive:
                if input("Move on? (y/n): ").lower() != "y":
                    sys.exit()

    if "GLON_" in df_new.keys():
        # Check for close GCs
        if GCs_check(logging, df_GCs, newDB_json, df_new):
            if flag_interactive:
                if input("Move on? (y/n): ").lower() != "y":
                    sys.exit()


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
    bad_names = {"vdberghhagen", "vdbergh", "vandenberghhagen", "vandenbergh", "bh"}
    digit_re = re.compile(r"\d")  # compile once

    vds_found = []
    for i, new_cl in enumerate(df_new[newDB_json["names"]]):
        # Remove all numbers
        new_cl_r = digit_re.sub("", new_cl)
        # Normalize
        new_cl_r = normalize_name(new_cl_r)
        all_names = new_cl_r.split(",")

        # If the root of the name is a proper name, skip check
        if "vdbh" in all_names or "vdb" in all_names:
            continue

        results = [
            process.extractOne(q, bad_names, scorer=fuzz.ratio) for q in all_names
        ]
        best_match, score, index = max(results, key=lambda x: x[1])
        if score > 50:
            vds_found.append([i, new_cl, best_match, score])

    vdb_flag = False
    if len(vds_found) > 0:
        vdb_flag = True
        logging.info(f"\nFound {len(vds_found)} entries that could need name editing")
        logging.info("* BH ; vdBergh-Hagen --> VDBH")
        logging.info("* vdBergh            --> VDB")
        for i, new_cl, name_check, sm_ratio in vds_found:
            logging.info(f"{i}, {new_cl.strip()} --> {name_check} (P={sm_ratio})")

    return vdb_flag


def close_OC_inner_check(
    logging,
    newDB_json: dict,
    df_new: pd.DataFrame,
    leven_rad: float = 0.9,
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
    coords = np.array(
        [df_new[newDB_json["pos"]["RA"]], df_new[newDB_json["pos"]["DEC"]]]
    ).T
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
        logging, dist, db_matches, col_1, col_2, ID_call, leven_rad, sep
    )


def close_OC_UCC_check(
    logging,
    dbs_merged_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
    df_new: pd.DataFrame,
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
    coords_new = np.array([df_new["GLON_"], df_new["GLAT_"]]).T
    coords_UCC = np.array([dbs_merged_new["GLON"], dbs_merged_new["GLAT"]]).T

    # Find the distances to all clusters, for all clusters (in arcmin)
    dist = cdist(coords_new, coords_UCC) * 60

    col_1 = dbs_merged_new["fnames"]
    col_2 = new_DB_fnames
    ID_call = "UCC"

    return close_OC_check(
        logging, dist, db_matches, col_1, col_2, ID_call, leven_rad, sep
    )


def close_OC_check(
    logging,
    dist,
    db_matches,
    col_1,
    col_2,
    ID_call: str,
    leven_rad: float,
    sep: str,
    rad_dup: float = 15,
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
            norm_cl_names = normalize_name(cl_name).split(",")
            norm_dup_names = normalize_name(col_1[j]).split(",")
            results = [
                process.extractOne(q, norm_dup_names, scorer=fuzz.ratio)
                for q in norm_cl_names
            ]
            # Divide by 100 to make the range [0, 1]
            L_ratio = max(results, key=lambda x: x[1])[1] / 100

            if L_ratio > leven_rad:
                N_inner_dups += 1
                dups_list.append(col_1[j])
                dups.append(col_1[j])
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
                f"{i:<6} {cl_name.strip():<15} (N={N_inner_dups}) --> "
                + f"{';'.join([_.strip() for _ in dups]):<15} | d={';'.join(dist)}, L={';'.join(L_ratios)}"
            )

    return dups_flag


def GCs_check(
    logging,
    df_GCs: pd.DataFrame,
    newDB_json: dict,
    df_new: pd.DataFrame,
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
    glon_glat = np.array([df_new["GLON_"], df_new["GLAT_"]]).T

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
            names_l.append(normalize_name(name))
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


def normalize_name(name: str) -> str:
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
    df_UCC: pd.DataFrame, new_DB_fnames: list[list[str]], sep: str = ";"
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
    # db_matches = []
    # # For each fname(s) for each entry in the new DB
    # for new_cl_fnames in new_DB_fnames:
    #     found = None
    #     # For each fname in this new DB entry
    #     for new_cl_fname in new_cl_fnames:
    #         # For each fname(s) for each entry in the UCC
    #         for j, ucc_cl_fnames in enumerate(df_UCC["fnames"]):
    #             # Check if the fname in the new DB is included in the UCC
    #             if new_cl_fname in ucc_cl_fnames.split(";"):
    #                 # If it is, return the UCC index
    #                 found = j
    #                 break
    #         if found is not None:
    #             break
    #     db_matches.append(found)

    # Build lookup dictionary from fname -> UCC index
    fname_to_idx = {}
    for idx, fnames in enumerate(df_UCC["fnames"]):
        for fname in fnames.split(sep):
            fname_to_idx[fname] = idx

    db_matches = []
    for cl_fnames in new_DB_fnames:
        found = next((fname_to_idx[f] for f in cl_fnames if f in fname_to_idx), None)
        db_matches.append(found)

    return db_matches


def check_new_DB_vs_UCC(
    logging,
    dbs_merged_new,
    df_new,
    new_DB_fnames,
    db_matches,
    flag_interactive: bool,
) -> None:
    """
    - Checks for duplicate entries between the new database and the UCC.
    - Checks for OCs very close to each other between the new database and the UCC.
    - Checks positions and flags for attention if required.
    """
    # Check all fnames in the new DB against all fnames in the UCC
    # logging.info("\nChecking uniqueness of fnames")
    if fnames_check_UCC_new_DB(logging, dbs_merged_new, new_DB_fnames):
        raise ValueError("\nResolve the above issues before moving on")

    # # Check the first fname for all entries in the new DB
    # logging.info("\nChecking for entries that must be combined")
    # if dups_fnames_inner_check(logging, new_DB, newDB_json, df_new, new_DB_fnames):
    #     raise ValueError("\nResolve the above issues before moving on")

    # # Check for duplicate entries in the new DB that also exist in the UCC
    # if dups_check_newDB_UCC(logging, new_DB, df_UCC, new_DB_fnames, db_matches):
    #     raise ValueError("\nResolve the above issues before moving on")

    if "GLON_" in df_new.keys():
        # Check for OCs very close to other OCs in the UCC
        if close_OC_UCC_check(
            logging, dbs_merged_new, new_DB_fnames, db_matches, df_new
        ):
            if flag_interactive:
                if input("Move on? (y/n): ").lower() != "y":
                    sys.exit()

    # Check positions and flag for attention if required
    if positions_check(
        logging,
        dbs_merged_new,
        df_new,
        new_DB_fnames,
        db_matches,
    ):
        if flag_interactive:
            if input("\nMove on? (y/n): ").lower() != "y":
                sys.exit()


def positions_check(
    logging,
    dbs_merged_new: pd.DataFrame,
    df_new,
    new_DB_fnames,
    db_matches,
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
    df_new_glon, df_new_glat = [np.array([np.nan] * len(df_new))] * 2
    if "GLON_" in df_new.keys():
        df_new_glon, df_new_glat = (
            df_new["GLON_"].to_numpy(),
            df_new["GLAT_"].to_numpy(),
        )
    df_UCC_glon, df_UCC_glat = (
        dbs_merged_new["GLON"].to_numpy(),
        dbs_merged_new["GLAT"].to_numpy(),
    )

    ocs_attention = []
    for i, j in enumerate(db_matches):
        # Check centers if the OC is already present in the UCC
        if j is not None:
            d_arcmin = (
                np.sqrt(
                    (df_UCC_glon[j] - df_new_glon[i]) ** 2
                    + (df_UCC_glat[j] - df_new_glat[i]) ** 2
                )
                * 60
            )
            # Store information on the OCs that require attention
            if d_arcmin > rad_dup:
                ocs_attention.append([i, new_DB_fnames[i][0], d_arcmin])

    attention_flag = False
    if len(ocs_attention) > 0:
        attention_flag = True
        logging.info(
            f"\nEntries with coords far from those in the UCC (N={len(ocs_attention)})"
        )
        logging.info(f"{'DB_idx':<6} {'name':<20} {'d [arcmin]':<5}")
        for i, fname, d_arcmin in ocs_attention:
            logging.info(f"{i:<6} {fname:<20} {d_arcmin:<5.0f}")

    return attention_flag


def fnames_check_UCC_new_DB(
    logging, df_UCC: pd.DataFrame, new_DB_fnames: list[list[str]], sep: str = ";"
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
        for fname in fnames.split(sep):
            filename_map[fname].append(i)

    # Find matches between UCC fnames and new_DB_fnames
    fnames_ucc_idxs = {}
    for k, fnames in enumerate(new_DB_fnames):
        fnames_ucc_idxs[k] = []
        for fname in fnames:
            if fname in filename_map:  # Check if the filename exists in df_fnames
                # 'filename_map[fname]' will always contain a single element
                fnames_ucc_idxs[k].append(filename_map[fname][0])

    # Extract bad entries
    bad_entries = []
    for k, v in fnames_ucc_idxs.items():
        if len(list(set(v))) > 1:
            bad_entries.append([k, v])

    # Check if any new entry has more than one entry in the UCC associated to it
    dup_flag = False
    if bad_entries:
        dup_flag = True
        logging.info(
            f"\nFound {len(bad_entries)} entries in the new DB with duplicated fnames"
        )
        for k, v in bad_entries:
            new_db_entries = f"({k}) {', '.join(new_DB_fnames[k])}"
            ucc_entries = ", ".join([f"({_}) {df_UCC.iloc[_]['fnames']}" for _ in v])
            logging.info(f"{new_db_entries} --> {ucc_entries}")

    return dup_flag


def combine_UCC_new_DB(
    logging,
    new_DB: str,
    newDB_json: dict,
    df_UCC: pd.DataFrame,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
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
    pos_cols = newDB_json["pos"]
    # Faster access below compared to using iloc
    ucc_dict_rows = df_UCC.to_dict(orient="records")

    new_db_dict = {_: [] for _ in df_UCC.keys()}
    # For each entry in the new DB

    for i_new_cl, (_, row_n) in enumerate(df_new.iterrows()):
        # Rename certain entries (ESO, FSR, etc)
        oc_names = rename_standard(str(row_n[newDB_json["names"]]))

        row_ucc = {}
        ucc_index = db_matches[i_new_cl]
        if ucc_index is not None:  # The cluster is already present in the UCC
            # Row in UCC where this match is located
            row_ucc = ucc_dict_rows[ucc_index]

        new_db_dict = updt_new_DB(
            new_DB,
            new_db_dict,
            str(i_new_cl),
            new_DB_fnames[i_new_cl],
            oc_names,
            row_n,
            row_ucc,
            pos_cols,
        )

    N_new = db_matches.count(None)
    N_updt = len(db_matches) - N_new
    logging.info(f"\nNew entries: {N_new}, Updated entries: {N_updt}")

    return pd.DataFrame(new_db_dict)


def rename_standard(all_names: str, sep_in: str = ",", sep_out: str = ";") -> str:
    """
    Standardize the naming of these clusters

    FSR XXX w leading zeros
    FSR XXX w/o leading zeros
    FSR_XXX w leading zeros
    FSR_XXX w/o leading zeros

    --> FSR_XXXX (w leading zeroes)

    ESO XXX-YY w leading zeros
    ESO XXX-YY w/o leading zeros
    ESO_XXX_YY w leading zeros
    ESO_XXX_YY w/o leading zeros
    ESO_XXX-YY w leading zeros
    ESO_XXX-YY w/o leading zeros
    ESOXXX_YY w leading zeros (LOKTIN17)

    --> ESO_XXX_YY (w leading zeroes)

    Parameters
    ----------
    name : str
        Name of the cluster.

    Returns
    -------
    str
        Standardized name of the cluster.
    """
    # For each comma separated name for this OC in the new DB
    oc_names = all_names.split(sep_in)

    new_names_rename = []
    for name in oc_names:
        name = name.strip()

        if name.startswith("FSR"):
            if " " in name or "_" in name:
                if "_" in name:
                    n2 = name.split("_")[1]
                else:
                    n2 = name.split(" ")[1]
                n2 = int(n2)
                if n2 < 10:
                    n2 = "000" + str(n2)
                elif n2 < 100:
                    n2 = "00" + str(n2)
                elif n2 < 1000:
                    n2 = "0" + str(n2)
                else:
                    n2 = str(n2)
                name = "FSR_" + n2

        if name.startswith("ESO"):
            if name[:4] not in ("ESO_", "ESO "):
                # E.g.: LOKTIN17, BOSSINI19
                name = "ESO_" + name[3:]

            if " " in name[4:]:
                n1, n2 = name[4:].split(" ")
            elif "_" in name[4:]:
                n1, n2 = name[4:].split("_")
            elif "-" in name[4:]:
                n1, n2 = name[4:].split("-")
            else:
                # This assumes that all ESo clusters are names as: 'ESO XXX YY'
                n1, n2 = name[4 : 4 + 3], name[4 + 3 :]

            n1 = int(n1)
            if n1 < 10:
                n1 = "00" + str(n1)
            elif n1 < 100:
                n1 = "0" + str(n1)
            else:
                n1 = str(n1)
            n2 = int(n2)
            if n2 < 10:
                n2 = "0" + str(n2)
            else:
                n2 = str(n2)
            name = "ESO_" + n1 + "_" + n2

        if "UBC" in name and "UBC " not in name and "UBC_" not in name:
            name = name.replace("UBC", "UBC ")
        if "UBC_" in name:
            name = name.replace("UBC_", "UBC ")

        if "UFMG" in name and "UFMG " not in name and "UFMG_" not in name:
            name = name.replace("UFMG", "UFMG ")

        if (
            "LISC" in name
            and "LISC " not in name
            and "LISC_" not in name
            and "LISC-" not in name
        ):
            name = name.replace("LISC", "LISC ")

        if "OC-" in name:
            name = name.replace("OC-", "OC ")

        # Removes duplicates such as "NGC_2516" and "NGC 2516"
        name = name.replace("_", " ")

        new_names_rename.append(name)
    oc_names = sep_out.join(new_names_rename)

    return oc_names


def extract_new_DB_coords(DB_ID, row_n, pos_cols):
    """ """
    ra_n, dec_n, lon_n, lat_n = [np.nan] * 4
    plx_n, pmra_n, pmde_n = [np.nan] * 3

    # Don't use KHARCHENKO2012 for coordinates
    if DB_ID != "KHARCHENKO2012":
        if "RA" in pos_cols:
            ra_n, dec_n = row_n[pos_cols["RA"]], row_n[pos_cols["DEC"]]
            lon_n, lat_n = row_n["GLON_"], row_n["GLAT_"]
    # Don't use LOKTIN2017 Pms, Plx values
    if DB_ID != "LOKTIN2017":
        if "plx" in pos_cols:
            plx_n = row_n[pos_cols["plx"]]
        if "pmra" in pos_cols:
            pmra_n = row_n[pos_cols["pmra"]]
        if "pmde" in pos_cols:
            pmde_n = row_n[pos_cols["pmde"]]

    return ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n


def updt_new_DB(
    new_DB: str,
    new_db_dict: dict,
    i_new_cl: str,
    fnames_new_cl: list[str],
    oc_names: str,
    row_n: pd.Series,
    row_ucc: dict,
    pos_cols: dict,
    sep: str = ";",
):
    """ """
    # Extract coordinates from new DB
    ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = extract_new_DB_coords(
        new_DB, row_n, pos_cols
    )

    if len(row_ucc) == 0:  # This OC is not present in the UCC
        # Extract name(s) and fname(s) from new DB
        DB_ID, DB_i = new_DB, i_new_cl
        names, fnames = oc_names, sep.join(fnames_new_cl)

    else:  # This OC is already present in the UCC
        # Attach name(s) and fname(s) present in new DB to the UCC
        DB_ID = row_ucc["DB"] + sep + new_DB
        DB_i = row_ucc["DB_i"] + sep + i_new_cl
        names = row_ucc["Names"] + sep + oc_names
        fnames = row_ucc["fnames"] + sep + sep.join(fnames_new_cl)

        # Only update values if nan is in place, else keep the FIRST values stored
        vals_new = [ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n]
        cols = ("RA_ICRS", "DE_ICRS", "GLON", "GLAT", "Plx", "pmRA", "pmDE")
        vals = [row_ucc[col] for col in cols]
        for i, val in enumerate(vals):
            if np.isnan(val):
                vals[i] = vals_new[i]
        ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = vals

        # If this is one of the OCs with selected center values, replace here
        idx = [
            i
            for i, val in enumerate(fnames_new_cl)
            if val in selected_center_coords.keys()
        ]
        if idx:
            ra_n, dec_n = selected_center_coords[fnames_new_cl[idx[0]]]
            lon_n, lat_n = radec2lonlat(ra_n, dec_n)

        # # Always update values unless nan
        # vals_new = [ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n]
        # cols = ("RA_ICRS", "DE_ICRS", "GLON", "GLAT", "Plx", "pmRA", "pmDE")
        # vals = [row_ucc[col] for col in cols]
        # for i, val in enumerate(vals_new):
        #     if np.isnan(val):
        #         vals_new[i] = vals[i]
        # ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = vals_new

    new_db_dict["DB"].append(DB_ID)
    new_db_dict["DB_i"].append(DB_i)
    new_db_dict["Names"].append(names)
    new_db_dict["fnames"].append(fnames)
    new_db_dict["RA_ICRS"].append(ra_n)
    new_db_dict["DE_ICRS"].append(dec_n)
    new_db_dict["GLON"].append(lon_n)
    new_db_dict["GLAT"].append(lat_n)
    new_db_dict["Plx"].append(plx_n)
    new_db_dict["pmRA"].append(pmra_n)
    new_db_dict["pmDE"].append(pmde_n)

    return new_db_dict


def rm_name_dups_order(df_new_db: pd.DataFrame, sep: str = ";") -> pd.DataFrame:
    """
    Removes duplicate names from a semicolon-separated string, considering variations
    with or without spaces and underscores, and orders according to given rules.

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

    def order_by_name(items: list[str]) -> np.ndarray:
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
        np.ndarray[int]
            Indexes that reorder the list accordingly.
        """
        idxs = []
        for item in items:
            idx = [np.inf]
            for j, p in enumerate(naming_order):
                if item.startswith(p):
                    idx.append(j)
                    break
            idxs.append(min(idx))
        return np.argsort(idxs)

    names_out, fnames_out = [], []
    for i, fnames in enumerate(df_new_db["fnames"]):
        # Remove duplicates from fnames
        unq_fnames, rm_idx = rm_dups(fnames.split(sep))
        # Remove same duplicates from names
        unq_names = str(df_new_db["Names"][i]).split(sep)
        if rm_idx:
            unq_names = [_ for i, _ in enumerate(unq_names) if i not in rm_idx]

        # Name ordering
        idx_order = order_by_name(unq_fnames)
        names_out.append(sep.join(unq_names[i] for i in idx_order))
        fnames_out.append(sep.join(unq_fnames[i] for i in idx_order))

    df_new_db["Names"] = names_out
    df_new_db["fnames"] = fnames_out

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
    dbs_merged_new = pd.concat([df_UCC_no_new, df_new_db], ignore_index=True)

    return dbs_merged_new


def date_order_DBs(dbs_merged_new: pd.DataFrame, sep: str = ";") -> pd.DataFrame:
    """
    Orders two semicolon-separated strings of database entries by the year extracted
    from each entry.
    """
    DB_all, DB_i_all = dbs_merged_new["DB"], dbs_merged_new["DB_i"]

    DB_all_sorted, DB_i_all_sorted = [], []
    for i, DBs in enumerate(DB_all):
        # Split lists
        all_dbs = DBs.split(sep)
        all_dbs_i = str(DB_i_all[i]).split(sep)

        # Extract years from DBs
        all_years = []
        for db in all_dbs:
            # Flip the year and DB name
            db1, db2 = db, ""
            if "_" in db:
                db1, db2 = db.split("_")
            year = db1[-4:] + db1[:-4] + db2
            all_years.append(year)
        # Sort and re-generate strings
        idx = np.argsort(all_years)

        DB_all_sorted.append(sep.join(list(np.array(all_dbs)[idx])))
        DB_i_all_sorted.append(sep.join(list(np.array(all_dbs_i)[idx])))

    dbs_merged_new["DB"] = DB_all_sorted
    dbs_merged_new["DB_i"] = DB_i_all_sorted

    return dbs_merged_new


def duplicates_fnames_check(
    logging, dbs_merged_new: pd.DataFrame, sep: str = ";"
) -> bool:
    """
    Check for duplicate filenames across rows in 'dbs_merged_new' DataFrame.

    The function scans the 'fnames' column of the DataFrame, where each entry
    may contain one or more filenames separated by semicolons. It identifies
    any filenames that appear in more than one row. For each duplicate, the
    row indices and the filename are logged, ensuring that duplicate pairs
    are reported only once.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    dbs_merged_new : pd.DataFrame
        DataFrame to check for duplicates.

    Returns
    -------
    bool
        True if duplicates are found, False otherwise.
    """
    # Create a dictionary to map filenames to their corresponding row indices
    filename_map = {}
    # Populate the dictionary
    for i, fnames in enumerate(dbs_merged_new["fnames"]):
        for fname in fnames.split(sep):
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
    dbs_merged_current_file: str,
    temp_JSON_file: str,
    all_dbs_data: dict,
    dbs_merged_new: pd.DataFrame,
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

    # Save dbs_merged_new to final folder
    save_df_UCC(logging, dbs_merged_new, dbs_merged_current_file, order_col="fnames")


if __name__ == "__main__":
    main()
