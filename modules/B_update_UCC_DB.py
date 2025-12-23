import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.spatial.distance import cdist

from .utils import (
    diff_between_dfs,
    final_fname_compare,
    logger,
    radec2lonlat,
    save_df_UCC,
)
from .variables import (
    GCs_cat,
    c_Ag,
    c_Ebv,
    c_Evi,
    c_z_sun,
    data_folder,
    dbs_folder,
    fpars_order,
    merged_dbs_file,
    name_DBs_json,
    naming_order,
    selected_center_coords,
    temp_folder,
)


def main():
    """
    Function to update the UCC (Unified Cluster Catalogue) with a new database.
    """
    logging = logger()

    skip_check_flag = False
    if input("\nSkip check for non new DBs? (y/n): ").lower() == "y":
        skip_check_flag = True

    # Generate paths and check for required folders and files
    temp_database_folder, df_UCC_B_current_file, temp_JSON_file = get_paths_check_paths(
        logging
    )

    (
        new_JSON,
        df_GCs,
        gcs_fnames,
        all_dbs_data,
        df_UCC_B_old,
        df_UCC_B,
        flag_interactive,
    ) = load_data(logging, df_UCC_B_current_file, temp_JSON_file, temp_database_folder)

    # profiler = Profiler()
    # profiler.start()

    for new_DB, (df_new, newDB_json) in all_dbs_data.items():
        logging.info("\n" + "-" * 40)
        logging.info(f"Adding {new_DB} to the UCC")

        # Add inner columns with galactic coordinates if possible
        if "RA" in newDB_json["pos"].keys():
            df_new["GLON_"], df_new["GLAT_"] = radec2lonlat(
                df_new[newDB_json["pos"]["RA"]].to_numpy(),
                df_new[newDB_json["pos"]["DEC"]].to_numpy(),
            )

        # Standardize names in the DB
        new_DB_fnames = get_fnames_new_DB(logging, df_new, newDB_json)

        # Check the DB for basic requirements
        if new_DB in flag_interactive or not skip_check_flag:
            check_new_DB(
                logging,
                df_GCs,
                gcs_fnames,
                new_DB,
                df_new,
                new_DB_fnames,
                newDB_json,
                flag_interactive,
            )

        # Match the new DB with the UCC
        db_matches = get_matches_new_DB(df_UCC_B, new_DB_fnames)

        # Report and check new entries (if any)
        if new_DB in flag_interactive or not skip_check_flag:
            N_new = db_matches.count(None)
            if N_new > 0:
                logging.info(f"\nFound {N_new} new entries")
                N_max, N_print = 50, 0
                for i, idx in enumerate(db_matches):
                    if idx is None:
                        logging.info(f"{i}    {new_DB_fnames[i][0]}")
                        N_print += 1
                    if N_print == N_max:
                        logging.info(f"... (only first {N_max} entries shown)")
                        break
                breakpoint()
            else:
                logging.info("\nNo new entries found")

            # Check the entries in the DB vs the entries in the UCC
            check_new_DB_vs_UCC(
                logging,
                new_DB,
                df_UCC_B,
                df_new,
                new_DB_fnames,
                db_matches,
                flag_interactive,
            )

        # Add fundamental parameters column
        df_new = add_fpars_col(newDB_json, df_new)

        # Combine the new DB with the UCC
        df_UCC_B = combine_UCC_new_DB(
            logging,
            new_DB,
            newDB_json,
            df_UCC_B,
            df_new,
            new_DB_fnames,
            db_matches,
        )

    # profiler.stop()
    # profiler.open_in_browser()

    logging.info("\n\n=====================")
    logging.info("Merging of DBs completed\n")

    df_UCC_B = sort_year_importance(new_JSON, df_UCC_B)

    # Last sanity check. Check every individual fname for duplicates
    exit_flag = duplicates_fnames_check(logging, df_UCC_B)
    if exit_flag:
        logging.info("\nDuplicated entries found in 'fnames' column")
        breakpoint()
        sys.exit(0)
    if exit_flag:
        raise ValueError("Duplicated entries found in 'fnames' column")

    # Generate diff files to open with Meld or similar
    diff_found = diff_between_dfs(logging, df_UCC_B_old, df_UCC_B, order_col="fnames")

    # Compare changes in 'fnames' columns between old and new df
    if diff_found:
        final_fname_compare(logging, df_UCC_B_old, df_UCC_B)

    file_path = df_UCC_B_current_file.replace(data_folder, temp_folder)
    save_df_UCC(logging, df_UCC_B, file_path, order_col="fnames")

    if input("\nMove files to their final paths? (y/n): ").lower() == "y":
        move_files(
            logging,
            df_UCC_B_current_file,
            new_JSON,
            temp_JSON_file,
            all_dbs_data,
            temp_database_folder,
        )


def get_paths_check_paths(
    logging,
) -> tuple[str, str, str]:
    """ """
    # Path to the current DBs merged file
    df_UCC_B_current_file = data_folder + merged_dbs_file

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

    return temp_database_folder, df_UCC_B_current_file, temp_JSON_file


def load_data(
    logging,
    dbs_merged_current_file: str,
    temp_JSON_file: str,
    temp_database_folder: str,
) -> tuple[
    dict,
    pd.DataFrame,
    dict,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    list,
]:
    """ """
    # Load current UCC file to extract the column names
    dbs_merged_current = pd.read_csv(dbs_merged_current_file)
    logging.info(
        f"\nUCC version {dbs_merged_current_file} loaded (N={len(dbs_merged_current)})"
    )
    # Empty dataframe
    df_UCC_B = pd.DataFrame(dbs_merged_current[0:0])

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    # Extract GCs fnames
    names_gcs = list(df_GCs["Name"] + ", " + df_GCs["OName"])
    gcs_fnames_lst = []
    for gcs_fname in get_fnames(names_gcs):
        if gcs_fname[1] != "":
            gcs_fnames_lst.append(gcs_fname)
        else:
            gcs_fnames_lst.append([gcs_fname[0]])
    # To dictionary for faster search
    gcs_fnames = {}
    for i, names_list in enumerate(gcs_fnames_lst):
        for name in names_list:
            if name not in gcs_fnames:
                gcs_fnames[name] = i

    # Load current JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)
    new_DBs = []
    if os.path.isfile(temp_JSON_file):
        logging.info("\n=== Adding new DBs to the UCC ===\n")
        # Load new temp JSON file
        with open(temp_JSON_file) as f:
            temp_JSON = json.load(f)
        new_JSON = temp_JSON
        # Extract new DB's name(s)
        new_DBs = list(set(temp_JSON.keys()) - set(current_JSON.keys()))
    else:
        logging.info("\n=== Rebuilding the UCC ===\n")
        new_JSON = current_JSON

    # Order new_JSON by the values in the 'received' keys
    new_JSON = dict(sorted(new_JSON.items(), key=lambda item: item[1]["received"]))

    all_dbs_data = {}
    # Load existing DBs
    for DB in new_JSON:
        if DB in new_DBs:
            df_new = pd.read_csv(temp_database_folder + DB + ".csv")
        else:
            df_new = pd.read_csv(dbs_folder + DB + ".csv")
        logging.info(f"{DB} loaded (N={len(df_new)})")
        newDB_json = new_JSON[DB]
        all_dbs_data[DB] = [df_new, newDB_json]

    # Re order 'all_dbs_data' dictionary so that ["DIAS2002", "BICA2019"] are first
    excluded_dbs = ["DIAS2002", "BICA2019"]
    core_dbs = [_ for _ in all_dbs_data.keys() if _ not in excluded_dbs]
    all_dbs = excluded_dbs + core_dbs
    # core_dbs = [_ for _ in all_dbs_data.keys() if _ not in excluded_dbs + new_DBs]
    # all_dbs = excluded_dbs + core_dbs + new_DBs
    all_dbs_data = {k: all_dbs_data[k] for k in all_dbs}

    # flag_interactive == new_DBs
    return (
        new_JSON,
        df_GCs,
        gcs_fnames,
        all_dbs_data,
        dbs_merged_current,
        df_UCC_B,
        new_DBs,
    )


def check_new_DB(
    logging,
    df_GCs: pd.DataFrame,
    gcs_fnames: dict,
    new_DB: str,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    newDB_json: dict,
    flag_interactive: list,
) -> None:
    """Check new DB for required columns, bad characters in names, wrong VDBH/BH naming,
    and close GCs.
    """
    # Extract required columns
    read_cols = [newDB_json["names"]]
    for _, v in newDB_json["pos"].items():
        if isinstance(v, list):
            read_cols += v
        else:
            read_cols.append(v)
    # Parameters (and their uncertainties are dictionaries)
    for entry in ("pars", "e_pars"):
        for _, pdict in newDB_json[entry].items():
            for _, v in pdict.items():
                if isinstance(v, list):
                    read_cols += v
                else:
                    read_cols.append(v)
    missing_cols = [col for col in read_cols if col not in df_new.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns in {new_DB}: {missing_cols}")

    # Check for bad characters in name column
    all_bad_names = []
    for new_cl in df_new[newDB_json["names"]]:
        if bool(re.search(r"[\[\]();*]", new_cl)):
            all_bad_names.append(new_cl)
    N_bad_names = len(all_bad_names)
    if N_bad_names > 0:
        logging.info(
            f"\nFound {N_bad_names} entries with bad characters in names '(, ), ;, *'"
        )
        if new_DB in flag_interactive:
            for new_cl in all_bad_names:
                logging.info(f"  {new_cl}")  # bad char found
            breakpoint()
        # raise ValueError("Resolve the above issues before moving on.")

    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    if vdberg_check(logging, newDB_json, df_new):
        if new_DB in flag_interactive:
            # if input("Move on? (y/n): ").lower() != "y":
            #     sys.exit()
            breakpoint()

    if "RA" in newDB_json["pos"].keys():
        # Check for OCs very close to each other in the new DB
        if close_OC_inner_check(logging, newDB_json, df_new):
            if new_DB in flag_interactive:
                # if input("Move on? (y/n): ").lower() != "y":
                #     sys.exit()
                breakpoint()

    if "GLON_" in df_new.keys():
        # Check for close GCs
        if GCs_check(logging, df_GCs, gcs_fnames, newDB_json, df_new, new_DB_fnames):
            if new_DB in flag_interactive:
                # if input("Move on? (y/n): ").lower() != "y":
                #     sys.exit()
                breakpoint()


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
    leven_rad: float = 0.05,
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
    cls_dist = cdist(coords_new, coords_UCC) * 60

    col_1 = dbs_merged_new["fnames"]
    col_2 = new_DB_fnames
    ID_call = "UCC"

    return close_OC_check(
        logging, cls_dist, db_matches, col_1, col_2, ID_call, leven_rad, sep
    )


def close_OC_check(
    logging,
    cls_dist,
    db_matches,
    col_1,
    col_2,
    ID_call: str,
    leven_rad: float,
    sep: str,
    rad_dup: float = 10,
    N_max: int = 50,
):
    """ """
    idxs = np.arange(0, len(col_1))
    all_dups, dups_list = [], []
    for i, cl_d in enumerate(cls_dist):
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
        for idx in i_sort[:N_max]:
            i, cl_name, N_inner_dups, dups, dist, L_ratios = all_dups[idx]
            logging.info(
                f"{i:<6} {cl_name.strip():<15} (N={N_inner_dups}) --> "
                + f"{';'.join([_.strip() for _ in dups]):<15} | d={';'.join(dist)}, L={';'.join(L_ratios)}"
            )
        if dups_found > N_max:
            logging.info(f"... (only first {N_max} entries shown)")

    return dups_flag


def GCs_check(
    logging,
    df_GCs: pd.DataFrame,
    gcs_fnames: dict,
    newDB_json: dict,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    search_rad: float = 30,
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
        # d_arcmin = np.rad2deg(angular_separation(glon_i, glat_i, l_gc, b_gc)) * 60
        d_arcmin = np.sqrt((glon_i - l_gc) ** 2 + (glat_i - b_gc) ** 2) * 60
        gc_found_flag, gc_idx, gc_match, gc_dist = False, None, None, None

        # Check names match first
        for fname in new_DB_fnames[idx]:
            if fname in gcs_fnames:
                gc_found_flag = True
                gc_idx = gcs_fnames[fname]
                gc_match = df_GCs["Name"][gc_idx]
                gc_dist = d_arcmin[gc_idx]
                break

        if gc_found_flag is False:
            j1 = np.argmin(d_arcmin)
            if d_arcmin[j1] < search_rad:
                gc_found_flag = True
                gc_idx = j1
                gc_match = df_GCs["Name"][j1]
                gc_dist = d_arcmin[j1]

        if gc_found_flag:
            GCs_found += 1
            gc_all.append(
                [
                    idx,
                    df_new.iloc[idx][newDB_json["names"]],
                    gc_idx,
                    gc_match,
                    gc_dist,
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


def get_fnames(names_all, sep: str = ","):
    """ """
    fnames = []
    for names in names_all:
        names_l = []
        names_s = str(names).split(sep)
        for name in names_s:
            name = name.strip()
            name = rename_standard(name)
            names_l.append(normalize_name(name))
        fnames.append(names_l)
    return fnames


def get_fnames_new_DB(
    logging, df_new: pd.DataFrame, newDB_json: dict
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
    new_DB_fnames = get_fnames(df_new[newDB_json["names"]])

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
        logging.info(f"\nFound {len(duplicates)} duplicate fnames in new DB entries:")
        for name, idxs in duplicates.items():
            logging.info(f"'{name}' found in {len(idxs)} entries --> {sorted(idxs)}")
        breakpoint()
        sys.exit(0)

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
    df_UCC_B: pd.DataFrame, new_DB_fnames: list[list[str]], sep: str = ";"
) -> list[int | None]:
    """
    Get cluster matches for the new DB being added to the UCC

    Parameters
    ----------
    df_UCC_B : pd.DataFrame
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
    for idx, fnames in enumerate(df_UCC_B["fnames"]):
        for fname in fnames.split(sep):
            fname_to_idx[fname] = idx

    db_matches = []
    for cl_fnames in new_DB_fnames:
        found = next((fname_to_idx[f] for f in cl_fnames if f in fname_to_idx), None)
        db_matches.append(found)

    return db_matches


def check_new_DB_vs_UCC(
    logging,
    new_DB,
    df_UCC_B,
    df_new,
    new_DB_fnames,
    db_matches,
    flag_interactive: list,
) -> None:
    """
    - Checks for duplicate entries between the new database and the UCC.
    - Checks for OCs very close to each other between the new database and the UCC.
    - Checks positions and flags for attention if required.
    """
    # Check all fnames in the new DB against all fnames in the UCC
    # logging.info("\nChecking uniqueness of fnames")
    if fnames_check_UCC_new_DB(logging, new_DB, df_UCC_B, new_DB_fnames):
        logging.info("\nResolve the above issues before moving on")
        breakpoint()

    if "GLON_" in df_new.keys():
        # Check for OCs very close to other OCs in the UCC
        if close_OC_UCC_check(logging, df_UCC_B, new_DB_fnames, db_matches, df_new):
            if new_DB in flag_interactive:
                breakpoint()

    # Check positions and flag for attention if required
    if positions_check(
        logging,
        df_UCC_B,
        df_new,
        new_DB_fnames,
        db_matches,
    ):
        if new_DB in flag_interactive:
            breakpoint()


def positions_check(
    logging,
    df_UCC_B: pd.DataFrame,
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
        df_UCC_B["GLON"].to_numpy(),
        df_UCC_B["GLAT"].to_numpy(),
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
        N_max = 50
        for i, fname, d_arcmin in ocs_attention[:N_max]:
            logging.info(f"{i:<6} {fname:<20} {d_arcmin:<5.0f}")
        if len(ocs_attention) > N_max:
            logging.info(f"... (only first {N_max} entries shown)")

    return attention_flag


def fnames_check_UCC_new_DB(
    logging,
    new_DB: str,
    df_UCC_B: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    sep: str = ";",
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
    fname_ucc_map = defaultdict(list)
    for i, fnames in enumerate(df_UCC_B["fnames"]):
        for fname in fnames.split(sep):
            fname_ucc_map[fname].append(i)

    # Find matches between UCC fnames and new_DB_fnames
    fnames_ucc_idxs = {}
    for k, fnames in enumerate(new_DB_fnames):
        fnames_ucc_idxs[k] = []
        for fname in fnames:
            # Check if the filename exists in df_fnames
            if fname in fname_ucc_map:
                # 'fname_ucc_map[fname]' will always contain a single element
                f_ucc_idx = fname_ucc_map[fname][0]
                fnames_ucc_idxs[k].append(f_ucc_idx)

    # Check if any new entry has more than one entry in the UCC associated to it
    bad_entries = []
    for k, v in fnames_ucc_idxs.items():
        if len(list(set(v))) > 1:
            bad_entries.append([k, v])

    # Check if the 'fnames_ucc_idxs' dictionary contains repeated elements
    locations = defaultdict(set)
    for key, lst in fnames_ucc_idxs.items():
        for item in lst:
            locations[item].add(key)
    # Items that occur in more than one key
    duplicates = {item: keys for item, keys in locations.items() if len(keys) > 1}

    dup_flag = False

    if bad_entries:
        dup_flag = True
        logging.info(
            f"\nFound {len(bad_entries)} entries in {new_DB} with duplicated "
            + "fnames in the combined DB:"
        )
        for k, v in bad_entries:
            new_db_entries = f"({k}) {', '.join(new_DB_fnames[k])}"
            ucc_entries = ", ".join(
                [
                    f"({_}, {df_UCC_B.iloc[_]['DB']}) {df_UCC_B.iloc[_]['fnames']}"
                    for _ in v
                ]
            )
            logging.info(f"{new_db_entries} --> {ucc_entries}")

    if duplicates:
        dup_flag = True
        logging.info(
            f"\nFound {len(duplicates)} entries in {new_DB} with combined "
            + "fnames in the combined DB:"
        )
        for k, v in duplicates.items():
            new_db_entries = "; ".join([", ".join(new_DB_fnames[_]) for _ in v])
            u_dbs = ",".join(list(set(df_UCC_B["DB"][k].split(";"))))
            u_fnames = ",".join(list(set(df_UCC_B["fnames"][k].split(";"))))
            logging.info(f"{tuple(v)} {new_db_entries} --> ({k}) {u_dbs}: {u_fnames}")

    return dup_flag


def add_fpars_col(newDB_json, df_new, max_chars=7):
    """ """
    all_pars = {}
    # For each parameter
    for par_general in fpars_order:
        if par_general in newDB_json["pars"]:
            par = newDB_json["pars"][par_general]

            flag_mult = ""
            # err_v = , ""

            # Select the first item (in case there are more than one)
            par_format, par_db_col = [[k, v] for k, v in par.items()][0]
            if len(par.items()) > 1:
                flag_mult = "*"

            # Select the first column (in case there are more than one)
            if isinstance(par_db_col, list):
                par_db_col = par_db_col[0]
                flag_mult = "*"

            # Column values as strings
            s = df_new[par_db_col].astype(str)
            # Check for multiple values in the selected column
            mask = s.str.contains(r"[;,]", na=False)
            # Flag if multiple values are found
            flag_mult = "*" if mask.any() else flag_mult

            # Extract parameter value
            par_v = (
                s.str.split(r"[;,]", n=1)  # Split elements by ',;'
                .str[0]  # If more than one value in column, select the first one
                .str.slice(0, max_chars)  # Trim
                .str.replace(" ", "", regex=False)  # Some cleaning
            )

            # err_list = DBs_json[db]["e_pars"]['e_' + par_general]
            # if err_p:
            #     err_v = f"±{err_p:.4f}"
            # err_v = "-"
            # final_v = transf_par(par_format, par_v) + f" ± {err_v}" + flag_mult
            final_v = transf_par(par_format, par_v, flag_mult)
        else:
            # If the parameter is not present in the new DB, fill with '-'
            final_v = ["--"] * len(df_new)

        all_pars[par_general] = final_v

    df_new["fund_pars"] = [",".join(vals) for vals in zip(*all_pars.values())]

    return df_new


def transf_par(par_n2, par_v, flag_mult):
    """
    In case of multiple values the first one is always selected.
    """

    def fmt(x, expr):
        if pd.isna(x) or str(x).strip() == "" or str(x) == "nan":
            return "--"
        return expr(x)

    # Ages as [Myr]
    if par_n2 == "loga":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{10 ** float(_) / 1e6:.0f}{flag_mult}")
        ).tolist()
    elif par_n2 == "amyr":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.0f}{flag_mult}")
        ).tolist()
    elif par_n2 == "agyr":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{1000 * float(_):.0f}{flag_mult}")
        ).tolist()

    # Metallicity as [Fe/H]
    elif par_n2 == "feh":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.3f}{flag_mult}")
        ).tolist()
    elif par_n2 == "z":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{np.log(float(_) / c_z_sun):.3f}{flag_mult}")
        ).tolist()

    # Mass as [Msun]
    elif par_n2 == "mass":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.0f}{flag_mult}")
        ).tolist()
    elif par_n2 == "logm":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{10 ** float(_):.0f}{flag_mult}")
        ).tolist()

    # Binary fraction
    elif par_n2 == "bf":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.2f}{flag_mult}")
        ).tolist()

    # Blue stragglers
    elif par_n2 == "bs_f":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.2f}{flag_mult}")
        ).tolist()
    elif par_n2 == "bs_n":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.0f}{flag_mult}")
        ).tolist()

    # Extinction
    elif par_n2 == "Av":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.2f}{flag_mult}")
        ).tolist()
    elif par_n2 == "Ag":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{c_Ag * float(_):.2f}{flag_mult}")
        ).tolist()
    elif par_n2 == "Ebv":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{c_Ebv * float(_):.2f}{flag_mult}")
        ).tolist()
    elif par_n2 == "Evi":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{c_Evi * float(_):.2f}{flag_mult}")
        ).tolist()

    # Differential extinction
    elif par_n2 == "dAv":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.2f}{flag_mult}")
        ).tolist()

    # Distance in [Kpc]
    elif par_n2 == "dkpc":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_):.2f}{flag_mult}")
        ).tolist()
    elif par_n2 == "dpc":
        return par_v.apply(
            lambda x: fmt(x, lambda _: f"{float(_) / 1000:.2f}{flag_mult}")
        ).tolist()
    elif par_n2 == "dm":
        return par_v.apply(
            lambda x: fmt(
                x, lambda _: f"{10 ** (0.2 * float(_)) / 1000:.2f}{flag_mult}"
            )
        ).tolist()

    else:
        raise ValueError(f"Parameter format '{par_n2}' not recognized.")


def combine_UCC_new_DB(
    logging,
    new_DB: str,
    newDB_json: dict,
    df_UCC_B: pd.DataFrame,
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
    # Convert df_new to records once (O(N) operation) to avoid Series creation overhead
    new_db_rows = df_new.to_dict(orient="records")
    ucc_dict_rows = df_UCC_B.to_dict(orient="records")

    new_db_dict = {_: [] for _ in df_UCC_B.keys()}

    # Iterate over the list of dictionaries
    for i_new_cl, row_n in enumerate(new_db_rows):
        # row_n is now a standard dict, but key access syntax remains valid
        oc_names = rename_standard(str(row_n[newDB_json["names"]]))

        row_ucc = {}
        ucc_index = db_matches[i_new_cl]
        if ucc_index is not None:
            row_ucc = ucc_dict_rows[ucc_index]

        # Extract coordinates from new DB
        ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = extract_new_DB_coords(
            new_DB, row_n, newDB_json["pos"]
        )

        new_db_dict = updt_new_DB(
            new_DB,
            new_db_dict,
            str(i_new_cl),
            row_n["fund_pars"],
            new_DB_fnames[i_new_cl],
            oc_names,
            row_ucc,
            ra_n,
            dec_n,
            lon_n,
            lat_n,
            plx_n,
            pmra_n,
            pmde_n,
        )

    N_new = db_matches.count(None)
    N_updt = len(db_matches) - N_new

    # Drop OCs from the UCC that are present in the new DB
    # Remove 'None' entries first from the indexes list
    idx_rm_comb_db = [_ for _ in db_matches if _ is not None]
    df_UCC_no_new = df_UCC_B.drop(list(df_UCC_B.index[idx_rm_comb_db]))
    df_UCC_no_new.reset_index(drop=True, inplace=True)

    # Final UCC with the new DB incorporated
    dbs_merged_new = pd.concat(
        [df_UCC_no_new, pd.DataFrame(new_db_dict)], ignore_index=True
    )
    logging.info(f"\nNew entries: {N_new}, Updated entries: {N_updt}")

    return dbs_merged_new


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

        # Use spaces not underscores
        name = name.replace("_", " ")

        new_names_rename.append(name)
    oc_names = sep_out.join(new_names_rename)

    return oc_names


def extract_new_DB_coords(DB_ID, row_n, pos_cols):
    """ """
    ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = [np.nan] * 7

    if "RA" in pos_cols:
        ra_n, dec_n = row_n[pos_cols["RA"]], row_n[pos_cols["DEC"]]
        lon_n, lat_n = row_n["GLON_"], row_n["GLAT_"]

    # Don't use these for PMs or Plx values
    if DB_ID in ("DIAS2002", "LOKTIN2017") or "KHARCHENKO" in DB_ID:
        return ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n

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
    i_fund_pars: str,
    fnames_new_cl: list[str],
    oc_names: str,
    row_ucc: dict,
    ra_n,
    dec_n,
    lon_n,
    lat_n,
    plx_n,
    pmra_n,
    pmde_n,
    sep: str = ";",
):
    """ """

    def expand(val: str, n: int, pad=None):
        """Repeat val n times joined by sep"""
        if n == 1 or pad is None:
            return sep.join([val] * n)
        return sep.join([val] + [pad] * (n - 1))

    N = len(fnames_new_cl)
    fnames_joined = sep.join(fnames_new_cl)

    # Normalize new DB entries (always done the same way)
    new_DB_exp = expand(new_DB, N)
    i_new_cl_exp = expand(i_new_cl, N)
    # Use this padding for a smaller fund_pars column. Will be removed when sorting
    fund_exp = expand(i_fund_pars, N, "-")

    # OC not present in UCC
    if len(row_ucc) == 0:
        DB_ID = new_DB_exp
        DB_i = i_new_cl_exp
        names = oc_names
        fnames = fnames_joined
        fund_pars = fund_exp
    # OC already present in UCC
    else:
        DB_ID = row_ucc["DB"] + sep + new_DB_exp
        DB_i = row_ucc["DB_i"] + sep + i_new_cl_exp
        names = row_ucc["Names"] + sep + oc_names
        fnames = row_ucc["fnames"] + sep + fnames_joined
        fund_pars = row_ucc["fund_pars"] + sep + fund_exp

        # Only update values if nan is in place, else keep the FIRST values stored
        cols = ("RA_ICRS", "DE_ICRS", "GLON", "GLAT", "Plx", "pmRA", "pmDE")
        new_vals = [ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n]
        old_vals = [row_ucc[c] for c in cols]
        vals = [nv if np.isnan(ov) else ov for ov, nv in zip(old_vals, new_vals)]
        ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = vals

        # Manually fixed centers
        for fname in fnames_new_cl:
            if fname in selected_center_coords:
                ra_n, dec_n = selected_center_coords[fname]
                lon_n, lat_n = radec2lonlat(ra_n, dec_n)
                break

    # Append to output dictionary
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
    new_db_dict["fund_pars"].append(fund_pars)

    return new_db_dict


def sort_year_importance(new_JSON, df_UCC_B):
    """
    The fnames are first sorted by year, then by importance ('naming_order' variable,
    mostly for old clusters), and finally by the order in which they are stored in
    the DBs.
    """

    def order_by_name(items: np.ndarray) -> np.ndarray:
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
        return np.argsort(idxs, kind="stable")

    def rm_dups(strings: list | np.ndarray) -> tuple[list[str], list[int]]:
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

    new_rows = {
        "Names": df_UCC_B["Names"].to_list(),
        "fnames": df_UCC_B["fnames"].to_list(),
        "DB": df_UCC_B["DB"].to_list(),
        "DB_i": df_UCC_B["DB_i"].to_list(),
        "fund_pars": df_UCC_B["fund_pars"].to_list(),
    }
    for k, row in df_UCC_B.iterrows():
        dbs = row["DB"].split(";")
        if len(dbs) > 1:
            names = np.array(row["Names"].split(";"))
            fnames = np.array(row["fnames"].split(";"))
            DB = np.array(row["DB"].split(";"))
            DB_i = np.array(row["DB_i"].split(";"))
            fund_pars = np.array(row["fund_pars"].split(";"))

            if len(set(dbs)) > 1:
                # Sort by dates
                ryears = []
                for db in dbs:
                    ryears.append(new_JSON[db]["received"])
                isort = np.argsort(ryears, kind="stable")
                names = names[isort]
                fnames = fnames[isort]
                DB = DB[isort]
                DB_i = DB_i[isort]
                fund_pars = fund_pars[isort]

            # Sort by importance
            i_new = order_by_name(fnames)
            fnames = [fnames[_] for _ in i_new]
            names = [
                str(names[_]) for _ in i_new
            ]  # str() added to avoid Pyright warning

            # Remove duplicates elements without affecting the order
            fnames, rm_idx = rm_dups(fnames)
            if rm_idx:
                names = [_ for i, _ in enumerate(names) if i not in rm_idx]
            # Remove duplicates from DB, DB_i, and fund_pars
            DB, rm_idx = rm_dups(DB)
            if rm_idx:
                DB_i = [_ for i, _ in enumerate(DB_i) if i not in rm_idx]
                fund_pars = [_ for i, _ in enumerate(fund_pars) if i not in rm_idx]

            # Update corresponding row in df_UCC_B
            new_rows["Names"][k] = ";".join(names)
            new_rows["fnames"][k] = ";".join(fnames)
            new_rows["DB"][k] = ";".join(DB)
            new_rows["DB_i"][k] = ";".join(DB_i)
            new_rows["fund_pars"][k] = ";".join(fund_pars)

    df_UCC_B["Names"] = new_rows["Names"]
    df_UCC_B["fnames"] = new_rows["fnames"]
    df_UCC_B["DB"] = new_rows["DB"]
    df_UCC_B["DB_i"] = new_rows["DB_i"]
    df_UCC_B["fund_pars"] = new_rows["fund_pars"]

    return df_UCC_B


def duplicates_fnames_check(logging, df_UCC_B: pd.DataFrame, sep: str = ";") -> bool:
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
    for i, fnames in enumerate(df_UCC_B["fnames"]):
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
    new_json_dict: dict,
    temp_JSON_file: str,
    all_dbs_data: dict,
    temp_database_folder: str,
) -> None:
    """ """
    # Update JSON file with all the DBs and store the new DB in place
    if os.path.isfile(temp_JSON_file):
        # Save to (temp) JSON file
        with open(temp_JSON_file, "w") as f:
            json.dump(new_json_dict, f, indent=2)

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

    # Generate '.gz' compressed file for the old B file and archive it
    df_OLD_B = pd.read_csv(dbs_merged_current_file)
    now_time = pd.Timestamp.now().strftime("%y%m%d%H")
    archived_B_file = (
        data_folder
        + "ucc_archived_nogit/"
        + merged_dbs_file.replace(".csv", f"_{now_time}.csv.gz")
    )
    save_df_UCC(logging, df_OLD_B, archived_B_file, "fnames", "gzip")
    # Remove old B csv file
    os.remove(dbs_merged_current_file)
    logging.info(dbs_merged_current_file + " --> " + archived_B_file)
    logging.info("NOTICE: UCC_cat_B archiving NOT APPLIED")

    # Move new B file into place
    ucc_temp = temp_folder + merged_dbs_file
    os.rename(ucc_temp, dbs_merged_current_file)
    logging.info(ucc_temp + " --> " + dbs_merged_current_file)


if __name__ == "__main__":
    main()
