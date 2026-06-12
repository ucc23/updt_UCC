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
    final_fnames_compare,
    get_fnames,
    load_BC_cats,
    logger,
    normalize_name,
    radec2lonlat,
    rename_standard,
    save_df_UCC,
)
from .variables import (
    DB_coords_hierarchy,
    GCs_cat,
    all_OC_names,
    c_Ag,
    c_Ebprp,
    c_Ebv,
    c_Egrp,
    c_Evi,
    c_z_sun,
    data_folder,
    dbs_folder,
    fpars_order,
    merged_dbs_file,
    name_DBs_json,
    naming_order,
    selected_centers_f,
    temp_folder,
)

# Pre-compute to use in extract_new_DB_coords()
_NANS_7 = (np.nan,) * 7
mid_hierarchy_val = int(0.5 * max(DB_coords_hierarchy.values()))


def main():
    """
    Function to update the UCC (Unified Cluster Catalogue) with a new database.
    """
    logging = logger()

    skip_check_flag = False
    if input("\nSkip check for non new DBs? (y/n): ").lower() == "y":
        skip_check_flag = True

    # Generate paths and check for required folders and files
    temp_database_folder, temp_all_OC_names, df_UCC_B_path, temp_JSON_path = (
        get_paths_check_paths(logging)
    )

    (
        new_JSON,
        df_GCs,
        gcs_fnames,
        selected_center_coords,
        all_dbs_data,
        df_UCC_B_old,
        df_UCC_B,
        flag_interactive,
        all_names,
        all_names_dict,
    ) = load_data(logging, df_UCC_B_path, temp_JSON_path, temp_database_folder)

    for new_DB, (df_new, newDB_json) in all_dbs_data.items():
        logging.info("\n" + "-" * 40 + f"\nAdding {new_DB} ({len(df_new)}) to the UCC")

        # Determine the level of checking to perform for this DB
        if new_DB in flag_interactive:
            # For new DBS, perform all checks and stop if problems are found
            flag_check_stop = "check_stop"
        else:
            if skip_check_flag is False:
                flag_check_stop = "check_no_stop"  # Never used
            else:
                # For non-new DBs, skip all checks if skip_check_flag=True
                flag_check_stop = "no_check"

        # Add inner columns with galactic coordinates if possible
        if "RA" in newDB_json["pos"].keys():
            ra_col = newDB_json["pos"]["RA"]
            dec_col = newDB_json["pos"]["DEC"]
            # Wrap negative RA values to [0, 360)
            msk = df_new[ra_col] < 0
            if msk.any():
                logging.info(
                    f"\nFound {msk.sum()} entries with negative RA values. Wrapping to [0, 360)"
                )
                df_new.loc[msk, ra_col] = df_new.loc[msk, ra_col] + 360
            df_new["GLON_"], df_new["GLAT_"] = radec2lonlat(
                df_new[ra_col].to_numpy(),
                df_new[dec_col].to_numpy(),
            )

        # Extract fnames from the new DB
        new_DB_fnames = get_fnames(df_new[newDB_json["names"]])

        # Check the DB for basic requirements
        if flag_check_stop != "no_check":
            basic_new_DB_checks(
                logging,
                df_GCs,
                gcs_fnames,
                new_DB,
                df_new,
                new_DB_fnames,
                newDB_json,
                flag_check_stop,
            )

        # Standardize names in the DB
        new_DB_fnames = get_canonical_fnames(
            df_new, newDB_json, all_names_dict, new_DB_fnames
        )

        # Check that no fname is repeated across entries in the new DB
        check_new_DB_fnames(logging, df_new, newDB_json, new_DB_fnames)

        # Check uniqueness of fnames (fnames in new DB vs fnames in UCC so far)
        fnames_check_UCC_new_DB(
            logging, df_UCC_B, all_names_dict, new_DB_fnames, df_new
        )

        # Match the new DB with the UCC
        db_matches = get_matches_new_DB(df_UCC_B, new_DB_fnames)

        if flag_check_stop != "no_check":
            # Report and check new entries (if any)
            check_new_entries(logging, new_DB_fnames, db_matches, flag_check_stop)

            # Check positions in the DB vs the UCC
            check_positions(
                logging,
                df_UCC_B,
                df_new,
                new_DB_fnames,
                db_matches,
                flag_check_stop,
            )

        # Add fundamental parameters column
        df_new = add_fpars_col(newDB_json, df_new)

        # Combine the new DB with the UCC
        df_UCC_B = combine_UCC_new_DB(
            logging,
            selected_center_coords,
            new_DB,
            newDB_json,
            df_UCC_B,
            df_new,
            new_DB_fnames,
            db_matches,
        )

    logging.info("\n\n\n===================================================")
    logging.info("Merging of DBs completed\n")
    logging.info(f"Old B file: {len(df_UCC_B_old)}")
    logging.info(f"New B file: {len(df_UCC_B)}\n")

    df_UCC_B = sort_year_importance(new_JSON, df_UCC_B)

    # Add medians and STDDEVs of fundamental parameters
    df_UCC_B = add_fpars_stats(logging, df_UCC_B)

    # Generate new all_names and df_UCC_B files
    all_names_new, df_UCC_B_new = gen_new_files(df_UCC_B)

    # Mandatory sanity check
    final_sanity_check(logging, all_names, df_UCC_B)

    # Check for differences between old and new files, update if any are found
    update_final_files(
        logging,
        temp_all_OC_names,
        df_UCC_B_path,
        all_names,
        all_names_new,
        df_UCC_B_old,
        df_UCC_B_new,
    )

    #
    move_files(
        logging,
        df_UCC_B_path,
        df_UCC_B_old,
        new_JSON,
        temp_JSON_path,
        all_dbs_data,
        temp_all_OC_names,
        temp_database_folder,
    )


def get_paths_check_paths(
    logging,
) -> tuple[str, str, str, str]:
    """ """
    # Path to the temporary all_OC_names file
    temp_all_OC_names = temp_folder + all_OC_names

    # Path to the current DBs merged file
    df_UCC_B_path = data_folder + merged_dbs_file

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
    temp_JSON_path = temp_folder + name_DBs_json

    return temp_database_folder, temp_all_OC_names, df_UCC_B_path, temp_JSON_path


def load_data(
    logging,
    df_UCC_B_path: str,
    temp_JSON_path: str,
    temp_database_folder: str,
) -> tuple[
    dict,
    pd.DataFrame,
    dict,
    dict,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    list,
    pd.DataFrame,
    dict,
]:
    """ """
    df_UCC_B_old = load_BC_cats("B", df_UCC_B_path)

    logging.info(f"\nUCC version {df_UCC_B_path} loaded (N={len(df_UCC_B_old)})")
    # Empty dataframe
    df_UCC_B = pd.DataFrame(df_UCC_B_old[0:0])
    # Remove _median and _stddev columns from df_UCC_B. Also remove the "fname" column,
    # it will be generated at the end of the script
    fpars_order_lst = list(fpars_order)
    cols_to_remove = (
        [_ + "_median" for _ in fpars_order_lst]
        + [_ + "_stddev" for _ in fpars_order_lst]
        + ["fname"]
    )
    df_UCC_B = df_UCC_B.drop(columns=cols_to_remove)
    # Add empty ["fnames", "Names"] columns to df_UCC_B
    df_UCC_B["fnames"] = ""
    df_UCC_B["Names"] = ""

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    # Load selected centers coordinates
    selected_center_coords = (
        pd.read_csv(selected_centers_f)  # , keep_default_na=False)
        .set_index("fname")
        .apply(lambda r: r.tolist(), axis=1)
        .to_dict()
    )

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
    if os.path.isfile(temp_JSON_path):
        logging.info("\n=== Adding new DBs to the UCC ===\n")
        # Load new temp JSON file
        with open(temp_JSON_path) as f:
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
    for DB, vals in new_JSON.items():
        if vals["data_cmmts"] == "comments":
            logging.info(f"Skipping {DB} (marked as 'comments' in JSON)")
            continue
        new_db_flag = ""
        if DB in new_DBs:
            new_db_flag = "(new DB)"
            df_new = pd.read_csv(temp_database_folder + DB + ".csv")
        else:
            df_new = pd.read_csv(dbs_folder + DB + ".csv")

        # Check that the all columns specified in the JSON file are present in the DB
        json_cols = [
            vals["names"],
            *vals["pos"].values(),
            *(v for d in vals["pars"].values() for v in d.values()),
            *(v for d in vals["e_pars"].values() for v in d.values()),
        ]
        # Force flat
        cols = []
        for x in json_cols:
            if isinstance(x, (list, tuple, set)):
                cols.extend(x)
            else:
                cols.append(x)
        missing = [c for c in cols if c not in df_new.columns]
        if missing:
            raise ValueError(f"Missing columns in {DB}: {missing}")

        logging.info(f"{DB} loaded (N={len(df_new)}) {new_db_flag}")
        all_dbs_data[DB] = [df_new, new_JSON[DB]]

    all_names, all_names_dict = handle_all_names(logging)

    # flag_interactive == new_DBs
    return (
        new_JSON,
        df_GCs,
        gcs_fnames,
        selected_center_coords,
        all_dbs_data,
        df_UCC_B_old,
        df_UCC_B,
        new_DBs,
        all_names,
        all_names_dict,
    )


def handle_all_names(logging, sep=";") -> tuple[pd.DataFrame, dict]:
    """ """
    # Create a dictionary with all names and their canonical fnames and Names
    all_names = pd.read_csv(data_folder + all_OC_names)

    # Check for invalid characters in fnames (anything other than letters, numbers, and ';')
    bad_idx = all_names.index[
        all_names["fnames"].str.contains(r"[^A-Za-z0-9;]", regex=True, na=False)
    ]
    if len(bad_idx) > 0:
        logging.info(
            f"\nFound {len(bad_idx)} entries with invalid characters in"
            + f" 'fnames' (only letters, numbers, and '{sep}' allowed):"
        )
        for idx in bad_idx:
            logging.info(f"{idx}: {all_names.loc[idx, 'fnames']}")
        sys.exit(1)

    # Check if sorted
    all_n_fame0 = [_.split(sep)[0] for _ in all_names["fnames"]]
    for i in range(len(all_n_fame0) - 1):
        if all_n_fame0[i] > all_n_fame0[i + 1]:
            logging.info(
                f"Not sorted at index {i}: {all_n_fame0[i]!r} > {all_n_fame0[i + 1]!r}"
            )
            sys.exit(1)

    # Create all_names_dict mapping each alias to its canonical fname and Names
    all_names_dict, all_fnames_list, all_names_list = {}, [], []
    for i, row in enumerate(all_names.itertuples(index=False)):
        fnames = str(row.fnames).split(sep)
        names = str(row.Names).split(sep)

        all_fnames_list.append(fnames)
        all_names_list.append(row.Names)

        if any(fname.strip() == "" for fname in fnames):
            raise ValueError(f"\nEmpty fname found at index {i}: {row.fnames}")
        if any(name.strip() == "" for name in names):
            raise ValueError(f"\nEmpty Name found at index {i}: {row.Names}")

        if len(fnames) != len(names):
            raise ValueError(f"\nMismatch in number of fnames and Names at index {i}")

        n_canonical = names[0]
        f_canonical = fnames[0]
        for alias in fnames:
            if alias in all_names_dict:
                raise ValueError(f"Duplicate alias '{alias}' found at index {i}")
            all_names_dict[alias] = {"fnames": f_canonical, "Names": n_canonical}

    # Check for duplicates
    duplicates = check_duplicated_fnames(all_fnames_list)
    if duplicates:
        logging.info(f"\nFound {len(duplicates)} duplicate fnames in 'all_names':")
        for name, idxs in duplicates.items():
            logging.info(f"{sorted(idxs)} --> '{name}'")
        sys.exit(1)

    # Check that all fnames are equivalent to the normalized names
    all_fnames_norm = get_fnames(all_names_list, sep=sep)
    if (all_fnames_list == all_fnames_norm) is False:
        raise ValueError("Mismatch between all_fnames_list and new_DB_fnames")

    return all_names, all_names_dict


def basic_new_DB_checks(
    logging,
    df_GCs: pd.DataFrame,
    gcs_fnames: dict,
    new_DB: str,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    newDB_json: dict,
    flag_check_stop: str,
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
        if bool(re.search(r"[\[\]();*/\\]", new_cl)):
            all_bad_names.append(new_cl)
    N_bad_names = len(all_bad_names)
    if N_bad_names > 0:
        logging.info(
            f"\nFound {N_bad_names} entries with bad characters in names '(),[],;,*,/'"
        )
        for new_cl in all_bad_names:
            logging.info(f"  {new_cl}")  # bad char found
        if flag_check_stop == "check_stop":
            breakpoint()

    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    if vdberg_check(logging, newDB_json, df_new):
        if flag_check_stop == "check_stop":
            breakpoint()

    if "RA" in newDB_json["pos"].keys():
        # Check for OCs very close to each other in the new DB
        if close_OC_inner_check(logging, newDB_json, df_new, flag_check_stop):
            if flag_check_stop == "check_stop":
                breakpoint()

    # Check for close GCs
    if GCs_check(logging, df_GCs, gcs_fnames, newDB_json, df_new, new_DB_fnames):
        if flag_check_stop == "check_stop":
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
            logging.info(f"{i}, {new_cl.strip()} --> {name_check} (P={sm_ratio:.2f})")

    return vdb_flag


def close_OC_inner_check(
    logging,
    newDB_json: dict,
    df_new: pd.DataFrame,
    flag_check_stop: str,
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
        logging,
        dist,
        db_matches,
        col_1,
        col_2,
        ID_call,
        leven_rad,
        sep,
        flag_check_stop,
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
    flag_check_stop: str,
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
        start = 0
        total = len(i_sort)
        while start < total:
            end = min(start + N_max, total)
            for idx in i_sort[start:end]:
                i, cl_name, N_inner_dups, dups, dist, L_ratios = all_dups[idx]
                dups_unq = ";".join(
                    {
                        item.strip()
                        for s in dups
                        for item in s.split(";")
                        if item.strip()
                    }
                )
                logging.info(
                    f"{i:<6} {cl_name.strip():<30} (N={N_inner_dups}) --> "
                    + f"{dups_unq:<30} | d={';'.join(dist)}, L={';'.join(L_ratios)}"
                )

            if flag_check_stop != "check_stop":
                logging.info(f"... (only first {N_max} entries shown)")
                break

            start = end
            if start < total:
                ans = input(f"Show {N_max} more? (y/n): ").strip().lower()
                if ans != "y":
                    break

    return dups_flag


def GCs_check(
    logging,
    df_GCs: pd.DataFrame,
    gcs_fnames: dict,
    newDB_json: dict,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    search_rad: float = 60,
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
    gc_all = {}

    # Check names match first
    for idx, all_fnames in enumerate(new_DB_fnames):
        for fname in all_fnames:
            if fname in gcs_fnames:
                gc_idx = gcs_fnames[fname]
                gc_match = ",".join(
                    [df_GCs["Name"][gc_idx].strip(), df_GCs["OName"][gc_idx].strip()]
                )
                gc_all[idx] = [
                    df_new.iloc[idx][newDB_json["names"]],
                    gc_idx,
                    gc_match,
                    0.0,  # 0 distance
                ]
                break

    if "GLON_" in df_new.keys():
        # Extract arrays once
        glon = np.array(df_new["GLON_"])
        glat = np.array(df_new["GLAT_"])
        l_gc = np.array(df_GCs["GLON"])
        b_gc = np.array(df_GCs["GLAT"])

        # Broadcasted distance matrix (N_new × N_gc)
        d_arcmin = (
            np.sqrt(
                (glon[:, None] - l_gc[None, :]) ** 2
                + (glat[:, None] - b_gc[None, :]) ** 2
            )
            * 60.0
        )

        # Nearest GC index for each new object
        gc_idx = np.argmin(d_arcmin, axis=1)
        gc_dist = d_arcmin[np.arange(d_arcmin.shape[0]), gc_idx]

        # Apply radius mask
        mask = gc_dist < search_rad
        matched_idx = np.where(mask)[0]

        # Pre-extract needed columns to avoid repeated iloc calls
        names_new = df_new[newDB_json["names"]].to_numpy()
        names_gc = df_GCs["Name"].str.strip().to_numpy()
        onames_gc = df_GCs["OName"].str.strip().to_numpy()

        for i in matched_idx:
            # Only add if not already matched by name
            if i not in gc_all:
                gc_all[i] = [
                    names_new[i],
                    gc_idx[i],
                    f"{names_gc[gc_idx[i]]},{onames_gc[gc_idx[i]]}",
                    gc_dist[i],
                ]

    gc_flag = False
    if gc_all:
        GCs_found = len(gc_all)
        # Convert dictionary → list of tuples including the key
        gc_list = [
            (idx, row_id, idx_gc, df_gcs_name, float(d_arcmin))
            for idx, (row_id, idx_gc, df_gcs_name, d_arcmin) in gc_all.items()
        ]
        # Sort by distance
        gc_list.sort(key=lambda x: x[-1])
        gc_flag = True
        logging.info(f"\nFound {GCs_found} probable GCs:")
        for idx, row_id, idx_gc, df_gcs_name, d_arcmin in gc_list:
            row_id = row_id.strip()
            logging.info(
                f"{idx:<6} {row_id:<40} --> {idx_gc:<6} {df_gcs_name.strip():<25}"
                + f"d={round(d_arcmin, 2)}"
            )

    return gc_flag


def get_canonical_fnames(
    df_new: pd.DataFrame,
    newDB_json: dict,
    all_names_dict: dict,
    new_DB_fnames: list[list[str]],
) -> list[list[str]]:
    """
    Extract and standardize all names in the new catalogue

    Parameters
    ----------
    df_new : pd.DataFrame
        DataFrame of the new catalogue.
    newDB_json : dict
        Dictionary with the parameters of the new catalogue.
    all_names_dict : dict
        Dictionary mapping all known names to their canonical fnames and Names.
    new_DB_fnames : list
        List of lists, where each inner list contains the standardized names for
        each cluster in the new catalogue.

    Returns
    -------
    list
        List of lists, where each inner list contains the standardized names for
        each cluster in the new catalogue.
    """
    # Use 'all_names_dict' to add canonical names to the list of names for each cluster
    # in the new DB
    col_name = newDB_json["names"]
    names_list = df_new[col_name].tolist()
    for i, DB_fnames in enumerate(new_DB_fnames):
        # Find the first match in all_names_dict for any of the fnames in DB_fnames
        match = next(
            (all_names_dict[f] for f in DB_fnames if f in all_names_dict), None
        )
        if match is None:
            # No match found, this is a new OC with no known aliases
            continue
        # Add the canonical fname at the beginning of the list
        new_DB_fnames[i] = [match["fnames"]] + DB_fnames
        # Add the canonical name to the main Names column. This is required for
        # consistency regarding the number of elements per fnames/Names
        names_list[i] = f"{match['Names']}, {names_list[i]}"

    # Write back to the DataFrame in a single vectorized step
    df_new[col_name] = names_list

    return new_DB_fnames


def check_duplicated_fnames(fnames_list: list[list[str]]) -> dict:
    """Check that no fname is repeated across entries"""
    seen, duplicates = {}, {}
    for i, fnames in enumerate(fnames_list):
        # use set(fnames) so duplicates inside the same fnames are ignored
        for fname in set(fnames):
            if fname in seen and seen[fname] != i:
                duplicates.setdefault(fname, {seen[fname]}).add(i)
            else:
                seen[fname] = i
    return duplicates


def check_new_DB_fnames(
    logging, df_new: pd.DataFrame, newDB_json: dict, new_DB_fnames: list[list[str]]
):
    """Check that no fname is repeated across entries in the new DB"""
    duplicates = check_duplicated_fnames(new_DB_fnames)

    if duplicates:
        logging.info(f"\nFound {len(duplicates)} duplicate fnames within the DB:")
        for name, idxs in duplicates.items():
            idxs_s = sorted(idxs)
            logging.info(
                f"{idxs_s} {' | '.join([str(df_new.loc[_, newDB_json['names']]) for _ in idxs_s])}"
                + f" --> '{name}'"
            )
        breakpoint()
        sys.exit(1)


def fnames_check_UCC_new_DB(
    logging,
    df_UCC_B: pd.DataFrame,
    all_names_dict: dict,
    new_DB_fnames: list[list[str]],
    df_new: pd.DataFrame,
    sep: str = ";",
) -> None:
    """
    Check that no fname associated to each entry in the new DB is listed in more than
    one entry in the UCC.
    """

    new_fnames_dup = []
    for i, new_fnames in enumerate(new_DB_fnames):
        c_fname = []
        for new_fname in set(new_fnames):
            if new_fname in all_names_dict:
                c_fname.append(all_names_dict[new_fname]["fnames"])
        if len(set(c_fname)) > 1:
            new_fnames_dup.append(i)

    if new_fnames_dup:
        N = len(new_fnames_dup)
        txt = "y" if N == 1 else "ies"
        logging.info(f"\nFound {N} entr{txt} with multiple fnames in 'all_names':")
        for k in new_fnames_dup:
            logging.info(f"{df_new.iloc[k]['Name']}")
        breakpoint()
        sys.exit(1)

    return

    # Assign an index to each fname in the UCC for O(1) lookup
    fname_ucc_map = {
        fname: i
        for i, fnames in enumerate(df_UCC_B["fnames"])
        for fname in fnames.split(sep)
    }

    # Index matching: Set comprehensions ensure uniqueness at extraction time O(1)
    fnames_ucc_idxs = {}
    for k, fnames in enumerate(new_DB_fnames):
        matched_idxs = {fname_ucc_map[f] for f in fnames if f in fname_ucc_map}
        if matched_idxs:
            fnames_ucc_idxs[k] = matched_idxs

    # Detect bad entries
    bad_entries = {k: list(v) for k, v in fnames_ucc_idxs.items() if len(v) > 1}

    # Duplicates: Invert mapping efficiently
    locations = defaultdict(list)
    for k, matched_idxs in fnames_ucc_idxs.items():
        for idx in matched_idxs:
            locations[idx].append(k)
    duplicates = {item: keys for item, keys in locations.items() if len(keys) > 1}

    dup_flag = bool(bad_entries or duplicates)

    # Logging: Extract raw numpy arrays to bypass extreme pd.DataFrame.iloc latency
    if dup_flag:
        dbs_arr = df_UCC_B["DB"].values
        fnames_arr = df_UCC_B["fnames"].values

        if bad_entries:
            logging.info(
                f"\nFound {len(bad_entries)} entries in new_DB with duplicated "
                "fnames in the combined DB:"
            )
            for k, v in bad_entries.items():
                new_db_entries = f"({k}) {', '.join(new_DB_fnames[k])}"
                ucc_entries = ", ".join(
                    [f"({_}, {dbs_arr[_]}) {fnames_arr[_]}" for _ in v]
                )
                logging.info(f"{new_db_entries} --> {ucc_entries}")

        if duplicates:
            logging.info(
                f"\nFound {len(duplicates) * 2} entries in new_DB with combined "
                "fnames in the combined DB:"
            )
            for k, v in duplicates.items():
                new_db_entries = "; ".join([", ".join(new_DB_fnames[_]) for _ in v])
                u_dbs = ",".join(set(str(dbs_arr[k]).split(";")))
                u_fnames = ",".join(set(str(fnames_arr[k]).split(";")))
                logging.info(
                    f"{tuple(v)} {new_db_entries} --> ({k}) {u_dbs}: {u_fnames}"
                )

    if dup_flag:
        logging.info("\nResolve the above issues before moving on")
        breakpoint()
        sys.exit(1)


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
    # Build lookup dictionary from fname -> UCC index
    fname_to_idx = {}
    for idx, fnames in enumerate(df_UCC_B["fnames"]):
        for fname in fnames.split(sep):
            fname_to_idx[fname] = idx

    # Check if each fname in the new DB is included in the UCC (so far)
    db_matches = []
    for cl_fnames in new_DB_fnames:
        found = next((fname_to_idx[f] for f in cl_fnames if f in fname_to_idx), None)
        db_matches.append(found)

    return db_matches


def check_new_entries(logging, new_DB_fnames, db_matches, flag_check_stop: str):
    """ """
    N_new = db_matches.count(None)
    if N_new > 0:
        logging.info(f"\nFound {N_new} new entries")
        N_max = 50
        count = 0
        for i, idx in enumerate(db_matches):
            if idx is None:
                logging.info(f"{i}    {new_DB_fnames[i][0]}")
                count += 1
                if count % N_max == 0:
                    logging.info(f"... (shown {count} entries)")
                    if flag_check_stop != "check_stop":
                        break
                    ans = input("Show 50 more? (y/n) ").strip().lower()
                    if ans != "y":
                        break
        if count % N_max != 0:
            logging.info(f"... (shown {count} entries in total)")
        if flag_check_stop == "check_stop":
            breakpoint()
    else:
        logging.info("\nNo new entries found")


def check_positions(
    logging,
    df_UCC_B,
    df_new,
    new_DB_fnames,
    db_matches,
    flag_check_stop: str,
) -> None:
    """
    - Checks for OCs very close to each other between the new database and the UCC.
    - Checks positions and flags for attention if required.
    """
    if "GLON_" in df_new.keys():
        # Check for OCs very close to other OCs in the UCC
        if check_new_DB_UCC_positions(
            logging, df_UCC_B, new_DB_fnames, db_matches, df_new, flag_check_stop
        ):
            if flag_check_stop == "check_stop":
                breakpoint()

    # Check positions for matched entries and flag for attention if required
    if check_matched_entries(
        logging,
        df_UCC_B,
        df_new,
        new_DB_fnames,
        db_matches,
    ):
        if flag_check_stop == "check_stop":
            breakpoint()


def check_new_DB_UCC_positions(
    logging,
    df_UCC_B: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
    df_new: pd.DataFrame,
    flag_check_stop: str,
    leven_rad: float = 0.05,
    sep: str = ";",
) -> bool:
    """
    Compute a complete distance matrix between all objects in the new catalog and
    the reference catalog. The close_OC_check() function filters out any matches
    already established in db_matches.

    For objects falling within a specific spatial radius, compute a normalized
    Levenshtein ratio against the UCC designations to find renaming discrepancies
    or alternate designations for the same physical OC.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC_B : pd.DataFrame
        DataFrame of the UCC.
    new_DB_fnames : list
        List of lists, where each inner list contains the
        standardized names for each cluster in the new catalogue.
    db_matches : list
        List of indexes into the UCC pointing to each entry in the
        new DB.

    Returns
    -------
    bool
        Boolean flag indicating if probable UCC duplicates were found.
    """
    coords_new = np.array([df_new["GLON_"], df_new["GLAT_"]]).T
    coords_UCC = np.array([df_UCC_B["GLON"], df_UCC_B["GLAT"]]).T

    # Find the distances to all clusters, for all clusters (in arcmin)
    cls_dist = cdist(coords_new, coords_UCC) * 60

    col_1 = df_UCC_B["fnames"]
    col_2 = new_DB_fnames
    ID_call = "UCC"

    return close_OC_check(
        logging,
        cls_dist,
        db_matches,
        col_1,
        col_2,
        ID_call,
        leven_rad,
        sep,
        flag_check_stop,
    )


def check_matched_entries(
    logging,
    df_UCC_B: pd.DataFrame,
    df_new,
    new_DB_fnames,
    db_matches,
    rad_dup: float = 30.0,
) -> bool:
    """
    Checks the positions of clusters in the new database against those in the UCC.
    Iterates strictly over pre-matched index pairs.

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
            # Wrapped longitude difference (degrees)
            dlon = abs(df_UCC_glon[j] - df_new_glon[i])
            dlon = min(dlon, 360.0 - dlon)
            # Latitude difference (degrees)
            dlat = df_UCC_glat[j] - df_new_glat[i]
            # Angular separation (arcmin)
            d_arcmin = np.sqrt(dlon**2 + dlat**2) * 60.0

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
        # Order by distance
        ocs_attention = sorted(ocs_attention, key=lambda x: x[2], reverse=True)
        for i, fname, d_arcmin in ocs_attention[:N_max]:
            logging.info(f"{i:<6} {fname:<20} {d_arcmin:<5.0f}")
        if len(ocs_attention) > N_max:
            logging.info(f"... (only first {N_max} entries shown)")

    return attention_flag


def add_fpars_col(newDB_json, df_new, max_chars=7):
    """ """
    all_pars = {}
    N_rows = len(df_new)

    # For each parameter
    for par_general in fpars_order:
        if par_general in newDB_json["pars"]:
            par = newDB_json["pars"][par_general]

            flag_mult = [""] * N_rows
            # err_v = , ""

            # If the parameter has more than one value associated in different columns
            # with different formats. I.e: {'dkpc': 'dmode', 'dpc': 'Dist'}
            if len(par.items()) > 1:
                # Add '*' indicator for multiple values (total - 1)
                flag_mult = ["*" * (len(par.items()) - 1)] * N_rows

            # Select the first item (in case there are more than one)
            par_format, par_db_col = [[k, v] for k, v in par.items()][0]

            # If the parameters has more than one value associated in different columns
            # with equivalent formats, i.e: ['[Fe/H]_1', '[Fe/H]_2'], flag and select
            # the first one
            if isinstance(par_db_col, list):
                flag_mult = ["*" * (len(par_db_col) - 1)] * N_rows
                par_db_col = par_db_col[0]

            # Column value(s) as strings
            s = df_new[par_db_col].astype(str)

            # If the entire column has already been flagged for multiple values,
            # skip this step
            if "*" not in flag_mult[0]:
                # Check for multiple values in the selected column
                mask = s.str.contains(r"[;,]", na=False)
                # Flag if multiple values are found
                if mask.any():
                    # Number of extra values = number of delimiters
                    n_extra = s.str.count(r"[;,]")
                    #
                    flag_mult = np.array(flag_mult, dtype=object)
                    flag_mult[mask] = n_extra[mask].apply(
                        lambda n: "*" * int(n) if n > 0 else ""
                    )
                    flag_mult = flag_mult.tolist()

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
            # If the parameter is not present in the new DB, fill with nan
            final_v = [np.nan] * len(df_new)

        all_pars[par_general] = final_v

    for col in all_pars.keys():
        df_new[col] = all_pars[col]

    return df_new


def transf_par(par_format, par_v, flag_mult):
    """
    In case of multiple values the first one is always selected.
    """

    def valid(x):
        return not (pd.isna(x) or str(x).strip() in ("", "nan"))

    def apply(expr):
        out = []
        for x, f in zip(par_v, flag_mult):
            if not valid(x):
                out.append(np.nan)
            else:
                out.append(f"{expr(float(x))}{f}")
        return out

    fmt_map = {
        "loga": lambda x: f"{10**x / 1e6:.0f}",
        "amyr": lambda x: f"{x:.0f}",
        "agyr": lambda x: f"{1000 * x:.0f}",
        "feh": lambda x: f"{x:.3f}",
        "z": lambda x: f"{np.log10(x / c_z_sun):.3f}",
        "mass": lambda x: f"{x:.0f}",
        "logm": lambda x: f"{10**x:.0f}",
        "bf": lambda x: f"{x:.2f}",
        "bs_f": lambda x: f"{x:.2f}",
        "bs_n": lambda x: f"{x:.0f}",
        "av": lambda x: f"{x:.2f}",
        "ag": lambda x: f"{c_Ag * x:.2f}",
        "ebv": lambda x: f"{c_Ebv * x:.2f}",
        "evi": lambda x: f"{c_Evi * x:.2f}",
        "ebprp": lambda x: f"{c_Ebprp * x:.2f}",
        "egrp": lambda x: f"{c_Egrp * x:.2f}",
        "dav": lambda x: f"{x:.2f}",
        "dkpc": lambda x: f"{x:.2f}",
        "dpc": lambda x: f"{x / 1000:.2f}",
        "dm": lambda x: f"{10 ** (0.2 * (x + 5)) / 1000:.2f}",
    }

    return apply(fmt_map[par_format])


def combine_UCC_new_DB(
    logging,
    selected_center_coords: dict,
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
    # Fast dict mapping for only the needed UCC rows
    valid_ucc_idxs = [_ for _ in db_matches if _ is not None]
    ucc_dict_map = df_UCC_B.iloc[valid_ucc_idxs].to_dict(orient="index")

    # Fast generator for new DB rows
    new_db_cols = df_new.columns.tolist()
    new_db_rows = (
        dict(zip(new_db_cols, row)) for row in df_new.itertuples(index=False, name=None)
    )

    new_db_dict = {_: [] for _ in df_UCC_B.columns}

    # Iterate over each cluster in the new DB (row_n is a dict)
    for i_new_cl, row_n in enumerate(new_db_rows):
        # Standardized some naming conventions
        oc_names = rename_standard(str(row_n[newDB_json["names"]]))

        ucc_index = db_matches[i_new_cl]
        row_ucc = ucc_dict_map.get(ucc_index, {})

        # Extract coordinates from new DB
        ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n, DB_used = (
            extract_new_DB_coords(
                new_DB,
                row_n,
                newDB_json["pos"],
                new_DB_fnames[i_new_cl],
                row_ucc,
                selected_center_coords,
            )
        )

        new_db_dict = updt_new_DB(
            new_DB,
            new_db_dict,
            str(i_new_cl),
            row_n,
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
            DB_used,
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


def extract_new_DB_coords(
    DB_ID: str,
    row_n: dict,
    pos_cols: dict,
    fnames_new_cl: list[str],
    row_ucc: dict,
    selected_center_coords: dict,
    cols=("RA_ICRS", "DE_ICRS", "GLON", "GLAT", "Plx", "pmRA", "pmDE"),
) -> tuple[float, float, float, float, float, float, float, str]:
    """ """
    # Extract manually fixed centers (if any)
    ra_f, dec_f, lon_f, lat_f, plx_f, pmra_f, pmde_f = _NANS_7
    for fname in fnames_new_cl:
        if fname in selected_center_coords:
            ra_f, dec_f, plx_f, pmra_f, pmde_f, _ = selected_center_coords[fname]
            if not np.isnan(ra_f):
                lon_f, lat_f = radec2lonlat(ra_f, dec_f)
            break

    # Extract values from new DB
    ra_db, dec_db, lon_db, lat_db, plx_db, pmra_db, pmde_db = _NANS_7
    if "RA" in pos_cols:
        ra_db, dec_db = row_n[pos_cols["RA"]], row_n[pos_cols["DEC"]]
        lon_db, lat_db = row_n["GLON_"], row_n["GLAT_"]
    if "plx" in pos_cols:
        plx_db = row_n[pos_cols["plx"]]
    if "pmra" in pos_cols:
        pmra_db = row_n[pos_cols["pmra"]]
    if "pmde" in pos_cols:
        pmde_db = row_n[pos_cols["pmde"]]

    def updt_nans(primary, fallback):
        return [p if not np.isnan(p) else s for p, s in zip(primary, fallback)]

    # Always prefer manually fixed values over DB values
    new_DB_vals = updt_nans(
        [ra_f, dec_f, lon_f, lat_f, plx_f, pmra_f, pmde_f],
        [ra_db, dec_db, lon_db, lat_db, plx_db, pmra_db, pmde_db],
    )
    # Default: use new DB values
    DB_used = DB_ID
    ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = new_DB_vals

    # If this entry already has assigned values in the UCC
    if row_ucc:
        # Extract DB used for main coords up to this point
        db_prev = row_ucc["DB_coords_used"]
        # Get the hierarchies of both DBs: the one used and this one.
        z_prev = DB_coords_hierarchy.get(db_prev, mid_hierarchy_val)
        z_new = DB_coords_hierarchy.get(DB_ID, mid_hierarchy_val)

        ucc_vals = [row_ucc[c] for c in cols]
        if z_prev < z_new:
            # If the previous DB has higher priority (smaller z), keep UCC values
            DB_used = db_prev
            ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = updt_nans(
                ucc_vals, new_DB_vals
            )
        else:
            # If the z values are larger or equal, update with new DB values
            if all(np.isnan(new_DB_vals)):
                # If the new DB has no values, keep UCC values
                DB_used = db_prev
                ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = ucc_vals
            else:
                # Update with new DB values
                ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n = updt_nans(
                    new_DB_vals, ucc_vals
                )

    return ra_n, dec_n, lon_n, lat_n, plx_n, pmra_n, pmde_n, DB_used


def updt_new_DB(
    new_DB: str,
    new_db_dict: dict,
    i_new_cl: str,
    row_newdb: dict,
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
    DB_used,
    sep: str = ";",
):
    """ """
    N = len(fnames_new_cl)
    fnames_joined = sep.join(fnames_new_cl)

    # Expand to match number of fnames, purely for consistency
    new_DB_exp = sep.join([new_DB] * N)
    i_new_cl_exp = sep.join([i_new_cl] * N)

    fund_pars_exp = {_: "" for _ in fpars_order}
    for par in fpars_order:
        if par in row_newdb:
            fund_pars_exp[par] = sep.join([str(row_newdb[par])] * N)
        else:
            fund_pars_exp[par] = sep.join([str(np.nan)] * N)

    fund_pars = fund_pars_exp
    if len(row_ucc) == 0:
        # OC not present in UCC
        DB_ID = new_DB_exp
        DB_i = i_new_cl_exp
        names = oc_names
        fnames = fnames_joined
    else:
        # OC already present in UCC
        DB_ID = row_ucc["DB"] + sep + new_DB_exp
        DB_i = row_ucc["DB_i"] + sep + i_new_cl_exp
        names = row_ucc["Names"] + sep + oc_names
        fnames = row_ucc["fnames"] + sep + fnames_joined
        for par in fpars_order:
            fund_pars[par] = row_ucc[par] + sep + fund_pars_exp[par]

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
    new_db_dict["DB_coords_used"].append(DB_used)
    for par in fpars_order:
        new_db_dict[par].append(fund_pars[par])

    return new_db_dict


def sort_year_importance(new_JSON: dict, df_UCC_B: pd.DataFrame) -> pd.DataFrame:
    """
    The fnames are first sorted by year, then by importance ('naming_order' variable,
    mostly for old clusters), and finally by the order in which they are stored in
    the DBs.

    :param new_JSON: dict
        Dictionary containing the information of all the databases, including the
        'received' year for each DB.
    :param df_UCC_B: pd.DataFrame
        DataFrame of the UCC before sorting.

    :return: pd.DataFrame
        DataFrame of the UCC after sorting the fnames by year and importance.
    """

    # Pre-compute hash map for years to avoid repeated dictionary lookups
    db_years = {db: info["received"] for db, info in new_JSON.items()}

    # Only process rows that actually contain multiple entries
    mask = df_UCC_B["DB"].str.contains(";", na=False)
    if not mask.any():
        return df_UCC_B

    def get_importance(item: str) -> int | float:
        """Optimized importance map (stops at first match)"""
        for j, p in enumerate(naming_order):
            if item.startswith(p):
                return j
        return np.inf

    # Extract only the necessary subset into native Python lists
    cols = ["Names", "fnames", "DB", "DB_i"] + list(fpars_order)
    subset = df_UCC_B.loc[mask, cols].to_dict(orient="list")

    n_fpars = len(fpars_order)

    for k in range(len(subset["DB"])):
        # Split native strings directly
        dbs = subset["DB"][k].split(";")
        names = subset["Names"][k].split(";")
        fnames = subset["fnames"][k].split(";")
        db_i = subset["DB_i"][k].split(";")
        fpars = [subset[par][k].split(";") for par in fpars_order]

        if len(set(dbs)) > 1:
            # Sort all properties by year (Stable sort using Timsort)
            years = [db_years.get(db, 0) for db in dbs]
            i_year = sorted(range(len(years)), key=lambda x: years[x])

            names = [names[i] for i in i_year]
            fnames = [fnames[i] for i in i_year]
            dbs = [dbs[i] for i in i_year]
            db_i = [db_i[i] for i in i_year]
            fpars = [[par_list[i] for i in i_year] for par_list in fpars]

        # Sort names and fnames by importance
        importance = [get_importance(f) for f in fnames]
        i_imp = sorted(range(len(fnames)), key=lambda x: importance[x])

        fnames = [fnames[i] for i in i_imp]
        names = [names[i] for i in i_imp]

        # Single-pass deduplication for fnames/names
        seen_fnames = set()
        u_fnames, u_names = [], []
        for f, n in zip(fnames, names):
            if f not in seen_fnames:
                seen_fnames.add(f)
                u_fnames.append(f)
                u_names.append(n)
        # Single-pass deduplication for DB properties
        seen_db = set()
        u_dbs, u_db_i = [], []
        u_fpars = [[] for _ in range(n_fpars)]

        for i, db in enumerate(dbs):
            if db not in seen_db:
                seen_db.add(db)
                u_dbs.append(db)
                u_db_i.append(db_i[i])
                for j in range(n_fpars):
                    u_fpars[j].append(fpars[j][i])

        # Overwrite the subset lists with rejoined strings
        subset["Names"][k] = ";".join(u_names)
        subset["fnames"][k] = ";".join(u_fnames)
        subset["DB"][k] = ";".join(u_dbs)
        subset["DB_i"][k] = ";".join(u_db_i)
        for j, par in enumerate(fpars_order):
            subset[par][k] = ";".join(u_fpars[j])

    # Inject the processed subset back into the main DataFrame
    for col in cols:
        df_UCC_B.loc[mask, col] = subset[col]

    return df_UCC_B


def add_fpars_stats(logging, df):
    """For each column in 'fpars_order', calculate the median and stddev"""

    for par in fpars_order:
        # String cleaning and splitting
        temp_df = (
            df[par]
            .str.replace("*", "", regex=False)
            .str.split(";", expand=True)
            .apply(pd.to_numeric, errors="coerce")
        )

        if par == "blue_str":
            # Filter fractional values
            # Identify rows where ALL non-NaN values are fractional
            all_frac = ((temp_df > 0) & (temp_df < 1)).all(axis=1)
            if all_frac.sum() > 0:
                logging.info(
                    f"\nFound {all_frac.sum()} entries with all BSS values in (0, 1) range"
                )
            # Condition to keep non-fractional values
            cond = (temp_df == 0) | (temp_df >= 1)
            # apply filter only to rows that are NOT fully fractional
            temp_df = temp_df.where(cond | all_frac.to_frame().reindex_like(temp_df))

        # Statistics (NaNs automatically ignored, ddof=1 by default for std)
        med = temp_df.median(axis=1)
        std = temp_df.std(axis=1)

        dec = 0 if par in {"age", "mass", "blue_str"} else 4
        df[f"{par}_median"] = med.round(dec)
        df[f"{par}_stddev"] = std.round(dec)

    return df


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


def ra_dec_check(logging, df_UCC_B):
    """Check that all RA and Dec values are within valid ranges and not NaN."""

    ra = df_UCC_B["RA_ICRS"]
    dec = df_UCC_B["DE_ICRS"]

    invalid_ra = df_UCC_B[ra.isna() | (ra < 0) | (ra >= 360)]
    invalid_dec = df_UCC_B[dec.isna() | (dec < -90) | (dec > 90)]

    if not invalid_ra.empty:
        logging.info(f"\nFound {len(invalid_ra)} entries with invalid RA values:")
        for idx, row in invalid_ra.iterrows():
            logging.info(
                f"{idx} ({row['fnames'].split(';')[0]}) --> RA={row['RA_ICRS']}"
            )

    if not invalid_dec.empty:
        logging.info(f"\nFound {len(invalid_dec)} entries with invalid Dec values:")
        for idx, row in invalid_dec.iterrows():
            logging.info(
                f"{idx} ({row['fnames'].split(';')[0]}) --> Dec={row['DE_ICRS']}"
            )

    return not invalid_ra.empty or not invalid_dec.empty


def gen_new_files(df_UCC_B):
    """ """
    # Generate new all_OC_names file
    all_names_new = pd.DataFrame(df_UCC_B[["fnames", "Names"]])

    # Generate new df_UCC_B file
    # Generate 'fname' column with the first name in the list of 'fnames'
    df_UCC_B["fname"] = df_UCC_B["fnames"].str.split(";").str[0]
    # Move 'fname' to the first column position
    df_UCC_B.insert(0, "fname", df_UCC_B.pop("fname"))
    # Drop ["fnames", "Names"] columns
    df_UCC_B_new = df_UCC_B.drop(columns=["fnames", "Names"])

    return all_names_new, df_UCC_B_new


def final_sanity_check(logging, all_names, df_UCC_B):
    """ """
    # Check every individual fname for duplicates
    exit_flag = duplicates_fnames_check(logging, df_UCC_B)
    if exit_flag:
        logging.info("\nERROR: duplicated entries found in 'fnames' column. Fix this!")
        breakpoint()
        sys.exit(1)

    # Check that (RA, DEC) ranges are valid
    exit_flag = ra_dec_check(logging, df_UCC_B)
    if exit_flag:
        logging.info(
            "\nERROR: entries were found with missing (RA, DEC) values. Fix this!"
        )
        breakpoint()
        sys.exit(1)
    #
    final_fnames_compare(logging, all_names["fnames"], df_UCC_B["fnames"])


def update_final_files(
    logging,
    temp_all_OC_names,
    df_UCC_B_path,
    all_names,
    all_names_new,
    df_UCC_B_old,
    df_UCC_B_new,
):
    """
    Compare changes in all_names and df_UCC_B files and generate diff files
    (open with Meld or similar)
    """
    diff_found = diff_between_dfs(
        logging, "all_names", all_names, all_names_new, order_col="fnames"
    )
    if diff_found:
        save_df_UCC(logging, all_names_new, temp_all_OC_names, order_col="fnames")

    diff_found = diff_between_dfs(logging, "B cat", df_UCC_B_old, df_UCC_B_new)
    if diff_found:
        file_path = df_UCC_B_path.replace(data_folder, temp_folder)
        save_df_UCC(logging, df_UCC_B_new, file_path)


def move_files(
    logging,
    df_UCC_B_path: str,
    df_UCC_B_old: pd.DataFrame,
    new_json_dict: dict,
    temp_JSON_path: str,
    all_dbs_data: dict,
    temp_all_OC_names: str,
    temp_database_folder: str,
) -> None:
    """ """
    if input("\nMove files to their final paths? (y/n): ").lower() != "y":
        logging.info("\nFiles not moved.")
        return

    # Update JSON file with all the DBs and store the new DB in place
    if os.path.isfile(temp_JSON_path):
        # Save to (temp) JSON file
        with open(temp_JSON_path, "w") as f:
            json.dump(new_json_dict, f, indent=2)
        # Move JSON file from temp folder to final folder
        os.rename(temp_JSON_path, name_DBs_json)
        logging.info(temp_JSON_path + " --> " + name_DBs_json)

    for new_DB in all_dbs_data.keys():
        # Move new DB file
        new_DB_file = new_DB + ".csv"
        db_temp = temp_database_folder + new_DB_file
        if os.path.isfile(db_temp):
            db_stored = dbs_folder + new_DB_file
            os.rename(db_temp, db_stored)
            logging.info(db_temp + " --> " + db_stored)

    ucc_temp = temp_folder + merged_dbs_file
    if os.path.isfile(ucc_temp):
        # Generate '.gz' compressed file for the old B file and archive it
        now_time = pd.Timestamp.now().strftime("%y%m%d%H")
        archived_B_file = (
            data_folder
            + "ucc_archived_nogit/"
            + merged_dbs_file.replace(".csv", f"_{now_time}.csv.gz")
        )
        save_df_UCC(logging, df_UCC_B_old, archived_B_file, compression="gzip")
        # Remove old B csv file
        os.remove(df_UCC_B_path)
        logging.info(df_UCC_B_path + " --> " + archived_B_file)

        # Move new B file into place
        os.rename(ucc_temp, df_UCC_B_path)
        logging.info(ucc_temp + " --> " + df_UCC_B_path)

    # Move new all_OC_names
    if os.path.isfile(temp_all_OC_names):
        os.rename(temp_all_OC_names, data_folder + all_OC_names)
        logging.info(temp_all_OC_names + " --> " + data_folder + all_OC_names)


if __name__ == "__main__":
    main()
