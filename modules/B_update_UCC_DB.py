import csv
import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.spatial.distance import cdist

from .utils import (
    diff_between_dfs,
    get_fnames,
    load_BC_cats,
    normalize_name,
    prune_archive,
    radec2lonlat,
    rename_standard,
    save_df_UCC,
)
from .variables import (
    DB_coords_hierarchy,
    GCs_cat,
    UCC_cat_B_in,
    UCC_cat_B_out,
    UCC_cmmts_folder,
    UCC_excl_folder,
    all_OC_names,
    archive_folder_path,
    data_folder,
    dbs_folder,
    fmt_to_UCC,
    fpars_order,
    max_dist_suggest,
    name_DBs_json,
    naming_order,
    naming_order_exceptions,
    temp_folder,
)

# Pre-compute to use in extract_new_DB_coords()
_NANS_7 = (np.nan,) * 7
mid_hierarchy_val = int(0.5 * max(DB_coords_hierarchy.values()))


def main():
    """
    Function to update the UCC (Unified Cluster Catalogue) with a new database.
    """

    # Generate paths and check for required folders and files
    temp_database_folder, temp_all_OC_names, df_UCC_B_path, temp_JSON_path = (
        get_paths_check_paths()
    )

    (
        new_JSON,
        df_GCs,
        gcs_fnames,
        selected_center_coords,
        all_dbs_data,
        df_UCC_B_old,
        df_UCC_B_new,
        new_DBs_lst,
        all_names_old,
        all_names_dict,
    ) = load_data(df_UCC_B_path, temp_JSON_path, temp_database_folder)

    skip_check_flag = ""
    if input("\nSkip check for non new DBs? (y/n): ").lower() == "y":
        skip_check_flag = "y"
    else:
        skip_check_flag = input(
            "\nSelect old DBs (comma separated) to stop, or\n"
            "leave empty to check all never stopping: "
        )
        skip_check_flag = [_.strip().lower() for _ in skip_check_flag.split(",")]

    # Positions and earliest DB of the UCC entries, used to report conflicts
    ucc_info = get_ucc_entries_info(df_UCC_B_old, new_JSON)

    for new_DB, (df_new, newDB_json) in all_dbs_data.items():
        print("\n" + "-" * 40 + f"\nAdding {new_DB} ({len(df_new)}) to the UCC")

        # Determine the level of checking to perform for this DB
        if new_DB in new_DBs_lst:
            # For new DBS, always perform checks and stop if problems are found
            flag_check_stop = "check_stop"
        else:
            if skip_check_flag == "y":
                # For non-new DBs, skip all checks
                flag_check_stop = "no_check"
            else:
                # For non-new DBs
                if new_DB.lower() in skip_check_flag:
                    # check and stop
                    flag_check_stop = "check_stop"
                else:
                    flag_check_stop = "check_no_stop"  # Never used

        # Add inner columns with galactic coordinates if possible
        if "RA" in newDB_json["pos"]:
            ra_col = newDB_json["pos"]["RA"]
            dec_col = newDB_json["pos"]["DEC"]
            # Wrap negative RA values to [0, 360)
            msk = df_new[ra_col] < 0
            if msk.any():
                print(
                    f"\nFound {msk.sum()} entries with negative RA values. "
                    "Wrapping to [0, 360)"
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
                df_GCs,
                gcs_fnames,
                new_DB,
                df_new,
                new_DB_fnames,
                newDB_json,
                flag_check_stop,
            )

        # Check that no fname is repeated across entries in the DB's own names
        check_own_DB_fnames(df_new, newDB_json, new_DB_fnames)

        # Check that no entry points to multiple canonical entries in all_names
        all_names_canonical_check(
            new_DB, df_new, newDB_json, all_names_dict, new_DB_fnames, ucc_info
        )

        # Standardize names in the DB (keep a copy of the original fnames)
        orig_DB_fnames = [list(_) for _ in new_DB_fnames]
        new_DB_fnames = get_canonical_fnames(
            df_new, newDB_json, all_names_dict, new_DB_fnames
        )

        # Check that no canonical fname is shared across entries in the new DB
        check_new_DB_fnames(df_new, newDB_json, new_DB_fnames, orig_DB_fnames)

        # Check uniqueness of fnames (fnames in new DB vs fnames in UCC so far)
        fnames_check_UCC_new_DB(df_UCC_B_new, new_DB_fnames, orig_DB_fnames, df_new)

        # Match the new DB with the UCC
        db_matches = get_matches_new_DB(df_UCC_B_new, new_DB_fnames)

        if flag_check_stop != "no_check":
            # Report and check new entries (if any)
            check_new_entries(
                new_DB_fnames, orig_DB_fnames, db_matches, flag_check_stop, df_new.index
            )

            # Check positions in the DB vs the UCC
            check_positions(
                df_UCC_B_new,
                df_new,
                new_DB_fnames,
                orig_DB_fnames,
                db_matches,
                flag_check_stop,
            )

        # Add fundamental parameters column
        df_new = add_fpars_col(newDB_json, df_new)

        # Compute the F-score for the new DB entries
        db_fscore = get_db_fscore(new_JSON, new_DB, df_new)

        # Combine the new DB with the UCC
        df_UCC_B_new = combine_UCC_new_DB(
            selected_center_coords,
            new_DB,
            newDB_json,
            df_UCC_B_new,
            df_new,
            new_DB_fnames,
            db_matches,
            db_fscore,
        )

    print("\n\n\n===================================================")
    print("Merging of DBs completed\n")
    print(f"Old B file: {len(df_UCC_B_old)}")
    print(f"New B file: {len(df_UCC_B_new)}\n")

    df_UCC_B_new = sort_year_importance(new_JSON, df_UCC_B_new)

    # Add medians and STDDEVs of fundamental parameters
    df_UCC_B_new = add_fpars_stats(df_UCC_B_new)

    # Add Type column
    dbs_dfs = {DB: df_db for DB, (df_db, _) in all_dbs_data.items()}
    df_UCC_B_new = add_types(new_JSON, dbs_dfs, df_UCC_B_new)

    # Sanity check
    sanity_check(all_names_old, df_UCC_B_new)

    # Generate new all_names and final df_UCC_B dataframes
    all_names_new, df_UCC_B_final = generate_dfs(df_UCC_B_new)

    # Add any missing fnames to all_names_new (if any), using all_names_dict as
    # reference
    all_names_final = add_lost_fnames(all_names_dict, all_names_new)

    # Add the names of the entries excluded as duplicates to their main entries
    all_names_final = add_excl_duplicate_names(new_JSON, all_names_final)

    # Check that the names in all_names are 1 to 1 with the names in the DBs
    db_paths = {
        DB: (temp_database_folder if DB in new_DBs_lst else dbs_folder) + DB + ".csv"
        for DB in new_JSON
    }
    all_names_DBs_check(all_names_final, new_JSON, db_paths)

    # Compare final 'fnames' column with the old one to check for unexpected changes
    final_fnames_compare(all_names_old["fnames"], all_names_final["fnames"])

    # Check for differences between old and final files, update if any are found
    update_final_files(
        temp_all_OC_names,
        df_UCC_B_path,
        all_names_old,
        all_names_final,
        df_UCC_B_old,
        df_UCC_B_final,
    )

    move_files(
        df_UCC_B_path,
        df_UCC_B_old,
        new_JSON,
        temp_JSON_path,
        all_dbs_data,
        temp_all_OC_names,
        temp_database_folder,
    )


def get_paths_check_paths() -> tuple[str, str, str, str]:
    """
    Generate paths for temporary files and check for required folders and files.
    """
    # Path to the temporary all_OC_names file
    temp_all_OC_names = temp_folder + all_OC_names

    # Path to the current DBs merged file
    df_UCC_B_path = data_folder + UCC_cat_B_out

    # If file exists, read and return it
    # last_version = get_last_version_UCC(data_folder)
    temp_merged_f = temp_folder + UCC_cat_B_out
    if os.path.isfile(temp_merged_f):
        print(f"WARNING: file {temp_merged_f} exists. Moving on will re-write it")
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
    """
    Load necessary data for updating the UCC.
    """
    df_UCC_B_old = load_BC_cats("B", df_UCC_B_path)
    print(f"\nUCC version {df_UCC_B_path} loaded (N={len(df_UCC_B_old)})")

    # Empty dataframe
    df_UCC_B_new = pd.DataFrame(df_UCC_B_old[0:0])
    # Remove _median and _stddev columns from df_UCC_B. Also remove the "fname" column,
    # it will be generated at the end of the script
    fpars_order_lst = list(fpars_order)
    cols_to_remove = (
        [_ + "_median" for _ in fpars_order_lst]
        + [_ + "_stddev" for _ in fpars_order_lst]
        + ["fname"]
    )
    df_UCC_B_new = df_UCC_B_new.drop(columns=cols_to_remove)
    # Add empty ["fnames", "Names"] columns to df_UCC_B_new
    df_UCC_B_new["fnames"] = ""
    df_UCC_B_new["Names"] = ""

    # Load all_names and all_names_dict
    all_names, all_names_dict = handle_all_names()

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    # Load selected centers coordinates
    selected_center_coords = (
        pd.read_csv(UCC_cat_B_in)  # , keep_default_na=False)
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

    # Load new temp JSON file
    if os.path.isfile(temp_JSON_path):
        print("\n=== Adding new DBs to the UCC ===\n")
        with open(temp_JSON_path) as f:
            temp_JSON = json.load(f)
        new_JSON = temp_JSON
        # Extract new DB's name(s) from the temp JSON file
        new_DBs = list(set(temp_JSON.keys()) - set(current_JSON.keys()))
    else:
        print("\n=== Rebuilding the UCC ===\n")
        new_JSON = current_JSON
        new_DBs = []

    # Order new_JSON by the values in the 'received' keys
    new_JSON = dict(sorted(new_JSON.items(), key=lambda item: item[1]["received"]))

    all_dbs_data = {}
    # Load existing DBs
    for DB, vals in new_JSON.items():
        if DB in new_DBs:
            new_db_flag = "(new DB)"
            db_csv_path = temp_database_folder + DB + ".csv"
        else:
            new_db_flag = ""
            db_csv_path = dbs_folder + DB + ".csv"
        excl_path = get_excl_path(DB)

        if os.path.isfile(db_csv_path) is True:
            df_new = pd.read_csv(db_csv_path)
        else:
            # If the DB was not found as a data file, check if it is only a comments file
            cmmts_path = UCC_cmmts_folder + DB + ".csv"
            if os.path.isfile(cmmts_path) is False:
                raise FileNotFoundError(
                    f"Neither {db_csv_path} nor {cmmts_path} found for {DB}"
                )
            print(f"Skipping {DB} (only stored as 'comments' file)")
            continue

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
        if vals["types"]["col_name"] not in ("N/A", "ALL"):
            cols.append(vals["types"]["col_name"])
        missing = [c for c in cols if c not in df_new.columns]
        if missing:
            raise ValueError(f"Missing columns in {DB}: {missing}")

        df_new = exclude_entries(DB, df_new, vals["names"], excl_path)

        print(f"{DB} loaded (N={len(df_new)}) {new_db_flag}")
        all_dbs_data[DB] = [df_new, vals]

    return (
        new_JSON,
        df_GCs,
        gcs_fnames,
        selected_center_coords,
        all_dbs_data,
        df_UCC_B_old,
        df_UCC_B_new,
        new_DBs,
        all_names,
        all_names_dict,
    )


def get_excl_path(DB: str) -> str:
    """
    Path to the exclusion file of a DB. A file in the temporary folder (new or
    updated exclusions) takes precedence over the stored one.
    """
    temp_excl_path = temp_folder + UCC_excl_folder + DB + ".csv"
    if os.path.isfile(temp_excl_path):
        return temp_excl_path
    return UCC_excl_folder + DB + ".csv"


def exclude_entries(
    DB: str, df_new: pd.DataFrame, names_col: str, excl_path: str
) -> pd.DataFrame:
    """
    Drop the entries of the DB listed in its exclusion file (if any), i.e. entries
    identified as duplicates, GCs, non-clusters, etc.

    The exclusion file has the columns:
    - row: position (0-based) of the entry in the DB file
    - name: value of the DB's names column for that entry, used to validate 'row'
    - category: duplicate, GC, not_cluster, other
    - comment: free text

    The index of the returned DataFrame is NOT reset: it holds the original row
    positions in the DB file, which are stored in the 'DB_i' column and used by
    all the scripts to access the DB entries.
    """
    if not os.path.isfile(excl_path):
        return df_new

    df_excl = pd.read_csv(excl_path)
    rows = df_excl["row"].to_numpy()

    out_of_range = [r for r in rows if not 0 <= r < len(df_new)]
    if out_of_range:
        raise ValueError(f"{excl_path}: rows out of range {out_of_range}")
    rows_unq, counts = np.unique(rows, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"{excl_path}: repeated rows {list(rows_unq[counts > 1])}")

    # Validate rows using the names
    names_db = df_new[names_col].astype(str).str.strip().to_numpy()[rows]
    names_excl = df_excl["name"].astype(str).str.strip().to_numpy()
    msk = names_db != names_excl
    if msk.any():
        print(f"\n{excl_path}: names do not match the DB file")
        for r, n_excl, n_db in zip(rows[msk], names_excl[msk], names_db[msk]):
            print(f"  row {r}: '{n_excl}' (exclusion file) != '{n_db}' (DB file)")
        raise ValueError(f"Fix the rows in {excl_path}")

    print(f"{DB}: {len(rows)} entries excluded")
    return df_new.drop(index=rows)


def handle_all_names(sep=";") -> tuple[pd.DataFrame, dict]:
    """
    Handle the all_OC_names file:
        - Load the all_OC_names file
        - Check for invalid characters in fnames
        - Check if sorted
        - Create a dictionary mapping each alias to its canonical fname and Names
        - Check for duplicates
        - Check that all fnames are equivalent to the normalized names
    """
    all_names = pd.read_csv(data_folder + all_OC_names)

    # Check for invalid characters in fnames (anything other than letters,
    # numbers, and ';')
    bad_idx = all_names.index[
        all_names["fnames"].str.contains(r"[^A-Za-z0-9;]", regex=True, na=False)
    ]
    if len(bad_idx) > 0:
        print(
            f"\nFound {len(bad_idx)} entries with invalid characters in"
            + f" 'fnames' (only letters, numbers, and '{sep}' allowed):"
        )
        for idx in bad_idx:
            print(f"{idx}: {all_names.loc[idx, 'fnames']}")
        sys.exit(1)

    # Check if sorted by the first fname
    all_names["fname0"] = all_names["fnames"].str.split(sep).str[0]
    if not all_names["fname0"].is_monotonic_increasing:
        print("\n'all_names' file is not sorted by the first fname")
        if input("Sort and store as temp file? (y/n): ").lower() == "y":
            all_names = (
                all_names.sort_values(by="fname0")
                .drop(columns="fname0")
                .reset_index(drop=True)
            )
            print("Sorted all_names by the first fname.\n\nMANUALLY UPDATE THE FILE")
            all_names.to_csv(
                temp_folder + all_OC_names,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
        sys.exit(1)
    all_names = all_names.drop(columns="fname0")

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
        for j, alias in enumerate(fnames):
            if alias in all_names_dict:
                raise ValueError(f"Duplicate alias '{alias}' found at index {i}")
            all_names_dict[alias] = {
                "fnames": f_canonical,
                "Names": n_canonical,
                "alias_name": names[j],
            }

    # Check for duplicates
    duplicates = check_duplicated_fnames(all_fnames_list)
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate fnames in 'all_names':")
        for name, idxs in duplicates.items():
            print(f"{sorted(idxs)} --> '{name}'")
        sys.exit(1)

    # Sanity check: all fnames are equivalent to the normalized names
    all_fnames_norm = get_fnames(all_names_list, sep=sep)
    if all_fnames_list != all_fnames_norm:
        n_max = 10
        diffs = [
            i
            for i, (a, b) in enumerate(zip(all_fnames_list, all_fnames_norm))
            if a != b
        ]
        len_a, len_b = len(all_fnames_list), len(all_fnames_norm)
        if len_a != len_b:
            print(f"Length mismatch: {len_a} vs {len_b}")
            # Indexes present in only one of the lists
            diffs += list(range(min(len_a, len_b), max(len_a, len_b)))

        for i in diffs[:n_max]:
            a = all_fnames_list[i] if i < len_a else None
            b = all_fnames_norm[i] if i < len_b else None
            print(f"[{i}] {a} != {b}")
        if len(diffs) > n_max:
            print(f"... and {len(diffs) - n_max} more")
        raise ValueError("Mismatch between all_fnames_list and new_DB_fnames")

    return all_names, all_names_dict


def basic_new_DB_checks(
    df_GCs: pd.DataFrame,
    gcs_fnames: dict,
    new_DB: str,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    newDB_json: dict,
    flag_check_stop: str,
) -> None:
    """Check new DB for:
    - required columns
    - bad characters in names
    - wrong VDBH/BH naming
    - close GCs
    """
    # Extract required columns
    read_cols = [newDB_json["names"]]
    for v in newDB_json["pos"].values():
        if isinstance(v, list):
            read_cols += v
        else:
            read_cols.append(v)
    # Parameters (and their uncertainties are dictionaries)
    for entry in ("pars", "e_pars"):
        for pdict in newDB_json[entry].values():
            for v in pdict.values():
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
        print(
            f"\nFound {N_bad_names} entries with bad characters in names '(),[],;,*,/'"
        )
        for new_cl in all_bad_names:
            print(f"  {new_cl}")  # bad char found
        if flag_check_stop == "check_stop":
            breakpoint()  # noqa: T100

    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    if vdberg_check(newDB_json, df_new) and flag_check_stop == "check_stop":
        breakpoint()  # noqa: T100

    # Check for OCs very close to each other in the new DB
    if (
        "RA" in newDB_json["pos"]
        and close_OC_inner_check(newDB_json, df_new, flag_check_stop)
        and flag_check_stop == "check_stop"
    ):
        breakpoint()  # noqa: T100

    # Check for close GCs
    if (
        GCs_check(df_GCs, gcs_fnames, newDB_json, df_new, new_DB_fnames)
        and flag_check_stop == "check_stop"
    ):
        breakpoint()  # noqa: T100


def vdberg_check(newDB_json: dict, df_new: pd.DataFrame) -> bool:
    """
    Check for instances of 'vdBergh-Hagen' and 'vdBergh'

    Per CDS recommendation:

    * BH, VDBergh-Hagen --> VDBH
    * VDBergh           --> VDB

    Parameters
    ----------
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
        best_match, score, _ = max(results, key=lambda x: x[1])
        if score > 50:
            vds_found.append([i, new_cl, best_match, score])

    vdb_flag = False
    if len(vds_found) > 0:
        vdb_flag = True
        print(f"\nFound {len(vds_found)} entries that could need name editing")
        print("* BH ; vdBergh-Hagen --> VDBH")
        print("* vdBergh            --> VDB")
        for i, new_cl, name_check, sm_ratio in vds_found:
            print(
                f"{df_new.index[i]}, {new_cl.strip()} --> {name_check} "
                f"(P={sm_ratio:.2f})"
            )

    return vdb_flag


def close_OC_inner_check(
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

    col_1 = df_new[newDB_json["names"]].to_numpy()
    col_2 = None
    db_matches = None
    ID_call = "inner"

    return close_OC_check(
        dist,
        db_matches,
        col_1,
        col_2,
        ID_call,
        leven_rad,
        sep,
        flag_check_stop,
        df_new.index.to_numpy(),
    )


def close_OC_check(
    cls_dist,
    db_matches,
    col_1,
    col_2,
    ID_call: str,
    leven_rad: float,
    sep: str,
    flag_check_stop: str,
    row_ids: np.ndarray,
    rad_dup: float = 10,
    N_max: int = 50,
):
    """
    Check for close OCs in the new DB (or between new DB and UCC) whose names are
    somewhat similar (Levenshtein distance).

    row_ids: original row positions in the new DB file, used for printing
    """
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

        # If cl_name is present in the UCC (df_UCC['fnames'][db_matches[i]])
        if ID_call == "UCC" and db_matches[i] is not None:
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

        print(f"\nFound {dups_found} probable {ID_call} duplicates")
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
                print(
                    f"{row_ids[i]:<6} [d={';'.join(dist)}, L={';'.join(L_ratios)}] {cl_name.strip():<30} (N={N_inner_dups}) --> "
                    + f"{dups_unq:<30}"
                )

            if flag_check_stop != "check_stop":
                print(f"... (only first {N_max} entries shown)")
                break

            start = end
            if start < total:
                ans = input(f"Show {N_max} more? (y/n): ").strip().lower()
                if ans != "y":
                    break

    return dups_flag


def GCs_check(
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

    if "GLON_" in df_new:
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
        print(f"\nFound {GCs_found} probable GCs:")
        for idx, row_id, idx_gc, df_gcs_name, d_arcmin in gc_list:
            row_id = row_id.strip()
            print(
                f"{df_new.index[idx]:<6} {row_id:<40} --> {idx_gc:<6} "
                f"{df_gcs_name.strip():<25}" + f"d={round(d_arcmin, 2)}"
            )

    return gc_flag


def get_ucc_entries_info(df_UCC_B: pd.DataFrame, new_JSON: dict) -> dict:
    """
    Map the canonical fname of each UCC entry to its (GLON, GLAT) position, the
    DBs (and rows in them) that list it, and the earliest DB (by 'received' date)
    that lists it. Used to report the conflicts in all_names_canonical_check()
    """
    ucc_info = {}
    for fname, dbs, dbs_i, lon, lat in zip(
        df_UCC_B["fname"],
        df_UCC_B["DB"],
        df_UCC_B["DB_i"],
        df_UCC_B["GLON"],
        df_UCC_B["GLAT"],
    ):
        dbs_known = [db for db in str(dbs).split(";") if db in new_JSON]
        db0 = min(dbs_known, key=lambda db: new_JSON[db]["received"], default=None)
        # (DB, row in the DB file, names column of the DB)
        dbs_rows = [
            (db, row, new_JSON[db]["names"] if db in new_JSON else None)
            for db, row in zip(str(dbs).split(";"), str(dbs_i).split(";"))
        ]
        ucc_info[fname] = {
            "pos": (lon, lat),
            "DBs": dbs_rows,
            "DB": db0,
            "received": new_JSON[db0]["received"] if db0 else "99999999",
        }
    return ucc_info


def get_DB_entry_names(
    DB: str, row: str, names_col: str | None, DBs_names: dict, sep: str = ","
) -> str:
    """
    Names of the entry stored in a given row of a DB file, as they appear in the
    DB. The names column of each DB is loaded once and stored in 'DBs_names'
    """
    if DB not in DBs_names:
        names = None
        for folder in (temp_folder + dbs_folder, dbs_folder):
            path = folder + DB + ".csv"
            if names_col is not None and os.path.isfile(path):
                df = pd.read_csv(path, usecols=[names_col])
                names = df[names_col].astype(str).to_numpy()
                break
        DBs_names[DB] = names
    names = DBs_names[DB]
    try:
        return ", ".join(_.strip() for _ in names[int(row)].split(sep))
    except (TypeError, ValueError, IndexError):
        return "?"


def lonlat_dist(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Approximate distance in arcmin between two (GLON, GLAT) positions"""
    dlon = ((lon1 - lon2 + 180) % 360 - 180) * np.cos(np.deg2rad(0.5 * (lat1 + lat2)))
    return float(np.hypot(dlon, lat1 - lat2) * 60)


def all_names_canonical_check(
    DB: str,
    df_new: pd.DataFrame,
    newDB_json: dict,
    all_names_dict: dict,
    new_DB_fnames: list[list[str]],
    ucc_info: dict,
) -> None:
    """
    Check that the fnames of each new-DB entry do not point to multiple canonical
    entries in all_names. Must run before get_canonical_fnames(), which assigns
    the canonical of the first fname found in all_names

    For each conflict, the distances between the DB entry and the UCC entries
    involved are shown, along with the suggested main name: the one that appeared
    first in the literature (same criteria used by sort_year_importance()). No
    main name is suggested if any distance is larger than 'max_dist_suggest'
    (arcmin). These entries are printed first in each section
    """
    bad_all_names = []
    for i, fnames in enumerate(new_DB_fnames):
        canonical = {
            all_names_dict[f]["fnames"] for f in set(fnames) if f in all_names_dict
        }
        if len(canonical) > 1:
            bad_all_names.append(i)
    if not bad_all_names:
        return

    # DB entries pointing to each canonical entry in all_names
    canonical_entries = {}
    for i, fnames in enumerate(new_DB_fnames):
        for f in fnames:
            if f in all_names_dict:
                canonical_entries.setdefault(all_names_dict[f]["fnames"], set()).add(i)

    def fmt(names: list[str]) -> str:
        return names[0] if len(names) == 1 else "[" + ", ".join(names) + "]"

    def c_name(c_fname: str) -> str:
        return all_names_dict[c_fname]["Names"]

    def fmt_dist(d: float) -> str:
        return "nan" if np.isnan(d) else f"{d:.1f}'"

    # Names column of each DB, loaded only if required by details()
    DBs_names = {}

    def ucc_pos(c_fname: str) -> tuple[float, float]:
        return ucc_info.get(c_fname, {}).get("pos", (np.nan, np.nan))

    def details(i: int, main_c_fname: str, dup_c_fnames: list[str]) -> tuple[str, bool]:
        """
        Distances between the entries in conflict, and suggested main name. Also
        returns True if the entries are too far apart to suggest a main name
        """
        c_fnames = [main_c_fname] + dup_c_fnames
        # Distances between the main UCC entry and each duplicate UCC entry
        lon_m, lat_m = ucc_pos(main_c_fname)
        d_main = {c: lonlat_dist(lon_m, lat_m, *ucc_pos(c)) for c in dup_c_fnames}
        # Distances between the new-DB entry and each UCC entry
        d_db = {}
        if "GLON_" in df_new.columns:
            lon, lat = df_new["GLON_"].iloc[i], df_new["GLAT_"].iloc[i]
            d_db = {c: lonlat_dist(lon, lat, *ucc_pos(c)) for c in c_fnames}

        dists = [
            f"{c_name(main_c_fname)} <-> {c_name(c)} = {fmt_dist(d)}"
            for c, d in d_main.items()
        ]
        txt = "    Distances: " + ", ".join(dists)
        if d_db:
            dists = [f"{c_name(c)} = {fmt_dist(d)}" for c, d in d_db.items()]
            txt += " | DB entry <-> " + ", ".join(dists)

        # Entries too far apart are not the same object: no main name to suggest
        all_d = list(d_main.values()) + list(d_db.values())
        max_d = np.nanmax(all_d) if not np.isnan(all_d).all() else np.nan
        if max_d > max_dist_suggest:
            # One line per UCC entry with its distances, followed by the DBs that
            # list it and the names used in them
            db_entry = f"{DB} ({df_new.index[i]})"
            lines = []
            for c in c_fnames:
                if c == main_c_fname:
                    line = f"    {c_name(c)} (main name):"
                    dists = [f"{fmt_dist(d_db[c])} from {db_entry}"] if d_db else []
                else:
                    line = f"    {c_name(c)} (duplicate):"
                    dists = [f"{fmt_dist(d_db[c])} from {db_entry}"] if d_db else []
                    dists.append(f"{fmt_dist(d_main[c])} from {c_name(main_c_fname)}")
                lines.append(line + (" " + ", ".join(dists) if dists else ""))
                dbs = [
                    f"{db} [{get_DB_entry_names(db, row, col, DBs_names)}]"
                    for db, row, col in ucc_info.get(c, {}).get("DBs", [])
                ]
                lines.append("        listed in: " + ("; ".join(dbs) or "?"))
            return "\n".join(lines), True

        # Suggest the main name: naming exceptions first, then by importance
        # ('naming_order'), then by the earliest DB that lists the entry
        received = {
            c: ucc_info.get(c, {}).get("received", "99999999") for c in c_fnames
        }
        suggested = min(
            c_fnames,
            key=lambda c: (
                c not in naming_order_exceptions,
                get_importance(c),
                received[c],
            ),
        )
        firsts = []
        for c in c_fnames:
            db0 = ucc_info.get(c, {}).get("DB")
            firsts.append(
                f"{c_name(c)}: {db0} ({received[c][:4]})" if db0 else f"{c_name(c)}: ?"
            )
        reason = (
            "" if received[suggested] == min(received.values()) else ", naming order"
        )
        txt += (
            f"\n    Suggested main name: {c_name(suggested)}"
            + f" [first listed in {'; '.join(firsts)}{reason}]"
        )
        return txt, False

    # Each conflicting entry is assigned to the section of how it should be resolved
    sections = {
        "in_DB": (
            "The duplicate entry in 'all_names' is also matched by another entry"
            + " of this DB.\nResolve: exclude one of the DB entries as 'duplicate'"
            + " in the DB's exclusion file"
        ),
        "secondary": (
            "The duplicate name is a secondary name of another entry in"
            + " 'all_names'.\nResolve: move (or remove) that name in 'all_names'"
        ),
        "canonical_exact": (
            "The duplicate name is the main name of another entry in 'all_names',"
            + " written exactly as in 'all_names'.\nResolve: merge both entries in"
            + " 'all_names', or fix the DB entry"
        ),
        "canonical": (
            "The duplicate name is the main name of another entry in 'all_names',"
            + " written differently or along with other names (-->).\nResolve:"
            + " merge both entries in 'all_names', or fix the DB entry"
        ),
    }
    section_lines = {k: [] for k in sections}
    for i in bad_all_names:
        names = str(df_new.iloc[i][newDB_json["names"]]).split(",")
        # Group the names by the canonical entry they point to. The main entry is
        # the one of the first fname found in all_names (see get_canonical_fnames)
        groups, not_in = {}, []
        for name, fname in zip(names, new_DB_fnames[i]):
            if fname in all_names_dict:
                c_fname = all_names_dict[fname]["fnames"]
                groups.setdefault(c_fname, []).append((name.strip(), fname))
            else:
                not_in.append(name.strip())
        main_c_fname = next(iter(groups))
        dup_c_fnames = [c for c in groups if c != main_c_fname]

        def is_exact(c_fname: str) -> bool:
            # The DB names in the group are just the canonical name of the entry
            c_names = [n for n, _ in groups[c_fname]]
            return c_names == [all_names_dict[c_fname]["Names"]]

        def link(c_fname: str, label: str) -> str:
            canonical = all_names_dict[c_fname]["Names"]
            c_names = [n for n, _ in groups[c_fname]]
            if is_exact(c_fname):
                return f"{canonical} ({label})"
            return f"{fmt(c_names)} --> {canonical} ({label})"

        links = [link(c, "duplicate") for c in dup_c_fnames]
        links.append(link(main_c_fname, "main name"))
        if not_in:
            links.append(f"{fmt(not_in)} (not in all_names)")

        if any(canonical_entries[c] - {i} for c in dup_c_fnames):
            section = "in_DB"
        elif any(all(f != c for _, f in groups[c]) for c in dup_c_fnames):
            section = "secondary"
        elif all(is_exact(c) for c in dup_c_fnames):
            section = "canonical_exact"
        else:
            section = "canonical"
        txt_details, too_far = details(i, main_c_fname, dup_c_fnames)
        section_lines[section].append(
            (too_far, f"({df_new.index[i]}) " + "; ".join(links) + "\n" + txt_details)
        )

    N = len(bad_all_names)
    txt = "y" if N == 1 else "ies"
    print(f"\nFound {N} entr{txt} pointing to multiple entries in 'all_names'")
    for section, lines in section_lines.items():
        if not lines:
            continue
        print(f"\n--- (N={len(lines)}) {sections[section]}")
        # Entries with no suggested main name (too far apart) first
        far = [line for too_far, line in lines if too_far]
        close = [line for too_far, line in lines if not too_far]
        if far:
            print(
                f"\nNo suggested main name (N={len(far)}), entries are far apart"
                + f" (>{max_dist_suggest}') and it's likely a wrong cross-ID:\n"
            )
            print("\n".join(far))
        if close:
            print(f"\nSuggested main name (N={len(close)}):\n")
            print("\n".join(close))
    breakpoint()  # noqa: T100
    sys.exit(1)


def entry_names(
    df_new: pd.DataFrame,
    names_col: str,
    new_DB_fnames: list[list[str]],
    orig_DB_fnames: list[list[str]],
    idx: int,
) -> tuple[str | None, list[str]]:
    """
    Return the canonical name assigned to a new-DB entry through all_names (None
    if the entry is not in all_names) and the entry's own names in the DB
    """
    dnames = [_.strip() for _ in str(df_new.iloc[idx][names_col]).split(",")]
    if len(new_DB_fnames[idx]) > len(orig_DB_fnames[idx]):
        return dnames[0], dnames[1:]
    return None, dnames


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


def check_own_DB_fnames(
    df_new: pd.DataFrame, newDB_json: dict, new_DB_fnames: list[list[str]]
):
    """
    Check that no fname is repeated across entries in the DB's own names. Must
    run before get_canonical_fnames(), i.e. on the fnames of the DB alone. The
    first entry of each group is kept, the rest are marked as elements that
    should be removed from the DB
    """
    duplicates = check_duplicated_fnames(new_DB_fnames)
    if not duplicates:
        return

    def names(idx: int) -> list[str]:
        return [
            _.strip() for _ in str(df_new.iloc[idx][newDB_json["names"]]).split(",")
        ]

    print(
        f"\nFound {len(duplicates)} groups of duplicated fnames repeated within "
        "the DB names:"
    )
    for N_g, (fname, idxs) in enumerate(duplicates.items(), 1):
        print(f"Group {N_g}: {fname}")
        idxs = sorted(idxs)
        keep_name = names(idxs[0])[0]
        for idx_s in idxs[1:]:
            print(
                f" - {', '.join(names(idx_s))} --> {keep_name} ({df_new.index[idx_s]})"
            )
    breakpoint()  # noqa: T100
    sys.exit(1)


def check_new_DB_fnames(
    df_new: pd.DataFrame,
    newDB_json: dict,
    new_DB_fnames: list[list[str]],
    orig_DB_fnames: list[list[str]],
):
    """Check that no canonical fname is shared across entries in the new DB

    Runs after check_own_DB_fnames() so every duplicate here comes from the
    canonical fnames added through all_names. Duplicates are split into two groups:
    - the canonical entry is present in the DB (it is kept)
    - the canonical entry is not present in the DB (no entry is kept)
    """
    duplicates = check_duplicated_fnames(new_DB_fnames)

    if duplicates:
        # If the fname is among the original names of an entry in the group of
        # duplicates (at most one, see check_own_DB_fnames()), keep that entry and
        # mark the rest as elements that should be removed from the DB. If it is
        # not, mark all the entries in the group as elements that should be
        # removed (none is kept)
        dups_all_names, dups_all_names_noDB = {}, {}
        for fname, idxs in duplicates.items():
            idxs = sorted(idxs)
            in_orig = [idx for idx in idxs if fname in orig_DB_fnames[idx]]
            keep_idx = in_orig[0] if in_orig else None
            rm_idxs = [_ for _ in idxs if _ != keep_idx]
            if in_orig:
                dups_all_names[fname] = (keep_idx, rm_idxs)
            else:
                dups_all_names_noDB[fname] = (keep_idx, rm_idxs)

        def names(idx: int) -> tuple[str | None, list[str]]:
            return entry_names(
                df_new, newDB_json["names"], new_DB_fnames, orig_DB_fnames, idx
            )

        def print_groups(dups: dict, msg: str):
            if not dups:
                return
            print(f"\nFound {len(dups)} groups of duplicated fnames {msg}:")
            for N_g, (fname, (keep_idx, idxs)) in enumerate(dups.items(), 1):
                print(f"Group {N_g}: {fname}")
                # Name of the entry that is kept in the DB. If none is kept, use the
                # canonical name assigned (through all_names) to all the entries
                canonical, own = names(idxs[0] if keep_idx is None else keep_idx)
                keep_name = canonical or own[0]
                for idx_s in idxs:
                    own_s = ", ".join(names(idx_s)[1])
                    print(f" - {own_s} --> {keep_name} ({df_new.index[idx_s]})")

        print_groups(
            dups_all_names,
            "assigned the same canonical entry in all_names (present in the DB)",
        )
        print_groups(
            dups_all_names_noDB,
            "assigned the same canonical entry in all_names (NOT present in the DB)",
        )
        breakpoint()  # noqa: T100
        sys.exit(1)


def UCC_matches_check(
    new_DB_fnames: list[list[str]],
    orig_DB_fnames: list[list[str]],
    fname_map: dict,
    ucc_str: Callable[[int], str],
    row_ids,
) -> bool:
    """
    Check the matches of the new DB against the UCC (so far), flagging:
    - new-DB entries matching multiple UCC entries
    - UCC entries matched by multiple new-DB entries

    Each case is split by the source of the matched fnames: the entry's own names
    in the DB, or the canonical fname added through all_names (marked with '*')

    Parameters
    ----------
    fname_map : dict
        Maps each fname in the UCC to its UCC entry index.
    ucc_str : Callable
        Returns the string that describes a UCC entry, given its index.
    row_ids :
        Original row positions in the new DB file, used for printing.

    Returns
    -------
    bool
        True if any issue was found.
    """
    # matches[i][j]: fnames of new-DB entry i that match UCC entry j
    matches = {}
    for i, fnames in enumerate(new_DB_fnames):
        for f in dict.fromkeys(fnames):
            if f in fname_map:
                matches.setdefault(i, {}).setdefault(fname_map[f], []).append(f)

    def via_own(i: int, fnames: list[str]) -> bool:
        return any(f in orig_DB_fnames[i] for f in fnames)

    def tag(i: int, fnames: list[str]) -> str:
        return ", ".join(f if f in orig_DB_fnames[i] else f + "*" for f in fnames)

    # One new-DB entry matching multiple UCC entries. If at least two of them are
    # matched by the entry's own names, the DB merges UCC entries. Otherwise the
    # canonical from all_names points to a different UCC entry than the own names
    multi_own, multi_all_names = {}, {}
    for i, ucc_m in matches.items():
        if len(ucc_m) > 1:
            n_own = sum(via_own(i, fs) for fs in ucc_m.values())
            (multi_own if n_own > 1 else multi_all_names)[i] = ucc_m

    # Multiple new-DB entries matching the same UCC entry
    ucc_to_new = {}
    for i, ucc_m in matches.items():
        for j, fs in ucc_m.items():
            ucc_to_new.setdefault(j, {})[i] = fs
    dups_own, dups_all_names = {}, {}
    for j, new_m in ucc_to_new.items():
        if len(new_m) > 1:
            all_own = all(via_own(i, fs) for i, fs in new_m.items())
            (dups_own if all_own else dups_all_names)[j] = new_m

    if not (multi_own or multi_all_names or dups_own or dups_all_names):
        return False

    msg_own = "through the DB's own names"
    msg_all_names = "through the canonical fname from all_names"
    for multi, msg in ((multi_own, msg_own), (multi_all_names, msg_all_names)):
        if not multi:
            continue
        print(
            f"\nFound {len(multi)} entries in new_DB matching multiple entries in "
            f"the combined DB {msg}:"
        )
        for i, ucc_m in multi.items():
            print(f"({row_ids[i]}) {', '.join(orig_DB_fnames[i])}")
            for j, fs in ucc_m.items():
                print(f"   {tag(i, fs)} --> {ucc_str(j)}")

    for dups, msg in ((dups_own, msg_own), (dups_all_names, msg_all_names)):
        if not dups:
            continue
        print(
            f"\nFound {len(dups)} UCC entries matched by multiple entries in "
            f"new_DB {msg}:"
        )
        for j, new_m in dups.items():
            print(ucc_str(j))
            for i, fs in new_m.items():
                print(
                    f"   ({row_ids[i]}) {', '.join(orig_DB_fnames[i])} [{tag(i, fs)}]"
                )

    print("(*: canonical fname added through all_names)")
    return True


def fnames_check_UCC_new_DB(
    df_UCC_B: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    orig_DB_fnames: list[list[str]],
    df_new: pd.DataFrame,
    sep: str = ";",
) -> None:
    """
    Check consistency of fnames between the new DB and the UCC. Checks that:
    - Each new-DB entry matches at most one existing UCC entry.
    - Multiple new-DB entries do not match the same existing UCC entry.
    """
    # Map each fname currently in the UCC to its row index
    fname_ucc_map = {
        fname: i
        for i, fnames in enumerate(df_UCC_B["fnames"])
        for fname in fnames.split(sep)
    }

    dbs_arr = df_UCC_B["DB"].to_numpy()
    fnames_arr = df_UCC_B["fnames"].to_numpy()

    def ucc_str(j: int) -> str:
        return f"({j}, {dbs_arr[j]}) {fnames_arr[j]}"

    if UCC_matches_check(
        new_DB_fnames, orig_DB_fnames, fname_ucc_map, ucc_str, df_new.index
    ):
        print("\nResolve the above issues before moving on")
        breakpoint()  # noqa: T100
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


def check_new_entries(
    new_DB_fnames, orig_DB_fnames, db_matches, flag_check_stop: str, row_ids
):
    """
    Check for new entries in the new DB that are not present in the UCC.

    New entries are split into those not found in all_names (new OCs) and those
    found in all_names (known OCs not added to the UCC so far, e.g. because the
    DBs that list them come later or their entries were excluded)

    row_ids: original row positions in the new DB file, used for printing
    """
    new_idxs = [i for i, idx in enumerate(db_matches) if idx is None]
    if not new_idxs:
        print("\nNo new entries found")
        return

    print(f"\nFound {len(new_idxs)} new entries")
    in_all_names = [
        i for i in new_idxs if len(new_DB_fnames[i]) > len(orig_DB_fnames[i])
    ]
    not_all_names = [
        i for i in new_idxs if len(new_DB_fnames[i]) == len(orig_DB_fnames[i])
    ]
    N_max = 50
    for idxs, msg in (
        (not_all_names, "not in all_names"),
        (in_all_names, "in all_names but not in the UCC so far"),
    ):
        if not idxs:
            continue
        print(f"\n{len(idxs)} entries {msg}:")
        for count, i in enumerate(idxs, 1):
            if len(new_DB_fnames[i]) > len(orig_DB_fnames[i]):
                print(
                    f"{row_ids[i]}    {orig_DB_fnames[i][0]} --> {new_DB_fnames[i][0]}"
                )
            else:
                print(f"{row_ids[i]}    {new_DB_fnames[i][0]}")
            if count % N_max == 0 and count < len(idxs):
                print(f"... (shown {count} entries)")
                if flag_check_stop != "check_stop":
                    break
                ans = input("Show 50 more? (y/n) ").strip().lower()
                if ans != "y":
                    break
        else:
            print(f"... (shown {len(idxs)} entries in total)")
    if flag_check_stop == "check_stop":
        breakpoint()  # noqa: T100


def check_positions(
    df_UCC_B,
    df_new,
    new_DB_fnames,
    orig_DB_fnames,
    db_matches,
    flag_check_stop: str,
) -> None:
    """
    - Checks for OCs very close to each other between the new database and the UCC.
    - Checks positions and flags for attention if required.
    """
    # Check for OCs very close to other OCs in the UCC
    if (
        "GLON_" in df_new
        and check_new_DB_UCC_positions(
            df_UCC_B, new_DB_fnames, db_matches, df_new, flag_check_stop
        )
        and flag_check_stop == "check_stop"
    ):
        breakpoint()  # noqa: T100

    # Check positions for matched entries and flag for attention if required
    if (
        check_matched_entries(
            df_UCC_B,
            df_new,
            new_DB_fnames,
            orig_DB_fnames,
            db_matches,
        )
        and flag_check_stop == "check_stop"
    ):
        breakpoint()  # noqa: T100


def check_new_DB_UCC_positions(
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
        cls_dist,
        db_matches,
        col_1,
        col_2,
        ID_call,
        leven_rad,
        sep,
        flag_check_stop,
        df_new.index.to_numpy(),
    )


def check_matched_entries(
    df_UCC_B: pd.DataFrame,
    df_new,
    new_DB_fnames,
    orig_DB_fnames,
    db_matches,
    rad_dup: float = 30.0,
    sep: str = ";",
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
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_db_info : dict
        Dictionary with the information of the new database.
    rad_dup : float
        Search radius in arcmin.

    Entries are split by the source of the match: the entry's own names in the
    DB, or only the canonical fname added through all_names

    Returns
    -------
    bool
        Boolean flag indicating if OCs were flagged for attention.
    """
    df_new_glon, df_new_glat = [np.array([np.nan] * len(df_new))] * 2
    if "GLON_" in df_new:
        df_new_glon, df_new_glat = (
            df_new["GLON_"].to_numpy(),
            df_new["GLAT_"].to_numpy(),
        )
    df_UCC_glon, df_UCC_glat = (
        df_UCC_B["GLON"].to_numpy(),
        df_UCC_B["GLAT"].to_numpy(),
    )
    df_UCC_fnames = df_UCC_B["fnames"].to_numpy()

    ocs_attention = {True: [], False: []}
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
                names = orig_DB_fnames[i][0]
                if len(new_DB_fnames[i]) > len(orig_DB_fnames[i]):
                    names += f" ({new_DB_fnames[i][0]})"
                # Matched through the entry's own names or only through the
                # canonical fname added from all_names
                via_own = not set(orig_DB_fnames[i]).isdisjoint(
                    str(df_UCC_fnames[j]).split(sep)
                )
                ocs_attention[via_own].append([df_new.index[i], names, d_arcmin])

    attention_flag = False
    N_max = 50
    for via_own, msg in (
        (True, "matched through the DB's own names"),
        (False, "matched through the canonical fname from all_names"),
    ):
        ocs = ocs_attention[via_own]
        if not ocs:
            continue
        attention_flag = True
        print(f"\nEntries with coords far from those in the UCC, {msg} (N={len(ocs)})")
        print(f"{'DB_idx':<6} {'name':<30} {'d [arcmin]':<5}")
        # Order by distance
        ocs = sorted(ocs, key=lambda x: x[2], reverse=True)
        for i, fname, d_arcmin in ocs[:N_max]:
            print(f"{i:<6} {fname:<30} {d_arcmin:<5.0f}")
        if len(ocs) > N_max:
            print(f"... (only first {N_max} entries shown)")

    return attention_flag


def add_fpars_col(newDB_json, df_new, max_chars=7):
    """
    Add columns for the fundamental parameters (fpars) to the new DB DataFrame.
    """
    all_pars = {}
    N_rows = len(df_new)

    # For each parameter
    for par_general in fpars_order:
        if par_general in newDB_json["pars"]:
            par = newDB_json["pars"][par_general]

            N_par_values = len(par.items())

            # If the parameter has more than one value associated in different columns
            # with different formats. I.e: {'dkpc': 'dmode', 'dpc': 'Dist'}
            flag_mult = [""] * N_rows
            if N_par_values > 1:
                # Add '*' indicator for multiple values (total - 1)
                flag_mult = ["*" * (N_par_values - 1)] * N_rows

            # Select the first item (in case there are more than one)
            # Equivalent to: [[k, v] for k, v in par.items()][0]
            par_format, par_db_col = next(iter(par.items()))

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

    for col, val in all_pars.items():
        df_new[col] = val

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

    return apply(fmt_to_UCC[par_format])


def combine_UCC_new_DB(
    selected_center_coords: dict,
    new_DB: str,
    newDB_json: dict,
    df_UCC_B: pd.DataFrame,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
    db_fscore: str,
) -> pd.DataFrame:
    """
    Combines a new database with the UCC, handling new and existing entries.

    Parameters
    ----------
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
            str(df_new.index[i_new_cl]),
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
            db_fscore,
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
    print(f"\nNew entries: {N_new}, Updated entries: {N_updt}")

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
    """
    Extracts the coordinates from the new database, prioritizing manually fixed values
    over database values, and considering existing UCC values if present.
    """
    # Extract manually fixed centers (if any)
    ra_f, dec_f, lon_f, lat_f, plx_f, pmra_f, pmde_f = _NANS_7
    manual_cents_used = False
    for fname in fnames_new_cl:
        if fname in selected_center_coords:
            ra_f, dec_f, plx_f, pmra_f, pmde_f, _ = selected_center_coords[fname]
            manual_cents_used = True
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

    if manual_cents_used:
        DB_used = "MANUAL"

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
    db_fscore: str,
    sep: str = ";",
    oc_type="OC",
):
    """
    Update the new database dictionary with the information from the new DB and the
    UCC.

    By default all entries are OC, will be updated later
    """
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
        type = oc_type
        F_score = db_fscore
    else:
        # OC already present in UCC
        fnames = row_ucc["fnames"] + sep + fnames_joined
        DB_ID = row_ucc["DB"] + sep + new_DB_exp
        DB_i = row_ucc["DB_i"] + sep + i_new_cl_exp
        F_score = row_ucc["F_score"] + sep + db_fscore
        names = row_ucc["Names"] + sep + oc_names
        type = row_ucc["Type"] + sep + oc_type
        for par in fpars_order:
            fund_pars[par] = row_ucc[par] + sep + fund_pars_exp[par]

    # Append to output dictionary
    new_db_dict["fnames"].append(fnames)
    new_db_dict["Type"].append(type)
    new_db_dict["DB"].append(DB_ID)
    new_db_dict["DB_i"].append(DB_i)
    new_db_dict["F_score"].append(F_score)
    new_db_dict["Names"].append(names)
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


def get_db_fscore(
    new_JSON: dict,
    new_DB: str,
    df_new: pd.DataFrame,
    focus_decay_per_year: float = 0.01,
    focus_log_ref: float = 500.0,
    focus_min: float = 0.1,
) -> str:
    """
    F_score: per-article focus score combining recency and specificity

    focus_decay_per_year : float
        Drop per decade at N_row=1
    focus_log_ref : float
        N_row at which S reaches 0.5
    focus_min : float
        Absolute F_score floor
    """

    db_year = int(new_JSON[new_DB]["year"])
    db_N_rows = len(df_new)
    y_now = datetime.now().year
    R = np.clip(1 - focus_decay_per_year * (y_now - db_year), 0.0, 1.0)
    S = np.clip(1 - 0.5 * np.log(db_N_rows) / np.log(focus_log_ref), 0.0, 1.0)
    db_fscore = f"{np.maximum(focus_min, R * S):.2f}"

    return db_fscore


def get_importance(item: str) -> int | float:
    """Optimized importance map (stops at first match)"""
    for j, p in enumerate(naming_order):
        if item.startswith(p):
            return j
    return np.inf


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
            # Sort all properties by year
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

        # Check for order exceptions
        for i, fname in enumerate(fnames):
            if fname in naming_order_exceptions:
                fnames.insert(0, fnames.pop(i))
                names.insert(0, names.pop(i))
                break

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


def add_fpars_stats(df_UCC_B_new: pd.DataFrame) -> pd.DataFrame:
    """
    For each column in 'fpars_order', calculate the median and stddev.
    """
    for par in fpars_order:
        # String cleaning and splitting
        temp_df = (
            df_UCC_B_new[par]
            .str.replace("*", "", regex=False)
            .str.split(";", expand=True)
            .apply(pd.to_numeric, errors="coerce")
        )

        if par == "blue_str":
            # Filter fractional values
            # Identify rows where ALL non-NaN values are fractional
            all_frac = ((temp_df > 0) & (temp_df < 1)).all(axis=1)
            if all_frac.sum() > 0:
                print(
                    f"\nFound {all_frac.sum()} entries with all BSS values in "
                    "(0, 1) range"
                )
            # Condition to keep non-fractional values
            cond = (temp_df == 0) | (temp_df >= 1)
            # apply filter only to rows that are NOT fully fractional
            temp_df = temp_df.where(cond | all_frac.to_frame().reindex_like(temp_df))

        # Statistics (NaNs automatically ignored, ddof=1 by default for std)
        med = temp_df.median(axis=1)
        std = temp_df.std(axis=1)

        dec = 0 if par in {"age", "mass", "blue_str"} else 4
        df_UCC_B_new[f"{par}_median"] = med.round(dec)
        df_UCC_B_new[f"{par}_stddev"] = std.round(dec)

    return df_UCC_B_new


# def add_fpars_stats_new(
#     logging, df_UCC_B_new: pd.DataFrame, mad_k: float = 3.0
# ) -> pd.DataFrame:
#     """
#     For each column in 'fpars_order', reject outliers (row-wise MAD clipping)
#     and calculate the F_Score-weighted mean and weighted stddev.
#     """
#     weights_df = (
#         df_UCC_B_new["F_score"]
#         .str.split(";", expand=True)
#         .apply(pd.to_numeric, errors="coerce")
#     )

#     for par in fpars_order:
#         temp_df = (
#             df_UCC_B_new[par]
#             .str.replace("*", "", regex=False)
#             .str.split(";", expand=True)
#             .apply(pd.to_numeric, errors="coerce")
#         )

#         if par == "blue_str":
#             all_frac = ((temp_df > 0) & (temp_df < 1)).all(axis=1)
#             if all_frac.sum() > 0:
#                 logging.info(
#                     f"\nFound {all_frac.sum()} entries with all BSS values in "
#                     "(0, 1) range"
#                 )
#             cond = (temp_df == 0) | (temp_df >= 1)
#             temp_df = temp_df.where(cond | all_frac.to_frame().reindex_like(temp_df))

#         # --- Outlier rejection: row-wise MAD clipping ---
#         n_valid = temp_df.notna().sum(axis=1)
#         row_med = temp_df.median(axis=1)
#         abs_dev = temp_df.sub(row_med, axis=0).abs()
#         mad = abs_dev.median(axis=1)
#         # Consistency constant so MAD approximates stddev under normality
#         mad_scaled = mad * 1.4826

#         # Rows where MAD collapses to 0 (e.g. duplicate values): fall back to
#         # no rejection rather than nuking everything that differs at all
#         threshold = mad_scaled.where(mad_scaled > 0, np.inf)
#         is_outlier = abs_dev.gt(mad_k * threshold, axis=0)
#         # Only apply rejection where there are enough points to judge (n >= 3)
#         is_outlier = is_outlier.where(n_valid.ge(3), other=False)

#         n_rejected = is_outlier.sum(axis=1)
#         if n_rejected.sum() > 0:
#             logging.info(
#                 f"\n{par}: rejected {int(n_rejected.sum())} outlier value(s) "
#                 f"across {int((n_rejected > 0).sum())} row(s)"
#             )

#         temp_df = temp_df.mask(is_outlier)

#         # --- Weighted stats on cleaned values ---
#         w = weights_df.reindex(columns=temp_df.columns, index=temp_df.index)
#         w = w.where(temp_df.notna())

#         w_sum = w.sum(axis=1, min_count=1)
#         wmean = (temp_df * w).sum(axis=1, min_count=1) / w_sum

#         n_clean = temp_df.notna().sum(axis=1)
#         wvar = (temp_df.sub(wmean, axis=0) ** 2 * w).sum(axis=1, min_count=1) / w_sum
#         wstd = wvar.pow(0.5).where(n_clean >= 2)

#         dec = 0 if par in {"age", "mass", "blue_str"} else 4
#         df_UCC_B_new[f"{par}_median"] = wmean.round(dec)
#         df_UCC_B_new[f"{par}_stddev"] = wstd.round(dec)

#     return df_UCC_B_new


def add_types(
    new_JSON: dict, dbs_dfs: dict[str, pd.DataFrame], df_UCC_B_new: pd.DataFrame
) -> pd.DataFrame:
    """
    Add the 'Type' column.

    Embedded clusters (ECs) are identified in each DB using the 'types' key of its
    JSON entry: 'col_name' is the column that holds the type information ('ALL' if
    every entry in the DB is an EC, 'N/A' if the DB does not identify ECs) and
    'ec_id' lists the values in that column that identify an EC.
    """
    # Dictionary mapping each DB to the set of row indexes (as stored in the 'DB_i'
    # column) catalogued as ECs, or to None if all its entries are ECs
    embedded_rows = {}
    for DB, df_db in dbs_dfs.items():
        col_name = new_JSON[DB]["types"]["col_name"]
        if col_name == "N/A":
            continue
        if col_name == "ALL":
            embedded_rows[DB] = None
            continue
        ec_id = new_JSON[DB]["types"]["ec_id"]
        msk = df_db[col_name].astype(str).str.strip().isin(ec_id).to_numpy()
        # Index of the DBs holds the original row positions (see exclude_entries)
        embedded_rows[DB] = {str(i) for i in df_db.index[msk]}

    type_col = []
    for dbs, dbs_i in zip(df_UCC_B_new["DB"], df_UCC_B_new["DB_i"]):
        is_ec = [
            db in embedded_rows
            and (embedded_rows[db] is None or db_i in embedded_rows[db])
            for db, db_i in zip(dbs.split(";"), dbs_i.split(";"))
        ]

        if not any(is_ec):
            # Object is never catalogued as an EC
            type_col.append("OC")
        elif all(is_ec):
            # Object is catalogued as an EC in all DBs
            type_col.append("EC")
        else:
            # Object is catalogued as an EC in some DBs, but not in all DBs
            type_col.append("EC;OC")

    # Add the 'Type' column
    df_UCC_B_new["Type"] = type_col

    return df_UCC_B_new


def duplicates_fnames_check(df_UCC_B: pd.DataFrame, sep: str = ";") -> bool:
    """
    Check for duplicate filenames across rows in 'dbs_merged_new' DataFrame.

    The function scans the 'fnames' column of the DataFrame, where each entry
    may contain one or more filenames separated by semicolons. It identifies
    any filenames that appear in more than one row. For each duplicate, the
    row indices and the filename are logged, ensuring that duplicate pairs
    are reported only once.

    Parameters
    ----------
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
                            print(f"{i}, {j}, {fname}")
                            printed_pairs.add(pair)

    return dup_flag


def ra_dec_check(df_UCC_B):
    """Check that all RA and Dec values are within valid ranges and not NaN."""

    ra = df_UCC_B["RA_ICRS"]
    dec = df_UCC_B["DE_ICRS"]

    invalid_ra = df_UCC_B[ra.isna() | (ra < 0) | (ra >= 360)]
    invalid_dec = df_UCC_B[dec.isna() | (dec < -90) | (dec > 90)]

    if not invalid_ra.empty:
        print(f"\nFound {len(invalid_ra)} entries with invalid RA values:")
        for idx, row in invalid_ra.iterrows():
            print(f"{idx} ({row['fnames'].split(';')[0]}) --> RA={row['RA_ICRS']}")

    if not invalid_dec.empty:
        print(f"\nFound {len(invalid_dec)} entries with invalid Dec values:")
        for idx, row in invalid_dec.iterrows():
            print(f"{idx} ({row['fnames'].split(';')[0]}) --> Dec={row['DE_ICRS']}")

    return not invalid_ra.empty or not invalid_dec.empty


def sanity_check(all_names_old, df_UCC_B, N_max=50):
    """
    Perform sanity checks on the updated UCC DataFrame after merging with a new
    database.
    """
    # Check every individual fname for duplicates
    exit_flag = duplicates_fnames_check(df_UCC_B)
    if exit_flag:
        print("\nERROR: duplicated entries found in B cat 'fnames' column. Fix this!")
        breakpoint()  # noqa: T100
        sys.exit(1)

    # Check that (RA, DEC) ranges are valid
    exit_flag = ra_dec_check(df_UCC_B)
    if exit_flag:
        print(
            "\nERROR: entries were found with missing (RA, DEC) values in B cat. "
            "Fix this!"
        )
        breakpoint()  # noqa: T100
        sys.exit(1)

    # Check fnames consistency between old 'all_names' and new 'df_UCC_B'
    fnames_old = [_.split(";") for _ in all_names_old["fnames"]]
    fnames_new = [_.split(";") for _ in df_UCC_B["fnames"]]
    fname0_old_lst = [_[0] for _ in fnames_old]
    fname0_new_lst = [_[0] for _ in fnames_new]

    # Build a lookup: each secondary name --> the full entry it appears in
    lookup_new = {}
    for full in fnames_new:
        for name in full:
            lookup_new[name] = full

    # Compute differences
    set_new0 = set(fname0_new_lst)
    diffs_1 = []
    for fname0_old in fname0_old_lst:
        if fname0_old not in set_new0:
            # Old canonical fname is not in the new canonical fnames, check if it
            # appears as a secondary fname
            full = lookup_new.get(fname0_old)
            if full:
                diffs_1.append(f"{fname0_old} --> {';'.join(full)}")
            else:
                raise ValueError(
                    f"Canonical fname '{fname0_old}' from old 'all_names' not found in"
                    + " new 'df_UCC_B', neither as canonical nor as secondary fname"
                )

    N_diff = len(diffs_1)
    if N_diff > 0:
        print(f"Found {N_diff} canonical fnames that changed:\n")
        for _ in diffs_1[:N_max]:
            print(_)
        if N_diff > N_max:
            print(f"... and {N_diff - N_max} more")


def generate_dfs(
    df_UCC_B_new: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate new 'all_names' and 'df_UCC_B' files"""
    # Generate new all_names file
    all_names_new = pd.DataFrame(df_UCC_B_new[["fnames", "Names"]]).reset_index(
        drop=True
    )

    # Create copy and drop columns
    df_UCC_B_final = df_UCC_B_new.drop(columns=["fnames", "Names"]).copy()
    # Generate 'fname' column and insert at the first position
    df_UCC_B_final.insert(
        0,
        "fname",
        df_UCC_B_new["fnames"].str.partition(";")[0],
    )

    return all_names_new, df_UCC_B_final


def add_lost_fnames(
    all_names_dict_old: dict,
    all_names_new: pd.DataFrame,
    sep: str = ";",
) -> pd.DataFrame:
    """
    Recover lost fnames from 'all_names_dict_old' and add them to 'all_names_new'.
    This process DOES NOT ALTER the canonical fnames in 'all_names_new'
    """
    # Create all_names_dict mapping each alias to its position index
    all_names_dict_new = {}
    for i, row in enumerate(all_names_new.itertuples(index=False)):
        for alias in str(row.fnames).split(sep):
            # if alias in all_names_dict_new:
            #     raise ValueError(f"Duplicate alias '{alias}' found at index {i}")
            all_names_dict_new[alias] = {"idx": i}

    # fnames only in the old dictionary, lost in the new one
    lost_fnames = [
        fname for fname in all_names_dict_old if fname not in all_names_dict_new
    ]
    # logging.info(f"\nLost fnames: N={len(lost_fnames)}")

    for lost_fname in lost_fnames:
        # Canonical fname attached to the lost fname in the old dictionary
        old_c_fname = all_names_dict_old[lost_fname]["fnames"]

        # Index in the new dictionary of the old canonical fname attached to the lost
        # fname
        new_c_idx = all_names_dict_new[old_c_fname]["idx"]

        # Add the lost fname to 'all_names_new', at the 'new_c_idx' index
        all_names_new.at[new_c_idx, "fnames"] += sep + lost_fname
        all_names_new.at[new_c_idx, "Names"] += (
            sep + all_names_dict_old[lost_fname]["alias_name"]
        )

    return all_names_new


def add_excl_duplicate_names(
    new_JSON: dict,
    all_names_new: pd.DataFrame,
    sep: str = ";",
) -> pd.DataFrame:
    """
    Add the names of the DB entries excluded with the 'duplicate' category to the
    entry in 'all_names_new' of their main name (stored in the 'comment' column of
    the exclusion file). This way the duplicated naming is not lost.
    This process DOES NOT ALTER the canonical fnames in 'all_names_new'
    """
    # Map each alias to its position index
    alias_idx = {}
    for i, row in enumerate(all_names_new.itertuples(index=False)):
        for alias in str(row.fnames).split(sep):
            alias_idx[alias] = i

    N_added, conflicts = 0, []
    for DB in new_JSON:
        excl_path = get_excl_path(DB)
        if not os.path.isfile(excl_path):
            continue
        df_excl = pd.read_csv(excl_path)
        df_dup = df_excl[df_excl["category"] == "duplicate"]

        for row in df_dup.itertuples(index=False):
            main_fname = get_fnames([row.comment])[0][0]
            if main_fname not in alias_idx:
                raise ValueError(
                    f"{excl_path}, row {row.row}: main name '{row.comment}' not "
                    + "found in 'all_names'"
                )
            idx = alias_idx[main_fname]

            dup_names = rename_standard(str(row.name)).split(sep)
            dup_fnames = get_fnames(dup_names, sep=sep)
            for dup_name, (dup_fname,) in zip(dup_names, dup_fnames):
                if dup_fname in alias_idx:
                    if alias_idx[dup_fname] != idx:
                        conflicts.append(
                            f"{DB} row {row.row}: '{dup_name}' --> "
                            + f"{all_names_new.at[alias_idx[dup_fname], 'fnames']}"
                            + f" (expected main: {all_names_new.at[idx, 'fnames']})"
                        )
                    continue
                all_names_new.at[idx, "fnames"] += sep + dup_fname
                all_names_new.at[idx, "Names"] += sep + dup_name.strip()
                alias_idx[dup_fname] = idx
                N_added += 1

    if N_added > 0:
        print(f"\nAdded {N_added} names from excluded duplicates to 'all_names'")
    if conflicts:
        print(
            f"\nWARNING: {len(conflicts)} names from excluded duplicates are "
            + "assigned to a different entry in 'all_names' (not added):"
        )
        for _ in conflicts:
            print(_)

    return all_names_new


def all_names_DBs_check(
    all_names: pd.DataFrame,
    new_JSON: dict,
    db_paths: dict,
    N_max: int = 50,
    sep: str = ";",
) -> None:
    """
    Check that the names in 'all_names' are 1 to 1 correlated with the names in
    the DBs, i.e.:
    - every fname in 'all_names' is listed by at least one DB entry
    - every fname listed by a DB entry is in 'all_names'
    - no fname appears more than once in 'all_names'

    The DB names are read again from the DB files. Entries excluded as
    'duplicate' count as DB entries (their names are stored along the main name),
    entries excluded under any other category do not.

    The full list of issues is stored in the temporary folder.
    """
    # fname --> (name as in the DB, list of 'DB (row)' that list it)
    db_fnames = {}
    for DB, path in db_paths.items():
        if not os.path.isfile(path):
            # Only stored as 'comments' file
            continue
        names_col = new_JSON[DB]["names"]
        names = pd.read_csv(path, usecols=[names_col])[names_col]
        excl_rows = set()
        excl_path = get_excl_path(DB)
        if os.path.isfile(excl_path):
            df_excl = pd.read_csv(excl_path)
            excl_rows = set(df_excl.loc[df_excl["category"] != "duplicate", "row"])
        for row, (name, fnames) in enumerate(zip(names, get_fnames(names))):
            if row in excl_rows:
                continue
            for n, f in zip(str(name).split(","), fnames):
                db_fnames.setdefault(f, (n.strip(), []))[1].append(f"{DB} ({row})")

    # fname --> (alias name, main name) of the entry in 'all_names'
    an_fnames, an_counts = {}, {}
    for fnames, names in zip(all_names["fnames"], all_names["Names"]):
        fnames, names = str(fnames).split(sep), str(names).split(sep)
        for f, n in zip(fnames, names):
            an_fnames[f] = (n, names[0])
            an_counts[f] = an_counts.get(f, 0) + 1

    issues = []
    for f, (n, main) in an_fnames.items():
        if f not in db_fnames:
            where = "" if n == main else f"(entry: {main})"
            issues.append(("not in any DB", f, n, where))
    for f, (n, dbs) in db_fnames.items():
        if f not in an_fnames:
            where = "listed in: " + "; ".join(dbs)
            issues.append(("not in all_names", f, n if n else "(empty name)", where))
    for f, N in an_counts.items():
        if N > 1:
            issues.append(("repeated in all_names", f, an_fnames[f][0], f"{N} times"))
    if not issues:
        return

    msgs = {
        "not in any DB": "Names in 'all_names' that are not listed in any DB",
        "not in all_names": "Names listed in the DBs that are not in 'all_names'",
        "repeated in all_names": "Names repeated in 'all_names'",
    }
    print("\nThe names in 'all_names' are not 1 to 1 with the names in the DBs")
    for issue, msg in msgs.items():
        lines = [i for i in issues if i[0] == issue]
        if not lines:
            continue
        print(f"\n{msg} (N={len(lines)}):")
        for _, f, n, where in lines[:N_max]:
            print(f"  {n} {where}".rstrip())
        if len(lines) > N_max:
            print(f"  ... and {len(lines) - N_max} more")

    out_path = temp_folder + "all_names_DBs_check.csv"
    pd.DataFrame(issues, columns=["issue", "fname", "name", "where"]).to_csv(
        out_path, index=False
    )
    print(f"\nFull list stored in {out_path}")
    if input("Continue anyway? (y/n): ").lower() != "y":
        sys.exit(1)


def final_fnames_compare(
    fnames_old: pd.Series, fnames_new: pd.Series, N_max: int = 50
) -> None:
    """Find fnames in old 'all_names' that are missing in new 'all_names' and
    raise an error if any are found.

    Raises
    ------
    ValueError
        If one or more fnames from the old catalogue are missing in the new
        catalogue.
    """

    # Extract all individual fnames from the semicolon-separated strings in both Series
    fnames_old_set = {x for sub in fnames_old.str.split(";") for x in sub}
    fnames_new_set = {x for sub in fnames_new.str.split(";") for x in sub}

    # Find fnames in old 'all_names' that are missing in new 'all_names'
    missing_old_in_new = sorted(fnames_old_set - fnames_new_set)

    N_missing = len(missing_old_in_new)
    if N_missing > 0:
        print(f"\nFound {N_missing} missing fnames in new 'all_names':")
        # Print first batch
        for fmiss in missing_old_in_new[:N_max]:
            print(f"{fmiss}")
        remaining = N_missing - N_max
        # Ask whether to show more
        while remaining > 0:
            show_more = (
                input(f"\nShow {min(N_max, remaining)} more? (y/n): ").strip().lower()
            )
            if show_more != "y":
                break
            start = N_missing - remaining
            end = start + min(N_max, remaining)
            for fmiss in missing_old_in_new[start:end]:
                print(fmiss)
            remaining -= end - start

        raise ValueError(f"Found {N_missing} missing fnames")


def update_final_files(
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
        "all_names", all_names, all_names_new, order_col="fnames"
    )
    if diff_found:
        save_df_UCC(all_names_new, temp_all_OC_names, order_col="fnames")

    diff_found = diff_between_dfs("B cat", df_UCC_B_old, df_UCC_B_new)
    if diff_found:
        file_path = df_UCC_B_path.replace(data_folder, temp_folder)
        save_df_UCC(df_UCC_B_new, file_path)


def move_files(
    df_UCC_B_path: str,
    df_UCC_B_old: pd.DataFrame,
    new_json_dict: dict,
    temp_JSON_path: str,
    all_dbs_data: dict,
    temp_all_OC_names: str,
    temp_database_folder: str,
) -> None:
    """
    Move the updated files from temporary locations to their final paths.
    """
    files_to_move = []

    if os.path.isfile(temp_JSON_path):
        files_to_move.append(temp_JSON_path)

    for new_DB in all_dbs_data:
        db_temp = temp_database_folder + new_DB + ".csv"
        if os.path.isfile(db_temp):
            files_to_move.append(db_temp)
        excl_temp = temp_folder + UCC_excl_folder + new_DB + ".csv"
        if os.path.isfile(excl_temp):
            files_to_move.append(excl_temp)

    ucc_temp = temp_folder + UCC_cat_B_out
    if os.path.isfile(ucc_temp):
        files_to_move.append(ucc_temp)

    if os.path.isfile(temp_all_OC_names):
        files_to_move.append(temp_all_OC_names)

    if not files_to_move:
        print("No files to move.")
        return

    print(f"{len(files_to_move)} file(s) ready to move")
    if input("\nMove files to their final paths? (y/n): ").lower() != "y":
        print("Files not moved.")
        return

    # Update JSON file with all the DBs and store the new DB in place
    if os.path.isfile(temp_JSON_path):
        with open(temp_JSON_path, "w") as f:
            json.dump(new_json_dict, f, indent=2)

        os.rename(temp_JSON_path, name_DBs_json)
        print(temp_JSON_path + " --> " + name_DBs_json)

    for new_DB in all_dbs_data:
        new_DB_file = new_DB + ".csv"
        db_temp = temp_database_folder + new_DB_file

        if os.path.isfile(db_temp):
            db_stored = dbs_folder + new_DB_file
            os.rename(db_temp, db_stored)
            print(db_temp + " --> " + db_stored)

        excl_temp = temp_folder + UCC_excl_folder + new_DB_file
        if os.path.isfile(excl_temp):
            os.makedirs(UCC_excl_folder, exist_ok=True)
            excl_stored = UCC_excl_folder + new_DB_file
            os.rename(excl_temp, excl_stored)
            print(excl_temp + " --> " + excl_stored)

    if os.path.isfile(ucc_temp):
        now_time = pd.Timestamp.now().strftime("%y%m%d%H")
        archived_B_file = archive_folder_path + UCC_cat_B_out.replace(
            ".csv", f"_{now_time}.csv.gz"
        )
        save_df_UCC(df_UCC_B_old, archived_B_file, compression="gzip")
        prune_archive(archived_B_file)

        os.remove(df_UCC_B_path)
        print(df_UCC_B_path + " --> " + archived_B_file)

        os.rename(ucc_temp, df_UCC_B_path)
        print(ucc_temp + " --> " + df_UCC_B_path)

    if os.path.isfile(temp_all_OC_names):
        final_all_OC_names = data_folder + all_OC_names
        os.rename(temp_all_OC_names, final_all_OC_names)
        print(temp_all_OC_names + " --> " + final_all_OC_names)


if __name__ == "__main__":
    main()
