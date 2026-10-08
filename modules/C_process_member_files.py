import csv
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np
import pandas as pd

from .C_funcs.classification import get_classif
from .C_funcs.member_files_updt_funcs import (
    get_fastMP_membs,
    get_gaia_frame,
    get_new_cl_data,
    save_cl_datafile,
    updt_UCC_new_cl_data,
)
from .utils import (
    diff_between_dfs,
    load_BC_cats,
    logger,
    normalize_name,
    prune_archive,
    round_columns,
    save_df_UCC,
)
from .variables import (
    C_lit_max,
    GCs_cat,
    N_archive_versions,
    P_dup_max,
    UCC_cat_B_out,
    UCC_cat_C_in,
    UCC_cat_C_out,
    UCC_members_file,
    UTI_max,
    all_OC_names,
    archive_folder_path,
    data_folder,
    frame_limits_cols,
    md_folder,
    name_DBs_json,
    path_gaia_frames_ranges,
    plots_folder,
    root_ucc_path,
    temp_folder,
    temp_members_folder,
    zenodo_cat_fname,
    zenodo_folder,
)


def main():
    """
    Second script to update the UCC (Unified Cluster Catalogue)

    The names in final (updated) C catalogue should match those in the members file.

    """
    logging = logger()

    # Generate paths and check for required folders and files
    ucc_B_file_out, ucc_C_file_in, ucc_C_file_out, temp_zenodo_fold = (
        get_paths_check_paths(logging)
    )

    (
        gaia_frames_data,
        current_JSON,
        new_ocs_manual_pars,
        df_GCs,
        df_members,
        all_names,
        df_UCC_B_out,
        df_UCC_C,
        old_zenodo_cat,
    ) = load_data(logging, ucc_B_file_out, ucc_C_file_in, ucc_C_file_out)

    # Initial check that the names in the members file match those in the UCC C
    # dataframe
    check_sorted_names(logging, df_UCC_C, df_members)

    # Detect entries to be processed
    rename_C_fname, C_not_in_B, B_not_in_C, C_reprocess = detect_entries_to_process(
        logging, all_names, df_UCC_B_out, df_UCC_C
    )

    load_file = False
    temp_UCC_updt_file = temp_folder + "df_UCC_C_updt.csv"
    if os.path.isfile(temp_UCC_updt_file) and (
        input(f"\nLoad existing '{temp_UCC_updt_file}' file? (y/n): ").lower() == "y"
    ):
        load_file = True

    if load_file:
        # Load file if it already exists and the .parquet files were generated
        df_UCC_C_updt = load_temp_C_updt_file(
            logging, temp_UCC_updt_file, B_not_in_C, C_reprocess
        )
    else:
        # Generate dataframe to store data extracted from the OCs to be processed
        df_add_reprocess = process_entries(
            df_UCC_B_out, new_ocs_manual_pars, B_not_in_C, C_reprocess
        )
        # Generate member files for new OCs and obtain their data
        df_UCC_C_updt = member_files_updt(
            logging, gaia_frames_data, df_GCs, df_UCC_C, df_add_reprocess
        )

    df_UCC_C_new = update_C_cat(C_not_in_B, rename_C_fname, df_UCC_C, df_UCC_C_updt)
    logging.info(f"\nUCC database C updated (N={len(df_UCC_C_new)})\n")

    logging.info("Updating members file...")
    # Check that the names in the members file match those in the UCC C dataframe
    # before combining
    check_names_before_members_combine(logging, B_not_in_C, C_reprocess)

    # Concatenate all temporary DataFrames into one
    df_comb = gen_comb_members_file(logging)
    flag_membs_changed, df_members_new = update_membs_file(
        rename_C_fname, C_not_in_B, df_members, df_comb
    )

    # Final check that the names in the members file match those in the UCC C dataframe
    check_sorted_names(logging, df_UCC_C_new, df_members_new)

    if flag_membs_changed is True:
        logging.info(
            f"Zenodo '{UCC_members_file}' file updated "
            f"(N={len(df_members)}->{len(df_members_new)})\n"
        )
        # Find shared members between OCs and update df_UCC_C_new dataframe
        df_UCC_C_final = find_shared_members(logging, df_UCC_C_new, df_members_new)
        logging.info("Shared members data updated in UCC\n")
    else:
        logging.info("No changes in members file. Skipping shared members check")
        df_UCC_C_final = df_UCC_C_new.copy()

    # Sort df_UCC_B by fname column to match 'df_UCC_C_final'
    df_UCC_B = df_UCC_B_out.sort_values("fname").reset_index(drop=True)
    # Check the 'fnames' columns in df_UCC_B and df_UCC_C_final dataframes are equal
    if not df_UCC_B["fname"].to_list() == df_UCC_C_final["fname"].to_list():
        raise ValueError("The 'fname' columns in B and final C dataframes differ")

    # Check that all entries in df_UCC_C_final have process='n'
    bad_process = ~df_UCC_C_final["process"].eq("n").fillna(False)
    if bad_process.any():
        bad = df_UCC_C_final.loc[bad_process, ["fname", "process"]]
        raise ValueError(
            "Some entries in final C do not have process='n':\n"
            + bad.to_string(index=False)
        )

    # Add C coefficients, UTI values, duplicate probabilities and 'bad_oc' flags
    df_UCC_C_final = add_info_to_C(current_JSON, df_UCC_B, df_UCC_C_final)

    # Check that the number of elements per unique 'name' group in
    # df_members_new matched the N_clust column in df_UCC_C_final
    check_N_clust(logging, df_UCC_C_final, df_members_new)

    # Split C database into in and out dataframes
    cols_in = [
        "fname",
        "process",
        "frame_limit",
        "N_clust",
        "N_clust_max",
        "use_mag_fastmp",
    ]
    cols_out = ["fname"] + [col for col in df_UCC_C_final.columns if col not in cols_in]
    df_UCC_C_in = df_UCC_C_final[cols_in].copy()
    df_UCC_C_out = df_UCC_C_final[cols_out].copy()

    # Check differences between the original and final C dataframes
    c_dict = {
        "in": (UCC_cat_C_in, df_UCC_C[cols_in], df_UCC_C_in),
        "out": (UCC_cat_C_out, df_UCC_C[cols_out], df_UCC_C_out),
    }
    for c_id, c_tuple in c_dict.items():
        file_c, df_C_old, df_C_new = c_tuple
        diff_found = diff_between_dfs(logging, f"C_{c_id} cat", df_C_old, df_C_new)
        if diff_found:
            # Save updated UCC to temporary CSV file
            save_df_UCC(logging, df_C_new, temp_folder + file_c)

    # Save the generated data to temporary files before moving them
    update_zenodo_files(
        logging,
        temp_zenodo_fold,
        old_zenodo_cat,
        all_names,
        df_UCC_B,
        df_UCC_C_final,
        df_members_new,
        flag_membs_changed,
    )

    if input("\nMove files to their final paths? (y/n): ").lower() == "y":
        files_moved = move_files(
            logging,
            temp_zenodo_fold,
            rename_C_fname,
            C_not_in_B,
            list(df_UCC_C_final["fname"]),
        )
        # Final check that the names in the members file match those in the UCC
        # C dataframe after moving files, and that the number of members per cluster
        # matches the N_clust column
        if files_moved:
            df_C_out_check = load_BC_cats("C", data_folder + UCC_cat_C_out)
            df_M_check = pd.read_parquet(zenodo_folder + UCC_members_file)
            check_sorted_names(logging, df_C_out_check, df_M_check)
            check_N_clust(logging, df_C_out_check, df_M_check)


def get_paths_check_paths(logging) -> tuple[str, str, str, str]:
    """
    Generate paths for required files and check for their existence.
    """
    txt = ""
    # Check for Gaia files
    if not os.path.isfile(path_gaia_frames_ranges):
        # raise FileNotFoundError(f"File {path_gaia_frames_ranges} is not present")
        txt += f"File {path_gaia_frames_ranges} is not present"
    if txt != "":
        logging.info(txt)
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit(1)

    # If temp file exists, warn
    c_temp_files = []
    for UCC_cat_C in (UCC_cat_C_in, UCC_cat_C_out):
        temp_f = temp_folder + UCC_cat_C
        if os.path.isfile(temp_f):
            c_temp_files.append(temp_f)
    if c_temp_files:
        logging.warning(
            f"WARNING: file(s) {c_temp_files} exists. Moving on will DELETE it"
        )
        if input("Move on? (y/n): ").lower() == "y":
            # Delete file
            for temp_f in c_temp_files:
                logging.warning(f"Removing stale temporary file: {temp_f}")
                os.remove(temp_f)
        else:
            sys.exit(1)

    # Create folder to store the per-cluster parquet member files
    if not os.path.exists(temp_members_folder):
        os.makedirs(temp_members_folder)
    else:
        # This could be desired to avoid re-estimating membership for OCs already
        # processed, hence the user is asked how to proceed
        if len(os.listdir(temp_members_folder)) > 0:
            logging.warning(
                f"WARNING: There are .parquet files in '{temp_members_folder}'. If "
                "left there,\nthey will be used when the script combines the final "
                "members data"
            )
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit(1)

    # Temporary zenodo/ folder
    temp_zenodo_fold = temp_folder + zenodo_folder
    # Create if required
    if not os.path.exists(temp_zenodo_fold):
        os.makedirs(temp_zenodo_fold)
    # Remove stale staged outputs from previous runs
    for fname in ("README.txt", zenodo_cat_fname, UCC_members_file):
        fpath = temp_zenodo_fold + fname
        if os.path.isfile(fpath):
            logging.warning(f"Removing stale temporary file: {fpath}")
            os.remove(fpath)

    # Path to the current UCC csv files
    ucc_B_file_out = data_folder + UCC_cat_B_out
    ucc_C_file_in = data_folder + UCC_cat_C_in
    ucc_C_file_out = data_folder + UCC_cat_C_out

    return ucc_B_file_out, ucc_C_file_in, ucc_C_file_out, temp_zenodo_fold


def load_data(
    logging,
    ucc_B_file_out: str,
    ucc_C_file_in: str,
    ucc_C_file_out: str,
    sep: str = ";",
) -> tuple[
    pd.DataFrame,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load required data files for processing the UCC (Unified Cluster Catalogue).
    """
    # Load file with Gaia frames ranges
    gaia_frames_data = pd.DataFrame([])
    if os.path.isfile(path_gaia_frames_ranges):
        gaia_frames_data = pd.read_csv(path_gaia_frames_ranges)

    # Load clusters data in JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    new_ocs_manual_pars = pd.DataFrame([])
    new_ocs_manual_pars_fpath = data_folder + "new_ocs_manual_pars.csv"
    if os.path.isfile(new_ocs_manual_pars_fpath):
        new_ocs_manual_pars = pd.read_csv(new_ocs_manual_pars_fpath)
        # 'frame_limit' column should be string and NaN values "nan"
        new_ocs_manual_pars["frame_limit"] = (
            new_ocs_manual_pars["frame_limit"].astype("string").fillna("nan")
        )
        # build normalized key
        new_ocs_manual_pars["fname"] = new_ocs_manual_pars["Name"].map(normalize_name)
        new_ocs_manual_pars.drop(columns=["Name"], inplace=True)
        new_ocs_manual_pars = new_ocs_manual_pars.set_index("fname")

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    # Load current members file
    df_members = pd.read_parquet(zenodo_folder + UCC_members_file)

    # Load current CSV data files
    all_names = pd.read_csv(data_folder + all_OC_names)
    df_UCC_B_out = load_BC_cats("B", ucc_B_file_out)
    logging.info(f"\nFile {ucc_B_file_out} loaded ({len(df_UCC_B_out)} entries)")
    df_UCC_C_in = load_BC_cats("C", ucc_C_file_in)
    logging.info(f"File {ucc_C_file_in}  loaded ({len(df_UCC_C_in)} entries)")
    df_UCC_C_out = load_BC_cats("C", ucc_C_file_out)
    logging.info(f"File {ucc_C_file_out} loaded ({len(df_UCC_C_out)} entries)")

    # Check that every alias in all_names["fnames"] occurs only once
    all_aliases = [
        fname for fnames in all_names["fnames"] for fname in fnames.split(sep)
    ]
    alias_counts = Counter(all_aliases)
    if any(count > 1 for count in alias_counts.values()):
        duplicates = {
            fname: count for fname, count in alias_counts.items() if count > 1
        }
        raise ValueError(
            f"Duplicate aliases found in all_names['fnames']: {duplicates}"
        )
    # Validate that df_UCC_B["fname"] and df_UCC_C["fname"] are unique
    if df_UCC_B_out["fname"].duplicated().any():
        raise ValueError("Duplicate 'fname' values found in df_UCC_B_out")
    if df_UCC_C_in["fname"].duplicated().any():
        raise ValueError("Duplicate 'fname' values found in df_UCC_C_in")
    if df_UCC_C_out["fname"].duplicated().any():
        raise ValueError("Duplicate 'fname' values found in df_UCC_C_out")
    if not (df_UCC_C_in["fname"] == df_UCC_C_out["fname"]).all():
        raise ValueError("Mismatch between df_UCC_C_in/out 'fname' values")
    else:
        # Merge the two C dataframes df_UCC_C_in,df_UCC_C_out using the fname column
        df_UCC_C = pd.merge(
            df_UCC_C_in,
            df_UCC_C_out,
            on="fname",
        )

    old_zenodo_cat = pd.read_csv(
        zenodo_folder + zenodo_cat_fname,
        dtype={"name": "string[python]"},
    )

    return (
        gaia_frames_data,
        current_JSON,
        new_ocs_manual_pars,
        df_GCs,
        df_members,
        all_names,
        df_UCC_B_out,
        df_UCC_C,
        old_zenodo_cat,
    )


def detect_entries_to_process(
    logging,
    all_names: pd.DataFrame,
    df_UCC_B: pd.DataFrame,
    df_UCC_C: pd.DataFrame,
    sep: str = ";",
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """
    The all_names["fnames"] column contains all the possible names (called 'fnames'
    for 'file names') associated ot a cluster, with the canonical fname positioned
    first. There are no repeated fnames in this column (checked in an earlier script).
    The canonical name is the one that is used in the UCC to identify a cluster.

    The logic to detect which entries in C should be renames or removed is as
    follows:

    for each fname in C:
        if it is in B:
            keep it in C
        else:
            find its canonical name in all_names
            if the canonical name is not found:
                raise an error (this should never happen)
            else:
                if the C fname does not match the canonical:
                    if the canonical name is in C:
                        This means that the fname in C is a merge and should be removed
                    else:
                        This means that the fname in C is a rename to the canonical
                        name
                else:
                    raise an error (this should never happen)

    C_not_in_B     --> Remove from C
    rename_C_fname --> Rename in C
    B_not_in_C     --> Add to C
    C_reprocess    --> Reprocess in C

    The logic to detect entries in C that

    """
    # Build a mapping of all aliases to their canonical names in B
    alias_to_canonical = {}
    for fnames in all_names["fnames"]:
        fnames_lst = fnames.split(sep)
        canonical = fnames_lst[0]
        for fname in fnames_lst:
            alias_to_canonical[fname] = canonical

    # Find entries in C that must be renamed or removed
    C_to_remove = list(df_UCC_C[~df_UCC_C["fname"].isin(df_UCC_B["fname"])]["fname"])

    # Find entries in 'C_to_remove' that are present in 'B' but with a different main
    # name (i.e. they just need renaming in C_not_in_B
    rename_C_fname, C_not_in_B = {}, {}
    for C_fname in C_to_remove:
        # Find the canonical name for this fname in C that is not in B
        canonical = alias_to_canonical.get(C_fname)
        if canonical is None:
            # This means that a fname was completely removed which should (almost)
            # never happen
            raise ValueError(
                f"Name {C_fname} in C not found in all 'fnames'\n"
                "This means that an object was completely removed which should\n"
                "(almost) never happen. If it does, manual editing of this database\n"
                "and also the members file is required"
            )
        else:
            if C_fname != canonical:
                if canonical in df_UCC_C["fname"].values:
                    # The canonical of C_fname is present in C fnames. This means that
                    # this is a merge 'C_fname --> canonical' and C_fname must be
                    # removed
                    C_not_in_B[C_fname] = canonical
                else:
                    # The canonical of C_fname is not present in C fnames. This means
                    # that C_fname was renamed to canonical
                    rename_C_fname[C_fname] = canonical
            else:
                # C_fname was not found in B fnames but matches a canonical fname.
                # This should never happen because a cluster is never fully removed
                # without merging or renaming it
                raise ValueError(
                    f"Name {C_fname} in C not found in B but is a canonical fname"
                )

    # Check for multiple C entries mapping to the same canonical name.
    # This would create duplicate 'fname' values in the final C catalogue.
    if len(rename_C_fname) > 0:
        # Count how many times each canonical name appears in the rename mapping
        canonical_counts = Counter(rename_C_fname.values())
        # Find canonical names that have more than one old name mapping to them
        duplicates = {
            canon: [old for old, new in rename_C_fname.items() if new == canon]
            for canon, count in canonical_counts.items()
            if count > 1
        }
        if duplicates:
            details = "\n".join(
                f"  {canon} <- {', '.join(olds)}" for canon, olds in duplicates.items()
            )
            raise ValueError(
                "Multiple C entries map to the same canonical name:\n"
                f"{details}\n"
                "This would create duplicate 'fname' values in the C catalogue. "
                "Please resolve manually (e.g., merge the aliases explicitly)."
            )

    # Entries in B that must be added to C
    B_not_in_C = df_UCC_B[~df_UCC_B["fname"].isin(df_UCC_C["fname"])]
    if len(rename_C_fname) > 0:
        # Remove the entries that just need renaming
        B_not_in_C = B_not_in_C[~B_not_in_C["fname"].isin(rename_C_fname.values())]

    # Entries manually marked for re-processing in C
    msk = df_UCC_C["process"] == "y"
    C_reprocess = df_UCC_C[msk].copy()

    ###############################################################################
    # Sanity check
    reprocess_fnames = set(C_reprocess["fname"])

    overlap = reprocess_fnames & set(C_not_in_B.keys())
    if overlap:
        details = ", ".join(f"{f} --> {C_not_in_B[f]}" for f in overlap)
        raise ValueError(
            f"Entries marked process='y' are flagged for removal (merge): {details}"
        )

    overlap = reprocess_fnames & set(rename_C_fname.keys())
    if overlap:
        details = ", ".join(f"{f} --> {rename_C_fname[f]}" for f in overlap)
        raise ValueError(
            f"Entries marked process='y' are also flagged for renaming: {details}"
        )

    ###############################################################################
    # Print summary of results
    def show_items(label, items, formatter=str, limit=50):
        n = len(items)
        logging.info(f"\n{label:20}: {n}")
        if n == 0:
            return
        if n <= limit or input("Show list? (y/n): ").strip().lower() == "y":
            for item in items:
                logging.info(formatter(item))

    logging.info("\nProcessing:")
    show_specs = [
        (
            "1. ADD (B entries to C)",
            list(B_not_in_C.itertuples(index=False)),
            lambda row: f"    {row.fname:<20}{f'({row.DB})':>20}",
        ),
        (
            "2. RENAME (changed main fname)",
            list(rename_C_fname.items()),
            lambda x: f"    {x[0]:10} --> {x[1]}",
        ),
        (
            "3. REMOVE (merged into another fname)",
            list(C_not_in_B.items()),
            lambda x: f"    {x[0]:10} --> {x[1]}",
        ),
        (
            "4. RE-PROCESS (C entries)",
            list(C_reprocess.itertuples(index=False)),
            lambda row: f"    {row.fname}",
        ),
    ]

    for label, items, formatter in show_specs:
        show_items(label, items, formatter)

    # Total number of entries to process
    N_process = (
        len(rename_C_fname) + len(B_not_in_C) + len(C_not_in_B) + len(C_reprocess)
    )
    msg = "no"
    if N_process > 0:
        msg = f"{N_process}"
    if input(f"\nThere are {msg} entries to process. Continue? (y/n): ").lower() != "y":
        sys.exit(1)

    return rename_C_fname, C_not_in_B, B_not_in_C, C_reprocess


def load_temp_C_updt_file(
    logging,
    temp_UCC_updt_file: str,
    B_not_in_C: pd.DataFrame,
    C_reprocess: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load and validate the temporary update file for the Unified Cluster Catalogue (UCC)
    """
    # Load previously generated update file
    str_cols = ["frame_limit", "use_mag_fastmp", "shared_members", "shared_members_p"]
    df_UCC_C_updt = pd.read_csv(
        temp_UCC_updt_file, dtype={c: "string" for c in str_cols}
    )
    df_UCC_C_updt[str_cols] = df_UCC_C_updt[str_cols].fillna("nan")
    # Clusters that should be present in the update file for this run
    expected_names = set(B_not_in_C["fname"]) | set(C_reprocess["fname"])

    # Check for duplicated names
    duplicated = sorted(
        df_UCC_C_updt.loc[
            df_UCC_C_updt["fname"].duplicated(keep=False), "fname"
        ].unique()
    )

    if duplicated:
        raise ValueError(
            "Duplicated 'fname' entries in df_UCC_C_updt:\n"
            + "\n".join(f"  {name}" for name in duplicated)
        )

    # Check that the loaded file corresponds exactly to this run
    loaded_names = set(df_UCC_C_updt["fname"])

    unexpected = sorted(loaded_names - expected_names)
    missing = sorted(expected_names - loaded_names)

    if unexpected or missing:
        logging.info("\nLoaded df_UCC_C_updt does not match the current run.")
        if unexpected:
            logging.info("\nUnexpected entries in df_UCC_C_updt:")
            logging.info("  " + "\n  ".join(unexpected))
        if missing:
            logging.info("\nExpected entries missing from df_UCC_C_updt:")
            logging.info("  " + "\n  ".join(missing))
        raise ValueError(
            "The loaded df_UCC_C_updt.csv does not correspond to the current run."
        )

    logging.info(
        f"\nTemp file df_UCC_C_updt loaded and validated (N={len(df_UCC_C_updt)})"
    )

    return df_UCC_C_updt


def process_entries(
    df_UCC_B_out: pd.DataFrame,
    new_ocs_manual_pars: pd.DataFrame,
    B_not_in_C: pd.DataFrame,
    C_reprocess: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process entries to be added or reprocessed in the UCC
    """
    # Rows to reprocess. Merge with df_UCC_B to recover B columns
    part_C = C_reprocess.merge(
        df_UCC_B_out,  # .drop(columns=["fnames"]),
        on="fname",
        how="left",
    )
    # Combine both blocks
    df_add_reprocess = pd.concat([B_not_in_C, part_C], ignore_index=True).replace(
        {pd.NA: "nan"}
    )

    if not df_add_reprocess.empty and not new_ocs_manual_pars.empty:
        # index df_add_reprocess temporarily on fname
        df_add_reprocess = df_add_reprocess.set_index("fname")
        # replace values for matching entries
        common = df_add_reprocess.index.intersection(new_ocs_manual_pars.index)
        cols = list(new_ocs_manual_pars.keys())
        df_add_reprocess.loc[common, cols] = new_ocs_manual_pars.loc[
            common, cols
        ].values
        # restore fname as column
        df_add_reprocess = df_add_reprocess.reset_index()

    if (
        any(np.isnan(df_add_reprocess["N_clust_max"]))
        and input("\nSet a general N_clust_max value? (y/n): ").lower() == "y"
    ):
        N_clust_max_general = int(input("Enter N_clust_max value: "))
        # Update the 'df_add_reprocess['N_clust_max']' column with this value
        df_add_reprocess["N_clust_max"] = N_clust_max_general

    return df_add_reprocess


def member_files_updt(
    logging,
    gaia_frames_data,
    df_GCs: pd.DataFrame,
    df_UCC_C: pd.DataFrame,
    df_UCC_C_updt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Updates the Unified Cluster Catalogue (UCC) with new open clusters (OCs).
    """
    if df_UCC_C_updt.empty:
        return df_UCC_C_updt

    # Used by close objects check
    df_UCC_m = df_UCC_C[
        ["fname", "GLON_m", "GLAT_m", "Plx_m", "pmRA_m", "pmDE_m"]
    ].copy()

    N_tot = len(df_UCC_C_updt)
    for idx in df_UCC_C_updt.index:
        cl_row = df_UCC_C_updt.loc[idx]
        # Extract some data
        fname0, ra_c, dec_c, glon_c, glat_c, pmra_c, pmde_c, plx_c = (
            cl_row["fname"],
            float(cl_row["RA_ICRS"]),
            float(cl_row["DE_ICRS"]),
            float(cl_row["GLON"]),
            float(cl_row["GLAT"]),
            float(cl_row["pmRA"]),  # This can be nan
            float(cl_row["pmDE"]),  # This can be nan
            float(cl_row["Plx"]),  # This can be nan
        )
        logging.info(f"\n{idx + 1}/{N_tot} Processing {fname0}")

        # Extract manual parameters if any
        N_clust, N_clust_max, use_mag_fastmp, frame_limit = cl_row[
            ["N_clust", "N_clust_max", "use_mag_fastmp", "frame_limit"]
        ]

        # Generate frame_limits dictionary
        frame_limit_dict = {}
        if str(frame_limit) != "nan":
            # Extract possible manual frame limits
            for fm in frame_limit.split(";"):
                vals = fm.split("_")
                if vals[0] not in frame_limits_cols:
                    raise ValueError(f"Unknown frame limit '{vals[0]}'")
                frame_limit_dict[vals[0]] = float(vals[1])

        # Obtain the full Gaia frame
        gaia_frame = get_gaia_frame(
            logging, gaia_frames_data, fname0, ra_c, dec_c, plx_c, frame_limit_dict
        )
        # gaia_frame.to_csv("temp_clust.csv", index=False)

        # Obtain the cluster members (and field stars) using fastMP
        df_field, df_membs = get_fastMP_membs(
            logging,
            df_GCs,
            df_UCC_m,
            fname0,
            ra_c,
            dec_c,
            glon_c,
            glat_c,
            pmra_c,
            pmde_c,
            plx_c,
            N_clust,
            N_clust_max,
            use_mag_fastmp,
            gaia_frame,
        )

        # This should never happen, check anyway
        if len(df_membs) == 0:
            raise ValueError(
                f"No members found for {fname0}. Check parameters and try again"
            )

        # Write selected member stars to file
        save_cl_datafile(logging, fname0, df_membs)

        # Classification data
        C1, C2, C3 = get_classif(df_membs, df_field)

        # Extract data from the members array
        (
            c_lon,
            c_lat,
            c_ra,
            c_dec,
            c_plx,
            e_plx,
            c_pmRA,
            c_pmDE,
            e_pmRA,
            e_pmDE,
            c_Rv,
            e_Rv,
            N_Rv,
            X_GC,
            Y_GC,
            Z_GC,
            R_GC,
            N_membs,
            r_50,
            r_core,
            dens_core,
        ) = get_new_cl_data(df_membs)

        # update 'df_UCC_C_updt' with the extracted data for this cluster
        df_UCC_C_updt = updt_UCC_new_cl_data(
            idx,
            df_UCC_C_updt,
            C1,
            C2,
            C3,
            c_lon,
            c_lat,
            c_ra,
            c_dec,
            c_plx,
            e_plx,
            c_pmRA,
            c_pmDE,
            e_pmRA,
            e_pmDE,
            c_Rv,
            e_Rv,
            N_Rv,
            X_GC,
            Y_GC,
            Z_GC,
            R_GC,
            N_membs,
            r_50,
            r_core,
            dens_core,
        )

        # Update file with information. Do this for each iteration to avoid
        # losing data if something goes wrong with any cluster
        df_UCC_C_updt.to_csv(
            temp_folder + "df_UCC_C_updt.csv", index=False, na_rep="nan"
        )

    return df_UCC_C_updt


def update_C_cat(
    C_not_in_B: dict,
    rename_C_fname: dict,
    df_UCC_C: pd.DataFrame,
    df_UCC_C_updt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Update the UCC database using the data extracted from the processed OCs'
    members.
    """
    df_UCC_C_new = df_UCC_C.copy()

    # Rename entries
    if len(rename_C_fname) > 0:
        msk = df_UCC_C_new["fname"].isin(rename_C_fname.keys())
        df_UCC_C_new.loc[msk, "fname"] = df_UCC_C_new.loc[msk, "fname"].map(
            rename_C_fname
        )

    # Remove entries in C_not_in_B from df_UCC_C
    if len(C_not_in_B) > 0:
        msk = ~df_UCC_C_new["fname"].isin(C_not_in_B.keys())
        df_UCC_C_new = df_UCC_C_new[msk]

    # Update df_UCC_C_new using data from df_UCC_C_updt
    # Ensure 'fname' is the index in both DataFrames
    A = df_UCC_C_new.set_index("fname")
    B = df_UCC_C_updt.set_index("fname")

    # Update existing rows in A with values from B
    # Overwrite existing rows in A with ALL values from B, including NaN.
    # The line `A.update(B)` skips NaN in B, so we assign directly for matching
    # indices/columns.
    common_idx = A.index.intersection(B.index)
    common_cols = A.columns.intersection(B.columns)
    if len(common_idx) == 0 and len(common_cols) == 0:
        raise ValueError("No common indices or columns found in 'update_C_cat()'")
    A.loc[common_idx, common_cols] = B.loc[common_idx, common_cols]

    # Identify new rows in B
    new_rows = B.loc[~B.index.isin(A.index)]
    # Drop completely empty columns
    new_rows = new_rows.dropna(axis=1, how="all")

    # Concatenate and sort
    A = pd.concat([A, new_rows], axis=0)
    # Restore 'fnames' as a column
    df_UCC_C_new = A.reset_index()

    # Reset indexes and restore column order
    df_UCC_C_new = df_UCC_C_new.reindex(columns=df_UCC_C.columns)
    df_UCC_C_new = df_UCC_C_new.sort_values("fname")
    df_UCC_C_new = df_UCC_C_new.reset_index(drop=True)

    return df_UCC_C_new


def check_names_before_members_combine(
    logging, B_not_in_C: pd.DataFrame, C_reprocess: pd.DataFrame
):
    """
    Check that the temporary member files in 'temp_members_folder' match the expected
    clusters to be processed (B_not_in_C and C_reprocess). If there are discrepancies,
    log the unexpected or missing files and raise a ValueError.
    """
    # Expected clusters with newly generated member files
    expected_temp_names = set(B_not_in_C["fname"]) | set(C_reprocess["fname"])

    # Actual cluster names found in temp_members_folder
    temp_names = {
        file.removesuffix(".parquet")
        for file in os.listdir(temp_members_folder)
        if file.endswith(".parquet")
    }

    if temp_names != expected_temp_names:
        only_temp = sorted(temp_names - expected_temp_names)
        missing_temp = sorted(expected_temp_names - temp_names)
        logging.info("\nTemporary member files do not match expected clusters:")
        if only_temp:
            logging.info("\nUnexpected files in temp_members_folder:")
            logging.info("  " + "\n  ".join(only_temp))

        if missing_temp:
            logging.info("\nExpected member files not present:")
            logging.info("  " + "\n  ".join(missing_temp))
        raise ValueError(
            "Temporary member files do not match the clusters being processed."
        )


def gen_comb_members_file(logging) -> pd.DataFrame:
    """Combine individual parquet files into a single temporary one"""

    # Path to folder with individual .parquet files
    member_files = [
        file for file in os.listdir(temp_members_folder) if file.endswith(".parquet")
    ]
    if len(member_files) == 0:
        return pd.DataFrame([])

    logging.info(f"Combining {len(member_files)} .parquet files...")
    tmp = []
    for file in member_files:
        df = pd.read_parquet(temp_members_folder + file)

        # Round before storing
        df[["RA_ICRS", "DE_ICRS", "GLON", "GLAT"]] = df[
            ["RA_ICRS", "DE_ICRS", "GLON", "GLAT"]
        ].round(6)
        df[
            [
                "Plx",
                "e_Plx",
                "pmRA",
                "e_pmRA",
                "pmDE",
                "e_pmDE",
                "RV",
                "e_RV",
                "Gmag",
                "BP-RP",
                "e_Gmag",
                "e_BP-RP",
                "probs",
            ]
        ] = df[
            [
                "Plx",
                "e_Plx",
                "pmRA",
                "e_pmRA",
                "pmDE",
                "e_pmDE",
                "RV",
                "e_RV",
                "Gmag",
                "BP-RP",
                "e_Gmag",
                "e_BP-RP",
                "probs",
            ]
        ].round(4)

        fname = file.replace(".parquet", "")
        df.insert(loc=0, column="name", value=fname)
        tmp.append(df)

    # Concatenate all temporary DataFrames into one
    df_comb = pd.concat(tmp, ignore_index=True)

    return df_comb


def update_membs_file(
    rename_C_fname: dict,
    C_not_in_B: dict,
    df_members: pd.DataFrame,
    df_comb: pd.DataFrame,
) -> tuple[bool, pd.DataFrame]:
    """
    Update the parquet file containing estimated members from the
    Unified Cluster Catalog (UCC) dataset, formatted for storage in the Zenodo
    repository.
    """
    # df_updated = df_members.copy()

    # # Rename entries
    # if len(rename_C_fname) > 0:
    #     msk = df_updated["name"].isin(rename_C_fname.keys())
    #     df_updated.loc[msk, "name"] = df_updated.loc[msk, "name"].map(rename_C_fname)

    # # Remove entries in C_not_in_B
    # if len(C_not_in_B) > 0:
    #     msk = ~df_updated["name"].isin(C_not_in_B.keys())
    #     df_updated = pd.DataFrame(df_updated[msk])

    # if not df_comb.empty:
    #     # Get the list of names in each DataFrame
    #     names_df1 = set(df_updated["name"])
    #     names_df2 = set(df_comb["name"])
    #     # Identify names in df_updated not in df_comb
    #     extra_names = names_df1 - names_df2
    #     # Filter df_updated for those extra groups
    #     df1_extra = df_updated[df_updated["name"].isin(extra_names)]
    #     # Concatenate df_comb with the extra df_members groups
    #     df_members_new = pd.concat([df_comb, df1_extra], ignore_index=True)
    #     df_members_new = pd.DataFrame(df_members_new).sort_values("name")
    # else:
    #     df_members_new = df_updated.copy()
    # df_members_new = df_members_new.sort_values("name").reset_index(drop=True)

    # Fast exit: nothing to do
    if not rename_C_fname and not C_not_in_B and df_comb.empty:
        return False, df_members

    df_updated = df_members

    # Rename entries only if necessary
    if rename_C_fname:
        msk = df_updated["name"].isin(rename_C_fname)
        if msk.any():
            df_updated = df_updated.copy()
            df_updated.loc[msk, "name"] = df_updated.loc[msk, "name"].map(
                rename_C_fname
            )

    # Remove entries only if necessary
    if C_not_in_B:
        msk = ~df_updated["name"].isin(C_not_in_B)
        if not msk.all():
            if df_updated is df_members:
                df_updated = df_updated.copy()
            df_updated = df_updated.loc[msk]

    if df_comb.empty:
        df_members_new = df_updated.sort_values("name").reset_index(drop=True)
        return True, df_members_new

    # Append groups present only in df_updated
    extra = df_updated.loc[~df_updated["name"].isin(df_comb["name"])]
    df_members_new = (
        pd.concat([df_comb, extra], ignore_index=True)
        .sort_values("name")
        .reset_index(drop=True)
    )

    return True, df_members_new


def check_sorted_names(
    logging, df_UCC_C_new: pd.DataFrame, df_members_new: pd.DataFrame
) -> None:
    """
    Check that the names in the members file match those in the UCC C dataframe
    """
    # Check that members file is sorted by 'name' and that each name appears
    # in a single contiguous block.
    if not df_members_new["name"].is_monotonic_increasing:
        raise ValueError("Members file is not sorted by 'name'")
    # Check that each name appears in a single contiguous block
    seen = set()
    current = None
    for name in df_members_new["name"]:
        if name != current:
            if name in seen:
                raise ValueError(
                    f"Name '{name}' appears in non-contiguous blocks in members file"
                )
            seen.add(name)
            current = name

    # Check that the names in the members file match those in the UCC C dataframe
    names_members = sorted(df_members_new["name"].unique())
    names_C = sorted(df_UCC_C_new["fname"])
    if names_members != names_C:
        only_members = sorted(set(names_members) - set(names_C))
        only_C = sorted(set(names_C) - set(names_members))
        duplicated_C = sorted(
            df_UCC_C_new.loc[
                df_UCC_C_new["fname"].duplicated(keep=False), "fname"
            ].unique()
        )
        logging.info("\nError found:")
        logging.info(f"N unique members names: {len(names_members)}")
        logging.info(f"N C rows              : {len(names_C)}")
        logging.info(f"N unique C names      : {len(set(names_C))}")
        if only_members:
            logging.info("\nPresent only in members:")
            logging.info("  " + "\n  ".join(only_members))
        if only_C:
            logging.info("\nPresent only in C:")
            logging.info("  " + "\n  ".join(only_C))
        if duplicated_C:
            logging.info("\nDuplicated in C:")
            logging.info("  " + "\n  ".join(duplicated_C))
        logging.info("\n")
        raise ValueError("Final names do not match between members and C dataframes.")


def find_shared_members(
    logging, df_UCC_C_new: pd.DataFrame, df_members_new: pd.DataFrame
) -> pd.DataFrame:
    """
    Find shared members between OCs and update df_UCC_C_new dataframe.
    """
    logging.info("Finding shared members...")

    # Find OCs that intersect. This helps to speed up the process
    intersection_map = find_intersections(df_members_new)

    # Group members by 'fname' (called 'name' in df_members_new)
    grouped = df_members_new.groupby("name")["Source"].apply(set)
    N_total = len(grouped)
    results = {
        "fname": grouped.keys().tolist(),
        "shared_members": ["nan"] * N_total,  # Requires "nan" strings
        "shared_members_p": ["nan"] * N_total,  # Requires "nan" strings
    }

    # Compute shared elements and percentages
    for idx, (fname, sources) in enumerate(grouped.items()):
        if fname not in intersection_map:
            continue

        shared_info, percentage_info, percentage_vals = [], [], []
        for other_fname in intersection_map[fname]:
            other_sources = grouped[other_fname]

            shared = sources & other_sources
            if shared:
                shared_info.append(other_fname)
                percentage = len(shared) / len(sources) * 100
                percentage_vals.append(percentage)
                percentage_info.append(f"{percentage:.1f}")

        if shared_info:
            if len(shared_info) > 1:
                # Sort by max values first and name second
                i_sort = np.lexsort((shared_info, -np.array(percentage_vals)))
                shared_info = [shared_info[i] for i in i_sort]
                percentage_info = [percentage_info[i] for i in i_sort]
            results["shared_members"][idx] = ";".join(shared_info)
            results["shared_members_p"][idx] = ";".join(percentage_info)

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    result_df["fname"] = pd.Categorical(
        result_df["fname"], categories=df_UCC_C_new["fname"], ordered=True
    )
    result_df = result_df.sort_values("fname").reset_index(drop=True)

    if result_df["fname"].tolist() != df_UCC_C_new["fname"].tolist():
        raise ValueError("The 'fname' columns do not match in 'find_shared_members'")

    # Update data columns for shared members
    df_UCC_C_new = df_UCC_C_new.reset_index(drop=True)
    df_UCC_C_new[["shared_members", "shared_members_p"]] = result_df[
        ["shared_members", "shared_members_p"]
    ]
    df_UCC_C_final = df_UCC_C_new.sort_values("fname").reset_index(drop=True)

    return df_UCC_C_final


def find_intersections(df_members: pd.DataFrame) -> dict:
    """
    Find OCs that share at least one member 'Source', regardless of
    spatial separation between the OCs.
    """
    # Identify sources that appear in more than one row
    source_counts = df_members["Source"].value_counts()
    shared_sources = source_counts[source_counts > 1].index

    # Restrict to only the rows carrying a shared source before grouping
    df_shared = df_members[df_members["Source"].isin(shared_sources)]

    # Now group only over the reduced set
    source_to_ocs = df_shared.groupby("Source")["name"].apply(set)

    intersection_map: dict[str, set] = {}
    for ocs in source_to_ocs:
        if len(ocs) > 1:
            for name in ocs:
                intersection_map.setdefault(name, set()).update(ocs - {name})

    return intersection_map


def add_info_to_C(
    current_JSON: dict,
    df_UCC_B: pd.DataFrame,
    df_UCC_C: pd.DataFrame,
    N_memb_min: int = 5,
    max_dens: float = 5.0,
    N_lit_min: int = 2,
    C_lit_perc_max: float = 0.5,
) -> pd.DataFrame:
    """
    Compute quality metrics and the Unified Trust Index (UTI) for all catalogue
    entries.

    The following normalized metrics are computed:

    - C_N: confidence based on the number of members.
    - C_dens: confidence based on projected core stellar density.
    - C_C3: confidence derived from the C3 classification.
    - C_lit: confidence based on literature coverage.
    - C_dup: confidence that the entry is *not* a duplicate.
    - UTI: overall quality score combining the previous metrics.

    Duplicate confidence is estimated by comparing publication dates and member
    overlap with other catalogue entries.

    Parameters
    ----------
    current_JSON : dict
        Literature metadata indexed by database name. Each entry must contain a
        "received" publication date.
    df_UCC_B : pandas.DataFrame
        Catalogue table containing literature information.
    df_UCC_C : pandas.DataFrame
        Catalogue table containing cluster properties.
    N_memb_min : int, default=5
        Number of members below which C_N is forced to zero.
    max_dens : float, default=5
        Core stellar density (pc^-2) corresponding to C_dens = 1.
    N_lit_min : int, default=2
        Number of literature references below which C_lit is zero.
    C_lit_perc_max : float, default=0.5
        Fraction of the maximum literature coverage corresponding to C_lit = 1.

    Returns
    -------
    pandas.DataFrame
        df_UCC_C with the quality metrics, duplication statistics,
        UTI, and bad-object flag added.
    """

    def normalize(N, arr, Nmin, Nmax, vmin, vmax):
        msk2 = (N >= Nmin) & (N < Nmax)
        arr[msk2] = vmin + ((N[msk2] - Nmin) / (Nmax - Nmin)) * (vmax - vmin)

    #
    # C_N_membs
    N_membs = df_UCC_C["N_membs"].to_numpy(dtype=float)
    C_N_membs = np.ones(len(N_membs))
    C_N_membs[N_membs < N_memb_min] = 0.0
    # Define intervals and mapping ranges
    bounds = (0.0, 0.05, 0.5, 0.75, 0.9)
    Nvals = (N_memb_min, 20, 50, 75, 100)
    for i in range(1, len(bounds)):
        normalize(N_membs, C_N_membs, Nvals[i - 1], Nvals[i], bounds[i - 1], bounds[i])

    #
    # C_dens
    C_dens = np.clip((df_UCC_C["dens_core_pc2"] - 0) / (max_dens - 0), 0, 1)

    #
    # C_C3
    C3 = df_UCC_C["C3"].to_numpy(dtype=str)
    C3_SCORE = {
        "A": 1.00,
        "B": 0.50,
        "C": 0.25,
        "D": 0.00,
    }
    C_C3 = np.array([C3_SCORE[a[0]] + C3_SCORE[a[1]] for a in C3], dtype=float) * 0.5

    #
    # C_lit
    # Count number of times each OC is mentioned in the literature
    N_lit = np.array([len(_.split(";")) for _ in df_UCC_B["DB"]])
    # Normalizing value: max number of DBs for a single OC
    N_lit_tot = max(N_lit)
    N_lit_max = C_lit_perc_max * N_lit_tot
    C_lit = np.ones(len(N_lit))
    C_lit[N_lit <= N_lit_min] = 0.0
    # Define intervals and mapping ranges. Values with N_lit>=N_lit_max stay at 1
    bounds = (0, 0.5, 0.99)
    Nvals = (N_lit_min, 10, N_lit_max)
    for i in range(1, len(bounds)):
        normalize(N_lit, C_lit, Nvals[i - 1], Nvals[i], bounds[i - 1], bounds[i])

    #
    # C_dup
    # Estimate duplication probability by comparing each cluster with all
    # clusters sharing members. Older publications take precedence over newer
    # ones; publication received dates are used to break ties.
    # C_dup indicates the confidence that an entry is a duplicate of a previously
    # reported object. A value of 1 means not at all a duplicate
    #

    # Extract the first DB, year of publication and canonical fname for each entry
    # (assumes the DBs are already ordered by year)
    dbs = [_.split(";")[0] for _ in df_UCC_B["DB"]]
    f_year = [int(_.split("_")[0][-4:]) for _ in dbs]
    fnames = df_UCC_B["fname"]
    # Map years and dbs to canonical fnames
    fname_to_db_info = {
        name: {"year": year, "db": db} for name, year, db in zip(fnames, f_year, dbs)
    }

    C_dup = [100.0] * len(df_UCC_C)
    C_dup_same_db = [100.0] * len(df_UCC_C)
    for idx in df_UCC_C.index:
        cl = df_UCC_C.loc[idx]
        if str(cl["shared_members"]) == "nan":
            # This OC does not share members with any other
            continue

        # Extract the years and dbs associated to the entries that share members
        # with 'cl'
        shared_members = cl["shared_members"].split(";")
        fyears_shared, dbs_shared = [], []
        for s in shared_members:
            info = fname_to_db_info.get(s, {})
            fyears_shared.append(info["year"])
            dbs_shared.append(info["db"])
            # fyears_shared.append(fname_db_to_year[s][0])
            # dbs_shared.append(fname_db_to_year[s][1])

        # # Year of publication of 'cl'
        # f_year_cl = f_year[idx]
        # Publication details of 'cl'
        cl_info = fname_to_db_info.get(cl["fname"], {})
        f_year_cl = cl_info["year"]
        db_cl = cl_info["db"]

        if min(fyears_shared) > f_year_cl:
            # All entries that share members with 'cl' where published *after* 'cl',
            # 'cl' thus cannot be a duplicate of any of them
            continue

        # List of percentages of shared members between 'cl' and the entries that share
        # members with it
        shared_members_p = list(map(float, cl["shared_members_p"].split(";")))

        # Date when the initial article for  'cl' was received for publication
        date_received_cl = int(current_JSON[db_cl]["received"])

        shared_p = {"other_db": 0.0, "same_db": 0.0}
        for j, f_year_shared in enumerate(fyears_shared):
            if f_year_cl > f_year_shared:
                # If 'cl' is more recent than this entry, 'cl' is the duplicate
                shared_p["other_db"] = max(shared_p["other_db"], shared_members_p[j])
            elif f_year_cl == f_year_shared:
                # If the years are equal, use the received date to disambiguate
                date_received_shared = int(current_JSON[dbs_shared[j]]["received"])
                if date_received_cl > date_received_shared:
                    # If 'cl' is more recent than this entry, 'cl' is the duplicate
                    shared_p["other_db"] = max(
                        shared_p["other_db"], shared_members_p[j]
                    )
                elif date_received_cl == date_received_shared:
                    if db_cl == dbs_shared[j]:
                        # These entries share members AND they belong to the same DB
                        shared_p["same_db"] = max(
                            shared_p["same_db"], shared_members_p[j]
                        )
                    else:
                        # This should never happen
                        raise ValueError(
                            f"({idx}) {cl['fname']} & {shared_members[j]} share members"
                            f" and a publication date ({date_received_cl}), but are "
                            f"mentioned in different DBs ({db_cl, dbs_shared[j]})."
                            " This makes it impossible to disambiguate which one is the"
                            " duplicate of the other."
                        )
            # else:
            #     # If 'cl' is older than this entry,
            #     f_year_cl < f_year_shared
            #     'cl' cannot be a duplicate of it

        if shared_p["other_db"] > 0.0:
            # At least one entry that shares members with 'cl' belongs to a
            # different DB
            C_dup[idx] -= shared_p["other_db"]
        if shared_p["same_db"] > 0.0:
            # All entries that share members with 'cl' belong to the same DB
            C_dup_same_db[idx] -= shared_p["same_db"]

    C_dup = np.array(C_dup) / 100
    C_dup_same_db = np.array(C_dup_same_db) / 100

    # Store before modifying C_dup values. This is used by the 'D' script to flag
    # entries that share a members with other entries in others or the same DB
    df_UCC_C["C_dup_info"] = np.char.add(
        np.char.add(np.round(C_dup, 2).astype(str), ";"),
        np.round(C_dup_same_db, 2).astype(str),
    )

    # Replace 'C_dup' values with a smaller value only when 'C_dup_same_db<0.5', ie:
    # only entries that share a significant fraction of members with other entries in
    # the same DB.
    msk = C_dup_same_db < 0.5
    C_dup[msk] = np.minimum(C_dup[msk], C_dup_same_db[msk])

    #
    # Final UTI
    UTI = np.clip(0.2 * (C_N_membs + C_dens + C_C3 + 2 * C_lit) * C_dup, 0, 1)

    # Add data to df
    df_UCC_C["C_N"] = np.round(C_N_membs, 2)
    df_UCC_C["C_dens"] = np.round(C_dens, 2)
    df_UCC_C["C_C3"] = np.round(C_C3, 2)
    df_UCC_C["C_lit"] = np.round(C_lit, 2)
    df_UCC_C["C_dup"] = np.round(C_dup, 2)
    df_UCC_C["P_dup"] = np.round(1 - df_UCC_C["C_dup"], 2)
    df_UCC_C["UTI"] = np.round(UTI, 2)

    # All entries are by default "good" entries
    df_UCC_C["bad_oc"] = "n"
    # Flag as "bad_oc" possible asterisms, moving groups, or artifacts of some kind
    msk = (
        # Only include entries with very low UTI
        (df_UCC_C["UTI"] < UTI_max)
        # Only include entries with very low probability of duplication
        & (df_UCC_C["P_dup"] < P_dup_max)
        # Only include entries not studied in the literature
        & (df_UCC_C["C_lit"] < C_lit_max)
    )
    df_UCC_C.loc[msk, "bad_oc"] = "y"

    return df_UCC_C


def check_N_clust(
    logging, df_UCC_C_final: pd.DataFrame, df_members_new: pd.DataFrame
) -> None:
    """Check that the number of elements per unique 'name' group in df_members_new
    matched the N_clust column in df_UCC_C_final"""
    logging.info(
        "Checking that the number of members per cluster "
        "matches the N_membs column...\n"
    )

    # Check for duplicated (name, Source) pairs in df_members_new
    dup = df_members_new.duplicated(["name", "Source"], keep=False)
    if dup.any():
        bad = df_members_new.loc[dup, ["name", "Source"]].sort_values(
            ["name", "Source"]
        )
        raise ValueError(
            "Duplicated (name, Source) pairs found in members file:\n"
            + bad.to_string(index=False)
        )

    # Group by 'name' and count unique 'Source'
    member_counts = df_members_new.groupby("name")["Source"].nunique().reset_index()
    member_counts.rename(columns={"Source": "N_clust_actual"}, inplace=True)

    # Merge with df_UCC_C_final to compare with 'N_clust'
    merged = pd.merge(
        df_UCC_C_final,
        member_counts,
        left_on="fname",
        right_on="name",
        how="left",
    )

    # Check for mismatches
    mismatches = merged[merged["N_membs"] != merged["N_clust_actual"]]
    if not mismatches.empty:
        # The only allowed mismatch is:
        # C reports <25 members but the stored file contains the minimum 25.
        allowed = (mismatches["N_membs"] < 25) & (mismatches["N_clust_actual"] == 25)

        bad = mismatches[~allowed]
        if not bad.empty:
            for row in bad.itertuples():
                logging.warning(
                    f"  Cluster '{row.fname}': "
                    f"N_membs={row.N_membs} vs stars in members "
                    f"file={row.N_clust_actual}"
                )
            raise ValueError(
                "Member counts do not match between C and the members file"
            )
    else:
        logging.info("All clusters have matching member counts\n")


def update_zenodo_files(
    logging,
    temp_zenodo_fold: str,
    old_zenodo_cat: pd.DataFrame,
    all_names: pd.DataFrame,
    df_UCC_B: pd.DataFrame,
    df_UCC_C_final: pd.DataFrame,
    df_members_new: pd.DataFrame,
    flag_membs_changed: bool,
):
    """
    Update the Zenodo files with the latest catalogue and members data.
    """
    # Generate updated full UCC catalogue
    logging.info("Update Zenodo files:")

    fpath = temp_zenodo_fold + zenodo_cat_fname
    df_UCC_C_copy = df_UCC_C_final.copy()
    flag_zen_cat_changed = updt_zenodo_csv(
        logging, all_names, df_UCC_B, df_UCC_C_copy, old_zenodo_cat, fpath
    )

    if flag_membs_changed:
        zenodo_members_file_temp = temp_zenodo_fold + UCC_members_file
        df_members_new.to_parquet(zenodo_members_file_temp, index=False)
        logging.info(f"Zenodo members file: '{zenodo_members_file_temp}'")

    if flag_zen_cat_changed or flag_membs_changed:
        # No changes detected in Zenodo files. No updates needed
        N_clusters, N_members = len(df_UCC_C_final), len(df_members_new)
        updt_readme(logging, N_clusters, N_members, temp_zenodo_fold)
    else:
        logging.info("No changes detected in Zenodo files. No updates needed.")


def updt_zenodo_csv(
    logging,
    all_names: pd.DataFrame,
    df_UCC_B: pd.DataFrame,
    df_UCC_C: pd.DataFrame,
    old_zenodo_cat: pd.DataFrame,
    file_path: str,
) -> bool:
    """
    Generates a CSV file containing a reduced Unified Cluster Catalog
    (UCC) dataset, which can be stored in the Zenodo repository.
    """

    # Check that the df_UCC_C["fname"] column matches the first string of the
    # all_names["fnames"] column (strings separated by ';') before moving on
    fnames_from_all_names = (
        all_names["fnames"]
        .str.split(";")
        .str[0]
        .astype("string")
        .reset_index(drop=True)
    )
    fnames_from_C = df_UCC_C["fname"].astype("string").reset_index(drop=True)
    if not fnames_from_C.equals(fnames_from_all_names):
        raise ValueError(
            "The 'fname' column in df_UCC_C does not match the canonical fname in "
            "the 'fnames' column in all_names."
        )

    # Add the 'Names' column from all_names
    df_UCC_C["Names"] = all_names["Names"]

    # Add columns from B to C
    for col in (
        "dist_median",
        "dist_stddev",
        "av_median",
        "av_stddev",
        "diff_ext_median",
        "diff_ext_stddev",
        "age_median",
        "age_stddev",
        "met_median",
        "met_stddev",
        "mass_median",
        "mass_stddev",
        "bi_frac_median",
        "bi_frac_stddev",
        "blue_str_median",
        "blue_str_stddev",
        "Type",
    ):
        df_UCC_C[col] = df_UCC_B[col]

    # Round columns
    cols = [
        "age_median",
        "age_stddev",
        "mass_median",
        "mass_stddev",
        "blue_str_median",
        "blue_str_stddev",
    ]
    df_UCC_C[cols] = df_UCC_C[cols].round(0)

    # Re-name columns
    df_UCC_C.rename(
        columns={
            "Names": "Name(s)",
            "fname": "name",
            "RA_ICRS_m": "RA_ICRS",
            "DE_ICRS_m": "DE_ICRS",
            "GLON_m": "GLON",
            "GLAT_m": "GLAT",
            "Plx_m": "Plx",
            "pmRA_m": "pmRA",
            "pmDE_m": "pmDE",
            "Rv_m": "Rv",
            "r_core_pc": "r_core",
            "dist_median": "Dist_[kpc]",
            "dist_stddev": "Dist_STDDEV",
            "av_median": "Av_[mag]",
            "av_stddev": "Av_STDDEV",
            "diff_ext_median": "Diff_ext_[mag]",
            "diff_ext_stddev": "Diff_ext_STDDEV",
            "age_median": "Age_[Myr]",
            "age_stddev": "Age_STDDEV",
            "met_median": "FeH_[dex]",
            "met_stddev": "FeH_STDDEV",
            "mass_median": "Mass_[Msun]",
            "mass_stddev": "Mass_STDDEV",
            "bi_frac_median": "Binary_fr",
            "bi_frac_stddev": "Binary_fr_STDDEV",
            "blue_str_median": "BSS",
            "blue_str_stddev": "BSS_STDDEV",
        },
        inplace=True,
    )

    # Re-order columns
    zenodo_UCC_cat = pd.DataFrame(
        df_UCC_C[
            [
                "Name(s)",
                "name",
                "Type",
                "N_membs",
                "r_50",
                "r_core",
                "RA_ICRS",
                "DE_ICRS",
                "GLON",
                "GLAT",
                "Plx",
                "pmRA",
                "pmDE",
                "Rv",
                "N_Rv",
                "Dist_[kpc]",
                "Dist_STDDEV",
                "Av_[mag]",
                "Av_STDDEV",
                "Diff_ext_[mag]",
                "Diff_ext_STDDEV",
                "Age_[Myr]",
                "Age_STDDEV",
                "FeH_[dex]",
                "FeH_STDDEV",
                "Mass_[Msun]",
                "Mass_STDDEV",
                "Binary_fr",
                "Binary_fr_STDDEV",
                "BSS",
                "BSS_STDDEV",
                "C3",
                "P_dup",
                "UTI",
                "bad_oc",
            ]
        ]
    )

    # Check if the new Zenodo catalogue differs from the old one
    flag_zen_cat_changed = not old_zenodo_cat.equals(zenodo_UCC_cat)
    if flag_zen_cat_changed:
        # Store to csv file
        zenodo_UCC_cat = round_columns(zenodo_UCC_cat)
        zenodo_UCC_cat.to_csv(
            file_path,
            na_rep="nan",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        logging.info(f"Zenodo '.csv' file: '{file_path}'")

        # Check differences between the original and final UCC_cat files
        diff_between_dfs(
            logging, "zenodo cat", old_zenodo_cat, zenodo_UCC_cat, order_col="name"
        )

    return flag_zen_cat_changed


def updt_readme(
    logging, N_clusters: int, N_members: int, temp_zenodo_fold: str
) -> None:
    """Update info number in README file uploaded to Zenodo"""

    XXXX = pd.Timestamp.now().strftime("%y%m%d")  # Date in format YYMMDD
    YYYY = str(N_clusters)
    ZZZZ = str(N_members)
    txt = [
        (
            f"These files correspond to the {XXXX} version of the UCC "
            "database (https://ucc.ar),\n"
        ),
        f"composed of {YYYY} clusters with a combined {ZZZZ} members.\n",
    ]

    # Load the main file
    in_file_path = zenodo_folder + "README.txt"
    with open(in_file_path, "r") as f:
        dataf = f.readlines()
        # Replace lines
        dataf[2:4] = txt

    # Store updated file
    out_file_path = temp_zenodo_fold + "README.txt"
    with open(out_file_path, "w") as f:
        f.writelines(dataf)

    logging.info(f"Zenodo 'README' file: '{out_file_path}'")


def move_files(
    logging,
    temp_zenodo_fold: str,
    rename_C_fname: dict,
    C_not_in_B: dict,
    C_fnames: list,
) -> bool:
    """Move files to the appropriate folders"""
    post_actions = []

    # Move Zenodo README
    file_path_temp = temp_zenodo_fold + "README.txt"
    if os.path.isfile(file_path_temp):
        file_path = zenodo_folder + "README.txt"
        post_actions.append(("move", file_path_temp, file_path))

    # Move Zenodo catalogue
    file_path_temp = temp_zenodo_fold + zenodo_cat_fname
    if os.path.isfile(file_path_temp):
        file_path = zenodo_folder + zenodo_cat_fname
        post_actions.append(("move", file_path_temp, file_path))

    # Move Zenodo members file
    file_path_temp = temp_zenodo_fold + UCC_members_file
    if os.path.isfile(file_path_temp):
        file_path = zenodo_folder + UCC_members_file
        date = pd.Timestamp.now().strftime("%y%m%d%H")
        archived_members = archive_folder_path + UCC_members_file.replace(
            ".parquet", f"_{date}.parquet"
        )
        # Copy current members file to archive
        post_actions.append(("archive_parquet", file_path, archived_members))
        # Replace current members file with new one
        post_actions.append(("replace", file_path_temp, file_path))

    # Move C catalogue files (in / out)
    for UCC_cat_C in [UCC_cat_C_in, UCC_cat_C_out]:
        ucc_temp = temp_folder + UCC_cat_C
        if os.path.isfile(ucc_temp):
            # Archive old C catalogue
            ucc_stored = data_folder + UCC_cat_C
            now_time = pd.Timestamp.now().strftime("%y%m%d%H")
            archived_C_file = archive_folder_path + UCC_cat_C.replace(
                ".csv", f"_{now_time}.csv.gz"
            )
            post_actions.append(("archive_csv", ucc_stored, archived_C_file))
            # Move new C file into place
            post_actions.append(("move", ucc_temp, ucc_stored))


    # Collect rename operations
    md_root = root_ucc_path + md_folder
    for name in os.listdir(md_root):
        # 'name' should not contain '.' except for the extension
        mdfile = name.split(".")[0]
        if mdfile in rename_C_fname:
            old_fpath = os.path.join(md_root, mdfile + ".md")
            new_fpath = os.path.join(md_root, rename_C_fname[mdfile] + ".md")
            post_actions.append(("rename", old_fpath, new_fpath))
    # WEBP files
    for root, dirs, files in os.walk(root_ucc_path + plots_folder):
        dirs[:] = [d for d in dirs if d != ".git"]  # exclude .git
        for name in files:
            if not name.endswith(".webp"):
                continue
            webpfile = name.rsplit(".", 1)[0]
            if webpfile in rename_C_fname:
                old_fpath = os.path.join(root, webpfile + ".webp")
                new_fname = rename_C_fname[webpfile]
                prefix, rest = root.split("plots_", 1)
                _, suffix = rest.split("/", 1)
                new_root = f"{prefix}plots_{new_fname[0]}/{suffix}"
                new_fpath = os.path.join(new_root, new_fname + ".webp")
                post_actions.append(("rename", old_fpath, new_fpath))

    # Collect removal operations
    fname_C = set(C_fnames)
    # MD removals
    for name in os.listdir(root_ucc_path + md_folder):
        mdname = name.rsplit(".", 1)[0]
        if mdname not in fname_C and mdname not in rename_C_fname:
            post_actions.append(
                (
                    "remove",
                    os.path.join(root_ucc_path + md_folder, mdname + ".md"),
                    None,
                )
            )
    # WEBP removals
    rename_warnings = []
    for root, dirs, files in os.walk(root_ucc_path + plots_folder):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            if not name.endswith(".webp"):
                continue
            webpname = name.rsplit(".", 1)[0]
            if webpname not in fname_C and webpname not in rename_C_fname:
                old_fpath = os.path.join(root, webpname + ".webp")
                if "HUNT23" in root or "CANTAT20" in root:
                    if webpname in C_not_in_B:
                        new_fname = C_not_in_B[webpname]
                        # New root path
                        prefix, rest = root.split("plots_", 1)
                        _, suffix = rest.split("/", 1)
                        new_root = f"{prefix}plots_{new_fname[0]}/{suffix}"
                        new_fpath = os.path.join(new_root, new_fname + ".webp")
                        if os.path.exists(new_fpath):
                            rename_warnings.append(
                                f"File '{new_fname}.webp' already exists in '{root}'. "
                                f"Cannot rename '{webpname}.webp'"
                            )
                        else:
                            # logging.warning(
                            #     f"File '{new_fname}.webp' does not exist in {root}"
                            #     f"Rename '{webpname}.webp' --> '{new_fname}.webp'"
                            # )
                            post_actions.append(("rename", old_fpath, new_fpath))
                else:
                    post_actions.append(("remove", old_fpath, None))

    if not post_actions:
        logging.info("No changes to make.")
        return False

    if rename_warnings:
        logging.warning("\n=== RENAME WARNINGS ===")
        for warning in rename_warnings:
            logging.warning(warning)

    # Print actions and ask for confirmation
    logging.info("\n=== ACTIONS ===")
    for action_type, src, dst in post_actions:
        if action_type == "move":
            logging.info(f"MOVE:    {src} --> {dst}")
        elif action_type == "replace":
            logging.info(f"REPLACE: {src} --> {dst}")
        elif action_type == "archive_parquet":
            logging.info(
                f"ARCHIVE: {src} --> {dst} (keep last {N_archive_versions})"
            )
        elif action_type == "remove":
            logging.info(f"REMOVE:  {src}")
        elif action_type == "archive_csv":
            logging.info(
                f"ARCHIVE + GZIP: {src} --> {dst} "
                f"(keep last {N_archive_versions})"
            )
        elif action_type == "rename":
            logging.info(f"RENAME:  {src} --> {dst}")

    if input("\nProceed with these changes? [y/N]: ").strip().lower() != "y":
        logging.info("Aborted.")
        return False

    for action_type, src, dst in post_actions:
        if action_type == "move":
            os.rename(src, dst)
            logging.info(f"{src} --> {dst}")
        elif action_type == "archive_parquet":
            shutil.copy2(src, dst)
            logging.info(f"{src} --> {dst} (archived copy)")
            prune_archive(logging, dst)
        elif action_type == "replace":
            os.replace(src, dst)
            logging.info(f"{src} --> {dst} (replaced)")
        elif action_type == "archive_csv":
            df_OLD_C = pd.read_csv(src)
            save_df_UCC(logging, df_OLD_C, dst, compression="gzip")
            logging.info(f"{src} --> {dst} (archived)")
            prune_archive(logging, dst)
        elif action_type == "remove":
            if os.path.isfile(src):
                os.remove(src)
                logging.info(f"Removed: {src}")
        elif action_type == "rename":
            os.rename(src, dst)
            logging.info(f"Renamed: {src} --> {dst}")

    return True


if __name__ == "__main__":
    main()
