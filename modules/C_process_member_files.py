import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .C_funcs.member_files_updt_funcs import (
    get_fastMP_membs,
    save_cl_datafile,
    updt_UCC_new_cl_data,
)
from .utils import diff_between_dfs, load_BC_cats, logger, save_df_UCC
from .variables import (
    C_dup_min,
    C_lit_max,
    GCs_cat,
    UCC_members_file,
    UTI_max,
    data_folder,
    md_folder,
    merged_dbs_file,
    name_DBs_json,
    path_gaia_frames,
    path_gaia_frames_ranges,
    plots_folder,
    root_ucc_path,
    temp_folder,
    temp_members_folder,
    ucc_cat_file,
    zenodo_cat_fname,
    zenodo_folder,
)


def main():
    """Second function to update the UCC (Unified Cluster Catalogue)"""
    logging = logger()

    # Generate paths and check for required folders and files
    ucc_B_file, ucc_C_file, temp_zenodo_fold = get_paths_check_paths(logging)

    gaia_frames_data, current_JSON, df_GCs, df_members, df_UCC_B, df_UCC_C = load_data(
        logging, ucc_B_file, ucc_C_file
    )

    # Detect entries to be processed
    rename_C_fname, B_not_in_C, C_not_in_B, C_reprocess = detect_entries_to_process(
        df_UCC_B, df_UCC_C
    )

    N_process = (
        len(rename_C_fname) + len(B_not_in_C) + len(C_not_in_B) + len(C_reprocess)
    )

    logging.info(
        f"\nProcessing:\n-B entries not in C   : {len(B_not_in_C)}\n"
        + f"-C entries not in B   : {len(C_not_in_B)}\n"
        + f"-C entries to rename  : {len(rename_C_fname)}\n"
        + f"-C entries re-process : {len(C_reprocess)}"
    )

    if N_process == 0:
        if input("\nNo new OCs to process. Process anyway? (y/n): ").lower() != "y":
            sys.exit()

    load_file = False
    temp_UCC_updt_file = temp_folder + "df_UCC_C_updt.csv"
    if os.path.isfile(temp_UCC_updt_file):
        if (
            input(f"\nLoad existing '{temp_UCC_updt_file}' file? (y/n): ").lower()
            == "y"
        ):
            load_file = True

    if load_file:
        # Load file if it already exists and the .parquet files were generated
        df_UCC_C_updt = pd.read_csv(temp_UCC_updt_file)
        logging.info("\nTemp file df_UCC_C_updt loaded")
    else:
        # Generate dataframe to store data extracted from the OCs to be processed
        df_UCC_C_updt = process_entries(df_UCC_B, B_not_in_C, C_reprocess)
        # Generate member files for new OCs and obtain their data
        df_UCC_C_updt = member_files_updt(
            logging, gaia_frames_data, df_GCs, df_UCC_C, df_UCC_C_updt
        )

    df_UCC_C_new = update_C_cat(C_not_in_B, rename_C_fname, df_UCC_C, df_UCC_C_updt)
    logging.info(f"\nUCC database C updated (N={len(df_UCC_C_new)})\n")

    logging.info("Updating members file...")
    # Concatenate all temporary DataFrames into one
    df_comb = gen_comb_members_file(logging)
    df_members_new = update_membs_file(rename_C_fname, C_not_in_B, df_members, df_comb)
    logging.info(
        f"Zenodo '{UCC_members_file}' file updated (N={len(df_members)}->{len(df_members_new)})\n"
    )

    # Check that the 'name' column on the members file matches the fnames
    names0 = df_members_new["name"].unique().tolist()
    if not sorted(names0) == df_UCC_C_new["fname"].to_list():
        raise ValueError("'fname' and 'name'  columns do not match")

    # Find shared members between OCs and update df_UCC_C_new dataframe
    df_UCC_C_final = find_shared_members(logging, df_UCC_C_new, df_members_new)
    logging.info("Shared members data updated in UCC\n")

    # Sort df_UCC_B by fname column to match 'df_UCC_C_final'
    df_UCC_B = df_UCC_B.sort_values("fname").reset_index(drop=True)

    # Check the 'fnames' columns in df_UCC_B and df_UCC_C_final dataframes are equal
    if not df_UCC_B["fname"].to_list() == df_UCC_C_final["fname"].to_list():
        raise ValueError("The 'fname' columns in B and final C dataframes differ")

    # Check that all entries in df_UCC_C_final have process='n'
    if any(df_UCC_C_final["process"] == "y"):
        raise ValueError("Some entries in final C dataframe still have process='y'")

    # Add C coefficients, UTI values, duplicate probabilities and 'bad_oc' flags
    df_UCC_C_final = add_info_to_C(current_JSON, df_UCC_B, df_UCC_C_final)

    # Check differences between the original and final C dataframes
    diff_between_dfs(logging, "C cat", df_UCC_C, df_UCC_C_final)

    # Save the generated data to temporary files before moving them
    update_files(logging, temp_zenodo_fold, df_UCC_B, df_UCC_C_final, df_members_new)

    if input("\nMove files to their final paths? (y/n): ").lower() == "y":
        move_files(logging, temp_zenodo_fold, rename_C_fname, df_UCC_C_final)

    # if input("\nRemove temporary files and folders? (y/n): ").lower() == "y":
    #     # shutil.rmtree(temp_fold)
    #     logging.info(f"Folder removed: {temp_fold}")


def get_paths_check_paths(logging) -> tuple[str, str, str]:
    """ """
    txt = ""
    # Check for Gaia files
    if not os.path.isdir(path_gaia_frames):
        # raise FileNotFoundError(f"Folder {path_gaia_frames} is not present")
        txt += f"Folder {path_gaia_frames} is not present\n"
    if not os.path.isfile(path_gaia_frames_ranges):
        # raise FileNotFoundError(f"File {path_gaia_frames_ranges} is not present")
        txt += f"File {path_gaia_frames_ranges} is not present"
    if txt != "":
        logging.info(txt)
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # If temp file exists, warn
    temp_f = temp_folder + ucc_cat_file
    if os.path.isfile(temp_f):
        logging.warning(f"WARNING: file {temp_f} exists. Moving on will re-write it")
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Create folder to store the per-cluster parquet member files
    if not os.path.exists(temp_members_folder):
        os.makedirs(temp_members_folder)
    else:
        if len(os.listdir(temp_members_folder)) > 0:
            logging.warning(
                f"WARNING: There are .parquet files in '{temp_members_folder}'. If left "
                + "there,\nthey will be used when the script combines the final members data"
            )
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit()

    # Temporary zenodo/ folder
    temp_zenodo_fold = temp_folder + zenodo_folder
    # Create if required
    if not os.path.exists(temp_zenodo_fold):
        os.makedirs(temp_zenodo_fold)

    # Path to the current UCC csv files
    ucc_B_file = data_folder + merged_dbs_file
    ucc_C_file = data_folder + ucc_cat_file

    return ucc_B_file, ucc_C_file, temp_zenodo_fold


def load_data(
    logging, ucc_B_file, ucc_C_file
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ """
    # Load file with Gaia frames ranges
    gaia_frames_data = pd.DataFrame([])
    if os.path.isfile(path_gaia_frames_ranges):
        gaia_frames_data = pd.read_csv(path_gaia_frames_ranges)

    # Load clusters data in JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    # Load current members file
    df_members = pd.read_parquet(zenodo_folder + UCC_members_file)

    # Load current CSV data files
    df_UCC_B = load_BC_cats("B", ucc_B_file)
    logging.info(f"\nFile {ucc_B_file} loaded ({len(df_UCC_B)} entries)")
    df_UCC_C = load_BC_cats("C", ucc_C_file)
    logging.info(f"File {ucc_C_file} loaded ({len(df_UCC_C)} entries)")

    return gaia_frames_data, current_JSON, df_GCs, df_members, df_UCC_B, df_UCC_C


def detect_entries_to_process(
    df_UCC_B: pd.DataFrame, df_UCC_C: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    B_not_in_C  --> Add to C
    C_not_in_B  --> Remove from C
    C_reprocess --> Reprocess in C
    rename_C_fname --> Rename in C

    """
    df_UCC_B["fname"] = [_.split(";")[0] for _ in df_UCC_B["fnames"]]

    # Entries that must be added to C
    B_not_in_C = df_UCC_B[~df_UCC_B["fname"].isin(df_UCC_C["fname"])]

    # Entries that must be removed from C
    C_not_in_B = df_UCC_C[~df_UCC_C["fname"].isin(df_UCC_B["fname"])]

    # Entries in B_not_in_C that just need renaming in C_not_in_B
    rename_C_fname = {}
    C_fname_lst = C_not_in_B["fname"].to_list()
    for fnames in B_not_in_C["fnames"]:
        fnames = fnames.split(";")
        for fname in fnames:
            if fname in C_fname_lst:
                # This C entry needs ranaming: fname --> fnames[0]
                rename_C_fname[fname] = fnames[0]
                break

    if len(rename_C_fname) > 0:
        # Remove the entries that just need renaming
        B_not_in_C = B_not_in_C[~B_not_in_C["fname"].isin(rename_C_fname.values())]

    # Drop 'fnames' column from df_UCC_B
    B_not_in_C = B_not_in_C.drop(columns=["fnames"])

    # Entries manually marked for re-processing in C
    msk = df_UCC_C["process"] == "y"
    C_reprocess = df_UCC_C[msk].copy()

    # Check that these three dataframes don't share elements in their 'fname' columns
    df_names = ["B_not_in_C", "C_not_in_B", "C_reprocess"]
    for i, df1 in enumerate([B_not_in_C, C_not_in_B, C_reprocess]):
        for j, df2 in enumerate([B_not_in_C, C_not_in_B, C_reprocess]):
            if i >= j:
                continue
            shared = set(df1["fname"]) & set(df2["fname"])
            if len(shared) > 0:
                raise ValueError(
                    f"{df_names[i]} and {df_names[j]} share {len(shared)} elements"
                )

    return rename_C_fname, B_not_in_C, C_not_in_B, C_reprocess


def process_entries(
    df_UCC_B: pd.DataFrame, B_not_in_C: pd.DataFrame, C_reprocess: pd.DataFrame
) -> pd.DataFrame:
    """ """
    # Extract all columns except "fnames"
    B_cols = list(B_not_in_C.keys())
    B_cols.remove("fname")
    C_cols = list(C_reprocess.keys())
    C_cols.remove("fname")
    all_cols = B_cols + C_cols

    # Generate empty dictionary with all the fnames to be processed
    all_fnames = list(B_not_in_C["fname"]) + list(C_reprocess["fname"])
    df_UCC_updt = {"fname": all_fnames}
    N_tot = len(all_fnames)
    for k in all_cols:
        df_UCC_updt[k] = [np.nan] * N_tot

    # Add data from the B_not_in_C dataframe
    for i, fname in enumerate(B_not_in_C["fname"]):
        j = df_UCC_updt["fname"].index(fname)
        for col in B_cols:
            df_UCC_updt[col][j] = B_not_in_C[col].iloc[i]

    # Add data from the C_reprocess dataframe and information from df_UCC_B
    B_fnames_lst = list(df_UCC_B["fname"])
    for i, fname in enumerate(C_reprocess["fname"]):
        j = df_UCC_updt["fname"].index(fname)
        for col in C_cols:
            df_UCC_updt[col][j] = C_reprocess[col].iloc[i]

        # Add df_UCC_B data
        k = B_fnames_lst.index(fname)
        for col in B_cols:
            df_UCC_updt[col][j] = df_UCC_B[col].iloc[k]

    df_UCC_updt = pd.DataFrame(df_UCC_updt).replace({pd.NA: "nan"})
    # Set dtype of columns
    str_type = (
        "fname",
        "DB",
        "DB_i",
        "Names",
        "fund_pars",
        "plot_used",
        "process",
        "frame_limit",
        "C3",
        "shared_members",
        "shared_members_p",
        "bad_oc",
        "DB_coords_used",
    )
    for col in df_UCC_updt.columns:
        if col in str_type:
            df_UCC_updt[col] = df_UCC_updt[col].astype("string")
        else:
            df_UCC_updt[col] = df_UCC_updt[col].astype(float)

    return df_UCC_updt


def member_files_updt(
    logging, gaia_frames_data, df_GCs, df_UCC_C, df_UCC_C_updt
) -> pd.DataFrame:
    """
    Updates the Unified Cluster Catalogue (UCC) with new open clusters (OCs).
    """
    if df_UCC_C_updt.empty:
        return df_UCC_C_updt

    N_tot = len(df_UCC_C_updt)
    for idx, cl_row in df_UCC_C_updt.iterrows():
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
        N_clust, N_clust_max, N_box, frame_limit = np.nan, np.nan, np.nan, ""
        if str(cl_row["process"]) == "y":
            N_clust, N_clust_max, N_box, frame_limit = cl_row[
                ["N_clust", "N_clust_max", "N_box", "frame_limit"]
            ]
        if isinstance(frame_limit, float) or frame_limit == "nan":
            frame_limit = ""

        df_field, df_membs = get_fastMP_membs(
            logging,
            df_GCs,
            gaia_frames_data,
            df_UCC_C,  # Used by close objects check
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
            N_box,
            frame_limit,
        )

        # Write selected member stars to file
        save_cl_datafile(logging, fname0, df_membs)

        # Store data from this new entry used to update the UCC
        df_UCC_C_updt = updt_UCC_new_cl_data(idx, df_UCC_C_updt, df_field, df_membs)

        # Update file with information. Do this for each iteration to avoid
        # losing data if something goes wrong with any cluster
        df_UCC_C_updt.to_csv(
            temp_folder + "df_UCC_C_updt.csv", index=False, na_rep="nan"
        )

    return df_UCC_C_updt


def update_C_cat(
    C_not_in_B: pd.DataFrame,
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
        msk = ~df_UCC_C_new["fname"].isin(C_not_in_B["fname"])
        df_UCC_C_new = df_UCC_C_new[msk]

    # Update df_UCC_C_new using data from df_UCC_C_updt
    # Ensure 'fname' is the index in both DataFrames
    A = df_UCC_C_new.set_index("fname")
    B = df_UCC_C_updt.set_index("fname")
    # Update existing rows in A with values from B
    A.update(B)
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


def gen_comb_members_file(logging) -> pd.DataFrame:
    """Combine individual parquet files into a single temporary one"""

    # Path to folder with individual .parquet files
    member_files = os.listdir(temp_members_folder)
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
    C_not_in_B: pd.DataFrame,
    df_members: pd.DataFrame,
    df_comb: pd.DataFrame,
) -> pd.DataFrame:
    """
    Update the parquet file containing estimated members from the
    Unified Cluster Catalog (UCC) dataset, formatted for storage in the Zenodo
    repository.
    """
    df_updated = df_members.copy()

    # Rename entries
    if len(rename_C_fname) > 0:
        msk = df_updated["name"].isin(rename_C_fname.keys())
        df_updated.loc[msk, "name"] = df_updated.loc[msk, "name"].map(rename_C_fname)

    # Remove entries in C_not_in_B
    if len(C_not_in_B) > 0:
        msk = ~df_updated["name"].isin(C_not_in_B["fname"])
        df_updated = pd.DataFrame(df_updated[msk])

    if not df_comb.empty:
        # Get the list of names in each DataFrame
        names_df1 = set(df_updated["name"])
        names_df2 = set(df_comb["name"])
        # Identify names in df_updated not in df_comb
        extra_names = names_df1 - names_df2
        # Filter df_updated for those extra groups
        df1_extra = df_updated[df_updated["name"].isin(extra_names)]  # pyright: ignore
        # Concatenate df_comb with the extra df_members groups
        df_members_new = pd.concat([df_comb, df1_extra], ignore_index=True)
        df_members_new = pd.DataFrame(df_members_new).sort_values("name")
    else:
        df_members_new = df_updated.copy()
    df_members_new = df_members_new.sort_values("name").reset_index(drop=True)

    return df_members_new


def find_shared_members(logging, df_UCC_C_new, df_members):
    """ """
    logging.info("Finding shared members...")

    # Find OCs that intersect. This helps to speed up the process
    intersection_map = find_intersections(df_UCC_C_new, df_members)

    # Group members by 'fname'
    grouped = df_members.groupby("name")["Source"].apply(set)
    N_total = len(grouped)
    results = {
        "fname": grouped.keys().tolist(),
        "shared_members": ["nan"] * N_total,
        "shared_members_p": ["nan"] * N_total,
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
    df_UCC_C_new[["shared_members", "shared_members_p"]] = result_df[
        ["shared_members", "shared_members_p"]
    ]
    df_UCC_C_final = df_UCC_C_new.sort_values("fname").reset_index(drop=True)

    return df_UCC_C_final


def find_intersections(df_C, df_members):
    """ """

    # Find OCs that contain duplicated element in any other OC, also to speed up
    source_counts = df_members["Source"].value_counts()
    shared_sources = source_counts[source_counts > 1].index
    ocs_w_shared_sources = df_members[df_members["Source"].isin(shared_sources)][
        "name"
    ].unique()

    # Filter UCC df to only include OCs with shared sources
    # names = np.array([_.split(";")[0] for _ in df_C["fnames"]])
    arr2_set = set(ocs_w_shared_sources)
    msk = np.fromiter((x in arr2_set for x in df_C["fname"]), dtype=bool)
    df_msk = df_C[msk]

    # The search region is two times the r_50 radius
    radii = 2 * df_msk["r_50"].to_numpy() / 60

    # Compute pairwise distances
    coords = df_msk[["GLON_m", "GLAT_m"]].to_numpy()
    dists = cdist(coords, coords)

    # Compute pairwise sum of radii
    radii_sum = radii[:, None] + radii[None, :]

    # Intersection condition: distance <= sum of radii
    # Exclude only the diagonal (self-comparisons)
    not_self = ~np.eye(len(dists), dtype=bool)
    intersection_mask = (dists <= radii_sum) & not_self

    # Extract intersecting names
    names = np.array(df_msk["fname"])
    results = []
    for i, name in enumerate(names):
        intersecting = names[intersection_mask[i]]
        val = {}  # Empty set
        if intersecting.size > 0:
            val = set(intersecting)
        results.append({"name": name, "intersects_with": val})

    intersections = pd.DataFrame(results)

    intersection_map = intersections.set_index("name")["intersects_with"].to_dict()

    return intersection_map


def add_info_to_C(current_JSON, df_UCC_B, df_UCC_C, max_dens=5):
    """ """

    def normalize(N, arr, Nmin, Nmax, vmin, vmax):
        msk2 = (N >= Nmin) & (N < Nmax)
        arr[msk2] = vmin + ((N[msk2] - Nmin) / (Nmax - Nmin)) * (vmax - vmin)

    N_50 = df_UCC_C["N_50"].to_numpy()
    C_N50 = np.ones(len(N_50))
    C_N50[N_50 < 25] = 0.0
    # Define intervals and mapping ranges
    bounds = (0.25, 0.5, 0.75, 0.9)
    Nvals = (25, 50, 100, 500)
    for i in range(1, len(bounds)):
        normalize(N_50, C_N50, Nvals[i - 1], Nvals[i], bounds[i - 1], bounds[i])

    r_50 = df_UCC_C["r_50"].to_numpy()
    dist_pc = 1000 / np.clip(df_UCC_C["Plx_m"], 0.01, 50)
    r_pc = dist_pc * np.tan(np.deg2rad(r_50 / 60))
    dens = N_50 / r_pc**2
    C_dens = np.clip((dens - 0) / (max_dens - 0), 0, 1)

    # Assign a number to all elements in C3
    C3 = df_UCC_C["C3"].to_numpy()
    vals = {"A": 1, "B": 0.5, "C": 0.25, "D": 0}
    C_C3 = np.array([vals[a[0]] + vals[a[1]] for a in C3], dtype=float) * 0.5

    # Count number of times each OC is mentioned in the literature
    N_lit = np.array([len(_.split(";")) for _ in df_UCC_B["DB"]])
    # Normalizing value
    N_lit_tot = len({s for cell in df_UCC_B["DB"] for s in cell.split(";")})
    # N_lit_tot = max(N_lit)
    # Percentages that define the mapping ranges: 5%, 10%, 15%, 20%
    Nvals = [int(N_lit_tot * _) for _ in (0.05, 0.1, 0.15, 0.2)]
    C_lit = np.ones(len(N_lit))
    C_lit[N_lit < min(Nvals)] = 0.0
    # Define intervals and mapping ranges
    bounds = (0.25, 0.5, 0.75, 0.9)
    for i in range(1, len(bounds)):
        normalize(N_lit, C_lit, Nvals[i - 1], Nvals[i], bounds[i - 1], bounds[i])

    # C_dup indicates the confidence that an entry is a duplicate of a previously
    # reported object. A value of 1 means not at all a duplicate

    # Extract the first DB and year of publication for each entry (assumes the DBs
    # are already ordered by year)
    dbs = [_.split(";")[0] for _ in df_UCC_B["DB"]]
    f_year = [int(_.split("_")[0][-4:]) for _ in dbs]
    # Extract first fname
    fnames = [_.split(";")[0] for _ in df_UCC_B["fnames"]]
    # Map years and dbs to fnames
    fname_db_to_year = {name: [year, db] for name, year, db in zip(fnames, f_year, dbs)}

    C_dup = [100.0] * len(df_UCC_C)
    C_dup_same_db = [100.0] * len(df_UCC_C)
    for idx, cl in df_UCC_C.iterrows():
        if str(cl["shared_members"]) == "nan":
            # This OC does not share members with any other, move on to the next
            continue

        # # Extract the years and dbs associated to the entries that share members
        # with 'cl'
        shared_members = cl["shared_members"].split(";")
        fyears_shared, dbs_shared = [], []
        for s in shared_members:
            fyears_shared.append(fname_db_to_year[s][0])
            dbs_shared.append(fname_db_to_year[s][1])

        # Year of publication of 'cl'
        f_year_cl = f_year[idx]

        if min(fyears_shared) > f_year_cl:
            # All entries that share members with 'cl' where published *after* 'cl',
            # 'cl' thus cannot be a duplicate of any of them
            continue

        shared_members_p = list(map(float, cl["shared_members_p"].split(";")))
        date_received_cl = int(current_JSON[dbs[idx]]["received"])

        shared_p = {"n": 0.0, "y": 0.0}
        for j, f_year_shared in enumerate(fyears_shared):
            # shared_members_j, same_db = 0.0, 'n'
            if f_year_cl > f_year_shared:
                # If 'cl' is more recent than this entry, 'cl' is the duplicate
                shared_p["n"] = max(shared_p["n"], shared_members_p[j])
            elif f_year_cl == f_year_shared:
                # If the years are equal, use the received date to disambiguate
                date_received_shared = int(current_JSON[dbs_shared[j]]["received"])
                if date_received_cl > date_received_shared:
                    # If 'cl' is more recent than this entry, 'cl' is the duplicate
                    shared_p["n"] = max(shared_p["n"], shared_members_p[j])
                elif date_received_cl == date_received_shared:
                    if dbs[idx] != dbs_shared[j]:
                        # This should never happen
                        raise ValueError(
                            f"({idx}) {cl['fname']} & {shared_members[j]} share members "
                            f"and a publication date ({date_received_cl}), but are "
                            f"mentioned in different DBs ({dbs[idx], dbs_shared[j]}). "
                            "This makes it impossible to disambiguate which one is the "
                            "duplicate of the other."
                        )
                    # These entries share members BUT they belong to the same DB.
                    shared_p["y"] = max(shared_p["y"], shared_members_p[j])

        if shared_p["n"] > 0.0:
            # At least one entry that shares members with 'cl' belongs to a
            # different DB
            C_dup[idx] -= shared_p["n"]
        if shared_p["y"] > 0.0:
            # All entries that share members with 'cl' belong to the same DB
            C_dup_same_db[idx] -= shared_p["y"]

    C_dup = np.array(C_dup) / 100
    C_dup_same_db = np.array(C_dup_same_db) / 100

    # Final UTI
    UTI = np.clip(0.2 * (C_N50 + C_dens + C_C3 + 2 * C_lit) * C_dup, 0, 1)

    # Add data to df
    df_UCC_C["C_N"] = np.round(C_N50, 2)
    df_UCC_C["C_dens"] = np.round(C_dens, 2)
    df_UCC_C["C_C3"] = np.round(C_C3, 2)
    df_UCC_C["C_lit"] = np.round(C_lit, 2)
    df_UCC_C["C_dup"] = np.round(C_dup, 2)
    df_UCC_C["C_dup_same_db"] = np.round(C_dup_same_db, 2)
    df_UCC_C["P_dup"] = np.round(1 - df_UCC_C["C_dup"], 2)
    df_UCC_C["UTI"] = np.round(UTI, 2)

    # Flag entries that have bad values and are possibly asterisms, moving groups,
    # or artifacts of some kind.
    msk = (
        (df_UCC_C["UTI"] < UTI_max)
        & (df_UCC_C["C_dup"] > C_dup_min)
        & (df_UCC_C["C_lit"] < C_lit_max)
    )
    df_UCC_C.loc[msk, "bad_oc"] = "y"

    return df_UCC_C


# def get_bad_ocs(df):
#     """Flag entries that have bad values and are possibly asterisms, moving groups,
#     or artifacts of some kind.
#     """
#     msk = (df["UTI"] < UTI_max) & (df["C_dup"] > C_dup_min) & (df["C_lit"] < C_lit_max)
#     df.loc[msk, "bad_oc"] = "y"

#     return df


def update_files(
    logging,
    temp_zenodo_fold: str,
    df_UCC_B: pd.DataFrame,
    df_UCC_C_final: pd.DataFrame,
    df_members_new: pd.DataFrame,
):
    """ """
    # Generate updated full UCC catalogue
    logging.info("Update files:")

    # Save updated UCC to temporary CSV file
    save_df_UCC(logging, df_UCC_C_final, temp_folder + ucc_cat_file, "fname")

    fpath = temp_zenodo_fold + zenodo_cat_fname
    df_UCC_C_copy = df_UCC_C_final.copy()
    updt_zenodo_csv(logging, df_UCC_B, df_UCC_C_copy, fpath)

    N_clusters, N_members = len(df_UCC_C_final), len(df_members_new)
    updt_readme(logging, N_clusters, N_members, temp_zenodo_fold)

    zenodo_members_file_temp = temp_zenodo_fold + UCC_members_file
    df_members_new.to_parquet(zenodo_members_file_temp, index=False)
    logging.info(f"Zenodo members file: '{zenodo_members_file_temp}'")


def updt_zenodo_csv(
    logging, df_UCC_B: pd.DataFrame, df_UCC_C: pd.DataFrame, file_path: str
) -> None:
    """
    Generates a CSV file containing a reduced Unified Cluster Catalog
    (UCC) dataset, which can be stored in the Zenodo repository.
    """
    # Add columns from B to C
    for col in (
        "Names",
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
        "blue_str_values",
    ):
        df_UCC_C[col] = df_UCC_B[col]

    # Round columns
    df_UCC_C["P_dup"] = np.round(1 - df_UCC_C["C_dup"], 2)
    df_UCC_C["age_median"] = np.round(df_UCC_C["age_median"], 0)
    df_UCC_C["age_stddev"] = np.round(df_UCC_C["age_stddev"], 0)
    df_UCC_C["mass_median"] = np.round(df_UCC_C["mass_median"], 0)
    df_UCC_C["mass_stddev"] = np.round(df_UCC_C["mass_stddev"], 0)

    # Re-name columns
    df_UCC_C.rename(
        columns={
            "Names": "Name(s)",
            "fname": "name",
            "RA_ICRS_m": "RA_ICRS",
            "DE_ICRS_m": "DEC_ICRS",
            "GLON_m": "GLON",
            "GLAT_m": "GLAT",
            "Plx_m": "Plx",
            "pmRA_m": "pmRA",
            "pmDE_m": "pmDE",
            "Rv_m": "Rv",
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
            "blue_str_values": "Blue_str",
        },
        inplace=True,
    )

    # Re-order columns
    df_UCC_C = pd.DataFrame(
        df_UCC_C[
            [
                "Name(s)",
                "name",
                "N_50",
                "r_50",
                "RA_ICRS",
                "DEC_ICRS",
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
                "Blue_str",
                "C3",
                "P_dup",
                "UTI",
                "bad_oc",
            ]
        ]
    )

    # Store to csv file
    df_UCC_C.to_csv(
        file_path,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )

    logging.info(f"Zenodo '.csv' file: '{file_path}'")


def updt_readme(
    logging, N_clusters: int, N_members: int, temp_zenodo_fold: str
) -> None:
    """Update info number in README file uploaded to Zenodo"""

    XXXX = pd.Timestamp.now().strftime("%y%m%d")  # Date in format YYMMDD
    YYYY = str(N_clusters)
    ZZZZ = str(N_members)
    txt = [
        f"These files correspond to the {XXXX} version of the UCC database (https://ucc.ar),\n",
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
    logging, temp_zenodo_fold: str, rename_C_fname: dict, df_UCC_C_final: pd.DataFrame
) -> None:
    """Move files to the appropriate folders"""
    post_actions = []

    # Move README
    file_path_temp = temp_zenodo_fold + "README.txt"
    file_path = zenodo_folder + "README.txt"
    post_actions.append(("move", file_path_temp, file_path))

    # Move Zenodo catalogue
    file_path_temp = temp_zenodo_fold + zenodo_cat_fname
    file_path = zenodo_folder + zenodo_cat_fname
    post_actions.append(("move", file_path_temp, file_path))

    # Combined members parquet
    file_path_temp = temp_zenodo_fold + UCC_members_file
    if os.path.isfile(file_path_temp):
        file_path = zenodo_folder + UCC_members_file
        date = pd.Timestamp.now().strftime("%y%m%d%H")
        archived_members = (
            data_folder
            + "ucc_archived_nogit/"
            + UCC_members_file.replace(".parquet", f"_{date}.parquet")
        )
        # Archive-copy
        post_actions.append(("archive_parquet", file_path, archived_members))
        # Move new parquet
        post_actions.append(("move", file_path_temp, file_path))

    # Archive old C catalogue
    ucc_stored = data_folder + ucc_cat_file
    now_time = pd.Timestamp.now().strftime("%y%m%d%H")
    archived_C_file = (
        data_folder
        + "ucc_archived_nogit/"
        + ucc_cat_file.replace(".csv", f"_{now_time}.csv.gz")
    )
    post_actions.append(("archive_csv", ucc_stored, archived_C_file))
    # # Remove old C file
    # post_actions.append(("remove", ucc_stored, None))
    # Move new C file into place
    ucc_temp = temp_folder + ucc_cat_file
    post_actions.append(("move", ucc_temp, ucc_stored))

    # Collect rename operations
    md_root = root_ucc_path + md_folder
    for name in os.listdir(md_root):
        mdfile = name.split(".")[0]
        if mdfile in rename_C_fname:
            old_fpath = os.path.join(md_root, mdfile + ".md")
            new_fpath = os.path.join(md_root, rename_C_fname[mdfile] + ".md")
            post_actions.append(("rename", old_fpath, new_fpath))
    # now webp files
    for root, dirs, files in os.walk(root_ucc_path + plots_folder):
        dirs[:] = [d for d in dirs if d != ".git"]  # exclude .git
        for name in files:
            if not name.endswith(".webp"):
                continue
            webpfile = name.split(".")[0]
            if webpfile in rename_C_fname:
                old_fpath = os.path.join(root, webpfile + ".webp")
                new_fname = rename_C_fname[webpfile]
                new_root = root.replace(f"plots_{webpfile[0]}", f"plots_{new_fname[0]}")
                new_fpath = os.path.join(new_root, new_fname + ".webp")
                post_actions.append(("rename", old_fpath, new_fpath))

    # Collect removal operations
    fname_C = set(df_UCC_C_final["fname"].tolist())
    # MD removals
    for name in os.listdir(root_ucc_path + md_folder):
        webname = name.rsplit(".", 1)[0]
        if webname not in fname_C:
            # remove_actions.append(os.path.join(root_ucc_path + md_folder, webname + ".md"))
            post_actions.append(
                (
                    "remove",
                    os.path.join(root_ucc_path + md_folder, webname + ".md"),
                    None,
                )
            )
    # WEBP removals
    for root, dirs, files in os.walk(root_ucc_path + plots_folder):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            if not name.endswith(".webp"):
                continue
            webname = name.rsplit(".", 1)[0]
            if webname not in fname_C:
                post_actions.append(
                    ("remove", os.path.join(root, webname + ".webp"), None)
                )

    logging.info("\n=== ACTIONS ===")
    for action_type, src, dst in post_actions:
        if action_type == "move":
            logging.info(f"MOVE:  {src} --> {dst}")
        elif action_type == "archive_parquet":
            logging.info(f"ARCHIVE: {src} --> {dst}")
        elif action_type == "remove":
            logging.info(f"REMOVE: {src}")
        elif action_type == "archive_csv":
            logging.info(f"ARCHIVE + GZIP: {src} --> {dst}")
        elif action_type == "rename":
            logging.info(f"RENAME: {src} --> {dst}")

    if input("\nProceed with these changes? [y/N]: ").strip().lower() != "y":
        logging.info("Aborted.")
        return

    for action_type, src, dst in post_actions:
        if action_type == "move":
            os.rename(src, dst)
            logging.info(f"{src} --> {dst}")
        elif action_type == "archive_parquet":
            os.rename(src, dst)
            logging.info(f"{src} --> {dst}")
        elif action_type == "archive_csv":
            df_OLD_C = pd.read_csv(src)
            save_df_UCC(logging, df_OLD_C, dst, "fname", "gzip")
            logging.info(f"{src} --> {dst} (archived)")
        elif action_type == "remove":
            if os.path.isfile(src):
                os.remove(src)
                logging.info(f"Removed: {src}")
        elif action_type == "rename":
            os.rename(src, dst)
            logging.info(f"Renamed: {src} --> {dst}")


if __name__ == "__main__":
    main()
