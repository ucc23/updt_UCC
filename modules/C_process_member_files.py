import csv
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .C_funcs.member_files_updt_funcs import (
    get_fastMP_membs,
    save_cl_datafile,
    updt_UCC_new_cl_data,
)
from .utils import (
    diff_between_dfs,
    logger,
    save_df_UCC,
)
from .variables import (
    GCs_cat,
    UCC_members_file,
    class_order,
    data_folder,
    merged_dbs_file,
    path_gaia_frames,
    path_gaia_frames_ranges,
    temp_folder,
    temp_members_folder,
    ucc_cat_file,
    zenodo_cat_fname,
    zenodo_folder,
)


def main():
    """Second function to update the UCC (Unified Cluster Catalogue)"""
    logging = logger()
    logging.info("=== Running C script ===\n")

    # Generate paths and check for required folders and files
    ucc_B_file, ucc_C_file, temp_zenodo_fold = get_paths_check_paths(logging)

    (gaia_frames_data, df_GCs, df_members, df_UCC_B, df_UCC_C) = load_data(
        logging, ucc_B_file, ucc_C_file
    )

    # Detect entries to be processed
    B_not_in_C, C_not_in_B, C_reprocess = detect_entries_to_process(df_UCC_B, df_UCC_C)

    N_process = len(B_not_in_C) + len(C_not_in_B) + len(C_reprocess)
    if N_process == 0:
        logging.info("\nNo new OCs to process")
        # N_members = ParquetFile(UCC_members_file).count()
        return

    logging.info(
        f"\nProcessing:\n-B entries not in C : {len(B_not_in_C)}\n"
        + f"-C entries not in B : {len(C_not_in_B)}\n"
        + f"-Entries marked in C: {len(C_reprocess)}"
    )

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
        df_UCC_C_updt = pd.read_csv(temp_folder + "df_UCC_C_updt.csv")
        logging.info("\nTemp file df_UCC_C_updt loaded")
    else:
        # Generate dataframe to store data extracted from the OCs to be processed
        df_UCC_C_updt = process_entries(df_UCC_B, B_not_in_C, C_reprocess)

        # Generate member files for new OCs and obtain their data
        df_UCC_C_updt = member_files_updt(
            logging, gaia_frames_data, df_GCs, df_UCC_C, df_UCC_C_updt
        )

    # Update the UCC with the new OCs member's data. Remove here entries in C that
    # are no longer in B in both the C and members dataframes
    df_members_new, df_UCC_C_new = update_UCC_membs_data(
        df_members, C_not_in_B, df_UCC_C, df_UCC_C_updt
    )
    logging.info(
        f"UCC database C updated: ({len(df_UCC_C_new)} entries; {len(df_members_new)} members)"
    )

    # Updated members file
    df_members_new = update_membs_file(logging, df_members_new)
    logging.info(f"Zenodo '{UCC_members_file}' file updated")

    # Find shared members between OCs and update df_UCC_C_new dataframe
    df_UCC_C_final = find_shared_members(logging, df_UCC_C_new, df_members_new)
    logging.info("Shared members data updated in UCC")

    # Add UTI values
    df_UCC_C_final = get_UTI(df_UCC_B, df_UCC_C_final)

    # Check differences between the original and final C dataframes
    diff_between_dfs(logging, df_UCC_C, df_UCC_C_final)

    # Check the 'fnames' columns in df_UCC_B and df_UCC_C_final dataframes are equal
    if not df_UCC_B["fnames"].equals(df_UCC_C_final["fnames"]):
        raise ValueError("The 'fnames' columns in B and final C dataframes differ")

    # Save the generated data to temporary files before moving them
    update_files(logging, temp_zenodo_fold, df_UCC_B, df_UCC_C_final, df_members_new)

    if input("\nMove files to their final paths? (y/n): ").lower() == "y":
        move_files(logging, temp_zenodo_fold)

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
                f"WARNING: There are .parquet files in '{temp_members_folder}'. If left"
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ """
    # Load file with Gaia frames ranges
    gaia_frames_data = pd.DataFrame([])
    if os.path.isfile(path_gaia_frames_ranges):
        gaia_frames_data = pd.read_csv(path_gaia_frames_ranges)

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    # Load current members file
    df_members = pd.read_parquet(zenodo_folder + UCC_members_file)

    # Load current CSV data files
    df_UCC_B = pd.read_csv(ucc_B_file)
    logging.info(f"\nFile {ucc_B_file} loaded ({len(df_UCC_B)} entries)")
    df_UCC_C = pd.read_csv(ucc_C_file)
    logging.info(f"File {ucc_C_file} loaded ({len(df_UCC_C)} entries)")

    return gaia_frames_data, df_GCs, df_members, df_UCC_B, df_UCC_C


def detect_entries_to_process(
    df_UCC_B: pd.DataFrame, df_UCC_C: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    B_not_in_C  --> Add to C
    C_not_in_B  --> Remove from C
    C_reprocess --> Reprocess in C

    """

    # fnames_B = df_UCC_B["fnames"].to_list()
    # fnames_C = df_UCC_C["fnames"].to_list()

    # # Detect entries in C that are not in B and thus must be removed
    # remove_C_entries = []
    # for fname_C in fnames_C:
    #     if fname_C not in fnames_B:
    #         remove_C_entries.append(fname_C)

    # new_B_entries = []
    # for i, fname_B in enumerate(fnames_B):
    #     fname_B0 = fname_B.split(";")[0]

    #     fnameB_found = False
    #     for j, fname_C in enumerate(fnames_C[i:]):
    #         fname_C = fname_C.split(";")
    #         if fname_B0 in fname_C:
    #             if fname_B0 == fname_C[0]:
    #                 fnameB_found = True
    #                 break
    #             else:
    #                 raise ValueError(f"({i}) {fname_B} != ({j}) {fname_C}")
    #     if fnameB_found is False:
    #         new_B_entries.append(fname_B0)

    # Entries that must be added to C
    B_not_in_C = df_UCC_B[~df_UCC_B["fnames"].isin(df_UCC_C["fnames"])]

    # Entries that must be removed from C
    C_not_in_B = df_UCC_C[~df_UCC_C["fnames"].isin(df_UCC_B["fnames"])]

    # Entries manually marked for re-processing in C
    msk = df_UCC_C["process"] == "y"
    C_reprocess = df_UCC_C[msk].copy()

    # Check that these three dataframes don't share equal elements in their 'fnames' columns
    df_names = ["B_not_in_C", "C_not_in_B", "C_reprocess"]
    for i, df1 in enumerate([B_not_in_C, C_not_in_B, C_reprocess]):
        for j, df2 in enumerate([B_not_in_C, C_not_in_B, C_reprocess]):
            if i >= j:
                continue
            shared = set(df1["fnames"]) & set(df2["fnames"])
            if len(shared) > 0:
                raise ValueError(
                    f"{df_names[i]} and {df_names[j]} share {len(shared)} elements"
                )

    return pd.DataFrame(B_not_in_C), pd.DataFrame(C_not_in_B), pd.DataFrame(C_reprocess)


def process_entries(
    df_UCC_B: pd.DataFrame, B_not_in_C: pd.DataFrame, C_reprocess: pd.DataFrame
) -> pd.DataFrame:
    """ """
    # Extract all columns except "fnames"
    B_cols = list(B_not_in_C.keys())
    B_cols.remove("fnames")
    C_cols = list(C_reprocess.keys())
    C_cols.remove("fnames")
    all_cols = B_cols + C_cols

    # Generate empty dictionary with all the fnames to be processed
    all_fnames = list(B_not_in_C["fnames"]) + list(C_reprocess["fnames"])
    df_UCC_updt = {"fnames": all_fnames}
    N_tot = len(all_fnames)
    for k in all_cols:
        df_UCC_updt[k] = [np.nan] * N_tot

    # Add data from the B_not_in_C dataframe
    for i, fname in enumerate(B_not_in_C["fnames"]):
        j = df_UCC_updt["fnames"].index(fname)
        for col in B_cols:
            df_UCC_updt[col][j] = B_not_in_C[col].iloc[i]

    # Add data from the C_reprocess dataframe and information from df_UCC_B
    B_fnames_lst = list(df_UCC_B["fnames"])
    for i, fname in enumerate(C_reprocess["fnames"]):
        j = df_UCC_updt["fnames"].index(fname)
        for col in C_cols:
            df_UCC_updt[col][j] = C_reprocess[col].iloc[i]

        # Add df_UCC_B data
        k = B_fnames_lst.index(fname)
        for col in B_cols:
            df_UCC_updt[col][j] = df_UCC_B[col].iloc[k]

    return pd.DataFrame(df_UCC_updt)


def member_files_updt(
    logging, gaia_frames_data, df_GCs, df_UCC_C, df_UCC_C_updt
) -> pd.DataFrame:
    """
    Updates the Unified Cluster Catalogue (UCC) with new open clusters (OCs).
    """
    for idx, cl_row in df_UCC_C_updt.iterrows():
        # Extract some data
        fnames, ra_c, dec_c, glon_c, glat_c, pmra_c, pmde_c, plx_c = (
            cl_row["fnames"],
            float(cl_row["RA_ICRS"]),
            float(cl_row["DE_ICRS"]),
            float(cl_row["GLON"]),
            float(cl_row["GLAT"]),
            float(cl_row["pmRA"]),  # This can be nan
            float(cl_row["pmDE"]),  # This can be nan
            float(cl_row["Plx"]),  # This can be nan
        )
        fname0 = str(fnames).split(";")[0]
        logging.info(f"\n{idx} Processing {fname0}")

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

    # This dataframe (and file) contains the data extracted from all the new entries,
    # used to update the UCC
    df_UCC_updt = pd.DataFrame(df_UCC_C_updt)
    logging.info("\nTemp file df_UCC_C_updt saved")

    return df_UCC_updt


def update_UCC_membs_data(
    df_members: pd.DataFrame,
    C_not_in_B: pd.DataFrame,
    df_UCC_C: pd.DataFrame,
    df_UCC_C_updt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Update the UCC database using the data extracted from the processed OCs'
    members.
    """
    if len(C_not_in_B) > 0:
        remove_entries = [_.split(";")[0] for _ in C_not_in_B["fnames"]]

        # Remove entries from df_members in the 'name' column
        msk = ~df_members["name"].isin(remove_entries)
        df_members_new = pd.DataFrame(df_members[msk])

        # Remove elements in C_not_in_B["fnames"] from df_UCC_C
        msk = ~df_UCC_C["fnames"].isin(C_not_in_B["fnames"])
        df_UCC_C_new = pd.DataFrame(df_UCC_C[msk])
    else:
        df_members_new = df_members
        df_UCC_C_new = df_UCC_C.copy()

    # Update df_UCC_C_new using data from df_UCC_C_updt
    fnames_lst = list(df_UCC_C_new["fnames"])
    for _, row in df_UCC_C_updt.iterrows():
        fname = row["fnames"]
        if fname in fnames_lst:
            i = fnames_lst.index(fname)
            for col in df_UCC_C_new.columns:
                df_UCC_C_new.at[i, col] = row[col]

    return df_members_new, df_UCC_C_new


def gen_comb_members_file(logging) -> pd.DataFrame:
    """Combine individual parquet files into a single temporary one"""

    # Path to folder with individual .parquet files
    member_files = os.listdir(temp_members_folder)

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

        # # Update entry for this file
        # json_date_data[fname] = (
        #     f"updated ({datetime.datetime.now().strftime('%y%m%d%H')})"
        # )

    # Concatenate all temporary DataFrames into one
    df_comb = pd.concat(tmp, ignore_index=True)

    return df_comb


def update_membs_file(logging, df_members: pd.DataFrame) -> pd.DataFrame:
    """
    Update the parquet file containing estimated members from the
    Unified Cluster Catalog (UCC) dataset, formatted for storage in the Zenodo
    repository.
    """
    # Concatenate all temporary DataFrames into one
    df_comb = gen_comb_members_file(logging)

    #
    logging.info("Updating members file...")

    # Get the list of names in each DataFrame
    names_df1 = set(df_members["name"])
    names_df2 = set(df_comb["name"])

    # Identify names in df_members not in df_comb
    extra_names = names_df1 - names_df2

    # Filter df_members for those extra groups
    df1_extra = df_members[df_members["name"].isin(extra_names)]  # pyright: ignore

    # Concatenate df_comb with the extra df_members groups
    df_updated = pd.concat([df_comb, df1_extra], ignore_index=True)

    logging.info(f"N_membs={len(df_members)} --> N_membs={len(df_updated)}")

    return pd.DataFrame(df_updated)


def find_shared_members(logging, df_UCC, df_members):
    """ """
    logging.info("Finding shared members...")

    # Find OCs that intersect. This helps to speed up the process
    intersection_map = find_intersections(df_UCC, df_members)

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

    # Order 'result_df' according to the fnames order in the 'df_UCC'
    fnames = np.array([_.split(";")[0] for _ in df_UCC["fnames"]])
    result_df["fname"] = pd.Categorical(
        result_df["fname"], categories=fnames, ordered=True
    )
    result_df = result_df.sort_values("fname").reset_index(drop=True)

    # Update data columns for shared members
    df_UCC[["shared_members", "shared_members_p"]] = result_df[
        ["shared_members", "shared_members_p"]
    ]

    return df_UCC


def find_intersections(df, df_members):
    """ """

    # Find OCs that contain duplicated element in any other OC, also to speed up
    source_counts = df_members["Source"].value_counts()
    shared_sources = source_counts[source_counts > 1].index
    ocs_w_shared_sources = df_members[df_members["Source"].isin(shared_sources)][
        "name"
    ].unique()

    # Filter UCC df to only include OCs with shared sources
    names = np.array([_.split(";")[0] for _ in df["fnames"]])
    arr2_set = set(ocs_w_shared_sources)
    msk = np.fromiter((x in arr2_set for x in names), dtype=bool)
    df_msk = df[msk]

    # Convert to NumPy arrays for fast computation
    coords = df_msk[["GLON_m", "GLAT_m"]].to_numpy()
    names = np.array([_.split(";")[0] for _ in df_msk["fnames"]])

    # The search region is two times the r_50 radius
    radii = 2 * df_msk["r_50"].to_numpy() / 60

    # Compute pairwise distances
    dists = cdist(coords, coords)

    # Compute pairwise sum of radii
    radii_sum = radii[:, None] + radii[None, :]

    # Intersection condition: distance <= sum of radii
    # Exclude only the diagonal (self-comparisons)
    not_self = ~np.eye(len(dists), dtype=bool)
    intersection_mask = (dists <= radii_sum) & not_self

    # Extract intersecting names
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


def get_UTI(df_UCC_B, df_UCC_C):
    """ """

    N_50 = df_UCC_C["N_50"].to_numpy()
    r_50 = df_UCC_C["r_50"].to_numpy()
    dens_50 = N_50 / r_50

    # Linear transformation from values to [0, 1]
    C_N50 = np.clip((N_50 - 25) / (250 - 25), 0, 1)
    C_d50 = np.clip((dens_50 - 0) / (2 - 0), 0, 1)

    # Assign a number from 15 to 0 to all elements in C3, according to their
    # positions in 'class_order'
    order_map = {cls: i for i, cls in enumerate(class_order)}
    C3 = df_UCC_C["C3"].to_numpy()
    C_C3 = np.array([15 - order_map[x] for x in C3]) / 15

    # Count number of times each OC is mentioned in the literature
    N_lit = np.array([len(_.split(";")) for _ in df_UCC_B["DB"]])
    C_lit = np.clip((N_lit - 1) / (5 - 1), 0, 1)

    f_year = np.array([int(_.split(";")[0].split("_")[0][-4:]) for _ in df_UCC_B["DB"]])
    fnames = [_.split(";")[0] for _ in df_UCC_B["fnames"]]
    C_sha = [100] * len(df_UCC_C)
    for idx, cl in df_UCC_C.iterrows():
        shared_p = 0
        if str(cl["shared_members"]) == "nan":
            continue
        cl_year = f_year[idx]
        for j, cl_shared in enumerate(cl["shared_members"].split(";")):
            k = fnames.index(cl_shared)
            if cl_year <= f_year[k]:
                continue
            shared_p = max(shared_p, float(cl["shared_members_p"].split(";")[j]))

        C_sha[idx] -= shared_p
    C_sha = np.array(C_sha) / 100

    UTI = 0.25 * (C_N50 + C_d50 + C_C3 + C_lit) * C_sha
    df_UCC_C["UTI"] = np.round(UTI, 2)

    return df_UCC_C


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
    save_df_UCC(logging, df_UCC_C_final, temp_folder + ucc_cat_file, "fnames")

    fpath = temp_zenodo_fold + zenodo_cat_fname
    updt_zenodo_csv(logging, df_UCC_B, df_UCC_C_final, fpath)

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
    # Add column
    df_UCC_C["Name"] = df_UCC_B["Names"]

    # Re-order columns
    df_UCC_C = pd.DataFrame(
        df_UCC_C[
            [
                "Name",
                "RA_ICRS_m",
                "DE_ICRS_m",
                "GLON_m",
                "GLAT_m",
                "Plx_m",
                "pmRA_m",
                "pmDE_m",
                "Rv_m",
                "N_Rv",
                "N_50",
                "r_50",
                "C3",
                "shared_members",
                "shared_members_p",
            ]
        ]
    )
    # Re-name columns
    df_UCC_C.rename(
        columns={
            "RA_ICRS_m": "RA_ICRS",
            "DE_ICRS_m": "DEC_ICRS",
            "GLON_m": "GLON",
            "GLAT_m": "GLAT",
            "Plx_m": "Plx",
            "pmRA_m": "pmRA",
            "pmDE_m": "pmDE",
            "Rv_m": "Rv",
            "shared_members_p": "shared_members_perc",
        },
        inplace=True,
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


def move_files(logging, temp_zenodo_fold: str) -> None:
    """Move files to the appropriate folders"""

    # Move the README file
    file_path_temp = temp_zenodo_fold + "README.txt"
    file_path = zenodo_folder + "README.txt"
    os.rename(file_path_temp, file_path)
    logging.info(file_path_temp + " --> " + file_path)

    # Move the Zenodo catalogue file
    file_path_temp = temp_zenodo_fold + zenodo_cat_fname
    file_path = zenodo_folder + zenodo_cat_fname
    os.rename(file_path_temp, file_path)
    logging.info(file_path_temp + " --> " + file_path)

    # Move the final combined members parquet file
    file_path_temp = temp_zenodo_fold + UCC_members_file
    if os.path.isfile(file_path_temp):
        # Save a copy to the archive folder first
        archived_members = data_folder + "ucc_archived_nogit/" + UCC_members_file
        shutil.copy(file_path_temp, archived_members)
        logging.info(file_path_temp + " --> " + archived_members)
        # Now rename
        file_path = zenodo_folder + UCC_members_file
        os.rename(file_path_temp, file_path)
        logging.info(file_path_temp + " --> " + file_path)
    # Delete left over individual parquet files?

    # Generate '.gz' compressed file for the old C file and archive it
    df = pd.read_csv(data_folder + ucc_cat_file)
    now_time = pd.Timestamp.now().strftime("%y%m%d%H")
    archived_C_file = (
        data_folder
        + "ucc_archived_nogit/"
        + ucc_cat_file.replace(".csv", f"_{now_time}.csv.gz")
    )
    df.to_csv(
        archived_C_file,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        compression="gzip",
    )
    # Remove old C csv file
    os.remove(data_folder + ucc_cat_file)
    logging.info(data_folder + ucc_cat_file + " --> " + archived_C_file)
    # Move new C file into place
    ucc_stored = data_folder + ucc_cat_file
    ucc_temp = temp_folder + ucc_cat_file
    os.rename(ucc_temp, ucc_stored)
    logging.info(ucc_temp + " --> " + ucc_stored)


if __name__ == "__main__":
    main()
