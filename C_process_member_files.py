import csv
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from fastparquet import ParquetFile
from scipy.spatial.distance import cdist

from B_update_UCC_DB import run_mode
from modules.HARDCODED import (
    GCs_cat,
    UCC_archive,
    UCC_folder,
    UCC_members_file,
    manual_pars_file,
    members_folder,
    parquet_dates,
    path_gaia_frames,
    path_gaia_frames_ranges,
    temp_fold,
)
from modules.update_database.member_files_updt_funcs import get_fastMP_membs
from modules.utils import (
    diff_between_dfs,
    get_last_version_UCC,
    logger,
    save_df_UCC,
)


def main():
    """Second function to update the UCC (Unified Cluster Catalogue) with a new
    database
    """
    logging = logger()

    logging.info(f"=== Running C script in '{run_mode}' mode ===\n")

    # Generate paths and check for required folders and files
    (
        ucc_file,
        temp_zenodo_fold,
        UCC_new_version,
        new_ucc_file,
        archived_UCC_file,
    ) = get_paths_check_paths(logging)

    (
        gaia_frames_data,
        df_GCs,
        df_UCC_B,
        manual_pars,
    ) = load_data(logging)

    # Entries with no 'N_50' value are identified as new and processed with fastMP
    N_new = np.isnan(df_UCC_B["N_50"]).sum()

    if N_new == 0:
        logging.info("\nNo new OCs to process")
        df_UCC_final = df_UCC_B
        json_date_data, df_members = None, None
        # Count number of members in current file
        N_members = ParquetFile(UCC_folder + UCC_members_file).count()
    else:
        logging.info(f"\nProcessing {N_new} new OCs")

        # # Load file if it already exists and the .parquet files were generated
        # df_UCC_updt = pd.read_csv(temp_fold + "df_UCC_updt.csv")

        # Generate member files for new OCs and obtain their data
        df_UCC_updt = member_files_updt(
            logging, df_UCC_B, gaia_frames_data, df_GCs, manual_pars
        )

        # Update the UCC with the new OCs member's data
        df_UCC_new = update_UCC_membs_data(df_UCC_B, df_UCC_updt)
        logging.info(f"UCC database updated ({len(df_UCC_new)} entries)")
        diff_between_dfs(logging, df_UCC_B, df_UCC_new)

        # Combine individual parquet members file into a single one. Also update
        # dates of their updating
        df_comb, json_date_data = gen_comb_members_file(logging)
        logging.info(f"Combined members file generated ({len(df_comb)} stars)")

        # Generate final updated members file
        df_members = update_membs_file(logging, df_comb)
        N_members = len(df_members)
        logging.info(f"Zenodo '{UCC_members_file}' file updated ({N_members} stars)")

        # Find shared members between OCs
        df_UCC_final = find_shared_members(logging, df_UCC_new, df_members)
        logging.info("Shared members data updated in UCC")

    # Save the generated data to temporary files before moving them
    update_files(
        logging,
        UCC_new_version,
        new_ucc_file,
        temp_zenodo_fold,
        json_date_data,
        df_members,
        N_members,
        df_UCC_final,
    )

    if input("\nMove files to their final paths? (y/n): ").lower() != "y":
        sys.exit()
    move_files(logging, temp_zenodo_fold, ucc_file, archived_UCC_file, new_ucc_file)

    # if input("\nRemove temporary files and folders? (y/n): ").lower() == "y":
    #     # shutil.rmtree(temp_fold)
    #     logging.info(f"Folder removed: {temp_fold}")


def get_paths_check_paths(
    logging,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
]:
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

    # If file exists, warn
    if os.path.isfile(temp_fold + "df_UCC_updt.csv"):
        logging.warning(
            "WARNING: file 'df_UCC_updt.csv' exists. Moving on will re-write it"
        )
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Create folder to store parquet member files
    out_path = temp_fold + members_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        if len(os.listdir(out_path)) > 0:
            logging.warning(
                f"WARNING: There are .parquet files in '{out_path}'. If left there,"
                + "\nthey will be used when the script combines the"
                + " final members data"
            )
            if input("Move on? (y/n): ").lower() != "y":
                sys.exit()

    last_version = get_last_version_UCC(UCC_folder)
    # Path to the current UCC csv file
    ucc_file = UCC_folder + last_version

    # Temporary zenodo/ folder
    temp_zenodo_fold = temp_fold + UCC_folder
    # Create if required
    if not os.path.exists(temp_zenodo_fold):
        os.makedirs(temp_zenodo_fold)

    # Path to the new (temp) version of the UCC database
    new_version = datetime.datetime.now().strftime("%Y%m%d%H")[2:]
    new_ucc_file = "UCC_cat_" + new_version + ".csv"
    # # Check if file already exists
    # if os.path.exists(temp_zenodo_fold + new_ucc_file):
    #     logging.info(
    #         f"File {temp_zenodo_fold + new_ucc_file} already exists. "
    #         + "Moving on will re-write it"
    #     )
    #     if input("Move on? (y/n): ").lower() != "y":
    #         sys.exit()

    # Path to archive the current UCC csv file
    archived_UCC_file = UCC_archive + last_version.replace(".csv", ".csv.gz")

    return (
        ucc_file,
        temp_zenodo_fold,
        new_version,
        new_ucc_file,
        archived_UCC_file,
    )


def load_data(logging) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ """
    # Load file with Gaia frames ranges
    gaia_frames_data = pd.DataFrame([])
    if os.path.isfile(path_gaia_frames_ranges):
        gaia_frames_data = pd.read_csv(path_gaia_frames_ranges)
    # else:
    #     warnings.warn(f"File {path_gaia_frames_ranges} not found")

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)

    df_UCC_B = pd.read_csv(temp_fold + "df_UCC_B_updt.csv")
    logging.info(f"\nFile 'df_UCC_B_updt.csv' loaded ({len(df_UCC_B)} entries)")

    # Dummy
    manual_pars = pd.DataFrame()
    if run_mode == "manual":
        # Read OCs manual parameters
        manual_pars = pd.read_csv(manual_pars_file)

    return gaia_frames_data, df_GCs, df_UCC_B, manual_pars


def member_files_updt(
    logging, df_UCC, gaia_frames_data, df_GCs, manual_pars
) -> pd.DataFrame:
    """
    Updates the Unified Cluster Catalogue (UCC) with new open clusters (OCs).
    """
    # For each new OC
    df_UCC_updt = {
        "UCC_idx": [],
        "N_clust": [],
        "N_clust_max": [],
        "C3": [],
        "GLON_m": [],
        "GLAT_m": [],
        "RA_ICRS_m": [],
        "DE_ICRS_m": [],
        "Plx_m": [],
        "pmRA_m": [],
        "pmDE_m": [],
        "Rv_m": [],
        "N_Rv": [],
        "N_50": [],
        "r_50": [],
    }
    N_cl = 0
    for UCC_idx, new_cl in df_UCC.iterrows():
        # Check if this is a new OC that should be processed
        if not np.isnan(new_cl["N_50"]):
            continue

        fnames, ra_c, dec_c, glon_c, glat_c, pmra_c, pmde_c, plx_c = (
            new_cl["fnames"],
            float(new_cl["RA_ICRS"]),
            float(new_cl["DE_ICRS"]),
            float(new_cl["GLON"]),
            float(new_cl["GLAT"]),
            float(new_cl["pmRA"]),  # This can be nan
            float(new_cl["pmDE"]),  # This can be nan
            float(new_cl["Plx"]),  # This can be nan
        )
        fname0 = str(fnames).split(";")[0]
        logging.info(f"\n{N_cl} Processing {fname0} (idx={UCC_idx})")

        df_UCC_updt = get_fastMP_membs(
            logging,
            manual_pars,
            df_UCC,
            df_GCs,
            gaia_frames_data,
            UCC_idx,
            ra_c,
            dec_c,
            glon_c,
            glat_c,
            pmra_c,
            pmde_c,
            plx_c,
            fname0,
            df_UCC_updt,
        )

        # Update file with information. Do this for each iteration to avoid
        # losing data if something goes wrong with any cluster
        df = pd.DataFrame(df_UCC_updt)
        df.to_csv(temp_fold + "df_UCC_updt.csv", index=False, na_rep="nan")

        N_cl += 1

    # This dataframe (and file) contains the data extracted from all the new entries,
    # used to update the UCC
    df_UCC_updt = pd.DataFrame(df_UCC_updt)
    logging.info("\nTemp file df_UCC_updt saved")

    return df_UCC_updt


def update_UCC_membs_data(df_UCC, df_UCC_updt) -> pd.DataFrame:
    """
    Update the UCC database using the data extracted from the processed OCs'
    members.
    """
    # Generate copy to not disturb the dataframe given to this function which is
    # later used to generate the diffs files
    df_inner = df_UCC.copy()

    # Update 'df_inner' using 'df_UCC_updt' data
    for _, row in df_UCC_updt.iterrows():
        UCC_idx = row["UCC_idx"]
        for key, val in row.items():
            if key == "UCC_idx":
                continue
            df_inner.at[UCC_idx, key] = val

    return df_inner


def gen_comb_members_file(logging) -> tuple[pd.DataFrame, dict]:
    """Combine individual parquet files into a single temporary one"""

    # Path to folder with individual .parquet files
    path = temp_fold + members_folder
    member_files = os.listdir(path)

    logging.info(f"Combining {len(member_files)} .parquet files...")
    json_date_data, tmp = {}, []
    for file in member_files:
        df = pd.read_parquet(path + file)

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

        # Update entry for this file
        json_date_data[fname] = (
            f"updated ({datetime.datetime.now().strftime('%y%m%d%H')})"
        )

    # Concatenate all temporary DataFrames into one
    df_comb = pd.concat(tmp, ignore_index=True)

    return df_comb, json_date_data


def update_membs_file(logging, df_comb) -> pd.DataFrame:
    """
    Update the parquet file containing estimated members from the
    Unified Cluster Catalog (UCC) dataset, formatted for storage in the Zenodo
    repository.
    """
    logging.info("Updating members file...")

    # Load current members file
    zenodo_members_file = UCC_folder + UCC_members_file
    df_members = pd.read_parquet(zenodo_members_file)

    # Get the list of names in each DataFrame
    names_df1 = set(df_members["name"])
    names_df2 = set(df_comb["name"])

    # Identify names in df_members not in df_comb
    extra_names = pd.DataFrame(names_df1 - names_df2)

    # Filter df_members for those extra groups
    df1_extra = df_members[df_members["name"].isin(extra_names)]

    # Concatenate df_comb with the extra df_members groups
    df_updated = pd.concat([df_comb, df1_extra], ignore_index=True)

    return pd.DataFrame(df_updated)


def find_shared_members(logging, df_UCC, df_members):
    """ """
    logging.info("Finding shared members...")

    intersection_map = find_intersections(df_UCC)

    # Group members by 'fname'
    grouped = df_members.groupby("name")["Source"].apply(set)

    results = []
    # Compute shared elements and percentages
    for fname, sources in grouped.items():
        if intersection_map[fname] == "nan":
            results.append(
                {"fname": fname, "shared_members": "nan", "shared_members_p": "nan"}
            )
            continue

        fnames_process = [_ for _ in intersection_map[fname].split(",")]

        # shared_info, percentage_info = [], []
        # for other_fname in fnames_process:
        #     other_sources = set(df_members[df_members['name']==other_fname]["Source"])

        shared_info, percentage_info = [], []
        for other_fname, other_sources in grouped.items():
            # Only process intersecting OCs
            if other_fname not in fnames_process:
                continue

            shared = sources & other_sources
            if shared:
                shared_info.append(other_fname)
                percentage = len(shared) / len(sources) * 100
                percentage_info.append(f"{percentage:.1f}")

        if shared_info:
            results.append(
                {
                    "fname": fname,
                    "shared_members": ";".join(shared_info),
                    "shared_members_p": ";".join(percentage_info),
                }
            )
        else:
            results.append(
                {"fname": fname, "shared_members": "nan", "shared_members_p": "nan"}
            )

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


def find_intersections(df):
    # Convert to NumPy arrays for fast computation
    coords = df[["GLON_m", "GLAT_m"]].to_numpy()
    names = np.array([_.split(";")[0] for _ in df["fnames"]])

    # The search region is two times the r_50 radius
    radii = 2 * df["r_50"].to_numpy() / 60

    # Compute pairwise distances
    dists = cdist(coords, coords)

    # Compute pairwise sum of radii
    radii_sum = radii[:, None] + radii[None, :]

    # Intersection condition: distance <= sum of radii, excluding self-comparisons
    intersection_mask = (dists <= radii_sum) & (dists > 0)

    # Extract intersecting names
    results = []
    for i, name in enumerate(names):
        intersecting = names[intersection_mask[i]]
        val = "nan"
        if intersecting.size > 0:
            val = ",".join(intersecting)
        results.append({"name": name, "intersects_with": val})

    intersections = pd.DataFrame(results)

    intersection_map = intersections.set_index("name")["intersects_with"].to_dict()

    return intersection_map


def update_files(
    logging,
    UCC_new_version: str,
    new_ucc_file: str,
    temp_zenodo_fold: str,
    json_date_data: dict | None,
    df_members: pd.DataFrame | None,
    N_members: int,
    df_UCC_final: pd.DataFrame,
):
    """ """
    # Generate updated full UCC catalogue
    logging.info("\nUpdate files:")

    # Save updated UCC to temporary CSV file
    ucc_temp = temp_zenodo_fold + new_ucc_file
    save_df_UCC(logging, df_UCC_final, ucc_temp)

    file_path = temp_zenodo_fold + "UCC_cat.csv"
    updt_zenodo_csv(logging, df_UCC_final, file_path)

    N_clusters = len(df_UCC_final)
    updt_readme(logging, UCC_new_version, N_clusters, N_members, temp_zenodo_fold)

    if df_members is not None:
        # Update JSON dates file
        fname_json = UCC_folder + parquet_dates
        # Load the current JSON file
        with open(fname_json, "r") as f:
            json_data = json.load(f)
        # Update JSON data
        json_data.update(json_date_data)
        # Save the updated JSON to temp file
        fname_json_temp = temp_zenodo_fold + parquet_dates
        with open(fname_json_temp, "w") as f:
            json.dump(json_data, f, indent=2)
        logging.info(f"JSON file with dates updated: '{fname_json_temp}'")

        zenodo_members_file_temp = temp_zenodo_fold + UCC_members_file
        df_members.to_parquet(zenodo_members_file_temp, index=False)
        logging.info(f"Members file updated: '{zenodo_members_file_temp}'")


def updt_zenodo_csv(logging, df_UCC: pd.DataFrame, file_path: str) -> None:
    """
    Generates a CSV file containing a reduced Unified Cluster Catalog
    (UCC) dataset, which can be stored in the Zenodo repository.
    """

    # Add 'fname' column
    df = df_UCC.copy()
    df["fname"] = df["fnames"].str.split(";").str[0]

    # Re-order columns
    df = pd.DataFrame(
        df[
            [
                "UCC_ID",
                "ID",
                "RA_ICRS",
                "DE_ICRS",
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
    df.rename(
        columns={
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
    df.to_csv(
        file_path,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )

    logging.info(f"Zenodo '.csv' file generated: '{file_path}'")


def updt_readme(
    logging, new_version: str, N_clusters: int, N_members: int, temp_zenodo_fold: str
) -> None:
    """Update info number in README file uploaded to Zenodo"""

    XXXX = str(new_version[:-2])
    YYYY = str(N_clusters)
    ZZZZ = str(N_members)
    txt = [
        f"These files correspond to the {XXXX} version of the UCC database (https://ucc.ar),\n",
        f"composed of {YYYY} clusters with a combined {ZZZZ} members.",
    ]

    # Load the main file
    in_file_path = UCC_folder + "README.txt"
    with open(in_file_path, "r") as f:
        dataf = f.readlines()
        # Replace lines
        dataf[2:4] = txt

    # Store updated file
    out_file_path = temp_zenodo_fold + "README.txt"
    with open(out_file_path, "w") as f:
        f.writelines(dataf)

    logging.info(f"Zenodo README file updated: '{out_file_path}'")


def move_files(
    logging,
    temp_zenodo_fold: str,
    ucc_file: str,
    archived_UCC_file: str,
    new_ucc_file: str,
) -> None:
    """Move files to the appropriate folders"""

    # Move the README file
    file_path_temp = temp_zenodo_fold + "README.txt"
    file_path = UCC_folder + "README.txt"
    os.rename(file_path_temp, file_path)
    logging.info(file_path_temp + " --> " + file_path)

    # Move the data dates file
    file_path_temp = temp_zenodo_fold + parquet_dates
    file_path = UCC_folder + parquet_dates
    os.rename(file_path_temp, file_path)
    logging.info(file_path_temp + " --> " + file_path)

    # Move the Zenodo catalogue file
    file_path_temp = temp_zenodo_fold + "UCC_cat.csv"
    file_path = UCC_folder + "UCC_cat.csv"
    os.rename(file_path_temp, file_path)
    logging.info(file_path_temp + " --> " + file_path)

    # Generate '.gz' compressed file for the old UCC and archive it
    df = pd.read_csv(ucc_file)
    df.to_csv(
        archived_UCC_file,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    # Remove old UCC csv file
    os.remove(ucc_file)
    logging.info(ucc_file + " --> " + archived_UCC_file)
    # Move new UCC file into place
    ucc_stored = UCC_folder + new_ucc_file
    ucc_temp = temp_zenodo_fold + new_ucc_file
    os.rename(ucc_temp, ucc_stored)
    logging.info(ucc_temp + " --> " + ucc_stored)

    # Move the final combined parquet file
    file_path_temp = temp_zenodo_fold + UCC_members_file
    file_path = UCC_folder + UCC_members_file
    os.rename(file_path_temp, file_path)
    logging.info(file_path_temp + " --> " + file_path)

    # Delete all individual parquet files?


if __name__ == "__main__":
    main()
