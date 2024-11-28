import os.path
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("..")
from HARDCODED import (
    plots_folder,
)
from modules import ucc_plots


def main() -> None:
    """
    Main function to process and validate HUNT23 data and generate plots and parquet
    files.

    This script was used in November 2024, as a previous step to the implementation
    of the images carousel in the UCC cluster's pages.
    """
    df_UCC, hunt23_membs, df_H23_types = load_data()

    ucc_fnames_h23, ucc_quad_h23 = extract_H23(df_UCC)

    hunt23_membs, df_H23_types = apply_name_changes(hunt23_membs, df_H23_types)

    h23_names, h23_fnames, h23_types_fnames = generate_fnames(
        hunt23_membs, df_H23_types
    )

    fnames_members = get_UCC_fnames(
        ucc_fnames_h23,
        ucc_quad_h23,
        h23_names,
        h23_fnames,
        h23_types_fnames,
        df_H23_types,
    )

    files_check(fnames_members)

    create_parquet_and_webp_files(fnames_members, hunt23_membs)


def load_data() -> tuple:
    """
    Load the UCC database, HUNT23 members, and HUNT23 type information.

    Returns:
        tuple: A tuple containing the UCC dataframe, HUNT23 members dataframe,
        and HUNT23 types dataframe.
    """
    # Read UCC
    df_UCC = pd.read_csv("../../zenodo/UCC_cat_241106.csv")

    print("Reading HUNT23 members...\n")
    hunt23_membs = pd.read_parquet("HUNT23_members.parquet")
    hunt23_membs["Name"] = hunt23_membs["Name"].str.strip()

    # Read the file with type information
    df_H23_types = pd.read_csv("HUNT23_types.csv")
    df_H23_types["Name"] = df_H23_types["Name"].str.strip()

    return df_UCC, hunt23_membs, df_H23_types


def extract_H23(df_UCC: pd.DataFrame) -> tuple:
    """
    Extract HUNT23 open clusters from the UCC dataframe.

    Args:
        df_UCC (pd.DataFrame): The UCC dataframe.

    Returns:
        tuple: Two lists, filenames and quadrants of HUNT23 clusters in the UCC.
    """
    ucc_fnames_h23, ucc_quad_h23 = [], []
    for i, db in enumerate(df_UCC["DB"]):
        if "HUNT23" in db:
            ucc_fnames_h23.append(df_UCC["fnames"][i].split(";"))
            ucc_quad_h23.append(df_UCC["quad"][i])
    print(f"N={len(ucc_fnames_h23)} HUNT23 OCs in UCC\n")

    return ucc_fnames_h23, ucc_quad_h23


def apply_name_changes(hunt23_membs: pd.DataFrame, df_H23_types: pd.DataFrame) -> tuple:
    """
    Apply name corrections to the HUNT23 members and types dataframes.

    Args:
        hunt23_membs (pd.DataFrame): The HUNT23 members dataframe.
        df_H23_types (pd.DataFrame): The HUNT23 types dataframe.

    Returns:
        tuple: Updated dataframes for members and types.
    """
    name_changes = [
        ("ESO_429-429", "ESO_429-02"),
        ("CMa_2", "CMa_02"),
        ("AH03_J0748+26.9", "AH03_J0748-26.9"),
        ("Juchert_J0644.8+0925", "Juchert_J0644.8-0925"),
        ("Teutsch_J0718.0+1642", "Teutsch_J0718.0-1642"),
        ("Teutsch_J0924.3+5313", "Teutsch_J0924.3-5313"),
        ("Teutsch_J1037.3+6034", "Teutsch_J1037.3-6034"),
        ("Teutsch_J1209.3+6120", "Teutsch_J1209.3-6120"),
    ]
    for names in name_changes:
        msk = hunt23_membs["Name"] == names[0]
        hunt23_membs.loc[msk, "Name"] = names[1]

        msk = df_H23_types["Name"] == names[0]
        df_H23_types.loc[msk, "Name"] = names[1]

    return hunt23_membs, df_H23_types


def generate_fnames(hunt23_membs: pd.DataFrame, df_H23_types: pd.DataFrame) -> tuple:
    """
    Generate standardized filenames for HUNT23 members and types.

    Args:
        hunt23_membs (pd.DataFrame): The HUNT23 members dataframe.
        df_H23_types (pd.DataFrame): The HUNT23 types dataframe.

    Returns:
        tuple: Arrays of names, filenames, and type filenames.
    """
    # Extract names maintaining the order
    h23_names = np.array(list(dict.fromkeys(hunt23_membs["Name"]).keys()))
    h23_fnames = [
        _.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("+", "p")
        .replace(".", "")
        for _ in h23_names
    ]

    h23_types_names = df_H23_types["Name"].values
    h23_types_fnames = [
        _.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("+", "p")
        .replace(".", "")
        for _ in h23_types_names
    ]

    return h23_names, h23_fnames, h23_types_fnames


def get_UCC_fnames(
    ucc_fnames_h23: list,
    ucc_quad_h23: list,
    h23_names: np.ndarray,
    h23_fnames: list,
    h23_types_fnames: list,
    df_H23_types: pd.DataFrame,
) -> list:
    """
    Match HUNT23 members with UCC filenames.

    Args:
        ucc_fnames_h23 (list): Filenames of HUNT23 clusters in the UCC.
        ucc_quad_h23 (list): Quadrants of HUNT23 clusters in the UCC.
        h23_names (np.ndarray): Array of HUNT23 names.
        h23_fnames (list): List of standardized filenames for HUNT23 members.
        h23_types_fnames (list): List of standardized filenames for HUNT23 types.
        df_H23_types (pd.DataFrame): The HUNT23 types dataframe.

    Returns:
        list: Matched filenames and their associated quadrants.
    """
    # Removed GCs from the original HUNT23 database
    GCs_removed = (
        "palomar2",
        "ic1276",
        "palomar8",
        "palomar10",
        "palomar11",
        "palomar12",
        "1636283",
        "pismis26",
        "lynga7",
        "hsc2890",
        "hsc134",
        "teutsch182",  # This is not a GC but it was removed
    )

    fnames_members = []
    N = 0
    # For each OC in the HUNT23 members list
    for k, fname in enumerate(h23_fnames):
        # Find it in the list of types data
        i = h23_types_fnames.index(fname)
        # Only move forward for OCs
        if df_H23_types["Type"][i] != "o":
            continue
        # Skip removed entries
        if fname in GCs_removed:
            continue

        # Find this OC in the members file in the UCC database
        found_flag = False
        for q, fnames_h23 in enumerate(ucc_fnames_h23):
            if fname in fnames_h23:
                j = fnames_h23.index(fname)
                # If the fname of this members OCs is not the fname in the UCC
                if j != 0:
                    print(f"({h23_names[k]}) {fname} --> {fnames_h23[0]}")
                found_flag = True
                # Save full name of OC and its fname matching the UCC's fname
                fnames_members.append([h23_names[k], fnames_h23[0], ucc_quad_h23[q]])
                break

        if found_flag is False:
            print(fname, "NOT FOUND")
            N += 1

    if N > 0:
        print(f"\nOCs with members not found in UCC: {N}\n")
        raise SystemExit

    return fnames_members


def files_check(fnames_members: list) -> None:
    """
    Check the existence of parquet and webp files for HUNT23 members.

    Args:
        fnames_members (list): List of filenames and their associated quadrants.

    Raises:
        SystemExit: If any required parquet file is missing.
    """
    N1, N2 = 0, 0
    for name in fnames_members:
        name0, fname0, Qfold = name

        # Find file in 'path' folder. It should exist
        path = "../../../" + Qfold + "/datafiles/" + fname0 + ".parquet"
        if os.path.isfile(path) is False:
            print(f"{name0} parquet file not found")
            N1 += 1

        path = Path("../../../" + Qfold + f"/{plots_folder}/" + fname0 + ".webp")
        if os.path.isfile(path) is False:
            print(f"{name0} webp file not found")
            N2 += 1

    if N1 > 0:
        print(f"\nParquet files not found in Q folders: {N1}\n")
        raise SystemExit
    if N2 > 0:
        print(f"\nwebp files not found in Q folders: {N2}\n")
        raise SystemExit


def create_parquet_and_webp_files(
    fnames_members: list, hunt23_membs: pd.DataFrame
) -> None:
    """
    Generate parquet files and plots for specified HUNT23 members.

    Args:
        fnames_members (list): List of filenames and their associated quadrants.
        hunt23_membs (pd.DataFrame): The HUNT23 members dataframe.
    """
    hunt23_membs.rename(columns={"Prob": "probs"}, inplace=True)

    for idx, name in enumerate(fnames_members):
        fname0, Qfold = name[1], name[2]
        print(idx, name)

        # Create .parquet file
        msk = hunt23_membs["Name"] == name[0]

        df = hunt23_membs[msk]
        df = df[
            [
                "GaiaDR3",
                "probs",
                "RA_ICRS",
                "DE_ICRS",
                "GLON",
                "GLAT",
                "pmRA",
                "pmDE",
                "Plx",
                "Gmag",
                "BP-RP",
                "RV",
            ]
        ]
        path = "../../../" + Qfold + "/datafiles/" + fname0 + "_HUNT23.parquet"
        df.to_parquet(path, index=False)

        # Create plot file
        plot_fpath = Path(
            "../../../" + Qfold + f"/{plots_folder}/" + fname0 + "_HUNT23.webp"
        )
        title = r"Hunt \& Reffert (2023)"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ucc_plots.make_plot(plot_fpath, df, DRY_RUN=False, title=title)


if __name__ == "__main__":
    main()
