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
    Main function to process and validate CANTAT20 data and generate plots and parquet
    files.

    This script was used in November 2024, as a previous step to the implementation
    of the images carousel in the UCC cluster's pages.
    """
    df_UCC, cantat20_membs = load_data()

    ucc_fnames_c20, ucc_quad_c20 = extract_C20(df_UCC)

    cantat20_membs = apply_name_changes(cantat20_membs)

    c20_names, c20_fnames = generate_fnames(cantat20_membs)

    fnames_members = get_UCC_fnames(
        ucc_fnames_c20,
        ucc_quad_c20,
        c20_names,
        c20_fnames,
    )

    files_check(fnames_members)

    create_parquet_and_webp_files(fnames_members, cantat20_membs)


def load_data() -> tuple:
    """
    Load the UCC database and CANTAT20 members.

    Returns:
        tuple: A tuple containing the UCC dataframe and CANTAT20 members dataframe.
    """

    # Read UCC
    df_UCC = pd.read_csv("../../zenodo/UCC_cat_241106.csv")

    print("Reading CANTAT20 members...\n")
    cantat20_membs = pd.read_parquet("CANTAT20_members.parquet")
    cantat20_membs["Name"] = cantat20_membs["Cluster"]

    return df_UCC, cantat20_membs


def extract_C20(df_UCC: pd.DataFrame) -> tuple:
    """
    Extract CANTAT20 open clusters from the UCC dataframe.

    Args:
        df_UCC (pd.DataFrame): The UCC dataframe.

    Returns:
        tuple: Two lists, filenames and quadrants of CANTAT20 clusters in the UCC.
    """
    ucc_fnames_c20, ucc_quad_c20 = [], []
    for i, db in enumerate(df_UCC["DB"]):
        if "CANTAT20" in db:
            ucc_fnames_c20.append(df_UCC["fnames"][i].split(";"))
            ucc_quad_c20.append(df_UCC["quad"][i])
    print(f"N={len(ucc_fnames_c20)} CANTAT20 OCs in UCC\n")

    return ucc_fnames_c20, ucc_quad_c20


def apply_name_changes(cantat20_membs: pd.DataFrame) -> pd.DataFrame:
    """
    Apply name corrections to the CANTAT20 members.

    Args:
        hunt23_membs (pd.DataFrame): The CANTAT20 members dataframe.

    Returns:
        pd.DataFrame: Updated dataframes for members.
    """
    for i, name in enumerate(cantat20_membs["Name"]):
        if name.startswith("LP_"):
            nname = name.replace("LP_", "fof_")
            cantat20_membs.loc[i, "Name"] = nname

        if name.startswith("Sigma_Ori"):
            nname = name.replace("Sigma_Ori", "Sigma_Orionis")
            cantat20_membs.loc[i, "Name"] = nname

    return cantat20_membs


def generate_fnames(cantat20_membs: pd.DataFrame) -> tuple:
    """
    Generate standardized filenames for CANTAT20 members and types.

    Args:
        cantat20_membs (pd.DataFrame): The CANTAT20 members dataframe.

    Returns:
        tuple: Arrays of names and filenames.
    """
    # Extract names maintaining the order
    c20_names = np.array(list(dict.fromkeys(cantat20_membs["Name"]).keys()))
    c20_fnames = [
        _.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("+", "p")
        .replace(".", "")
        for _ in c20_names
    ]

    return c20_names, c20_fnames


def get_UCC_fnames(
    ucc_fnames_c20: list,
    ucc_quad_c20: list,
    c20_names: np.ndarray,
    c20_fnames: list,
) -> list:
    """
    Match CANTAT20 members with UCC filenames.

    Args:
        ucc_fnames_c20 (list): Filenames of CANTAT20 clusters in the UCC.
        ucc_quad_c20 (list): Quadrants of CANTAT20 clusters in the UCC.
        c20_names (np.ndarray): Array of CANTAT20 names.
        c20_fnames (list): List of standardized filenames for CANTAT20 members.

    Returns:
        list: Matched filenames and their associated quadrants.
    """
    fnames_members = []
    N = 0
    # For each OC in the HUNT23 members list
    for k, fname in enumerate(c20_fnames):
        # Find this OC in the members file in the UCC database
        found_flag = False
        for q, fnames_c20 in enumerate(ucc_fnames_c20):
            if fname in fnames_c20:
                j = fnames_c20.index(fname)
                # If the fname of this members OCs is not the fname in the UCC
                if j != 0:
                    print(f"({c20_names[k]}) {fname} --> {fnames_c20[0]}")
                found_flag = True
                # Save full name of OC and its fname matching the UCC's fname
                fnames_members.append([c20_names[k], fnames_c20[0], ucc_quad_c20[q]])
                break

        if found_flag is False:
            print(f"({c20_names[k]}) {fname}", "NOT FOUND")
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
    fnames_members: list, cantat20_membs: pd.DataFrame
) -> None:
    """
    Generate parquet files and plots for specified CANTAT20 members.

    Args:
        fnames_members (list): List of filenames and their associated quadrants.
        cantat20_membs (pd.DataFrame): The CANTAT20 members dataframe.
    """
    cantat20_membs.rename(columns={"proba": "probs"}, inplace=True)
    cantat20_membs.rename(columns={"pmRA*": "pmRA"}, inplace=True)

    for idx, name in enumerate(fnames_members):
        fname0, Qfold = name[1], name[2]

        print(idx, name)

        # Create .parquet file
        msk = cantat20_membs["Name"] == name[0]
        df = cantat20_membs[msk]
        df = df[
            [
                "GaiaDR2",
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
            ]
        ]
        path = "../../../" + Qfold + "/datafiles/" + fname0 + "_CANTAT20.parquet"
        df.to_parquet(path, index=False)

        # Create plot file
        plot_fpath = Path(
            "../../../" + Qfold + f"/{plots_folder}/" + fname0 + "_CANTAT20.webp"
        )
        title = r"Cantat-Gaudin et al. (2020)"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ucc_plots.make_plot(plot_fpath, df, DRY_RUN=False, title=title)


if __name__ == "__main__":
    main()
