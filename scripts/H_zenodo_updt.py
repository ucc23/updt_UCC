import csv
import os

import pandas as pd
from HARDCODED import UCC_folder, members_folder, root_UCC_path, zenodo_folder
from modules import UCC_new_match, logger


def main():
    """ """
    logging = logger.main()
    logging.info("\nRunning 'J_zenodo_updt' script\n")

    # Read latest version of the UCC
    df_UCC, UCC_cat = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    zenodo_UCC(df_UCC)
    logging.info("\n'UCC_cat.csv' file generated\n")

    print("Reading member files...")
    zenodo_membs(logging, root_UCC_path, members_folder)
    logging.info("\n'UCC_members.parquet' file generated")

    zenodo_readme(UCC_cat)
    logging.info("\nREADME file updated")


def zenodo_UCC(df_UCC: pd.DataFrame) -> None:
    """
    Generates a CSV file containing a reduced Unified Cluster Catalog
    (UCC) dataset, which can be stored in the Zenodo repository.

    Parameters:
    df_UCC (pd.DataFrame): DataFrame containing the UCC data with a wide set of columns.

    The function performs the following steps:
    1. Drops unnecessary columns from the input DataFrame.
    2. Re-orders the remaining columns in a specified order.
    3. Saves the resulting DataFrame to a CSV file, with specified formatting.

    Returns:
    None
    """
    # Define columns to drop
    drop_cols = [
        "DB",
        "DB_i",
        "GLON",
        "GLAT",
        "fnames",
        "quad",
        "dups_fnames",
        "dups_probs",
        "N_fixed",
        # "N_membs",
        "fixed_cent",
        "cent_flags",
        "GLON_m",
        "GLAT_m",
        "dups_fnames_m",
        "dups_probs_m",
    ]
    df = df_UCC.drop(columns=drop_cols)

    # Re-order columns
    df = df[
        [
            "ID",
            "RA_ICRS",
            "DE_ICRS",
            "Plx",
            "pmRA",
            "pmDE",
            "UCC_ID",
            "N_50",
            "r_50",
            "RA_ICRS_m",
            "DE_ICRS_m",
            "Plx_m",
            "pmRA_m",
            "pmDE_m",
            "Rv_m",
            "N_Rv",
            "C1",
            "C2",
            "C3",
        ]
    ]

    # Store to csv file
    df.to_csv(
        zenodo_folder + "UCC_cat.csv",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def zenodo_membs(logging, root_UCC_path: str, members_folder: str) -> None:
    """
    Generates a parquet file containing estimated members from the
    Unified Cluster Catalog (UCC) dataset, formatted for storage in the Zenodo
    repository.

    Parameters:
    logging: A logging object to record information and progress.
    root_UCC_path (str): Root directory of the UCC dataset.
    members_folder (str): Subdirectory containing the members files within each
    quadrant and latitude folder.

    The function performs the following steps:
    1. Iterates through quadrant and latitude directories (Q1P, Q1N, ..., Q4N) within
       the UCC structure.
    2. For each file in these directories:
        - Reads the data, adds a 'name' column based on the filename, and appends it
          to a temporary list.
    3. Concatenates all DataFrames in the list.
    4. Saves the combined DataFrame to a parquet file in the specified folder.

    Returns:
    None
    """
    # Initialize list for storing temporary DataFrames
    tmp = []
    for quad in ("1", "2", "3", "4"):
        for lat in ("P", "N"):
            logging.info(f"Processing Q{quad}{lat}")
            path = root_UCC_path + "Q" + quad + lat + f"/{members_folder}/"
            for file in os.listdir(path):
                # Ignore these dbs
                if "HUNT23" in file or "CANTAT20" in file:
                    continue
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

    # Concatenate all temporary DataFrames into one
    df_comb = pd.concat(tmp, ignore_index=True)
    df_comb.to_parquet(zenodo_folder + "UCC_members.parquet", index=False)


def zenodo_readme(UCC_cat) -> None:
    """Update version number in README file uploaded to Zenodo"""
    # Load the main file
    with open(UCC_folder + "README.txt", "r") as f:
        dataf = f.read()
        # Update the version number
        version = UCC_cat.split("/")[-1].split(".")[0].split("_cat_")[1]
        dataf = dataf.replace("XXXXXX", version)
    # Store updated file in the appropriate folder
    with open(zenodo_folder + "README.txt", "w") as f:
        f.write(dataf)


if __name__ == "__main__":
    main()
