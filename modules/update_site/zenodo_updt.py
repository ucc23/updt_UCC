import csv
import os

import pandas as pd


def create_csv_UCC(zenodo_file: str, df_UCC: pd.DataFrame) -> None:
    """
    Generates a CSV file containing a reduced Unified Cluster Catalog
    (UCC) dataset, which can be stored in the Zenodo repository.
    """
    # Re-order columns
    df = df_UCC[
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
        zenodo_file,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def create_membs_UCC(
    logging, root_UCC_path: str, members_folder: str, zenodo_members_file: str
) -> None:
    """
    Generates a parquet file containing estimated members from the
    Unified Cluster Catalog (UCC) dataset, formatted for storage in the Zenodo
    repository.
    """
    # Initialize list for storing temporary DataFrames
    tmp = []
    for quad in ("1", "2", "3", "4"):
        for lat in ("P", "N"):
            logging.info(f"Processing Q{quad}{lat}")
            path = root_UCC_path + "/Q" + quad + lat + f"/{members_folder}/"
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
    df_comb.to_parquet(zenodo_members_file, index=False)


def updt_readme(UCC_folder, last_version, zenodo_readme) -> None:
    """Update version number in README file uploaded to Zenodo"""

    # Load the main file
    with open(UCC_folder + "README.txt", "r") as f:
        dataf = f.read()
        # Update the version number. The [:-2] at the end removes the hour from the
        # version's name
        version = last_version.split("_cat_")[1].split(".csv")[0][:-2]
        dataf = dataf.replace("XXXXXX", version)

    # Store updated file in the appropriate folder
    with open(zenodo_readme, "w") as f:
        f.write(dataf)
