import csv
import os

import pandas as pd

from modules.HARDCODED import (
    UCC_folder,
    members_folder,
    temp_fold,
)
from modules.utils import get_last_version_UCC, logger


def main():
    logging = logger()

    # Get the latest version of the UCC catalogue
    UCC_last_version = get_last_version_UCC(UCC_folder)

    temp_zenodo_path = temp_fold + UCC_folder
    if not os.path.exists(temp_zenodo_path):
        os.makedirs(temp_zenodo_path)

    # Read current UCC csv file
    ucc_file_path = UCC_folder + UCC_last_version
    df_UCC = pd.read_csv(ucc_file_path)
    N_clusters = len(df_UCC)

    upld_zenodo_file = temp_zenodo_path + "UCC_cat.csv"
    create_csv_UCC(upld_zenodo_file, df_UCC)
    logging.info("\nZenodo 'UCC_cat.csv' file generated")

    zenodo_members_file = temp_zenodo_path + "UCC_members.parquet"
    N_members = create_membs_UCC(logging, members_folder, zenodo_members_file)
    logging.info("Zenodo 'UCC_members.parquet' file generated")

    zenodo_readme = temp_zenodo_path + "README.txt"
    updt_readme(UCC_folder, UCC_last_version, N_clusters, N_members, zenodo_readme)
    logging.info("Zenodo README file generated")

    logging.info(f"\nThe three files are stored in '{temp_zenodo_path}'")

    # zenodo_upload()


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


def create_membs_UCC(logging, members_folder: str, zenodo_members_file: str) -> int:
    """
    Generates a parquet file containing estimated members from the
    Unified Cluster Catalog (UCC) dataset, formatted for storage in the Zenodo
    repository.
    """
    logging.info("Reading member files...")

    # This assumes that the 'QXY' folders are located one directory above the
    # current script <-- !! IMPORTANT !!
    root_UCC_dir = ".."

    # Initialize list for storing temporary DataFrames
    tmp = []
    for quad in ("1", "2", "3", "4"):
        for lat in ("P", "N"):
            logging.info(f"Processing Q{quad}{lat}")
            path = root_UCC_dir + "/Q" + quad + lat + f"/{members_folder}/"
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

    N_members = len(df_comb)

    return N_members


def updt_readme(
    UCC_folder: str,
    last_version: str,
    N_clusters: int,
    N_members: int,
    zenodo_readme: str,
) -> None:
    """Update version number in README file uploaded to Zenodo"""

    # Load the main file
    with open(UCC_folder + "README.txt", "r") as f:
        dataf = f.read()
        # Update the version number. The [:-2] at the end removes the hour from the
        # version's name
        version = last_version.split("_cat_")[1].split(".csv")[0][:-2]
        dataf = dataf.replace("XXXXXX", version)

        # Update number of clusters and members
        dataf = dataf.replace("YYYYYY", str(N_clusters))
        dataf = dataf.replace("ZZZZZZ", str(N_members))

    # Store updated file in the appropriate folder
    with open(zenodo_readme, "w") as f:
        f.write(dataf)


def zenodo_upload():
    """

    zenodo-client:
    A tool for automated uploading and version management of scientific data to Zenodo

    https://github.com/cthoyt/zenodo-client
    """
    from zenodo_client import Creator, Metadata, ensure_zenodo

    # Define the metadata that will be used on initial upload
    data = Metadata(
        title="Test Upload 3",
        upload_type="dataset",
        description="test description",
        creators=[
            Creator(
                name="Hoyt, Charles Tapley",
                affiliation="Harvard Medical School",
                orcid="0000-0003-4423-4370",
            ),
        ],
    )
    res = ensure_zenodo(
        key="test3",  # this is a unique key you pick that will be used to store
        # the numeric deposition ID on your local system's cache
        data=data,
        paths=[
            "/Users/cthoyt/Desktop/test1.png",
        ],
        sandbox=True,  # remove this when you're ready to upload to real Zenodo
    )
    from pprint import pprint

    pprint(res.json())


if __name__ == "__main__":
    main()
