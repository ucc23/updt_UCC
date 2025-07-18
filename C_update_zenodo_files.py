import csv
import os

import pandas as pd
from fastparquet import ParquetFile
from scipy.spatial.distance import cdist

from modules.HARDCODED import UCC_folder, UCC_members_file
from modules.utils import get_last_version_UCC, logger


def main():
    """Steps:

    1. Read current UCC CSV file
    2. Create a CSV file with the UCC data
    3. Create a parquet file with the members data
    4. Create a README file with the version number and number of clusters and members

    All files are stored in the temporary folder.
    """
    logging = logger()

    # # Create temp folder if it does not exist
    # temp_zenodo_path = temp_fold + UCC_folder
    # if not os.path.exists(temp_zenodo_path):
    #     os.makedirs(temp_zenodo_path)

    # Get the latest version of the UCC catalogue
    UCC_last_version = get_last_version_UCC(UCC_folder)

    # Read current UCC csv file
    ucc_file_path = UCC_folder + UCC_last_version
    df_UCC = pd.read_csv(ucc_file_path)
    N_clusters = len(df_UCC)

    N_members, df_members = update_membs_UCC(logging)
    logging.info(f"Zenodo '{UCC_members_file}' file updated ({N_members} stars)")

    if df_members is not None:
        df_UCC = find_shared_members(df_UCC, df_members)

    create_csv_UCC(df_UCC)
    logging.info("\nZenodo 'UCC_cat.csv' file generated")

    updt_readme(UCC_folder, UCC_last_version, N_clusters, N_members)
    logging.info("Zenodo README file generated")

    logging.info(f"\nThe three files are stored in '{UCC_folder}'")

    # zenodo_upload()


def update_membs_UCC(logging) -> tuple[int, pd.DataFrame | None]:
    """
    Update the parquet file containing estimated members from the
    Unified Cluster Catalog (UCC) dataset, formatted for storage in the Zenodo
    repository.
    """
    zenodo_members_file = UCC_folder + UCC_members_file
    zenodo_members_file_temp = zenodo_members_file + ".temp"

    # Check if the temp file exists
    if not os.path.exists(zenodo_members_file_temp):
        logging.info("No temporary members file found")
        pf = ParquetFile(zenodo_members_file)
        N_members = pf.count()
        return N_members, None

    logging.info("Reading member files...")
    df_members = pd.read_parquet(zenodo_members_file)
    df_temp = pd.read_parquet(zenodo_members_file_temp)

    # Set index to 'name' for both DataFrames
    df_m_indexed = df_members.set_index("name")
    df_t_indexed = df_temp.set_index("name")

    # Update df_members with df_temp where names match
    df_m_indexed.update(df_t_indexed)

    # Concatenate rows from df_temp that are not in df_members
    missing = df_t_indexed[~df_t_indexed.index.isin(df_m_indexed.index)]
    df_updated = pd.concat([df_m_indexed, missing])

    # Reset index
    df_updated = df_updated.reset_index()

    # Update file
    df_updated.to_parquet(zenodo_members_file, index=False)
    logging.info(f"Members file '{zenodo_members_file}' updated")

    # Remove temporary file
    os.remove(zenodo_members_file_temp)
    logging.info(f"Temporary file '{zenodo_members_file_temp}' removed")

    N_members = len(df_updated)

    return N_members, df_updated


def find_shared_members(df_UCC, df_members):
    """ """
    intersection_map = find_intersections(df_UCC)

    # Group by 'fname'
    grouped = df_members.groupby("name")["Source"].apply(set)

    results = []
    # Compute shared elements and percentages
    for fname, sources in grouped.items():
        fnames_process = [_ for _ in intersection_map[fname].split(",")]

        if fnames_process[0] == "nan":
            continue

        shared_info = []
        percentage_info = []

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
                    "groups_shared": ",".join(shared_info),
                    "groups_percentage": ",".join(percentage_info),
                }
            )
        else:
            results.append(
                {"fname": fname, "groups_shared": "nan", "groups_percentage": "nan"}
            )

    # Convert to DataFrame
    result_df = pd.DataFrame(results)

    # Update UDD with this data
    df_UCC = df_UCC.merge(
        result_df, left_on="fname", right_on="fname", how="left", suffixes=("", "_y")
    )

    return df_UCC


def find_intersections(df):
    # Convert to NumPy arrays for fast computation
    coords = df[["GLON_m", "GLAT_m"]].to_numpy()
    names = df["fname"].to_numpy()

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


def create_csv_UCC(df_UCC: pd.DataFrame) -> None:
    """
    Generates a CSV file containing a reduced Unified Cluster Catalog
    (UCC) dataset, which can be stored in the Zenodo repository.
    """

    # Add 'fname' column
    df = df_UCC.copy()
    df["fname"] = df["fnames"].str.split(";").str[0]

    # Re-order columns
    df = df[
        [
            "fname",
            "GLON_m",
            "GLAT_m",
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
            "C3",
        ]
    ]

    print("TODO: fix this file and the README that describes it")
    # store a single set of coordinates, pms and plx
    # order columns properly
    # add info on clusters that share members

    zenodo_file = UCC_folder + "UCC_cat.csv"
    # Store to csv file
    df.to_csv(
        zenodo_file,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def updt_readme(
    UCC_folder: str,
    last_version: str,
    N_clusters: int,
    N_members: int,
) -> None:
    """Update version number in README file uploaded to Zenodo"""

    # Load the main file
    with open(UCC_folder + "README.static.txt", "r") as f:
        dataf = f.read()
        # Update the version number. The [:-2] at the end removes the hour from the
        # version's name
        version = last_version.split("_cat_")[1].split(".csv")[0][:-2]
        dataf = dataf.replace("XXXXXX", version)

        # Update number of clusters and members
        dataf = dataf.replace("YYYYYY", str(N_clusters))
        dataf = dataf.replace("ZZZZZZ", str(N_members))

    zenodo_readme = UCC_folder + "README.txt"
    # Store updated file in the appropriate folder
    with open(zenodo_readme, "w") as f:
        f.write(dataf)


# def zenodo_upload():
#     """

#     zenodo-client:
#     A tool for automated uploading and version management of scientific data to Zenodo

#     https://github.com/cthoyt/zenodo-client
#     """
#     from zenodo_client import Creator, Metadata, ensure_zenodo

#     # Define the metadata that will be used on initial upload
#     data = Metadata(
#         title="Test Upload 3",
#         upload_type="dataset",
#         description="test description",
#         creators=[
#             Creator(
#                 name="Hoyt, Charles Tapley",
#                 affiliation="Harvard Medical School",
#                 orcid="0000-0003-4423-4370",
#             ),
#         ],
#     )
#     res = ensure_zenodo(
#         key="test3",  # this is a unique key you pick that will be used to store
#         # the numeric deposition ID on your local system's cache
#         data=data,
#         paths=[
#             "/Users/cthoyt/Desktop/test1.png",
#         ],
#         sandbox=True,  # remove this when you're ready to upload to real Zenodo
#     )
#     from pprint import pprint

#     pprint(res.json())


if __name__ == "__main__":
    main()
