import csv
import os

import pandas as pd
from HARDCODED import (
    UCC_folder,
    members_folder,
    root_UCC_path,
)
from modules import UCC_new_match, logger


def main():
    """ """
    logging = logger.main()
    logging.info("\nRunning 'J_zenodo_updt' script\n")

    # Read latest version of the UCC
    df_UCC, _ = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    zenodo_UCC(df_UCC, UCC_folder)
    logging.info("\nCompressed 'UCC_cat.csv.gz' file generated\n")

    print("Compressing members...")
    zenodo_membs(logging, UCC_folder, root_UCC_path, members_folder)
    logging.info("\nCompressed 'UCC_members.parquet.gz' file generated")


def zenodo_UCC(df_UCC, UCC_folder):
    """
    Generate the compressed file with the reduced UCC DB that is stored in
    the Zenodo repository
    """
    # Only keep certain columns
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
        "N_membs",
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
            "plx",
            "pmRA",
            "pmDE",
            "UCC_ID",
            "N_50",
            "r_50",
            "RA_ICRS_m",
            "DE_ICRS_m",
            "plx_m",
            "pmRA_m",
            "pmDE_m",
            "Rv_m",
            "N_Rv",
            "C1",
            "C2",
            "C3",
        ]
    ]

    # Store to compressed file
    df.to_csv(
        UCC_folder + "UCC_cat.csv.gz",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        compression="gzip",
    )


def zenodo_membs(logging, UCC_folder, root_UCC_path, members_folder):
    """
    Generate the compressed file with the estimated members that is stored in
    the Zenodo repository
    """

    # Now update the members file
    tmp = []
    for quad in ("1", "2", "3", "4"):
        for lat in ("P", "N"):
            logging.info(f"Processing Q{quad}{lat}")
            path = root_UCC_path + "Q" + quad + lat + f"/{members_folder}/"
            for file in os.listdir(path):
                df = pd.read_parquet(path + file)
                fname = file.replace(".parquet", "")
                name_col = [fname for _ in range(len(df))]
                df.insert(loc=0, column="name", value=name_col)
                tmp.append(df)

    df_comb = pd.concat(tmp, ignore_index=True)
    df_comb.to_parquet(
        UCC_folder + "UCC_members.parquet.gz", index=False, compression="gzip"
    )


if __name__ == "__main__":
    main()
