import csv
import datetime
import logging
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from modules.variables import data_folder, temp_folder


def logger():
    """
    Sets up a logger that writes log messages to a file named with the current date
    and also outputs to the console.

    Returns:
        logging.Logger: Configured logger instance.
    """
    mypath = Path().absolute()

    # Name of log file using the date
    x = datetime.date.today()
    out_file = data_folder + "logs/" + str(x).replace("-", "_") + ".log"

    # Set up logging module
    level = logging.INFO
    frmt = "%(message)s"
    handlers = [
        logging.FileHandler(join(mypath, out_file), mode="a"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=level, format=frmt, handlers=handlers)

    # logging.info("\n------------------------------")
    logging.info(str(datetime.datetime.now()) + "\n")

    return logging


# def get_last_version_UCC(UCC_folder: str) -> str:
#     """Path to the latest version of the UCC catalogue"""

#     pattern = re.compile(r"UCC_cat_\d{8}\.csv")
#     ucc_file = [f for f in os.listdir(UCC_folder) if pattern.fullmatch(f)]

#     if len(ucc_file) == 0:
#         raise ValueError(f"UCC file not found in {UCC_folder}")
#     elif len(ucc_file) > 1:
#         raise ValueError(f"More than one UCC file found in {UCC_folder}")

#     last_version = ucc_file[0].split("_")[-1].split(".")[0]

#     return last_version


def radec2lonlat(
    ra: float | list | np.ndarray, dec: float | list | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Converts equatorial coordinates (RA, Dec) to galactic coordinates (lon, lat).

    Parameters
    ----------
    ra : float or list
        Right ascension in degrees.
    dec : float or list
        Declination in degrees.

    Returns
    -------
    tuple
        A tuple containing the galactic longitude and latitude in degrees.
    """
    gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)  # pyright: ignore
    lb = gc.transform_to("galactic")
    return lb.l.value, lb.b.value  # pyright: ignore


def round_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ """
    # Detect available columns to round
    f_id = ""
    if "GLON_m" in df.keys():
        f_id = "_m"
    df = df.round(
        {
            "RA_ICRS" + f_id: 5,
            "DE_ICRS" + f_id: 5,
            "GLON" + f_id: 5,
            "GLAT" + f_id: 5,
            "Plx" + f_id: 4,
            "pmRA" + f_id: 4,
            "pmDE" + f_id: 4,
        }
    )
    return df


def diff_between_dfs(
    logging,
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    # cols_exclude=None,
) -> None:
    """
    Order by (lon, lat) and change NaN as "nan".

    Compare two DataFrames, find non-matching rows while preserving order, and
    output these rows in two files.

    Args:
        df_old (pd.DataFrame): First DataFrame to compare.
        df_new (pd.DataFrame): Second DataFrame to compare.
        cols_exclude (list | None): List of columns to exclude from the diff
    """
    # Convert DataFrames to lists of tuples (rows) for comparison
    rows1 = [[str(_) for _ in row] for row in df_old.values]
    rows2 = [[str(_) for _ in row] for row in df_new.values]

    # Convert lists to sets for quick comparison
    set1, set2 = set(map(tuple, rows1)), set(map(tuple, rows2))

    # Get non-matching rows in original order
    non_matching1 = [row for row in rows1 if tuple(row) not in set2]
    non_matching2 = [row for row in rows2 if tuple(row) not in set1]

    if len(non_matching1) == 0 and len(non_matching2) == 0:
        logging.info("\nNo differences found\n")
        return

    if len(non_matching1) > 0:
        # Write intertwined lines to the output file
        with open(temp_folder + "UCC_diff_old.csv", "w", newline="") as out:
            writer = csv.writer(out)
            for row in non_matching1:
                writer.writerow(row)
    if len(non_matching2) > 0:
        with open(temp_folder + "UCC_diff_new.csv", "w", newline="") as out:
            writer = csv.writer(out)
            for row in non_matching2:
                writer.writerow(row)

    logging.info("\nFiles 'UCC_diff_xxx.csv' saved\n")


def save_df_UCC(logging, df: pd.DataFrame, file_path: str, order_col: str) -> None:
    """ """
    df = round_columns(df)

    # Order by 'order_col'
    df = df.sort_values(by=order_col).reset_index(drop=True)
    # Save UCC to CSV file
    df.to_csv(
        file_path,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    logging.info(f"UCC file (N={len(df)}): '{file_path}'")
