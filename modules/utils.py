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


def get_fnames(names_all, sep: str = ","):
    """ """
    fnames = []
    for names in names_all:
        names_l = []
        names_s = str(names).split(sep)
        for name in names_s:
            name = name.strip()
            name = rename_standard(name)
            names_l.append(normalize_name(name))
        fnames.append(names_l)
    return fnames


def rename_standard(all_names: str, sep_in: str = ",", sep_out: str = ";") -> str:
    """
    Standardize the naming of these clusters

    FSR XXX w leading zeros
    FSR XXX w/o leading zeros
    FSR_XXX w leading zeros
    FSR_XXX w/o leading zeros

    --> FSR_XXXX (w leading zeroes)

    ESO XXX-YY w leading zeros
    ESO XXX-YY w/o leading zeros
    ESO_XXX_YY w leading zeros
    ESO_XXX_YY w/o leading zeros
    ESO_XXX-YY w leading zeros
    ESO_XXX-YY w/o leading zeros
    ESOXXX_YY w leading zeros (LOKTIN17)

    --> ESO_XXX_YY (w leading zeroes)

    Parameters
    ----------
    name : str
        Name of the cluster.

    Returns
    -------
    str
        Standardized name of the cluster.
    """
    # For each comma separated name for this OC in the new DB
    oc_names = all_names.split(sep_in)

    new_names_rename = []
    for name in oc_names:
        name = name.strip()

        if name.startswith("FSR"):
            if " " in name or "_" in name:
                if "_" in name:
                    n2 = name.split("_")[1]
                else:
                    n2 = name.split(" ")[1]
                n2 = int(n2)
                if n2 < 10:
                    n2 = "000" + str(n2)
                elif n2 < 100:
                    n2 = "00" + str(n2)
                elif n2 < 1000:
                    n2 = "0" + str(n2)
                else:
                    n2 = str(n2)
                name = "FSR_" + n2

        if name.startswith("ESO"):
            if name[:4] not in ("ESO_", "ESO "):
                # E.g.: LOKTIN17, BOSSINI19
                name = "ESO_" + name[3:]

            if " " in name[4:]:
                n1, n2 = name[4:].split(" ")
            elif "_" in name[4:]:
                n1, n2 = name[4:].split("_")
            elif "-" in name[4:]:
                n1, n2 = name[4:].split("-")
            else:
                # This assumes that all ESo clusters are names as: 'ESO XXX YY'
                n1, n2 = name[4 : 4 + 3], name[4 + 3 :]

            n1 = int(n1)
            if n1 < 10:
                n1 = "00" + str(n1)
            elif n1 < 100:
                n1 = "0" + str(n1)
            else:
                n1 = str(n1)
            n2 = int(n2)
            if n2 < 10:
                n2 = "0" + str(n2)
            else:
                n2 = str(n2)
            name = "ESO_" + n1 + "_" + n2

        if "UBC" in name and "UBC " not in name and "UBC_" not in name:
            name = name.replace("UBC", "UBC ")
        if "UBC_" in name:
            name = name.replace("UBC_", "UBC ")

        if "UFMG" in name and "UFMG " not in name and "UFMG_" not in name:
            name = name.replace("UFMG", "UFMG ")

        if (
            "LISC" in name
            and "LISC " not in name
            and "LISC_" not in name
            and "LISC-" not in name
        ):
            name = name.replace("LISC", "LISC ")

        if "OC-" in name:
            name = name.replace("OC-", "OC ")

        # Use spaces not underscores
        name = name.replace("_", " ")

        new_names_rename.append(name)
    oc_names = sep_out.join(new_names_rename)

    return oc_names


def normalize_name(name: str) -> str:
    """
    Removes special characters from a cluster name and converts it to lowercase.

    Parameters
    ----------
    name : str
        The cluster name.

    Returns
    -------
    str
        The cluster name with special characters removed and converted to lowercase.
    """
    # We replace '+' with 'p' to avoid duplicating names for clusters
    # like 'Juchert J0644.8-0925' and 'Juchert_J0644.8+0925'
    name = (
        name.lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("'", "")
        .replace("+", "p")
    )
    return name


def plx_to_pc(plx, PZPO=-0.02, min_plx=0.035, max_plx=200):
    """
    Ding et al. (2025), Fig 8 shows several PZPO values
    https://ui.adsabs.harvard.edu/abs/2025AJ....169..211D/abstract

    We use -0.02 as a reasonable value here.
    """
    plx = np.array(plx) * 1.0

    # "the zero-point returned by the code should be subtracted from the parallax value"
    # https://gitlab.com/icc-ub/public/gaiadr3_zeropoint
    plx -= PZPO

    # Clip to reasonable values
    plx = np.clip(plx, min_plx, max_plx)
    # Convert to pc
    d_pc = 1000 / plx

    return d_pc


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
            "X_GC": 4,
            "Y_GC": 4,
            "Z_GC": 4,
            "R_GC": 4,
        }
    )
    return df


def diff_between_dfs(
    logging,
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    order_col: str = "fname",
    add_context=True,
) -> bool:
    """
    Order by (lon, lat) and change NaN as "nan".

    Compare two DataFrames, find non-matching rows while preserving order, and
    output these rows in two files.

    Args:
        df_old (pd.DataFrame): First DataFrame to compare.
        df_new (pd.DataFrame): Second DataFrame to compare.
        cols_exclude (list | None): List of columns to exclude from the diff
    """

    def get_clean_context_rows(source_rows, diff_indices):
        """
        Returns a list of rows containing the differences and their immediate
        neighbors (above/below), sorted by original order, without duplicates.
        """
        indices_to_keep = set()

        for i in diff_indices:
            # Add the difference row
            indices_to_keep.add(i)
            # Add the row above (if exists)
            if i > 0:
                indices_to_keep.add(i - 1)
            # Add the row below (if exists)
            if i < len(source_rows) - 1:
                indices_to_keep.add(i + 1)

        # Sort indices to maintain the original file order
        sorted_indices = sorted(indices_to_keep)

        return [source_rows[i] for i in sorted_indices]

    df_old = df_old.copy().sort_values(by=order_col).reset_index(drop=True)
    df_new = df_new.copy().sort_values(by=order_col).reset_index(drop=True)
    df_new = round_columns(df_new)

    # Convert DataFrames to lists of tuples (rows) for comparison
    rows1 = [[str(_) for _ in row] for row in df_old.values]
    rows2 = [[str(_) for _ in row] for row in df_new.values]

    # Convert lists to sets for quick comparison
    set1, set2 = set(map(tuple, rows1)), set(map(tuple, rows2))

    if add_context:
        # Get indices of non-matching rows
        diff_indices1 = [i for i, row in enumerate(rows1) if tuple(row) not in set2]
        diff_indices2 = [i for i, row in enumerate(rows2) if tuple(row) not in set1]
        non_matching1 = []
        if len(diff_indices1) > 0:
            non_matching1 = get_clean_context_rows(rows1, diff_indices1)
        non_matching2 = []
        if len(diff_indices2) > 0:
            non_matching2 = get_clean_context_rows(rows2, diff_indices2)
    else:
        # Get non-matching rows in original order
        non_matching1 = [row for row in rows1 if tuple(row) not in set2]
        non_matching2 = [row for row in rows2 if tuple(row) not in set1]

    if len(non_matching1) == 0 and len(non_matching2) == 0:
        logging.info("No differences found\n")
        # Return boolean indicating if any differences where found
        return False

    # Write to files
    if len(non_matching1) > 0:
        with open(temp_folder + "UCC_diff_old.csv", "w", newline="") as out:
            writer = csv.writer(out)
            for row in non_matching1:
                writer.writerow(row)
        logging.info("File 'UCC_diff_old.csv' saved")

    if len(non_matching2) > 0:
        with open(temp_folder + "UCC_diff_new.csv", "w", newline="") as out:
            writer = csv.writer(out)
            for row in non_matching2:
                writer.writerow(row)
        logging.info("File 'UCC_diff_new.csv' saved")

    logging.info("")

    return True


def final_fname_compare(logging, old_df, new_df, N_max=50):
    """ """
    fnames_old = [_.split(";") for _ in old_df["fnames"]]
    fnames_new = [_.split(";") for _ in new_df["fnames"]]
    fname0_old = [_[0] for _ in fnames_old]
    fname0_new = [_[0] for _ in fnames_new]

    # Build a lookup: each secondary name --> the full entry it appears in
    lookup_new = {}
    for full in fnames_new:
        for name in full:
            lookup_new[name] = full

    # Compute differences
    set_new0 = set(fname0_new)
    diffs_1 = []
    for name0 in fname0_old:
        if name0 not in set_new0:
            full = lookup_new.get(name0)
            if full:
                diffs_1.append(f"{name0} --> {';'.join(full)}")
    N_diff = len(diffs_1)
    if N_diff > 0:
        logging.info(f"Found {N_diff} fnames in old df that changed in new:\n")
        for _ in diffs_1[:N_max]:
            logging.info(_)
        if N_diff > N_max:
            logging.info(f"... and {N_diff - N_max} more differences")

    fnames_old_set = {x for sub in fnames_old for x in sub}
    fnames_new_set = {x for sub in fnames_new for x in sub}
    new_vs_old = list(set(fname0_new) - fnames_old_set)
    old_vs_new = list(set(fname0_old) - fnames_new_set)

    def dif_missig(sets_id):
        missing = new_vs_old
        if sets_id[0] == "old":
            missing = old_vs_new
        N_missing = len(missing)
        if N_missing > 0:
            logging.info(
                f"\nFound {N_missing} fnames in {sets_id[0]} df that are not in {sets_id[1]}:"
            )
            for fmiss in missing[:N_max]:
                logging.info(f"{fmiss}")
            if N_missing > N_max:
                logging.info(f"... and {N_missing - N_max} more differences")

    dif_missig(("new", "old"))
    dif_missig(("old", "new"))
    logging.info("")


def save_df_UCC(
    logging, df: pd.DataFrame, file_path: str, order_col: str, compression: str = ""
) -> None:
    """ """
    df = round_columns(df)

    if order_col == "fnames":
        fnames0 = [_.split(";")[0] for _ in df["fnames"]]
        idx = np.argsort(fnames0)
        df = df.reindex(idx)
    else:
        # Order by 'order_col'
        df = df.sort_values(by=order_col).reset_index(drop=True)

    # Save UCC to CSV file
    if compression == "gzip":
        df.to_csv(
            file_path,
            na_rep="nan",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            compression="gzip",
        )
    else:
        df.to_csv(
            file_path,
            na_rep="nan",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )
    logging.info(f"UCC file (N={len(df)}): '{file_path}'")
