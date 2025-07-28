import csv
import datetime
import logging
import os
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from modules.HARDCODED import md_folder, temp_fold


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
    out_file = "logs/" + str(x).replace("-", "_") + ".log"

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


def get_last_version_UCC(UCC_folder: str) -> str:
    """Path to the latest version of the UCC catalogue"""
    last_version = None
    for file in os.listdir(UCC_folder):
        if file.endswith("csv"):
            last_version = file
            break
    if last_version is None:
        raise ValueError(f"UCC file not found in {UCC_folder}")

    return last_version


def radec2lonlat(
    ra: float | list, dec: float | list
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


def check_centers(
    xy_c_m: tuple[float, float],
    vpd_c_m: tuple[float, float],
    plx_c_m: float,
    xy_c_n: tuple[float, float],
    vpd_c_n: tuple[float, float],
    plx_c_n: float,
    rad_dup: float = 5,
) -> tuple[str, float, float, float, float]:
    """
    Compares the centers of a cluster estimated from members with those from the
    literature.

    Parameters
    ----------
    xy_c_m : tuple
        Center coordinates (lon, lat) or (ra, dec) from estimated members.
    vpd_c_m : tuple
        Center proper motion (pmRA, pmDE) from estimated members.
    plx_c_m : float
        Center parallax from estimated members.
    xy_c_n : tuple
        Center coordinates (lon, lat) or (ra, dec) from the literature.
    vpd_c_n : tuple
        Center proper motion (pmRA, pmDE) from the literature.
    plx_c_n : float
        Center parallax from the literature.
    rad_dup : float, optional
        Maximum allowed distance between centers in arcmin. Default is 5.

    Returns
    -------
    str
        A string indicating the quality of the center comparison:
        - "nnn": Centers are in agreement.
        - "y": Indicates a significant difference in xy, pm, or plx,
        with each 'y' corresponding to a specific discrepancy.
    floats
        The distances estimated in arcmin, and the percentage differences
        in proper motion and parallax.
    """

    bad_center_xy, bad_center_pm, bad_center_plx = "n", "n", "n"

    # Max distance in arcmin, 'rad_dup' arcmin maximum
    d_arcmin = np.sqrt((xy_c_m[0] - xy_c_n[0]) ** 2 + (xy_c_m[1] - xy_c_n[1]) ** 2) * 60
    if d_arcmin > rad_dup:
        bad_center_xy = "y"

    # Relative difference
    pmra_p, pmde_p = np.nan, np.nan
    if not np.isnan(vpd_c_n[0]):
        pm_max = []
        for vpd_c_i in abs(np.array(vpd_c_m)):
            if vpd_c_i > 5:
                pm_max.append(10)
            elif vpd_c_i > 1:
                pm_max.append(15)
            elif vpd_c_i > 0.1:
                pm_max.append(20)
            elif vpd_c_i > 0.01:
                pm_max.append(25)
            else:
                pm_max.append(50)
        pmra_p = 100 * abs((vpd_c_m[0] - vpd_c_n[0]) / (vpd_c_m[0] + 0.001))
        pmde_p = 100 * abs((vpd_c_m[1] - vpd_c_n[1]) / (vpd_c_m[1] + 0.001))
        if pmra_p > pm_max[0] or pmde_p > pm_max[1]:
            bad_center_pm = "y"

    # Relative difference
    plx_p = np.nan
    if not np.isnan(plx_c_n):
        if plx_c_m > 0.2:
            plx_max = 25
        elif plx_c_m > 0.1:
            plx_max = 30
        elif plx_c_m > 0.05:
            plx_max = 35
        elif plx_c_m > 0.01:
            plx_max = 50
        else:
            plx_max = 70
        plx_p = 100 * abs(plx_c_m - plx_c_n) / (plx_c_m + 0.001)
        if abs(plx_p) > plx_max:
            bad_center_plx = "y"

    bad_center = bad_center_xy + bad_center_pm + bad_center_plx

    return bad_center, d_arcmin, pmra_p, pmde_p, plx_p


def rename_standard(name: str) -> str:
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

    # Removes duplicates such as "NGC_2516" and "NGC 2516"
    name = name.replace("_", " ")

    return name


def date_order_DBs(DB: str, DB_i: str) -> tuple[str, str]:
    """
    Orders two semicolon-separated strings of database entries by the year extracted
    from each entry.

    Parameters
    ----------
    DB : str
        A semicolon-separated string where each entry contains a year in
        the format "_YYYY".
    DB_i : str
        A semicolon-separated string with integers associated to `DB`.

    Returns
    -------
    tuple
        A tuple containing two semicolon-separated strings (`DB`, `DB_i`)
        ordered by year.
    """
    # Split lists
    all_dbs = DB.split(";")
    all_dbs_i = DB_i.split(";")

    # Extract years from DBs
    all_years = []
    for db in all_dbs:
        year = db.split("_")[0][-4:]
        all_years.append(year)

    # Sort and re-generate strings
    idx = np.argsort(all_years)
    DB = ";".join(list(np.array(all_dbs)[idx]))
    DB_i = ";".join(list(np.array(all_dbs_i)[idx]))
    return DB, DB_i


def file_checker(
    logging,
    N_UCC: int,
    root_UCC_fold: str,
) -> None:
    """Check the number and types of files in directories for consistency.

    Parameters:
    - logging: Logger instance for recording messages.
    - UCC_new: DataFrame containing the new UCC data.

    Returns:
    - None
    """
    logging.info(f"\nChecking number of files against N_UCC={N_UCC}")
    # if datafiles_only:
    #     folders_log = "parquet / extra"
    #     folders = ("datafiles",)
    #     logging.info("    parquet  extra")
    # else:
    folders_log = "webp / aladin / extra"
    folders = ("plots",)
    logging.info("    webp  aladin  extra")

    flag_error = False
    NT_webp, NT_webp_aladin, NT_extra = 0, 0, 0
    for qnum in range(1, 5):
        for lat in ("P", "N"):
            N_webp, N_webp_aladin, N_extra = 0, 0, 0
            for ffolder in folders:
                qfold = root_UCC_fold + "Q" + str(qnum) + lat + f"/{ffolder}/"
                # Read all files in Q folder
                for file in os.listdir(qfold):
                    if "HUNT23" in file or "CANTAT20" in file:
                        pass
                    elif "aladin" in file:
                        N_webp_aladin += 1
                        NT_webp_aladin += 1
                    # elif "parquet" in file:
                    #     N_parquet += 1
                    #     NT_parquet += 1
                    elif "webp" in file:
                        N_webp += 1
                        NT_webp += 1
                    else:
                        N_extra += 1
                        NT_extra += 1

            # if datafiles_only:
            #     mark = "V" if (N_extra == 0) else "X"
            #     logging.info(
            #         f"{str(qnum) + lat}:   {N_parquet}     {N_extra} <-- {mark}"
            #     )
            # else:
            mark = "V" if (N_webp == N_webp_aladin) else "X"
            logging.info(
                f"{str(qnum) + lat}:   {N_webp}  {N_webp_aladin}    {N_extra} <-- {mark}"
            )
            if N_extra > 0:
                mark = "X"

            if mark == "X":
                flag_error = True

    # if datafiles_only:
    #     logging.info(f"Total {folders_log}: {NT_parquet} / {NT_extra}")
    #     if not (NT_parquet == N_UCC) or NT_extra > 0:
    #         flag_error = True
    # else:
    logging.info(f"Total {folders_log}: {NT_webp} / {NT_webp_aladin} / {NT_extra}")
    if not (NT_webp == NT_webp_aladin == N_UCC) or NT_extra > 0:
        flag_error = True

    # Check .md files
    clusters_md_fold = root_UCC_fold + md_folder
    NT_md = len(os.listdir(clusters_md_fold))
    NT_extra = NT_md - N_UCC
    mark = "V" if (NT_extra == 0) else "X"
    logging.info("\nN_UCC   md      extra")
    logging.info(f"{N_UCC}   {NT_md}   {NT_extra} <-- {mark}")
    if mark == "X":
        flag_error = True

    if flag_error:
        raise ValueError("The file check was unsuccessful")


def list_duplicates(seq: list) -> list:
    """
    Identifies duplicate elements in a list.

    Args:
        seq: The input list.

    Returns:
        A list of duplicate elements.
    """
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to 'seen' and all other to 'seen_twice'
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)


def diff_between_dfs(
    logging,
    df_old: pd.DataFrame,
    df_new_in: pd.DataFrame,
    cols_exclude=None,
) -> pd.DataFrame:
    """
    Order by (lon, lat) and change NaN as "nan".

    Compare two DataFrames, find non-matching rows while preserving order, and
    output these rows in two files.

    Args:
        df_old (pd.DataFrame): First DataFrame to compare.
        df_new (pd.DataFrame): Second DataFrame to compare.
        cols_exclude (list | None): List of columns to exclude from the diff
    """
    df_new = df_new_in.copy()
    # Order by (lon, lat)
    df_new = df_new.sort_values(["GLON", "GLAT"])
    df_new = df_new.reset_index(drop=True)

    if cols_exclude is not None:
        logging.info(f"\n{cols_exclude} columns excluded")
        for col in cols_exclude:
            if col in df_old.keys():
                df_old = df_old.drop(columns=(col))
            if col in df_new.keys():
                df_new = df_new.drop(columns=(col))
    else:
        logging.info("\nNo columns excluded")
    df1 = df_old
    df2 = df_new

    # Convert DataFrames to lists of tuples (rows) for comparison
    rows1 = [[str(_) for _ in row] for row in df1.values]
    rows2 = [[str(_) for _ in row] for row in df2.values]

    # Convert lists to sets for quick comparison
    set1, set2 = set(map(tuple, rows1)), set(map(tuple, rows2))

    # Get non-matching rows in original order
    non_matching1 = [row for row in rows1 if tuple(row) not in set2]
    non_matching2 = [row for row in rows2 if tuple(row) not in set1]

    if len(non_matching1) == 0 and len(non_matching2) == 0:
        logging.info("No differences found\n")
        return df_new

    if len(non_matching1) > 0:
        # Write intertwined lines to the output file
        with open(temp_fold + "UCC_diff_old.csv", "w", newline="") as out:
            writer = csv.writer(out)
            for row in non_matching1:
                writer.writerow(row)
    if len(non_matching2) > 0:
        with open(temp_fold + "UCC_diff_new.csv", "w", newline="") as out:
            writer = csv.writer(out)
            for row in non_matching2:
                writer.writerow(row)

    logging.info("Files 'UCC_diff_xxx.csv' saved\n")
    return df_new


def save_df_UCC(logging, df_UCC: pd.DataFrame, file_path: str) -> None:
    """ """

    # Order by (lon, lat) first
    df_UCC = df_UCC.sort_values(["GLON", "GLAT"])
    df_UCC = df_UCC.reset_index(drop=True)
    # Save UCC to CSV file
    df_UCC.to_csv(
        file_path,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    logging.info(f"UCC databse stored: {file_path} (N={len(df_UCC)})")
