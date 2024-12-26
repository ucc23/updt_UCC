import datetime
import logging
from os.path import join
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


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

    logging.info("\n------------------------------")
    logging.info(str(datetime.datetime.now()) + "\n")

    return logging


def radec2lonlat(ra, dec) -> tuple:
    gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    lb = gc.transform_to("galactic")
    # return np.array(lb.l), np.array(lb.b)
    return lb.l.value, lb.b.value


def check_centers(xy_c_m, vpd_c_m, plx_c_m, xy_c_n, vpd_c_n, plx_c_n, rad_dup=5):
    """
    xy_c_m, vpd_c_m, plx_c_m: Centers from estimated members
    xy_c_n, vpd_c_n, plx_c_n: Centers from the literature
    """
    bad_center_xy, bad_center_pm, bad_center_plx = "n", "n", "n"

    # Max distance in arcmin, 'rad_dup' arcmin maximum
    d_arcmin = np.sqrt((xy_c_m[0] - xy_c_n[0]) ** 2 + (xy_c_m[1] - xy_c_n[1]) ** 2) * 60
    if d_arcmin > rad_dup:
        bad_center_xy = "y"

    # Relative difference
    if not np.isnan(vpd_c_m[0]):
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
    if not np.isnan(plx_c_m):
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

    return bad_center


def date_order_DBs(DB: str, DB_i: str) -> tuple:
    """
    Orders two semicolon-separated strings of database entries by the year extracted
    from each entry.

    Args:
        DB (str): A semicolon-separated string where each entry contains a year in
        the format "_YYYY".
        DB_i (str): A semicolon-separated string with integers associated to `DB`.

    Returns:
        tuple: A tuple containing two semicolon-separated strings (`DB`, `DB_i`)
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
    DB = ";".join(np.array(all_dbs)[idx].tolist())
    DB_i = ";".join(np.array(all_dbs_i)[idx].tolist())
    return DB, DB_i
