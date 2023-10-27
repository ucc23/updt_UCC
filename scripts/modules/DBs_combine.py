import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from string import ascii_lowercase
from . import duplicate_probs


"""
This module contains helper functions to generate the extended
combined DB when a new DB is added.
"""


def get_fnames_new_DB(df_new, json_pars, sep) -> list:
    """
    Extract and standardize all names in the new catalogue
    """
    names_all = df_new[json_pars["names"]]
    new_DB_fnames = []
    for i, names in enumerate(names_all):
        names_l = []
        names_s = names.split(sep)
        for name in names_s:
            name = name.strip()
            name = rename_standard(name)
            names_l.append(rm_chars_from_name(name))
        new_DB_fnames.append(names_l)

    return new_DB_fnames


def rename_standard(name):
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
            # This is a LOKTIN17 ESO cluster
            name = "ESO_" + name[3:]

        if " " in name[4:]:
            n1, n2 = name[4:].split(" ")
        elif "_" in name[4:]:
            n1, n2 = name[4:].split("_")
        elif "" in name[4:]:
            n1, n2 = name[4:].split("-")

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


def rm_chars_from_name(name):
    """ """
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


def rm_name_dups(names_l):
    """
    Remove duplicates of the kind: Berkeley 102, Berkeley102,
    Berkeley_102; keeping only the name with the space
    """
    if "NGC 1976" in names_l:
        print(names_l)
    for i, n in enumerate(names_l):
        n2 = n.replace(" ", "")
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n
        n2 = n.replace(" ", "_")
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n

    if "NGC 1976" in names_l:
        print(";".join(list(dict.fromkeys(names_l))))

    return ";".join(list(dict.fromkeys(names_l)))


def get_matches_new_DB(df_UCC, new_DB_fnames):
    """
    Get cluster matches for the new DB being added to the combined DB
    """

    def match_fname(new_cl):
        for name_new in new_cl:
            for j, old_cl in enumerate(df_UCC["fnames"]):
                for name_old in old_cl.split(";"):
                    if name_new == name_old:
                        return j
        return None

    db_matches = []
    for i, new_cl in enumerate(new_DB_fnames):
        # Check if this new fname is already in the old DBs list of fnames
        db_matches.append(match_fname(new_cl))

    return db_matches


def radec2lonlat(ra, dec):
    gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    lb = gc.transform_to("galactic")
    lon, lat = lb.l.value, lb.b.value
    return np.round(lon, 4), np.round(lat, 4)


def assign_UCC_ids(glon, glat, ucc_ids_old):
    """
    Format: UCC GXXX.X+YY.Y
    """
    ll = trunc(np.array([glon, glat]).T)
    lon, lat = str(ll[0]), str(ll[1])

    if ll[0] < 10:
        lon = "00" + lon
    elif ll[0] < 100:
        lon = "0" + lon

    if ll[1] >= 10:
        lat = "+" + lat
    elif ll[1] < 10 and ll[1] > 0:
        lat = "+0" + lat
    elif ll[1] == 0:
        lat = "+0" + lat.replace("-", "")
    elif ll[1] < 0 and ll[1] >= -10:
        lat = "-0" + lat[1:]
    elif ll[1] < -10:
        pass

    ucc_id = "UCC G" + lon + lat

    i = 0
    while True:
        if i > 25:
            ucc_id += "ERROR"
            print("ERROR NAMING")
            break
        if ucc_id in ucc_ids_old:
            if i == 0:
                # Add a letter to the end
                ucc_id += ascii_lowercase[i]
            else:
                # Replace last letter
                ucc_id = ucc_id[:-1] + ascii_lowercase[i]
            i += 1
        else:
            break

    return ucc_id


def trunc(values, decs=1):
    return np.trunc(values * 10**decs) / (10**decs)


def QXY_fold(UCC_ID):
    """ """
    # UCC_ID = cl['UCC_ID']
    lonlat = UCC_ID.split("G")[1]
    lon = float(lonlat[:5])
    try:
        lat = float(lonlat[5:])
    except ValueError:
        lat = float(lonlat[5:-1])

    Qfold = "Q"
    if lon >= 0 and lon < 90:
        Qfold += "1"
    elif lon >= 90 and lon < 180:
        Qfold += "2"
    elif lon >= 180 and lon < 270:
        Qfold += "3"
    elif lon >= 270 and lon < 3600:
        Qfold += "4"
    if lat >= 0:
        Qfold += "P"
    else:
        Qfold += "N"

    return Qfold


def dups_identify(df, prob_cut=0.5, Nmax=3):
    """
    Assign a 'duplicate probability' for each cluster in the UCC, based on the
    literature data
    """
    x, y = df["GLON"], df["GLAT"]
    pmRA, pmDE, plx = df["pmRA"], df["pmDE"], df["plx"]

    dups_fnames, dups_probs = duplicate_probs.main(
        df["fnames"], x, y, plx, pmRA, pmDE, prob_cut, Nmax
    )

    return dups_fnames, dups_probs


def check_cents_diff(xy_c_o, vpd_c_o, plx_c_o, xy_c_n, vpd_c_n, plx_c_n):
    """ """
    bad_center_xy, bad_center_pm, bad_center_plx = "0", "0", "0"

    # 5 arcmin maximum
    d_arcmin = np.sqrt((xy_c_o[0] - xy_c_n[0]) ** 2 + (xy_c_o[1] - xy_c_n[1]) ** 2) * 60
    if d_arcmin > 5:
        bad_center_xy = "1"

    # Relative difference
    if not np.isnan(vpd_c_o[0]):
        pm_max = []
        for vpd_c_i in abs(np.array(vpd_c_o)):
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
        pmra_p = 100 * abs((vpd_c_o[0] - vpd_c_n[0]) / (vpd_c_o[0] + 0.001))
        pmde_p = 100 * abs((vpd_c_o[1] - vpd_c_n[1]) / (vpd_c_o[1] + 0.001))
        if pmra_p > pm_max[0] or pmde_p > pm_max[1]:
            bad_center_pm = "1"

    # Relative difference
    if not np.isnan(plx_c_o):
        if plx_c_o > 0.2:
            plx_max = 25
        elif plx_c_o > 0.1:
            plx_max = 30
        elif plx_c_o > 0.05:
            plx_max = 35
        elif plx_c_o > 0.01:
            plx_max = 50
        else:
            plx_max = 70
        plx_p = 100 * abs(plx_c_o - plx_c_n) / (plx_c_o + 0.001)
        if abs(plx_p) > plx_max:
            bad_center_plx = "1"

    bad_center = bad_center_xy + bad_center_pm + bad_center_plx

    return bad_center
