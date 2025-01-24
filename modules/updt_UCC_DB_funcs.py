import warnings
from difflib import SequenceMatcher
from string import ascii_lowercase

import asteca
import Levenshtein
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from scipy.integrate import quad
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.special import loggamma
from scipy.stats import gaussian_kde


def radec2lonlat(ra: float | list, dec: float | list) -> tuple[float, float]:
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
    gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    lb = gc.transform_to("galactic")
    return lb.l.value, lb.b.value


def check_centers(
    xy_c_m: tuple[float, float],
    vpd_c_m: tuple[float, float],
    plx_c_m: float,
    xy_c_n: tuple[float, float],
    vpd_c_n: tuple[float, float],
    plx_c_n: float,
    rad_dup: float = 5,
) -> str:
    """
    Compares the centers of a cluster estimated from members with those from the
    literature.

    Parameters
    ----------
    xy_c_m : tuple
        Center coordinates (lon, lat) from estimated members.
    vpd_c_m : tuple
        Center proper motion (pmRA, pmDE) from estimated members.
    plx_c_m : float
        Center parallax from estimated members.
    xy_c_n : tuple
        Center coordinates (lon, lat) from the literature.
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
    """

    bad_center_xy, bad_center_pm, bad_center_plx = "n", "n", "n"

    # Max distance in arcmin, 'rad_dup' arcmin maximum
    d_arcmin = np.sqrt((xy_c_m[0] - xy_c_n[0]) ** 2 + (xy_c_m[1] - xy_c_n[1]) ** 2) * 60
    if d_arcmin > rad_dup:
        bad_center_xy = "y"

    # Relative difference
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

    return bad_center


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


#####################################################################################
#####################################################################################
# standardize_and_match


def get_fnames_new_DB(
    df_new: pd.DataFrame, newDB_json: dict, sep: str = ","
) -> list[list[str]]:
    """
    Extract and standardize all names in the new catalogue

    Parameters
    ----------
    df_new : pd.DataFrame
        DataFrame of the new catalogue.
    newDB_json : dict
        Dictionary with the parameters of the new catalogue.
    sep : str, optional
        Separator used to split the names in the new catalogue. Default is ",".

    Returns
    -------
    list
        List of lists, where each inner list contains the standardized names for
        each cluster in the new catalogue.
    """
    names_all = df_new[newDB_json["names"]]
    new_DB_fnames = []
    for names in names_all:
        names_l = []
        names_s = str(names).split(sep)
        for name in names_s:
            name = name.strip()
            name = rename_standard(name)
            names_l.append(rm_chars_from_name(name))
        new_DB_fnames.append(names_l)

    return new_DB_fnames


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


def rm_chars_from_name(name: str) -> str:
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


def get_matches_new_DB(
    df_UCC: pd.DataFrame, new_DB_fnames: list[list[str]]
) -> list[int | None]:
    """
    Get cluster matches for the new DB being added to the UCC

    Parameters
    ----------
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_DB_fnames : list
        List of lists, where each inner list contains the
        standardized names for each cluster in the new catalogue.

    Returns
    -------
    list
        List of indexes into the UCC pointing to each entry in the new DB.
        If an entry in the new DB is not present in the UCC, the corresponding
        index in the list will be None.
    """

    def match_fname(new_cl):
        for name_new in new_cl:
            for j, old_cl in enumerate(df_UCC["fnames"]):
                for name_old in old_cl.split(";"):
                    if name_new == name_old:
                        return j
        return None

    db_matches = []
    for new_cl in new_DB_fnames:
        # Check if this new fname is already in the UCC
        db_matches.append(match_fname(new_cl))

    return db_matches


#####################################################################################
#####################################################################################


#####################################################################################
#####################################################################################
# check_new_DB


def dups_check_newDB_UCC(
    logging,
    new_DB: str,
    df_UCC: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
) -> bool:
    """
    Checks for duplicate entries within a new database that also exist in the UCC.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    new_DB : str
        Name of the new database.
    df_UCC : pd.DataFrame
        DataFrame representing the UCC.
    new_DB_fnames : list
        List of lists, each containing standardized names for a
        cluster in the new database.
    db_matches : list
        List of indices mapping entries in the new database to
        entries in the UCC.

    Returns
    -------
    bool
        True if duplicate entries are found that need to be combined, False otherwise.
    """

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

    idxs_match = [_ for _ in db_matches if _ is not None]
    dup_idxs = list_duplicates(idxs_match)
    if len(dup_idxs) > 0:
        logging.info(f"WARNING! Found entries in {new_DB} that must be combined")
        print("")
        for didx in dup_idxs:
            for i, db_idx in enumerate(db_matches):
                if db_idx == didx:
                    logging.info(
                        f"  UCC {didx}: {df_UCC['fnames'][didx]} --> {i} {new_DB_fnames[i]}"
                    )
            print("")
        return True

    return False


def GCs_check(
    logging,
    GCs_path: str,
    newDB_json: dict,
    df_new: pd.DataFrame,
    search_rad: float = 15,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Check for nearby GCs for a new database

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    GCs_path : str
        Path to the GCs database.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_new : pd.DataFrame
        DataFrame of the new database.
    search_rad : float, optional
        Search radius in arcmin. Default is 15.

    Returns
    -------
    tuple
        A tuple containing:
        - Array of galactic longitudes of the clusters in the new database.
        - Array of galactic latitudes of the clusters in the new database.
        - Boolean flag indicating if probable GCs were found.
    """
    # Equatorial to galactic
    gc = SkyCoord(
        ra=df_new[newDB_json["pos"]["RA"]].values * u.deg,
        dec=df_new[newDB_json["pos"]["DEC"]].values * u.deg,
    )
    lb = gc.transform_to("galactic")
    glon_glat = list(zip(*[lb.l, lb.b]))

    # Read GCs DB
    df_gcs = pd.read_csv(GCs_path)
    l_gc, b_gc = df_gcs["GLON"].values * u.deg, df_gcs["GLAT"].values * u.deg

    gc_all, GCs_found = [], 0
    for idx, (glon_i, glat_i) in enumerate(glon_glat):
        d_arcmin = angular_separation(glon_i, glat_i, l_gc, b_gc).to("deg").value * 60
        j1 = np.argmin(d_arcmin)

        if d_arcmin[j1] < search_rad:
            GCs_found += 1
            gc_all.append(
                [
                    idx,
                    df_new.iloc[idx][newDB_json["names"]],
                    df_gcs["Name"][j1],
                    d_arcmin[j1],
                ]
            )

    gc_flag = False
    if GCs_found > 0:
        gc_all = np.array(gc_all).T
        i_sort = np.argsort(np.array(gc_all[-1], dtype=float))
        gc_all = gc_all[:, i_sort].T

        gc_flag = True
        logging.info(f"Found {GCs_found} probable GCs")
        for gc in gc_all:
            idx, row_id, df_gcs_name, d_arcmin = gc
            logging.info(
                f"{idx:<10} {row_id:<15} --> {df_gcs_name.strip():<15}"
                + f"d={round(float(d_arcmin), 2)}"
            )
    else:
        logging.info("No probable GCs found")

    return np.array(lb.l), np.array(lb.b), gc_flag


def close_OC_check(
    logging,
    newDB_json: dict,
    df_new: pd.DataFrame,
    rad_dup: float,
    leven_rad: float = 0.85,
) -> bool:
    """
    Looks for OCs in the new DB that are close to other OCs in the new DB (RA, DEC)
    whose names are somewhat similar (Levenshtein distance).

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_new : pd.DataFrame
        DataFrame of the new database.
    rad_dup : float
        Search radius in arcmin.
    leven_rad : float, optional
        Levenshtein ratio threshold. Default is 0.85.

    Returns
    -------
    bool
        Boolean flag indicating if probable inner duplicates were found.
    """
    x, y = (
        df_new[newDB_json["pos"]["RA"]].values,
        df_new[newDB_json["pos"]["DEC"]].values,
    )
    coords = np.array([x, y]).T
    # Find the distances to all clusters, for all clusters (in arcmin)
    dist = cdist(coords, coords) * 60
    # Change distance to itself from 0 to inf
    msk = dist == 0.0
    dist[msk] = np.inf

    idxs = np.arange(0, len(df_new))
    all_dups, dups_list = [], []
    for i, cl_d in enumerate(dist):
        msk = cl_d < rad_dup
        if msk.sum() == 0:
            continue

        cl_name = str(df_new[newDB_json["names"]][i]).strip()
        if cl_name in dups_list:
            # print(cl_name, "continue")
            continue

        N_inner_dups, dups, dist, L_ratios = 0, [], [], []
        for j in idxs[msk]:
            dup_name = str(df_new[newDB_json["names"]][j]).strip()

            L_ratio = Levenshtein.ratio(cl_name, dup_name)
            if L_ratio > leven_rad:
                N_inner_dups += 1
                dups_list.append(dup_name)
                dups.append(dup_name)
                dist.append(str(round(cl_d[j], 1)))
                L_ratios.append(str(round(L_ratio, 2)))
        if dups:
            all_dups.append([i, cl_name, N_inner_dups, dups, dist, L_ratios])

    inner_flag = False
    dups_found = len(all_dups)
    if dups_found > 0:
        inner_flag = True
        all_dists = []
        for dup in all_dups:
            all_dists.append(min([float(_) for _ in dup[-2]]))
        i_sort = np.argsort(all_dists)

        logging.info(f"Found {dups_found} probable inner duplicates")
        for idx in i_sort:
            i, cl_name, N_inner_dups, dups, dist, L_ratios = all_dups[idx]
            logging.info(
                f"{i:<10} {cl_name:<15} (N={N_inner_dups}) --> "
                + f"{';'.join(dups):<15} d={';'.join(dist)}, L={';'.join(L_ratios)}"
            )
    else:
        logging.info("No inner duplicates found")

    return inner_flag


def close_OC_UCC_check(
    logging,
    df_UCC: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
    glon: np.ndarray,
    glat: np.ndarray,
    rad_dup: float,
) -> bool:
    """
    Looks for OCs in the new DB that are close to OCs in the UCC (GLON, GLAT) but
    with different names.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_DB_fnames : list
        List of lists, where each inner list contains the
        standardized names for each cluster in the new catalogue.
    db_matches : list
        List of indexes into the UCC pointing to each entry in the
        new DB.
    glon : np.ndarray
        Array of galactic longitudes of the clusters in the new database.
    glat : np.ndarray
        Array of galactic latitudes of the clusters in the new database.
    rad_dup : float
        Search radius in arcmin.

    Returns
    -------
    bool
        Boolean flag indicating if probable UCC duplicates were found.
    """

    coords_new = np.array([glon, glat]).T
    coords_UCC = np.array([df_UCC["GLON"], df_UCC["GLAT"]]).T
    # Find the distances to all clusters, for all clusters (in arcmin)
    dist = cdist(coords_new, coords_UCC) * 60

    idxs = np.arange(0, len(df_UCC))
    dups_list, dups_found = [], []
    for i, cl_d in enumerate(dist):
        msk = cl_d < rad_dup
        if msk.sum() == 0:
            continue

        cl_name = ",".join(new_DB_fnames[i])
        if cl_name in dups_list:
            continue

        if db_matches[i] is not None:
            # cl_name is present in the UCC (df_UCC['fnames'][db_matches[i]])
            continue

        dups, dist = [], []
        for j in idxs[msk]:
            dup_name = df_UCC["fnames"][j]
            dups_list.append(dup_name)
            dups.append(dup_name)
            dist.append(str(round(cl_d[j], 1)))
        dups_found.append(
            f"{i} {cl_name} (N={msk.sum()}) --> "
            + f"{'|'.join(dups)}, d={'|'.join(dist)}"
        )

    dups_flag = True
    if len(dups_found) > 0:
        logging.info(f"Found {len(dups_found)} probable UCC duplicates")
        for dup in dups_found:
            logging.info(dup)
    else:
        dups_flag = False
        logging.info("No UCC duplicates found")

    return dups_flag


def vdberg_check(logging, newDB_json: dict, df_new: pd.DataFrame) -> bool:
    """
    Check for instances of 'vdBergh-Hagen' and 'vdBergh'

    Per CDS recommendation:

    * VDBergh-Hagen --> VDBH
    * VDBergh       --> VDB

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_new : pd.DataFrame
        DataFrame of the new database.

    Returns
    -------
    bool
        Boolean flag indicating if probable vdBergh-Hagen/vdBergh OCs were found.
    """
    names_lst = ["vdBergh-Hagen", "vdBergh", "van den Bergh–Hagen", "van den Bergh"]
    names_lst = [_.lower().replace("-", "").replace(" ", "") for _ in names_lst]

    vds_found = 0
    for i, new_cl in enumerate(df_new[newDB_json["names"]]):
        new_cl = (
            new_cl.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
        )
        for name_check in names_lst:
            sm_ratio = SequenceMatcher(None, new_cl, name_check).ratio()
            if sm_ratio > 0.5:
                vds_found += 1
                logging.info(f"{i}, {new_cl} --> {name_check} (P={round(sm_ratio, 2)})")
                break

    vdb_flag = True
    if vds_found == 0:
        vdb_flag = False
        logging.info("No probable vdBergh-Hagen/vdBergh OCs found")

    return vdb_flag


def prep_newDB(
    newDB_json: dict,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
) -> dict:
    """
    Prepare information from a new database matched with the UCC

    Parameters
    ----------
    newDB_json : dict
        Dictionary containing parameters specifying column names for
        position (RA, Dec, etc.).
    df_new : pd.DataFrame
        DataFrame containing information about the new database.
    new_DB_fnames : list
        List of lists, where each inner list contains file names
        for each cluster in the new database.
    db_matches : list
        List of indices representing matches in the UCC, or None if
        no match exists.

    Returns
    -------
    dict
        A dictionary containing:
            - "fnames" (list): List of concatenated filenames for each cluster.
            - "RA_ICRS" (list): List of right ascension (RA) values.
            - "DE_ICRS" (list): List of declination (Dec) values.
            - "pmRA" (list): List of proper motion in RA.
            - "pmDE" (list): List of proper motion in Dec.
            - "Plx" (list): List of parallax values.
            - "UCC_idx" (list): List of indices of matched clusters in the UCC, or None
              for new clusters.
            - "GLON" (list): List of Galactic longitude values.
            - "GLAT" (list): List of Galactic latitude values.
    """
    new_db_info = {
        "fnames": [],
        "RA_ICRS": [],
        "DE_ICRS": [],
        "pmRA": [],
        "pmDE": [],
        "Plx": [],
        "UCC_idx": [],
    }

    for i, fnames_lst in enumerate(new_DB_fnames):
        # Use semi-colon here to math the UCC format
        fnames = ";".join(fnames_lst)
        row_n = df_new.iloc[i]

        # Coordinates for this cluster in the new DB
        plx_n, pmra_n, pmde_n = np.nan, np.nan, np.nan
        ra_n, dec_n = row_n[newDB_json["pos"]["RA"]], row_n[newDB_json["pos"]["DEC"]]
        if "plx" in newDB_json["pos"]:
            plx_n = row_n[newDB_json["pos"]["plx"]]
        if "pmra" in newDB_json["pos"]:
            pmra_n = row_n[newDB_json["pos"]["pmra"]]
        if "pmde" in newDB_json["pos"]:
            pmde_n = row_n[newDB_json["pos"]["pmde"]]

        # Index of the match for this new cluster in the old DB (if any)
        db_match_j = db_matches[i]

        # If the cluster is already present in the UCC
        if db_match_j is not None:
            new_db_info["fnames"].append(fnames)
            new_db_info["RA_ICRS"].append(ra_n)
            new_db_info["DE_ICRS"].append(dec_n)
            new_db_info["pmRA"].append(pmra_n)
            new_db_info["pmDE"].append(pmde_n)
            new_db_info["Plx"].append(plx_n)
            new_db_info["UCC_idx"].append(db_match_j)
        else:
            # This is a new cluster
            new_db_info["fnames"].append(fnames)
            new_db_info["RA_ICRS"].append(np.nan)
            new_db_info["DE_ICRS"].append(np.nan)
            new_db_info["pmRA"].append(np.nan)
            new_db_info["pmDE"].append(np.nan)
            new_db_info["Plx"].append(plx_n)
            new_db_info["UCC_idx"].append(None)

    lon_n, lat_n = radec2lonlat(
        new_db_info["RA_ICRS"],
        new_db_info["DE_ICRS"],
    )
    new_db_info["GLON"] = list(np.round(lon_n, 4))
    new_db_info["GLAT"] = list(np.round(lat_n, 4))

    return new_db_info


def positions_check(
    logging, df_UCC: pd.DataFrame, new_db_info: dict, rad_dup: float
) -> bool:
    """
    Checks the positions of clusters in the new database against those in the UCC.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_db_info : dict
        Dictionary with the information of the new database.
    rad_dup : float
        Search radius in arcmin.

    Returns
    -------
    bool
        Boolean flag indicating if OCs were flagged for attention.
    """
    ocs_attention = []
    for i, fnames in enumerate(new_db_info["fnames"]):
        j = new_db_info["UCC_idx"][i]
        # If the OC is already present in the UCC
        if j is not None:
            bad_center = check_centers(
                (df_UCC["GLON_m"].iloc[j], df_UCC["GLAT_m"].iloc[j]),
                (df_UCC["pmRA_m"].iloc[j], df_UCC["pmDE_m"].iloc[j]),
                df_UCC["Plx_m"].iloc[j],
                (new_db_info["GLON"][i], new_db_info["GLAT"][i]),
                (new_db_info["pmRA"][i], new_db_info["pmDE"][i]),
                new_db_info["Plx"][i],
                rad_dup,
            )
            # Is the difference between the old vs new center values large?
            if bad_center == "nnn":
                continue
            # Store information on the OCs that require attention
            ocs_attention.append([fnames, i, j, bad_center])

    attention_flag = False
    if len(ocs_attention) > 0:
        attention_flag = True
        logging.info("\nOCs flagged for attention:\n")
        logging.info(
            "{:<15} {:<5} {}".format(
                "name", "cent_flag", "[arcmin] [pmRA %] [pmDE %] [plx %]"
            )
        )
        for oc in ocs_attention:
            fnames, i, j, bad_center = oc
            flag_log(logging, df_UCC, new_db_info, bad_center, fnames, i, j)
    else:
        logging.info("\nNo OCs flagged for attention")

    return attention_flag


def flag_log(
    logging,
    df_UCC: pd.DataFrame,
    new_db_info: dict,
    bad_center: str,
    fnames: str,
    i: int,
    j: int,
) -> None:
    """
    Logs details about OCs flagged for attention based on center comparison.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_db_info : dict
        Dictionary with the information of the new database.
    bad_center : str
        String indicating the quality of the center comparison.
    fnames : str
        String of fnames for the cluster.
    i : int
        Index of the cluster in the new database.
    j : int
        Index of the cluster in the UCC.
    """
    txt = ""
    if bad_center[0] == "y":
        d_arcmin = (
            np.sqrt(
                (df_UCC["GLON_m"][j] - new_db_info["GLON"][i]) ** 2
                + (df_UCC["GLAT_m"][j] - new_db_info["GLAT"][i]) ** 2
            )
            * 60
        )
        txt += "{:.1f} ".format(d_arcmin)
    else:
        txt += "-- "

    if bad_center[1] == "y":
        pmra_p = 100 * abs(
            (df_UCC["pmRA_m"][j] - new_db_info["pmRA"][i])
            / (df_UCC["pmRA_m"][j] + 0.001)
        )
        pmde_p = 100 * abs(
            (df_UCC["pmDE_m"][j] - new_db_info["pmDE"][i])
            / (df_UCC["pmDE_m"][j] + 0.001)
        )
        txt += "{:.1f} {:.1f} ".format(pmra_p, pmde_p)
    else:
        txt += "-- -- "

    if bad_center[2] == "y":
        plx_p = (
            100
            * abs(df_UCC["plx_m"][j] - new_db_info["plx"][i])
            / (df_UCC["plx_m"][j] + 0.001)
        )
        txt += "{:.1f}".format(plx_p)
    else:
        txt += "--"

    txt = txt.split()
    logging.info(
        "{:<15} {:<5} {:>12} {:>8} {:>8} {:>7}".format(fnames, bad_center, *txt)
    )


#####################################################################################
#####################################################################################


#####################################################################################
#####################################################################################

# add_new_DB


def combine_UCC_new_DB(
    logging,
    new_DB: str,
    newDB_json: dict,
    df_UCC: pd.DataFrame,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
    sep: str = ",",
) -> dict:
    """
    Combines a new database with the UCC, handling new and existing entries.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    new_DB : str
        Name of the new database.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    df_new : pd.DataFrame
        DataFrame of the new database.
    new_DB_fnames : list
        List of lists, where each inner list contains the
        standardized names for each cluster in the new catalogue.
    db_matches : list
        List of indexes into the UCC pointing to each entry in the
        new DB.
    sep : str, optional
        Separator used for splitting names in the new database. Default is ",".

    Returns
    -------
    dict
        Dictionary representing the updated database with new and modified entries.
    """
    new_db_dict = {_: [] for _ in df_UCC.keys()}
    # For each entry in the new DB
    for i_new_cl, new_cl in enumerate(new_DB_fnames):
        row_n = dict(df_new.iloc[i_new_cl])

        # For each comma separated name for this OC in the new DB
        oc_names = row_n[newDB_json["names"]].split(sep)
        # Rename certain entries (ESO, FSR, etc)
        new_names_rename = []
        for _ in oc_names:
            name = rename_standard(_.strip())
            new_names_rename.append(name)
        oc_names = ";".join(new_names_rename)

        # Index of the match for this new cluster in the UCC (if any)
        if db_matches[i_new_cl] is None:
            # The cluster is not present in the UCC, add to the new Db dictionary
            new_db_dict = new_OC_not_in_UCC(
                new_DB, new_db_dict, i_new_cl, new_cl, row_n, oc_names, newDB_json
            )
            logging.info(f"{i_new_cl} {','.join(new_cl)} is a new OC")
        else:
            # The cluster is already present in the UCC
            # Row in UCC where this match is located
            row = dict(df_UCC.iloc[db_matches[i_new_cl]])
            # Add to the new Db dictionary
            new_db_dict = OC_in_UCC(
                new_DB, new_db_dict, i_new_cl, new_cl, oc_names, row
            )
            logging.info(
                f"{i_new_cl} {','.join(new_cl)} is in the UCC: update DB indexes"
            )

    return new_db_dict


def new_OC_not_in_UCC(
    new_DB: str,
    new_db_dict: dict,
    i_new_cl: int,
    fnames_new_cl: list[str],
    row_n: dict,
    oc_names: str,
    newDB_json: dict,
) -> dict:
    """
    Adds a new OC, not present in the UCC, to the new database dictionary.

    Parameters
    ----------
    new_DB : str
        Name of the new database.
    new_db_dict : dict
        Dictionary representing the updated database.
    i_new_cl : int
        Index of the new cluster in the new database.
    fnames_new_cl : list
        List of standardized names for the new cluster.
    row_n : dict
        Dictionary representing the row of the new cluster in the new database.
    oc_names : str
        Semicolon-separated string of names for the new cluster.
    newDB_json : dict
        Dictionary with column name mappings for the new database.

    Returns
    -------
    dict
        Updated dictionary representing the database with the new OC added.
    """
    # Remove duplicates from names and fnames
    ID = oc_names
    if ";" in oc_names:
        ID = rm_name_dups(oc_names)
    fnames = ";".join(fnames_new_cl)
    if ";" in fnames:
        fnames = rm_name_dups(fnames)

    ra_n, dec_n = row_n[newDB_json["pos"]["RA"]], row_n[newDB_json["pos"]["DEC"]]
    # Galactic coordinates
    lon_n, lat_n = radec2lonlat(ra_n, dec_n)
    #
    plx_n = np.nan
    if "plx" in newDB_json["pos"]:
        plx_n = row_n[newDB_json["pos"]["plx"]]
    pmra_n = np.nan
    if "pmra" in newDB_json["pos"]:
        pmra_n = row_n[newDB_json["pos"]["pmra"]]
    pmde_n = np.nan
    if "pmde" in newDB_json["pos"]:
        pmde_n = row_n[newDB_json["pos"]["pmde"]]

    new_vals = {
        "DB": new_DB,
        "DB_i": str(i_new_cl),
        "ID": ID,
        "RA_ICRS": round(ra_n, 4),
        "DE_ICRS": round(dec_n, 4),
        "GLON": round(lon_n, 4),
        "GLAT": round(lat_n, 4),
        "Plx": round(plx_n, 4),
        "pmRA": round(pmra_n, 4),
        "pmDE": round(pmde_n, 4),
        "fnames": fnames,
    }
    new_db_dict = updt_new_db_dict(new_db_dict, new_vals)

    return new_db_dict


def OC_in_UCC(
    new_DB: str,
    new_db_dict: dict,
    i_new_cl: int,
    fnames_new_cl: list[str],
    oc_names: str,
    row: dict,
) -> dict:
    """
    Updates an existing OC in the UCC with information from the new database.

    Parameters
    ----------
    new_DB : str
        Name of the new database.
    new_db_dict : dict
        Dictionary representing the updated database.
    i_new_cl : int
        Index of the cluster in the new database.
    fnames_new_cl : list
        List of standardized names for the cluster.
    oc_names : str
        Semicolon-separated string of names for the cluster.
    row : dict
        Dictionary representing the row of the cluster in the UCC.

    Returns
    -------
    dict
        Updated dictionary representing the database with the modified OC.
    """
    DB_ID = row["DB"] + ";" + new_DB
    DB_i = row["DB_i"] + ";" + str(i_new_cl)
    # Order by years before storing
    DB_ID, DB_i = date_order_DBs(DB_ID, DB_i)

    # Attach name(s) and fname(s) present in new DB to the UCC, removing duplicates
    ID = row["ID"] + ";" + oc_names
    ID = rm_name_dups(ID)
    # The first fname is the most important one as all files for this OC use this
    # naming. The 'rm_name_dups' function will always keep this name first in line
    fnames = row["fnames"] + ";" + ";".join(fnames_new_cl)
    fnames = rm_name_dups(fnames)

    # Galactic coordinates
    lon_n, lat_n = radec2lonlat(row["RA_ICRS"], row["DE_ICRS"])

    new_vals = {
        "DB": DB_ID,
        "DB_i": DB_i,
        "ID": ID,
        "RA_ICRS": round(row["RA_ICRS"], 4),
        "DE_ICRS": round(row["DE_ICRS"], 4),
        "GLON": round(lon_n, 4),
        "GLAT": round(lat_n, 4),
        "Plx": round(row["Plx"], 4),
        "pmRA": round(row["pmRA"], 4),
        "pmDE": round(row["pmDE"], 4),
        "fnames": fnames,
        # Copy the remaining values from the UCC row
    }
    new_db_dict = updt_new_db_dict(new_db_dict, new_vals, row)

    return new_db_dict


def rm_name_dups(names: str) -> str:
    """
    Removes duplicate names from a semicolon-separated string, considering variations
    with or without spaces and underscores, and retains the first occurrence.

    Removes duplicates of the kind:

        Berkeley 102, Berkeley102, Berkeley_102

    keeping only the name with the space.

    Parameters
    ----------
    names : str
        A semicolon-separated string of names.

    Returns
    -------
    str
        A semicolon-separated string of unique names, with duplicates removed.
    """
    names_l = names.split(";")
    for n in names_l:
        n2 = n.replace(" ", "")
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n
        n2 = n.replace(" ", "_")
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n

    names = ";".join(list(dict.fromkeys(names_l)))

    return names


def updt_new_db_dict(
    new_db_dict: dict, new_vals: dict, row: dict | None = None
) -> dict:
    """
    Updates the new database dictionary with new values and, optionally, existing
    values from a row.

    Parameters
    ----------
    new_db_dict : dict
        Dictionary representing the updated database.
    new_vals : dict
        Dictionary of new values to add to the database.
    row : dict, optional
        Optional; Dictionary representing an existing row in the UCC to copy
        values from.

    Returns
    -------
    dict
        Updated dictionary representing the database with new and existing values.
    """
    for col in (
        "DB",
        "DB_i",
        "ID",
        "RA_ICRS",
        "DE_ICRS",
        "GLON",
        "GLAT",
        "Plx",
        "pmRA",
        "pmDE",
        "fnames",
    ):
        new_db_dict[col].append(new_vals[col])

    for col in (
        "UCC_ID",
        "quad",
        "dups_fnames",
        "dups_probs",
        "r_50",
        "N_50",
        "N_fixed",
        "fixed_cent",
        "cent_flags",
        "C1",
        "C2",
        "C3",
        "GLON_m",
        "GLAT_m",
        "RA_ICRS_m",
        "DE_ICRS_m",
        "Plx_m",
        "pmRA_m",
        "pmDE_m",
        "Rv_m",
        "N_Rv",
        "dups_fnames_m",
        "dups_probs_m",
    ):
        if row is None:
            new_db_dict[col].append(np.nan)
        else:
            new_db_dict[col].append(row[col])

    return new_db_dict


def assign_UCC_ids(logging, glon: float, glat: float, ucc_ids_old: list[str]) -> str:
    """
    Assigns a new UCC ID based on galactic coordinates, avoiding duplicates.

    Format: UCC GXXX.X+YY.Y

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    glon : float
        Galactic longitude.
    glat : float
        Galactic latitude.
    ucc_ids_old : list
        List of existing UCC IDs.

    Returns
    -------
    str
        A new, unique UCC ID.
    """

    def trunc(values, decs=1):
        return np.trunc(values * 10**decs) / (10**decs)

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
            logging.info(f"ERROR NAMING: {glon}, {glat} --> {ucc_id}")
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


def QXY_fold(UCC_ID: str) -> str:
    """
    Determines the quadrant and north/south designation of a cluster based on its UCC ID.

    Parameters
    ----------
    UCC_ID : str
        The UCC ID of the cluster.

    Returns
    -------
    str
        A string representing the quadrant and north/south designation (e.g., "Q1P",
        "Q3N").
    """
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


def duplicates_check(logging, df_all: pd.DataFrame) -> bool:
    """
    Checks for duplicate entries in specified columns of a DataFrame.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_all : pd.DataFrame
        DataFrame to check for duplicates.

    Returns
    -------
    bool
        True if duplicates are found in any of the specified columns, False otherwise.
    """

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
        # adds all elements it doesn't know yet to seen and all other to seen_twice
        seen_twice = set(x for x in seq if x in seen or seen_add(x))
        # turn the set into a list (as requested)
        return list(seen_twice)

    def dup_check(df_all: pd.DataFrame, col: str) -> bool:
        """
        Checks for duplicates in a specific column of a DataFrame.

        Args:
            df_all: DataFrame to check.
            col: Name of the column to check.

        Returns:
            True if duplicates are found, False otherwise.
        """
        dups = list_duplicates(list(df_all[col]))
        if len(dups) > 0:
            logging.info(f"\nWARNING! N={len(dups)} duplicates found in '{col}':")
            for dup in dups:
                print(dup)
            logging.info("UCC was not updated")
            return True
        else:
            return False

    for col in ("ID", "UCC_ID", "fnames"):
        dup_flag = dup_check(df_all, col)
        if dup_flag:
            return True

    return False


#####################################################################################
#####################################################################################


#####################################################################################
#####################################################################################
# possible_duplicates


def duplicate_probs(
    fnames: list[str],
    x: np.ndarray,
    y: np.ndarray,
    plx: np.ndarray,
    pmRA: np.ndarray,
    pmDE: np.ndarray,
    prob_cut: float,
    Nmax: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Identifies potential duplicate clusters based on proximity and similarity of
    parameters.

    Parameters
    ----------
    fnames :list[str]
        List of cluster names (filenames).
    x : np.ndarray
        Array of x-coordinates (e.g., GLON).
    y : np.ndarray
        Array of y-coordinates (e.g., GLAT).
    plx : np.ndarray
        Array of parallax values.
    pmRA : np.ndarray
        Array of proper motion in RA.
    pmDE : np.ndarray
        Array of proper motion in DE.
    prob_cut : float
        Probability threshold for considering a cluster a duplicate.
    Nmax : int, optional
        Maximum number of standard deviations for considering clusters close in
        a dimension. Default is 3.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - dups_fnames: List of semicolon-separated filenames of probable duplicates for
          each cluster.
        - dups_probs: List of semicolon-separated probabilities of being duplicates for
          each cluster.
    """
    # Find the (x, y) distances to all clusters, for all clusters
    coords = np.array([x, y]).T
    dist = cdist(coords, coords)

    dups_fnames, dups_probs = [], []
    for i, dists_i in enumerate(dist):
        # Only process relatively close clusters
        rad_max = Nmax * max_coords_rad(plx[i])
        msk_rad = dists_i <= rad_max
        idx_j = np.arange(0, dists_i.size)
        j_msk = idx_j[msk_rad]

        dups_fname_i, dups_prob_i = [], []
        for j in j_msk:
            # Skip itself
            if j == i:
                continue

            # Fetch duplicate probability for the i,j clusters
            dup_prob = dprob(x, y, pmRA, pmDE, plx, i, j)
            if dup_prob >= prob_cut:
                # Store just the first fname
                dups_fname_i.append(fnames[j].split(";")[0])
                dups_prob_i.append(dup_prob)

        if dups_fname_i:
            # Store list of probable duplicates respecting a given order
            dups_fname_i, dups_prob_i = sort_by_float_desc_then_alpha(
                dups_fname_i, dups_prob_i
            )
            dups_fname_i = ";".join(dups_fname_i)
            dups_prob_i = ";".join(dups_prob_i)
        else:
            dups_fname_i, dups_prob_i = "nan", "nan"

        dups_fnames.append(dups_fname_i)
        dups_probs.append(dups_prob_i)

    return dups_fnames, dups_probs


def dprob(
    x: np.ndarray,
    y: np.ndarray,
    pmRA: np.ndarray,
    pmDE: np.ndarray,
    plx: np.ndarray,
    i: int,
    j: int,
    Nmax: int = 2,
) -> float:
    """
    Calculate the probability of being duplicates for the i,j clusters

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates (e.g., GLON).
    y : np.ndarray
        Array of y-coordinates (e.g., GLAT).
    pmRA : np.ndarray
        Array of proper motion in RA.
    pmDE : np.ndarray
        Array of proper motion in DE.
    plx : np.ndarray
        Array of parallax values.
    i : int
        Index of the first cluster.
    j : int
        Index of the second cluster.
    Nmax : int, optional
        maximum number of times allowed for the two objects to be apart
        in any of the dimensions. If this happens for any of the dimensions,
        return a probability of zero. Default is 2.

    Returns
    -------
    float
        The probability of the two clusters being duplicates.
    """

    # Define reference parallax
    if np.isnan(plx[i]) and np.isnan(plx[j]):
        plx_ref = np.nan
    elif np.isnan(plx[i]):
        plx_ref = plx[j]
    elif np.isnan(plx[j]):
        plx_ref = plx[i]
    else:
        plx_ref = (plx[i] + plx[j]) * 0.5

    # Arbitrary 'duplicate regions' for different parallax brackets
    if np.isnan(plx_ref):
        rad, plx_r, pm_r = 2.5, np.nan, 0.2
    elif plx_ref >= 4:
        rad, plx_r, pm_r = 20, 0.5, 1
    elif 3 <= plx_ref < 4:
        rad, plx_r, pm_r = 15, 0.25, 0.75
    elif 2 <= plx_ref < 3:
        rad, plx_r, pm_r = 10, 0.2, 0.5
    elif 1.5 <= plx_ref < 2:
        rad, plx_r, pm_r = 7.5, 0.15, 0.35
    elif 1 <= plx_ref < 1.5:
        rad, plx_r, pm_r = 5, 0.1, 0.25
    elif 0.5 <= plx_ref < 1:
        rad, plx_r, pm_r = 2.5, 0.075, 0.2
    elif plx_ref < 0.5:
        rad, plx_r, pm_r = 2, 0.05, 0.15
    elif plx_ref < 0.25:
        rad, plx_r, pm_r = 1.5, 0.025, 0.1
    else:
        raise ValueError(
            "Could not define 'rad, plx_r, pm_r' values, plx_ref is out of bounds"
        )

    # Angular distance in arcmin
    d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) * 60
    # PMs distance
    pm_d = np.sqrt((pmRA[i] - pmRA[j]) ** 2 + (pmDE[i] - pmDE[j]) ** 2)
    # Parallax distance
    plx_d = abs(plx[i] - plx[j])

    # If *any* distance is *very* far away, return no duplicate disregarding
    # the rest with 0 probability
    if d > Nmax * rad or pm_d > Nmax * pm_r or plx_d > Nmax * plx_r:
        return 0

    d_prob = lin_relation(d, rad)
    pms_prob = lin_relation(pm_d, pm_r)
    plx_prob = lin_relation(plx_d, plx_r)

    # Combined probability
    prob = np.nanmean((d_prob, pms_prob, plx_prob))

    # pyright issue due to: https://github.com/numpy/numpy/issues/28076
    return round(prob, 2)


def lin_relation(dist: float, d_max: float) -> float:
    """
    Calculates a linear probability based on distance.

    d_min=0 is fixed
    Linear relation for: (0, d_max), (1, d_min)

    Parameters
    ----------
    dist : float
        The distance between two objects.
    d_max : float
        The maximum distance for considering two objects related.

    Returns
    -------
    float
        A probability value between 0 and 1, where 1 indicates maximum proximity
        and 0 indicates the maximum distance `d_max`.
    """
    # m, h = (d_min - d_max), d_max
    # prob = (dist - h) / m
    # m, h = -d_max, d_max
    p = (dist - d_max) / -d_max
    if p < 0:  # np.isnan(p) or
        return 0
    return p


def max_coords_rad(plx_i: float) -> float:
    """
    Defines a maximum coordinate radius based on parallax.

    Parameters
    ----------
    plx_i : float
        Parallax value.

    Returns
    -------
    float
        Maximum radius in degrees (parallax dependent).
    """
    if np.isnan(plx_i):
        rad = 5
    elif plx_i >= 4:
        rad = 20
    elif 3 <= plx_i < 4:
        rad = 15
    elif 2 <= plx_i < 3:
        rad = 10
    elif 1.5 <= plx_i < 2:
        rad = 7.5
    elif 1 <= plx_i < 1.5:
        rad = 5
    elif 0.5 <= plx_i < 1:
        rad = 2.5
    elif plx_i < 0.5:
        rad = 1.5
    else:
        raise ValueError("Could not define 'rad' values, plx_i is out of bounds")

    rad = rad / 60  # To degrees
    return rad


def sort_by_float_desc_then_alpha(
    strings: list[str], floats: list[float]
) -> tuple[list[str], list[str]]:
    """
    Sorts two parallel lists: one of strings and one of floats. The lists are sorted
    by the float values in descending order, and by the strings in ascending
    alphabetical order if float values are the same. Returns two lists:
    one with the sorted strings and another with the sorted floats as strings.

    Parameters
    ----------
    strings : list
        A list of strings to be sorted alphabetically when float
        values are equal.
    floats : list
        A list of float values to be sorted in descending order.

    Returns
    -------
    tuple
        A tuple containing two lists:
            - A list of strings sorted by the criteria.
            - A list of floats as strings, sorted by the criteria.
    """
    # Combine strings and floats into a list of tuples
    data = list(zip(strings, floats))

    # Sort by float in descending order and string alphabetically in ascending order
    sorted_data = sorted(data, key=lambda x: (-x[1], x[0].lower()))

    # Unzip the sorted data back into two separate lists
    sorted_strings, sorted_floats = zip(*sorted_data)

    return list(sorted_strings), [str(f) for f in sorted_floats]


#####################################################################################
#####################################################################################


#####################################################################################
#####################################################################################
# member_files_updt_UCC


def process_new_OC(
    logging,
    df_UCC: pd.DataFrame,
    frames_path: str,
    max_mag: float,
    frames_data: pd.DataFrame,
    df_gcs: pd.DataFrame,
    manual_pars: pd.DataFrame,
    tree: KDTree,
    UCC_idx: int,
    new_cl: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes a new OC, performs data extraction, runs fastMP, and updates the UCC.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    frames_path : str
        Path to the directory containing Gaia data frames.
    max_mag : float
        Maximum magnitude for Gaia data retrieval.
    frames_data : pd.DataFrame
        DataFrame containing information about Gaia data frames.
    df_gcs : pd.DataFrame
        DataFrame of globular clusters.
    manual_pars : pd.DataFrame
        DataFrame with manual parameters for specific clusters.
    tree : KDTree
        KDTree for nearest neighbor searches in the UCC.
    UCC_idx : int
        Index of the new cluster in the UCC.
    new_cl : pd.Series
        Series representing the new cluster.

    Returns
    -------
    tuple
        A tuple containing:
        - df_membs: DataFrame of cluster members.
        - df_UCC: Updated DataFrame of the UCC.
    """
    # Identify position in the UCC
    fname0 = str(new_cl["fnames"]).split(";")[0]

    # Generate frame
    box_s, plx_min = get_frame(new_cl)

    #
    fix_N_clust = np.nan
    for _, row_manual_p in manual_pars.iterrows():
        if fname0 == row_manual_p["fname"]:
            if row_manual_p["Nmembs"] != "nan":
                fix_N_clust = int(row_manual_p["Nmembs"])
                logging.info(f"Manual N_membs applied: {fix_N_clust}")

            if row_manual_p["box_s"] != "nan":
                box_s = float(row_manual_p["box_s"])
                logging.info(f"Manual box size applied: {box_s}")

    # Get close clusters coords
    centers_ex = get_close_cls(
        float(new_cl["GLON"]),
        float(new_cl["GLAT"]),
        tree,
        box_s,
        UCC_idx,
        df_UCC,
        str(new_cl["dups_fnames"]),
        df_gcs,
    )
    if len(centers_ex) > 0:
        logging.info("WARNING: there are clusters close by:")
        logging.info(
            f"*{new_cl['ID']}: {(new_cl['GLON'], new_cl['GLAT'])}"
            + f" {new_cl['pmRA'], new_cl['pmDE']} ({new_cl['Plx']})"
        )
        for gc in centers_ex:
            logging.info(gc)

    # Request data
    data = gaia_query_frames(
        logging,
        frames_path,
        frames_data,
        box_s,
        plx_min,
        max_mag,
        float(new_cl["RA_ICRS"]),
        float(new_cl["DE_ICRS"]),
    )

    # Extract center coordinates
    lonlat_c = (float(new_cl["GLON"]), float(new_cl["GLAT"]))
    vpd_c = (np.nan, np.nan)
    if not np.isnan(new_cl["pmRA"]):
        vpd_c = (float(new_cl["pmRA"]), float(new_cl["pmDE"]))
    plx_c = np.nan
    if not np.isnan(new_cl["Plx"]):
        plx_c = float(new_cl["Plx"])

    # If the cluster has no PM or Plx center values assigned, run fastMP with fixed
    # (lon, lat) centers
    fixed_centers = False
    if np.isnan(vpd_c[0]) and np.isnan(plx_c):
        fixed_centers = True

    # Process with fastMP
    while True:
        logging.info(f"Fixed centers?: {fixed_centers}")
        probs_all = run_fastMP(
            logging,
            data,
            (float(new_cl["RA_ICRS"]), float(new_cl["DE_ICRS"])),
            vpd_c,
            plx_c,
            fixed_centers,
        )

        xy_c_m, vpd_c_m, plx_c_m = extract_centers(data, probs_all)
        cent_flags = check_centers(xy_c_m, vpd_c_m, plx_c_m, lonlat_c, vpd_c, plx_c)

        if cent_flags == "nnn" or fixed_centers is True:
            break
        else:
            # Re-run with fixed centers
            fixed_centers = True

    xy_c_m, vpd_c_m, plx_c_m = extract_centers(data, probs_all)
    cent_flags = check_centers(xy_c_m, vpd_c_m, plx_c_m, lonlat_c, vpd_c, plx_c)
    logging.info("\nP>0.5={}, cents={}".format((probs_all > 0.5).sum(), cent_flags))

    df_membs, df_field = split_membs_field(data, probs_all)
    C1, C2, C3 = get_classif(df_membs, df_field)
    lon, lat, ra, dec, plx, pmRA, pmDE, Rv, N_Rv, N_50, r_50 = extract_cl_data(df_membs)
    logging.info(f"{new_cl['ID']}: {(lon, lat)} {(ra, dec)} {pmRA, pmDE} ({plx})")

    # Update UCC
    df_UCC.at[UCC_idx, "N_fixed"] = fix_N_clust
    df_UCC.at[UCC_idx, "fixed_cent"] = fixed_centers
    df_UCC.at[UCC_idx, "cent_flags"] = cent_flags
    df_UCC.at[UCC_idx, "C1"] = C1
    df_UCC.at[UCC_idx, "C2"] = C2
    df_UCC.at[UCC_idx, "C3"] = C3
    df_UCC.at[UCC_idx, "GLON_m"] = lon
    df_UCC.at[UCC_idx, "GLAT_m"] = lat
    df_UCC.at[UCC_idx, "RA_ICRS_m"] = ra
    df_UCC.at[UCC_idx, "DE_ICRS_m"] = dec
    df_UCC.at[UCC_idx, "Plx_m"] = plx
    df_UCC.at[UCC_idx, "pmRA_m"] = pmRA
    df_UCC.at[UCC_idx, "pmDE_m"] = pmDE
    df_UCC.at[UCC_idx, "Rv_m"] = Rv
    df_UCC.at[UCC_idx, "N_Rv"] = N_Rv
    df_UCC.at[UCC_idx, "N_50"] = N_50
    df_UCC.at[UCC_idx, "r_50"] = r_50
    logging.info(f"UCC entry for {new_cl['ID']} updated")

    return df_membs, df_UCC


def get_frame(cl: pd.Series) -> tuple[float, float]:
    """
    Determines the frame size and minimum parallax for data retrieval based on cluster
    properties.

    Parameters
    ----------
    cl : pd.Series
        Series representing the cluster.

    Returns
    -------
    tuple
        A tuple containing:
        - box_s_eq (float): Size of the box to query (in degrees).
        - plx_min (float): Minimum parallax value for data retrieval.
    """
    if not np.isnan(cl["Plx"]):
        c_plx = cl["Plx"]
    else:
        c_plx = None

    if c_plx is None:
        box_s_eq = 0.5
    else:
        if c_plx > 10:
            box_s_eq = 25
        elif c_plx > 8:
            box_s_eq = 20
        elif c_plx > 6:
            box_s_eq = 15
        elif c_plx > 5:
            box_s_eq = 10
        elif c_plx > 4:
            box_s_eq = 7.5
        elif c_plx > 2:
            box_s_eq = 5
        elif c_plx > 1.5:
            box_s_eq = 3
        elif c_plx > 1:
            box_s_eq = 2
        elif c_plx > 0.75:
            box_s_eq = 1.5
        elif c_plx > 0.5:
            box_s_eq = 1
        elif c_plx > 0.25:
            box_s_eq = 0.75
        elif c_plx > 0.1:
            box_s_eq = 0.5
        else:
            box_s_eq = 0.25  # 15 arcmin

    if "Ryu" in cl["ID"]:
        box_s_eq = 10 / 60

    # Filter by parallax if possible
    plx_min = -2
    if c_plx is not None:
        if c_plx > 15:
            plx_p = 5
        elif c_plx > 4:
            plx_p = 2
        elif c_plx > 2:
            plx_p = 1
        elif c_plx > 1:
            plx_p = 0.7
        else:
            plx_p = 0.6
        plx_min = c_plx - plx_p

    return box_s_eq, plx_min


def get_close_cls(
    x: float,
    y: float,
    tree: KDTree,
    box_s: float,
    idx: int,
    df_UCC: pd.DataFrame,
    dups_fnames: str,
    df_gcs: pd.DataFrame,
) -> list[str]:
    """
    Identifies clusters and globular clusters (GCs) close to the specified coordinates.

    Parameters
    ----------
    x : float
        Galactic longitude of the cluster.
    y : float
        Galactic latitude of the cluster.
    tree : KDTree
        KDTree for nearest neighbor searches in the UCC.
    box_s : float
        Size of the box to query (in degrees).
    idx : int
        Index of the cluster in the UCC.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    dups_fnames : str
        String of semicolon-separated duplicate filenames.
    df_gcs : pd.DataFrame
        DataFrame of globular clusters.

    Returns
    -------
    list
        A list of strings, each representing a nearby cluster or GC with its
        coordinates and properties.
    """

    # Radius that contains the entire frame
    rad = np.sqrt(2 * (box_s / 2) ** 2)
    # Indexes to the closest clusters in XY
    ex_cls_idx = list(tree.query_ball_point([x, y], rad))
    # Remove self cluster
    del ex_cls_idx[ex_cls_idx.index(idx)]

    duplicate_cls = []
    if str(dups_fnames) != "nan":
        duplicate_cls = dups_fnames.split(";")

    centers_ex = []
    for i in ex_cls_idx:
        # Check if this close cluster is identified as a probable duplicate
        # of this cluster. If it is, do not add it to the list of extra
        # clusters in the frame
        skip_cl = False
        if duplicate_cls:
            for dup_fname_i in str(df_UCC["fnames"][i]).split(";"):
                if dup_fname_i in duplicate_cls:
                    skip_cl = True
                    break
            if skip_cl:
                continue

        # If the cluster does not contain PM or Plx information, check its
        # distance in (lon, lat) with the main cluster. If the distance locates
        # this cluster within 0.75 of the frame's radius (i.e.: within the
        # expected region of the main cluster), don't store it for removal.
        #
        # This prevents clusters with no PM|Plx data from disrupting
        # neighboring clusters (e.g.: NGC 2516 disrupted by FSR 1479) and
        # at the same time removes more distant clusters that disrupt the
        # number of members estimation process in fastMP
        if np.isnan(df_UCC["pmRA"][i]) or np.isnan(df_UCC["Plx"][i]):
            xy_dist = np.sqrt(
                (x - df_UCC["GLON"][i]) ** 2 + (y - df_UCC["GLAT"][i]) ** 2
            )
            if xy_dist < 0.75 * rad:
                continue

        ex_cl_dict = f"{df_UCC['ID'][i]}: {(df_UCC['GLON'][i], df_UCC['GLAT'][i])}"
        if not np.isnan(df_UCC["pmRA"][i]):
            ex_cl_dict += f" {df_UCC['pmRA'][i], df_UCC['pmDE'][i]}"
        if not np.isnan(df_UCC["Plx"][i]):
            ex_cl_dict += f" ({df_UCC['Plx'][i]})"

        centers_ex.append(ex_cl_dict)

    # Add closest GC
    glon, glat = df_UCC["GLON"][idx], df_UCC["GLAT"][idx]
    gc_d = np.sqrt(
        (glon - df_gcs["GLON"].values) ** 2 + (glat - df_gcs["GLAT"].values) ** 2
    )
    for i, gc_di in enumerate(gc_d):
        if gc_di < rad:
            ex_cl_dict = (
                f"{df_gcs['ID'][i]}: {(df_gcs['GLON'][i], df_gcs['GLAT'][i])}"
                + f" {df_gcs['pmRA'][i], df_gcs['pmDE'][i]}"
                + f" ({df_gcs['Plx'][i]})"
            )
            centers_ex.append(ex_cl_dict)

    return centers_ex


def run_fastMP(
    logging,
    field_df: pd.DataFrame,
    radec_c: tuple[float, float],
    pms_c: tuple[float, float],
    plx_c: float,
    fixed_centers: bool,
) -> np.ndarray:
    """
    Runs the fastMP algorithm to estimate membership probabilities.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    field_df : pd.DataFrame
        DataFrame of the field stars.
    radec_c : tuple
        Center coordinates (RA, Dec) for fastMP.
    pms_c : tuple
        Center proper motion (pmRA, pmDE) for fastMP.
    plx_c : float
        Center parallax for fastMP.
    fixed_centers : bool
        Boolean indicating whether to use fixed centers.

    Returns
    -------
    np.ndarray
        Array of membership probabilities.
    """
    my_field = asteca.cluster(
        obs_df=field_df,
        ra="RA_ICRS",
        dec="DE_ICRS",
        pmra="pmRA",
        pmde="pmDE",
        plx="Plx",
        e_pmra="e_pmRA",
        e_pmde="e_pmDE",
        e_plx="e_Plx",
    )

    # Estimate the cluster's center coordinates
    my_field.get_center(radec_c=radec_c)
    if fixed_centers:
        my_field.radec_c = radec_c
        if not np.isnan(pms_c[0]):
            my_field.pms_c = pms_c
        if not np.isnan(plx_c):
            my_field.plx_c = plx_c
    logging.info("Center coordinates used:")
    logging.info(my_field.radec_c)
    logging.info(my_field.pms_c)
    logging.info(my_field.plx_c)

    # Estimate the number of cluster members
    my_field.get_nmembers()

    # Define a ``membership`` object
    memb = asteca.membership(my_field)

    # Run ``fastmp`` method
    probs_fastmp = memb.fastmp(fixed_centers=fixed_centers)

    return probs_fastmp


def extract_centers(
    data: pd.DataFrame,
    probs_all: np.ndarray,
    N_membs_min: int = 25,
    prob_cut: float = 0.5,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    Extracts the cluster center coordinates, proper motion, and parallax from
    high-probability members.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing cluster data.
    probs_all : np.ndarray
        Array of membership probabilities.
    N_membs_min : int, optional
        Minimum number of members to use for center estimation. Default is 25.
    prob_cut : float, optional
        Probability value to select members. Default is 0.5.

    Returns
    -------
    tuple
        A tuple containing:
        - xy_c_m: Center coordinates (lon, lat).
        - vpd_c_m: Center proper motion (pmRA, pmDE).
        - plx_c_m: Center parallax.
    """

    # Select high-quality members
    msk = probs_all > prob_cut
    # Use at least 'N_membs_min' stars
    if msk.sum() < N_membs_min:
        idx = np.argsort(probs_all)[::-1][:N_membs_min]
        msk = np.full(len(probs_all), False)
        msk[idx] = True

    # Centers of selected members
    xy_c_m = np.nanmedian([np.array(data["GLON"])[msk], np.array(data["GLAT"])[msk]], 1)
    vpd_c_m = np.nanmedian(
        [np.array(data["pmRA"])[msk], np.array(data["pmDE"])[msk]], 1
    )
    plx_c_m = np.nanmedian(np.array(data["Plx"])[msk])

    # pyright issue due to: https://github.com/numpy/numpy/issues/28076
    return xy_c_m, vpd_c_m, plx_c_m


def split_membs_field(
    data: pd.DataFrame,
    probs_all: np.ndarray,
    prob_cut: float = 0.5,
    N_membs_min: int = 25,
    perc_cut: int = 95,
    N_perc: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into member and field star DataFrames based on membership
    probabilities.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing cluster data.
    probs_all : np.ndarray
        Array of membership probabilities.
    prob_cut : float, optional
        Probability threshold for considering a star a member. Default is 0.5.
    N_membs_min : int, optional
        Minimum number of members to use for filtering. Default is 25.
    perc_cut : int, optional
        Percentile to use for distance-based filtering. Default is 95.
    N_perc : int, optional
        Number of times the percentile distance to use for filtering. Default is 2.

    Returns
    -------
    tuple
        A tuple containing:
        - df_membs: DataFrame of cluster members.
        - df_field: DataFrame of field stars.
    """
    # This first filter removes stars beyond 2 times the 95th percentile
    # of the most likely members

    # Select most likely members
    msk_membs = probs_all >= prob_cut
    if msk_membs.sum() < N_membs_min:
        # Select the 'N_membs_min' stars with the largest probabilities
        idx = np.argsort(probs_all)[::-1][:N_membs_min]
        msk_membs = np.full(len(probs_all), False)
        msk_membs[idx] = True

    # Find xy filter
    xy = np.array([data["GLON"].values, data["GLAT"].values]).T
    xy_c = np.nanmedian(xy[msk_membs], 0)
    xy_dists = cdist(xy, np.array([xy_c])).T[0]
    x_dist = abs(xy[:, 0] - xy_c[0])
    y_dist = abs(xy[:, 1] - xy_c[1])
    # 2x95th percentile XY mask
    xy_95 = np.percentile(xy_dists[msk_membs], perc_cut)
    xy_rad = xy_95 * N_perc
    msk_rad = (x_dist <= xy_rad) & (y_dist <= xy_rad)

    # Add a minimum probability mask to ensure that all stars with P>prob_min
    # are included
    msk_pmin = probs_all >= prob_cut

    # Combine masks with logical OR
    msk = msk_rad | msk_pmin

    # Generate filtered combined dataframe
    data["probs"] = np.round(probs_all, 5)
    # This dataframe contains both members and a selected portion of
    # field stars
    df_comb = data.loc[msk]
    df_comb.reset_index(drop=True, inplace=True)

    # Split into members and field, now using the filtered dataframe
    msk_membs = df_comb["probs"] > prob_cut
    if msk_membs.sum() < N_membs_min:
        idx = np.argsort(df_comb["probs"].values)[::-1][:N_membs_min]
        msk_membs = np.full(len(df_comb["probs"]), False)
        msk_membs[idx] = True
    df_membs, df_field = df_comb[msk_membs], df_comb[~msk_membs]

    return df_membs, df_field


def extract_cl_data(
    df_membs: pd.DataFrame, prob_cut: float = 0.5
) -> tuple[float, float, float, float, float, float, float, float, int, int, float]:
    """
    Extracts cluster parameters from the member DataFrame.

    Parameters
    ----------
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    prob_cut : float, optional
        Probability threshold for calculating N_50. Default is 0.5.

    Returns
    -------
    tuple
        A tuple containing:
        - lon: Median galactic longitude.
        - lat: Median galactic latitude.
        - ra: Median right ascension.
        - dec: Median declination.
        - plx: Median parallax.
        - pmRA: Median proper motion in RA.
        - pmDE: Median proper motion in DE.
        - RV: Median radial velocity.
        - N_Rv: Number of stars with RV measurements.
        - N_50: Number of stars with membership probability above prob_cut.
        - r_50: Radius containing half the members.
    """
    N_50 = int((df_membs["probs"] >= prob_cut).sum())
    lon, lat = np.nanmedian(df_membs["GLON"]), np.nanmedian(df_membs["GLAT"])
    ra, dec = np.nanmedian(df_membs["RA_ICRS"]), np.nanmedian(df_membs["DE_ICRS"])
    plx = np.nanmedian(df_membs["Plx"])
    pmRA, pmDE = np.nanmedian(df_membs["pmRA"]), np.nanmedian(df_membs["pmDE"])
    RV, N_Rv = np.nan, 0
    if not np.isnan(df_membs["RV"].values).all():
        RV = np.nanmedian(df_membs["RV"])
        N_Rv = int(len(df_membs["RV"]) - np.isnan(df_membs["RV"].values).sum())
    lon, lat = round(lon, 3), round(lat, 3)
    ra, dec = round(ra, 3), round(dec, 3)
    plx = round(plx, 3)
    pmRA, pmDE = round(pmRA, 3), round(pmDE, 3)
    RV = round(RV, 3)

    # Radius that contains half the members
    xy = np.array([df_membs["GLON"].values, df_membs["GLAT"].values]).T
    xy_dists = cdist(xy, np.array([[lon, lat]])).T[0]
    r50_idx = np.argsort(xy_dists)[int(len(df_membs) / 2)]
    r_50 = xy_dists[r50_idx]
    # To arcmin
    r_50 = float(round(r_50 * 60.0, 1))

    # pyright issue due to: https://github.com/numpy/numpy/issues/28076
    return lon, lat, ra, dec, plx, pmRA, pmDE, RV, N_Rv, N_50, r_50


def save_cl_datafile(
    logging, temp_database_folder: str, cl: pd.Series, df_membs: pd.DataFrame
) -> None:
    """
    Saves the cluster member data to a Parquet file.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    temp_database_folder : str
        Path to the temporary database folder.
    cl : pd.Series
        Series representing the cluster.
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    """
    fname0 = str(cl["fnames"]).split(";")[0]
    quad = cl["quad"] + "/"

    # Order by probabilities
    df_membs = df_membs.sort_values("probs", ascending=False)

    out_fname = temp_database_folder + quad + fname0 + ".parquet"
    df_membs.to_parquet(out_fname, index=False)
    logging.info(f"Saved file to: {out_fname}")


#####################################################################################
#####################################################################################


#####################################################################################
#####################################################################################


def gaia_query_frames(
    logging,
    frames_path: str,
    fdata: pd.DataFrame,
    box_s_eq: float,
    plx_min: float,
    max_mag: float,
    c_ra: float,
    c_dec: float,
) -> pd.DataFrame:
    """
    Queries Gaia data frames based on specified parameters and returns a combined
    DataFrame.

    ******** IMPORTANT ********
    Clusters that wrap around the edges of the (ra, dec) coordinates are not
    still properly process; e.g.: Blanco 1

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    frames_path : str
        Path to the directory containing Gaia data frames.
    fdata : pd.DataFrame
        DataFrame containing information about Gaia data frames.
    box_s_eq : float
        Size of the box to query (in degrees).
    plx_min : float
        Minimum parallax value for data retrieval.
    max_mag : float
        Maximum magnitude for data retrieval.
    c_ra : float
        Central right ascension for the query.
    c_dec : float
        Central declination for the query.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the combined Gaia data.
    """
    # Gaia EDR3 zero points. Sigmas are already squared here.
    Zp_G, sigma_ZG_2 = 25.6873668671, 0.00000759
    Zp_BP, sigma_ZBP_2 = 25.3385422158, 0.000007785
    Zp_RP, sigma_ZRP_2 = 24.7478955012, 0.00001428

    logging.info(
        "  ({:.3f}, {:.3f}); Box size: {:.2f}, Plx min: {:.2f}".format(
            c_ra, c_dec, box_s_eq, plx_min
        )
    )

    c_ra_l = [c_ra]
    if c_ra - box_s_eq < 0:
        logging.info("Split frame, c_ra + 360")
        c_ra_l.append(c_ra + 360)
    if c_ra > box_s_eq > 360:
        logging.info("Split frame, c_ra - 360")
        c_ra_l.append(c_ra - 360)

    dicts = []
    for c_ra in c_ra_l:
        data_in_files, xmin_cl, xmax_cl, ymin_cl, ymax_cl = findFrames(
            logging, c_ra, c_dec, box_s_eq, fdata
        )

        if len(data_in_files) == 0:
            continue

        all_frames = query(
            logging,
            Zp_G,
            c_ra,
            c_dec,
            box_s_eq,
            frames_path,
            max_mag,
            data_in_files,
            xmin_cl,
            xmax_cl,
            ymin_cl,
            ymax_cl,
            plx_min,
        )

        dicts.append(all_frames)

    if len(dicts) > 1:
        # Combine
        all_frames = (
            pd.concat([dicts[0], dicts[1]]).drop_duplicates().reset_index(drop=True)
        )
    else:
        all_frames = dicts[0]

    all_frames = uncertMags(
        Zp_G, Zp_BP, Zp_RP, sigma_ZG_2, sigma_ZBP_2, sigma_ZRP_2, all_frames
    )
    all_frames = all_frames.drop(columns=["FG", "e_FG", "FBP", "e_FBP", "FRP", "e_FRP"])

    return all_frames


def findFrames(
    logging, c_ra: float, c_dec: float, box_s_eq: float, fdata: pd.DataFrame
) -> tuple[list[str], float, float, float, float]:
    """
    Identifies Gaia data frames that overlap with the specified region.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    c_ra : float
        Central right ascension of the region.
    c_dec : float
        Central declination of the region.
    box_s_eq : float
        Size of the box to query (in degrees).
    fdata : pd.DataFrame
        DataFrame containing information about Gaia data frames.

    Returns
    -------
    tuple
        A tuple containing:
        - data_in_files: List of filenames of overlapping frames.
        - xmin_cl: Minimum RA of the cluster region.
        - xmax_cl: Maximum RA of the cluster region.
        - ymin_cl: Minimum Dec of the cluster region.
        - ymax_cl: Maximum Dec of the cluster region.
    """
    # These are the points that determine the range of *all* the frames
    ra_min, ra_max = fdata["ra_min"].values, fdata["ra_max"].values
    dec_min, dec_max = fdata["dec_min"].values, fdata["dec_max"].values

    # frame == 'galactic':
    box_s_eq = np.sqrt(2) * box_s_eq
    # Correct size in RA
    box_s_x = box_s_eq / np.cos(np.deg2rad(c_dec))

    xl, yl = box_s_x * 0.5, box_s_eq * 0.5

    # Limits of the cluster's region in Equatorial
    xmin_cl, xmax_cl = c_ra - xl, c_ra + xl
    ymin_cl, ymax_cl = c_dec - yl, c_dec + yl

    frame_intersec = np.full(len(fdata), False)
    # Identify which frames contain the cluster region
    l2 = (xmin_cl, ymax_cl)  # Top left
    r2 = (xmax_cl, ymin_cl)  # Bottom right
    for i, xmin_fr_i in enumerate(ra_min):
        l1 = (xmin_fr_i, dec_max[i])  # Top left
        r1 = (ra_max[i], dec_min[i])  # Bottom right
        frame_intersec[i] = doOverlap(l1, r1, l2, r2)

    data_in_files = list(fdata[frame_intersec]["filename"])
    logging.info(f"  Cluster is present in {len(data_in_files)} frames")

    return data_in_files, xmin_cl, xmax_cl, ymin_cl, ymax_cl


def doOverlap(
    l1: tuple[float, float],
    r1: tuple[float, float],
    l2: tuple[float, float],
    r2: tuple[float, float],
) -> bool:
    """
    Checks if two rectangles defined by their top-left and bottom-right coordinates
    overlap.

    Source: https://www.geeksforgeeks.org/find-two-rectangles-overlap/

    Parameters
    ----------
    l1 : tuple
        Top-left coordinate of the first rectangle (x, y).
    r1 : tuple
        Bottom-right coordinate of the first rectangle (x, y).
    l2 : tuple
        Top-left coordinate of the second rectangle (x, y).
    r2 : tuple
        Bottom-right coordinate of the second rectangle (x, y).

    Returns
    -------
    bool
        True if the rectangles overlap, False otherwise.
    """
    min_x1, max_y1 = l1
    max_x1, min_y1 = r1
    min_x2, max_y2 = l2
    max_x2, min_y2 = r2
    # If one rectangle is on left side of other
    if min_x1 > max_x2 or min_x2 > max_x1:
        return False
    # If one rectangle is above other
    if min_y1 > max_y2 or min_y2 > max_y1:
        return False
    return True


def query(
    logging,
    Zp_G: float,
    c_ra: float,
    c_dec: float,
    box_s_eq: float,
    frames_path: str,
    max_mag: float,
    data_in_files: list[str],
    xmin_cl: float,
    xmax_cl: float,
    ymin_cl: float,
    ymax_cl: float,
    plx_min: float,
) -> pd.DataFrame:
    """
    Queries individual Gaia data frames and combines the results.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    Zp_G : float
        Zero point for the G band.
    c_ra : float
        Central right ascension of the region.
    c_dec : float
        Central declination of the region.
    box_s_eq : float
        Size of the box to query (in degrees).
    frames_path : str
        Path to the directory containing Gaia data frames.
    max_mag : float
        Maximum magnitude for data retrieval.
    data_in_files : list
        List of filenames of overlapping frames.
    xmin_cl : float
        Minimum RA of the cluster region.
    xmax_cl : float
        Maximum RA of the cluster region.
    ymin_cl : float
        Minimum Dec of the cluster region.
    ymax_cl : float
        Maximum Dec of the cluster region.
    plx_min : float
        Minimum parallax value for data retrieval.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the combined Gaia data from the queried frames.
    """
    # Mag (flux) filter
    min_G_flux = 10 ** ((max_mag - Zp_G) / (-2.5))

    all_frames = []
    for i, file in enumerate(data_in_files):
        data = pd.read_parquet(frames_path + file)

        mx = (data["ra"] >= xmin_cl) & (data["ra"] <= xmax_cl)
        my = (data["dec"] >= ymin_cl) & (data["dec"] <= ymax_cl)
        m_plx = data["parallax"] > plx_min
        m_gmag = data["phot_g_mean_flux"] > min_G_flux
        msk = mx & my & m_plx & m_gmag

        logging.info(f"{i + 1}, {file} contains {msk.sum()} cluster stars")
        if msk.sum() == 0:
            continue

        all_frames.append(data[msk])
    all_frames = pd.concat(all_frames)

    logging.info(f"  {len(all_frames)} stars retrieved")

    c_ra, c_dec = c_ra, c_dec
    box_s_h = box_s_eq * 0.5
    gal_cent = radec2lonlat(c_ra, c_dec)

    if all_frames["l"].max() - all_frames["l"].min() > 180:
        logging.info("Frame wraps around 360 in longitude. Fixing..")

        lon = np.array(all_frames["l"])
        if gal_cent[0] > 180:
            msk = lon < 180
            lon[msk] += 360
        else:
            msk = lon > 180
            lon[msk] -= 360
        all_frames["l"] = lon

    xmin_cl, xmax_cl = gal_cent[0] - box_s_h, gal_cent[0] + box_s_h
    ymin_cl, ymax_cl = gal_cent[1] - box_s_h, gal_cent[1] + box_s_h
    mx = (all_frames["l"] >= xmin_cl) & (all_frames["l"] <= xmax_cl)
    my = (all_frames["b"] >= ymin_cl) & (all_frames["b"] <= ymax_cl)
    msk = mx & my
    all_frames = pd.DataFrame(all_frames[msk])

    all_frames = all_frames.rename(
        columns={
            "source_id": "Source",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "parallax": "Plx",
            "parallax_error": "e_Plx",
            "pmra": "pmRA",
            "pmra_error": "e_pmRA",
            "b": "GLAT",
            "pmdec": "pmDE",
            "pmdec_error": "e_pmDE",
            "l": "GLON",
            "phot_g_mean_flux": "FG",
            "phot_g_mean_flux_error": "e_FG",
            "phot_bp_mean_flux": "FBP",
            "phot_bp_mean_flux_error": "e_FBP",
            "phot_rp_mean_flux": "FRP",
            "phot_rp_mean_flux_error": "e_FRP",
            "radial_velocity": "RV",
            "radial_velocity_error": "e_RV",
        }
    )

    logging.info(f"  {len(all_frames)} stars final")
    return all_frames


def uncertMags(
    Zp_G: float,
    Zp_BP: float,
    Zp_RP: float,
    sigma_ZG_2: float,
    sigma_ZBP_2: float,
    sigma_ZRP_2: float,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates magnitudes and uncertainties in the G, BP, and RP bands.

    Gaia DR3 zero points: https://www.cosmos.esa.int/web/gaia/dr3-passbands
    "The GBP (blue curve), G (green curve) and GRP (red curve) passbands are
    applicable to both Gaia Early Data Release 3 as well as to the full Gaia
    Data Release 3"

    Parameters
    ----------
    Zp_G : float
        Zero point for the G band.
    Zp_BP : float
        Zero point for the BP band.
    Zp_RP : float
        Zero point for the RP band.
    sigma_ZG_2 : float
        Variance of the zero point for the G band.
    sigma_ZBP_2 : float
        Variance of the zero point for the BP band.
    sigma_ZRP_2 : float
        Variance of the zero point for the RP band.
    data : pd.DataFrame
        DataFrame containing Gaia data with flux measurements.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns for Gmag, BP-RP, e_Gmag, and e_BP-RP.
    """
    I_G, e_IG = np.array(data["FG"]), np.array(data["e_FG"])
    I_BP, e_IBP = np.array(data["FBP"]), np.array(data["e_FBP"])
    I_RP, e_IRP = np.array(data["FRP"]), np.array(data["e_FRP"])

    data["Gmag"] = Zp_G + -2.5 * np.log10(I_G)
    BPmag = Zp_BP + -2.5 * np.log10(I_BP)
    RPmag = Zp_RP + -2.5 * np.log10(I_RP)
    data["BP-RP"] = BPmag - RPmag

    e_G = np.sqrt(sigma_ZG_2 + 1.179 * (e_IG / I_G) ** 2)
    data["e_Gmag"] = e_G
    e_BP = np.sqrt(sigma_ZBP_2 + 1.179 * (e_IBP / I_BP) ** 2)
    e_RP = np.sqrt(sigma_ZRP_2 + 1.179 * (e_IRP / I_RP) ** 2)
    data["e_BP-RP"] = np.sqrt(e_BP**2 + e_RP**2)

    return data


#####################################################################################
#####################################################################################


#####################################################################################
#####################################################################################


def get_classif(
    df_membs: pd.DataFrame, df_field: pd.DataFrame
) -> tuple[float, float, str]:
    """
    Calculates classification metrics C1, C2, and C3 for a cluster.

    Parameters
    ----------
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    df_field : pd.DataFrame
        DataFrame of field stars.

    Returns
    -------
    tuple
        A tuple containing:
        - C1: Photometric classification metric.
        - C2: Density-based classification metric.
        - C3: Combined classification string (e.g., "AA", "BC").
    """
    C1 = lkl_phot(df_membs, df_field)
    C2 = dens_ratio(df_membs, df_field)

    def ABCD_classif(CC):
        """Obtain 'ABCD' classification"""
        if CC >= 0.75:
            return "A"
        elif CC >= 0.5:
            return "B"
        elif CC >= 0.25:
            return "C"
        else:
            return "D"

    C3 = ABCD_classif(C1) + ABCD_classif(C2)

    return round(C1, 2), round(C2, 2), C3


def lkl_phot(
    df_membs: pd.DataFrame, df_field: pd.DataFrame, max_mag_perc: int = 90
) -> float:
    """
    Calculates the photometric likelihood ratio (C1) for cluster classification.

    Parameters
    ----------
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    df_field : pd.DataFrame
        DataFrame of field stars.
    max_mag_perc : int, optional
        Percentile of maximum magnitude to consider. Default is 90.

    Returns
    -------
    float
        The C1 classification metric.
    """
    x_cl, y_cl = df_membs["Gmag"].values, df_membs["BP-RP"].values
    msk_nan = np.isnan(x_cl) | np.isnan(y_cl)
    x_cl, y_cl = x_cl[~msk_nan], y_cl[~msk_nan]

    #
    max_mag = np.percentile(x_cl, max_mag_perc)
    msk_mag = x_cl <= max_mag
    x_cl, y_cl = x_cl[msk_mag], y_cl[msk_mag]

    x_fl, y_fl = np.array(df_field["Gmag"]), np.array(df_field["BP-RP"])
    msk_mag = x_fl <= max_mag
    x_fl, y_fl = x_fl[msk_mag], y_fl[msk_mag]
    msk_nan = np.isnan(x_fl) | np.isnan(y_fl)
    x_fl, y_fl = x_fl[~msk_nan], y_fl[~msk_nan]

    N_membs, N_field = len(x_cl), len(x_fl)

    if N_field <= N_membs:
        # Not enough field stars
        warnings.warn(
            "Not enough field stars to obtain C1 class, will result in 'D' class"
        )
        return np.nan
    elif N_field < 1.5 * N_membs:
        runs = 10
    elif N_field < 2 * N_membs:
        runs = 25
    elif N_field < 5 * N_membs:
        runs = 75
    else:
        runs = 100

    idx = np.arange(0, N_field)
    prep_clust_cl = prep_data(x_cl, y_cl)
    lkl_cl_max = tremmel([x_cl, y_cl], prep_clust_cl)

    pv_cl, pv_fr = [], []
    for _ in range(runs):
        # Sample two field regions
        msk1 = np.random.choice(idx, N_membs, replace=False)
        field1 = [x_fl[msk1], y_fl[msk1]]
        msk2 = np.random.choice(idx, N_membs, replace=False)
        field2 = [x_fl[msk2], y_fl[msk2]]

        prep_clust_f1 = prep_data(field1[0], field1[1])
        lkl_fl_max = tremmel(field1, prep_clust_f1)

        lkl_cl = tremmel(field1, prep_clust_cl)
        lkl_fl = tremmel(field2, prep_clust_f1)

        pv_cl.append(lkl_cl - lkl_cl_max)
        pv_fr.append(lkl_fl - lkl_fl_max)

    C_lkl = KDEoverlap(pv_cl, pv_fr)
    return C_lkl


def prep_data(mag: np.ndarray, col: np.ndarray) -> list:
    """
    Prepares data for photometric likelihood calculation by creating histograms and
    identifying bins with stars.

    Parameters
    ----------
    mag : np.ndarray
        Array of magnitudes.
    col : np.ndarray
        Array of colors.

    Returns
    -------
    list
        A list containing:
        - bin_edges: Bin edges for each dimension.
        - cl_histo_f_z: Flattened histogram of observed cluster with empty bins removed.
        - cl_z_idx: Indices of bins where stars were observed in the cluster.
    """

    # Obtain bin edges for each dimension, defining a grid.
    bin_edges = bin_edges_f(mag, col)

    # Histogram for observed cluster.
    cl_histo = np.histogramdd([mag, col], bins=bin_edges)[0]

    # Flatten N-dimensional histograms.
    cl_histo_f = np.array(cl_histo).ravel()

    # Index of bins where stars were observed
    cl_z_idx = cl_histo_f != 0

    # Remove all bins where n_i=0 (no observed stars)
    cl_histo_f_z = cl_histo_f[cl_z_idx]

    return [bin_edges, cl_histo_f_z, cl_z_idx]


def bin_edges_f(
    mag: np.ndarray, col: np.ndarray, min_bins: int = 2, max_bins: int = 50
) -> list[np.ndarray]:
    """
    Calculates bin edges for magnitude and color, ensuring a minimum and maximum number
    of bins.

    Parameters
    ----------
    mag : np.ndarray
        Array of magnitudes.
    col : np.ndarray
        Array of colors.
    min_bins : int, optional
        Minimum number of bins per dimension. Default is 2.
    max_bins : int, optional
        Maximum number of bins per dimension. Default is 50.

    Returns
    -------
    list
        A list of arrays, where each array contains the bin edges for a dimension.
    """
    bin_edges = []
    b_num = int(round(max(2, (max(mag) - min(mag)) / 1.0)))
    bin_edges.append(np.histogram(mag, bins=b_num)[1])
    b_num = int(round(max(2, (max(col) - min(col)) / 0.5)))
    bin_edges.append(np.histogram(col, bins=b_num)[1])

    # Impose a minimum of 'min_bins' cells per dimension. The number of bins
    # is the number of edges minus 1.
    for i, be in enumerate(bin_edges):
        N_bins = len(be) - 1
        if N_bins < min_bins:
            bin_edges[i] = np.linspace(be[0], be[-1], min_bins + 1)

    # Impose a maximum of 'max_bins' cells per dimension.
    for i, be in enumerate(bin_edges):
        N_bins = len(be) - 1
        if N_bins > max_bins:
            bin_edges[i] = np.linspace(be[0], be[-1], max_bins)

    return bin_edges


def tremmel(field: list[np.ndarray], prep_clust: list) -> float:
    """
    Calculates the Tremmel et al. (2013) likelihood ratio for a synthetic cluster
    as defined in Eq 10 with v_{i,j}=1.

    Parameters
    ----------
    field : list
        List of arrays representing the synthetic cluster data (e.g.,
        [magnitudes, colors]).
    prep_clust : list
        Prepared data for the observed cluster, including bin edges,
        flattened histogram, and indices of non-empty bins.

    Returns
    -------
    float
        The Tremmel likelihood ratio.
    """
    # Observed cluster's data.
    bin_edges, cl_histo_f_z, cl_z_idx = prep_clust

    # Histogram of the synthetic cluster, using the bin edges calculated
    # with the observed cluster.
    syn_histo = np.histogramdd(field, bins=bin_edges)[0]

    # Flatten N-dimensional histogram.
    syn_histo_f = syn_histo.ravel()
    # Remove all bins where n_i = 0 (no observed stars).
    syn_histo_f_z = syn_histo_f[cl_z_idx]

    SumLogGamma = np.sum(
        loggamma(cl_histo_f_z + syn_histo_f_z + 0.5) - loggamma(syn_histo_f_z + 0.5)
    )

    # M = field.shape[0]
    # ln(2) ~ 0.693
    tremmel_lkl = SumLogGamma - 0.693 * len(field[0])

    return tremmel_lkl


def KDEoverlap(p_vals_cl: list[float], p_vals_fr: list[float]) -> float:
    """
    Calculates the overlap between two kernel density estimations (KDEs).

    Parameters
    ----------
    p_vals_cl : list
        List of p-values for the cluster.
    p_vals_fr : list
        List of p-values for the field.

    Returns
    -------
    float
        The overlap probability for the cluster.
    """
    if (np.median(p_vals_cl) - np.median(p_vals_fr)) > 2 * np.std(p_vals_cl):
        return 0.0

    def y_pts(pt: float) -> float:
        y_pt = min(kcl(pt), kfr(pt))
        return y_pt

    kcl, kfr = gaussian_kde(p_vals_cl), gaussian_kde(p_vals_fr)

    all_pvals = np.concatenate([p_vals_cl, p_vals_fr])
    pmin, pmax = all_pvals.min(), all_pvals.max()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlap = quad(y_pts, pmin, pmax)[0]

    # Probability value for the cluster.
    prob_cl = 1.0 - overlap

    return prob_cl


def dens_ratio(
    df_membs: pd.DataFrame,
    df_field: pd.DataFrame,
    perc: int = 95,
    N_neigh: int = 10,
    N_max: int = 1000,
    norm_v: int = 5,
) -> float:
    """
    Calculates the density ratio (C2) for cluster classification.

    Parameters
    ----------
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    df_field : pd.DataFrame
        DataFrame of field stars.
    perc : int, optional
        Percentile for radius calculation. Default is 95.
    N_neigh : int, optional
        Number of nearest neighbors for distance calculation. Default is 10.
    N_max : int, optional
        Maximum number of field stars to consider. Default is 1000.
    norm_v : int, optional
        Normalization value for the density ratio. Default is 5.

    Returns
    -------
    float
        The C2 density ratio.
    """
    # Obtain the median distance to the 'N_neigh' closest neighbors in 5D
    # for each member
    arr = np.array(df_membs[["GLON", "GLAT", "pmRA", "pmDE", "Plx"]])
    tree = KDTree(arr)
    dists = np.array(tree.query(arr, min(arr.shape[0], N_neigh))[0])
    med_d_membs = np.median(dists[:, 1:])

    # Radius that contains 'perc' of the members for the coordinates
    xy = np.array([df_membs["GLON"].values, df_membs["GLAT"].values]).T
    xy_c = np.nanmedian(xy, 0)
    xy_dists = cdist(xy, np.array([xy_c])).T[0]
    rad = np.percentile(xy_dists, perc)

    # Select field stars within the above radius from the member's center
    xy = np.array([df_field["GLON"].values, df_field["GLAT"].values]).T
    xy_dists = cdist(xy, np.array([xy_c])).T[0]
    msk = np.arange(0, len(xy_dists))[xy_dists < rad]
    if len(msk) > N_max:
        step = max(1, int(len(msk) / N_max))
        msk = msk[::step]

    # Median distance to the 'N_neigh' closest neighbours in 5D for field stars
    arr = np.array(df_field[["GLON", "GLAT", "pmRA", "pmDE", "Plx"]])
    if len(df_field) > 10:
        arr = arr[msk, :]
    tree = KDTree(arr)
    dists = np.array(tree.query(arr, min(arr.shape[0], N_neigh))[0])
    med_d_field = np.median(dists[:, 1:])

    d_ratio = min(med_d_field / med_d_membs, norm_v) / norm_v

    return d_ratio


#####################################################################################
#####################################################################################
# check_UCC_versions


def fnames_checker(df_UCC: pd.DataFrame) -> None:
    """
    Ensure that filenames in the DataFrame are unique.

    Parameters
    ----------
    df_UCC : pd.DataFrame
        DataFrame containing UCC data.

    Returns
    -------
    None
    """
    fname0_UCC = [_.split(";")[0] for _ in df_UCC["fnames"]]
    NT = len(fname0_UCC)
    N_unique = len(list(set(fname0_UCC)))
    if NT != N_unique:
        raise ValueError("Initial fnames are not unique")


def check_new_entries(fnames_old_all: list, i_new: int, fnames_new: str) -> None:
    """
    Check new entries in the UCC that are not present in the old UCC.

    Parameters
    ----------
    fnames_old_all : list
        List of filenames in the old UCC.
    i_new : int
        Index of the new filename being checked.
    fnames_new : str
        New filename to check.

    Returns
    -------
    None
    """
    fname_old_all_lst = []
    for i_old, fnames_old in enumerate(fnames_old_all):
        temp = []
        for fname_old in fnames_old.split(";"):
            temp.append(fname_old)
        fname_old_all_lst.append(temp)

    idxs_old_match = []
    for fname_new in fnames_new.split(";"):
        for i_old, fnames_old in enumerate(fname_old_all_lst):
            if fname_new in fnames_old:
                idxs_old_match.append(i_old)
    idxs_old_match = list(set(idxs_old_match))
    if len(idxs_old_match) > 1:
        raise ValueError(f"Duplicate fname, new:{i_new}, old:{idxs_old_match}")


def check_rows(
    logging, fnames: str, diff_cols: list[str], row_old: pd.Series, row_new: pd.Series
) -> None:
    """
    Compares rows from two DataFrames at specified indices and prints details
    of differences in selected columns if differences exceed specified
    thresholds or do not fall under certain exceptions.

    Parameters
    ----------
    logging : logging.Logger
        Logger instance for recording messages.
    fnames : str
        fname(s) of cluster
    diff_cols : list
        List of column names with differences
    row_old : pd.Series
        Index of the row in UCC_new to compare.
    row_new : pd.Series
        Index of the row in UCC_old to compare.

    Returns
    -------
    None
        Outputs differences directly to the console if they meet specified
        criteria, otherwise returns nothing.
    """
    txt = ""

    # Catch general differences *not* in these columns
    for col in diff_cols:
        if col not in (
            "DB",
            "DB_i",
            "RA_ICRS",
            "DE_ICRS",
            "GLON",
            "GLAT",
            "dups_fnames",
            "dups_probs",
            "dups_fnames_m",
            "dups_probs_m",
        ):
            txt += f"; {col}: {row_old[col]} | {row_new[col]}"

    # Catch specific differences in these columns

    # Check (ra, dec)
    if "RA_ICRS" in diff_cols:
        txt = coords_check(txt, "RA_ICRS", row_old, row_new)
    if "DE_ICRS" in diff_cols:
        txt = coords_check(txt, "DE_ICRS", row_old, row_new)

    # Check (lon, lat)
    if "GLON" in diff_cols:
        txt = coords_check(txt, "GLON", row_old, row_new)
    if "GLAT" in diff_cols:
        txt = coords_check(txt, "GLAT", row_old, row_new)

    # Check dups_fnames and dups_probs
    if "dups_fnames" in diff_cols:
        txt = dups_check(txt, "dups_fnames", row_old, row_new)
    if "dups_probs" in diff_cols:
        txt = dups_check(txt, "dups_probs", row_old, row_new)

    # Check dups_fnames_m and dups_probs_m
    if "dups_fnames_m" in diff_cols:
        txt = dups_check(txt, "dups_fnames_m", row_old, row_new)
    if "dups_probs_m" in diff_cols:
        txt = dups_check(txt, "dups_probs_m", row_old, row_new)

    if txt != "":
        logging.info(f"{fnames:<10} --> {txt}")
    return


def coords_check(
    txt: str,
    coord_id: str,
    row_old: pd.Series,
    row_new: pd.Series,
    deg_diff: float = 0.001,
) -> str:
    """
    Check differences in coordinate values and append to log message.

    Parameters
    ----------
    txt : str
        Existing log message text.
    coord_id : str
        Column name of the coordinate.
    row_old : pd.Series
        Series representing the old row.
    row_new : pd.Series
        Series representing the new row.
    deg_diff : float, optional
        Threshold for coordinate difference. Default is 0.001.

    Returns
    -------
    str
        Updated log message text.
    """
    if abs(row_old[coord_id] - row_new[coord_id]) > deg_diff:
        txt += f"; {coord_id}: " + str(abs(row_old[coord_id] - row_new[coord_id]))
    return txt


def dups_check(txt: str, dup_id: str, row_old: pd.Series, row_new: pd.Series) -> str:
    """
    Check differences in duplicate-related columns and append to log message.

    Parameters
    ----------
    txt : str
        Existing log message text.
    dup_id : str
        Column name of the duplicate field.
    row_old : pd.Series
        Series representing the old row.
    row_new : pd.Series
        Series representing the new row.

    Returns
    -------
    str
        Updated log message text.
    """
    aa = str(row_old[dup_id]).split(";")
    bb = str(row_new[dup_id]).split(";")
    if len(list(set(aa) - set(bb))) > 0:
        txt += f"; {dup_id}: " + str(row_old[dup_id]) + " | " + str(row_new[dup_id])
    return txt


#####################################################################################
#####################################################################################
