import re
from collections import defaultdict
from difflib import SequenceMatcher

import Levenshtein
import numpy as np
import pandas as pd
from astropy.coordinates import angular_separation
from scipy.spatial.distance import cdist

from ..utils import check_centers, list_duplicates, radec2lonlat


def fnames_check_UCC_new_DB(
    logging,
    df_UCC: pd.DataFrame,
    new_DB_fnames: list[list[str]],
) -> bool:
    """
    Check that no fname associated to each entry in the new DB is listed in more than
    one entry in the UCC.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame representing the UCC.
    new_DB_fnames : list
        List of lists, each containing standardized names for a cluster in the new DB.

    Returns
    -------
    bool
        True if duplicate entries are found, False otherwise.
    """
    logging.info("\nChecking uniqueness of fnames")

    # Create a dictionary to map filenames to their corresponding row indices
    # Using defaultdict eliminates the explicit check for key existence
    filename_map = defaultdict(list)
    for i, fnames in enumerate(df_UCC["fnames"]):
        for fname in fnames.split(";"):
            filename_map[fname].append(i)

    # Find matches between UCC fnames and new_DB_fnames
    fnames_ucc_idxs = {}
    for k, fnames in enumerate(new_DB_fnames):
        fnames_ucc_idxs[k] = []
        for fname in fnames:
            if fname in filename_map:  # Check if the filename exists in df_fnames
                # 'filename_map[fname]' will always contain a single element
                fnames_ucc_idxs[k].append(filename_map[fname][0])

    # Check if any new entry has more than one entry in the UCC associated to it
    dup_flag = False
    for k, v in fnames_ucc_idxs.items():
        if len(list(set(v))) > 1:
            dup_flag = True
            logging.info(f"{new_DB_fnames[k]} --> {v}")

    return dup_flag


def dups_fnames_inner_check(
    logging,
    new_DB: str,
    newDB_json,
    df_new: pd.DataFrame,
    new_DB_fnames: list[list[str]],
) -> bool:
    """
    Checks for duplicate fnames in the new DB.

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

    Returns
    -------
    bool
        True if duplicate entries are found, False otherwise.
    """
    logging.info("\nChecking for entries that must be combined")

    # Extract first fname for entries in new DB
    new_DB_fnames_0 = [fnames[0] for fnames in new_DB_fnames]

    dup_fnames = list_duplicates(new_DB_fnames_0)
    if len(dup_fnames) > 0:
        logging.info(f"\nEntries in {new_DB} share 'fname' and must be combined:\n")
        for i, fname0 in enumerate(new_DB_fnames_0):
            if fname0 in dup_fnames:
                logging.info(f"  {i}: {df_new[newDB_json['names']][i]} --> {fname0}")
        return True

    return False


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

    idxs_match = [_ for _ in db_matches if _ is not None]
    dup_idxs = list_duplicates(idxs_match)
    if len(dup_idxs) > 0:
        logging.info(f"\nEntries in {new_DB} must be combined:")
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
    df_GCs: pd.DataFrame,
    newDB_json: dict,
    df_new: pd.DataFrame,
    glon,
    glat,
    search_rad: float = 15,
) -> bool:
    """
    Check for nearby GCs for a new database

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_GCs : pd.DataFrame
        GCs database.
    newDB_json : dict
        Dictionary with the parameters of the new database.
    df_new : pd.DataFrame
        DataFrame of the new database.
    search_rad : float, optional
        Search radius in arcmin. Default is 15.

    Returns
    -------
    bool
        Flag indicating if probable GCs were found.
    """
    glon_glat = np.array([glon, glat]).T

    # Read GCs DB
    l_gc, b_gc = df_GCs["GLON"].values, df_GCs["GLAT"].values  # pyright: ignore

    gc_all, GCs_found = [], 0
    for idx, (glon_i, glat_i) in enumerate(glon_glat):
        d_arcmin = np.rad2deg(angular_separation(glon_i, glat_i, l_gc, b_gc)) * 60
        j1 = np.argmin(d_arcmin)

        if d_arcmin[j1] < search_rad:
            GCs_found += 1
            gc_all.append(
                [
                    idx,
                    df_new.iloc[idx][newDB_json["names"]],
                    j1,
                    df_GCs["Name"][j1],
                    d_arcmin[j1],
                ]
            )

    gc_flag = False
    if GCs_found > 0:
        gc_all = np.array(gc_all).T
        i_sort = np.argsort(np.array(gc_all[-1], dtype=float))
        gc_all = gc_all[:, i_sort].T

        gc_flag = True
        logging.info(f"Found {GCs_found} probable GCs:")
        for gc in gc_all:
            idx, row_id, idx_gc, df_gcs_name, d_arcmin = gc
            logging.info(
                f"{idx:<6} {row_id:<15} --> {idx_gc:<6} {df_gcs_name.strip():<15}"
                + f"d={round(float(d_arcmin), 2)}"
            )
    else:
        logging.info("No probable GCs found")

    return gc_flag


def close_OC_inner_check(
    logging,
    newDB_json: dict,
    df_new: pd.DataFrame,
    ra,
    dec,
    rad_dup: float,
    leven_rad: float = 0.85,
    sep: str = ",",
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
    coords = np.array([ra, dec]).T
    # Find the distances to all clusters, for all clusters (in arcmin)
    dist = cdist(coords, coords) * 60
    # Change distance to itself from 0 to inf
    msk = dist == 0.0
    dist[msk] = np.inf

    col_1 = df_new[newDB_json["names"]]
    col_2 = None
    db_matches = None
    ID_call = "inner"

    return close_OC_check(
        logging, dist, db_matches, col_1, col_2, ID_call, rad_dup, leven_rad, sep
    )


def close_OC_UCC_check(
    logging,
    df_UCC: pd.DataFrame,
    new_DB_fnames: list[list[str]],
    db_matches: list[int | None],
    glon,
    glat,
    rad_dup: float,
    leven_rad: float = 0.5,
    sep: str = ";",
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

    col_1 = df_UCC["fnames"]
    col_2 = new_DB_fnames
    ID_call = "UCC"

    return close_OC_check(
        logging, dist, db_matches, col_1, col_2, ID_call, rad_dup, leven_rad, sep
    )


def close_OC_check(
    logging,
    dist,
    db_matches,
    col_1,
    col_2,
    ID_call: str,
    rad_dup: float,
    leven_rad: float,
    sep: str,
):
    """ """
    idxs = np.arange(0, len(col_1))
    all_dups, dups_list = [], []
    for i, cl_d in enumerate(dist):
        # If no OC is within rad_dup for this OC, continue with next
        msk = cl_d < rad_dup
        if msk.sum() == 0:
            continue

        # Already in dups_list
        if ID_call == "inner":
            cl_name = str(col_1[i])
        else:
            cl_name = sep.join(col_2[i])
        if cl_name in dups_list:
            continue

        if ID_call == "UCC":
            # If cl_name is present in the UCC (df_UCC['fnames'][db_matches[i]])
            if db_matches[i] is not None:
                continue

        # For each OC within the rad_dup region
        N_inner_dups, dups, dist, L_ratios = 0, [], [], []
        for j in idxs[msk]:
            dup_names = str(col_1[j]).split(sep)

            L_ratio = 0
            for dup_name in dup_names:
                L_ratio = max(L_ratio, Levenshtein.ratio(cl_name, dup_name.strip()))

            if L_ratio > leven_rad:
                N_inner_dups += 1
                dups_list.append(sep.join(dup_names))
                dups.append(sep.join(dup_names))
                dist.append(str(round(cl_d[j], 1)))
                L_ratios.append(str(round(L_ratio, 2)))
        if dups:
            all_dups.append([i, cl_name, N_inner_dups, dups, dist, L_ratios])

    dups_flag = False
    dups_found = len(all_dups)
    if dups_found > 0:
        dups_flag = True

        # Extract indexes that sort by distance
        all_dists = []
        for dup in all_dups:
            all_dists.append(min([float(_) for _ in dup[-2]]))
        i_sort = np.argsort(all_dists)

        logging.info(f"Found {dups_found} probable {ID_call} duplicates")
        for idx in i_sort:
            i, cl_name, N_inner_dups, dups, dist, L_ratios = all_dups[idx]
            logging.info(
                f"{i:<6} {cl_name:<15} (N={N_inner_dups}) --> "
                + f"{';'.join(dups):<15} | d={';'.join(dist)}, L={';'.join(L_ratios)}"
            )
    else:
        logging.info(f"No {ID_call} duplicates found")

    return dups_flag


def vdberg_check(logging, newDB_json: dict, df_new: pd.DataFrame) -> bool:
    """
    Check for instances of 'vdBergh-Hagen' and 'vdBergh'

    Per CDS recommendation:

    * BH, VDBergh-Hagen --> VDBH
    * VDBergh           --> VDB

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
    names_lst = [
        "vdBergh-Hagen",
        "vdBergh",
        "van den Bergh–Hagen",
        "van den Bergh",
        "BH",
    ]
    names_lst = [_.lower().replace("-", "").replace(" ", "") for _ in names_lst]
    proper_names = ["VDBH", "VDB"]

    vds_found = []
    for i, new_cl in enumerate(df_new[newDB_json["names"]]):
        new_cl_r = re.sub(r"\d", "", new_cl)  # Remove all numbers

        # If the root of the name is a proper name, skip check
        if new_cl_r.split("_")[0].split(" ")[0] in proper_names:
            continue

        new_cl_r = (
            new_cl_r.lower().strip().replace(" ", "").replace("-", "").replace("_", "")
        )
        for name_check in names_lst:
            sm_ratio = SequenceMatcher(None, new_cl_r, name_check).ratio()
            if sm_ratio > 0.5:
                vds_found.append([i, new_cl, name_check, round(sm_ratio, 2)])
                break

    if len(vds_found) == 0:
        vdb_flag = False
        logging.info("No vdBergh-Hagen/vdBergh found that need renaming")
    else:
        vdb_flag = True
        logging.info(f"Found {len(vds_found)} entries that could need name editing")
        logging.info("* BH ; vdBergh-Hagen --> VDBH")
        logging.info("* vdBergh            --> VDB")
        for i, new_cl, name_check, sm_ratio in vds_found:
            logging.info(f"{i}, {new_cl} --> {name_check} (P={sm_ratio})")

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
        ra_n, dec_n, plx_n, pmra_n, pmde_n = np.nan, np.nan, np.nan, np.nan, np.nan
        if "RA" in newDB_json["pos"]:
            ra_n = row_n[newDB_json["pos"]["RA"]]
            dec_n = row_n[newDB_json["pos"]["DEC"]]
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

    The logic for flagging for attention is handled as follows:

    Is the OC already present in the UCC?
        |         |
        v         |--> No --> do nothing
       Yes
        |
        v
    Is the difference between the old vs new centers values large?
        |         |
        v         |--> No --> do nothing
       Yes
        |
        v
    Request attention

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
        # Check centers if the OC is already present in the UCC
        if j is not None:
            bad_center, d_arcmin, pmra_p, pmde_p, plx_p = check_centers(
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
            ocs_attention.append([fnames, bad_center, d_arcmin, pmra_p, pmde_p, plx_p])

    attention_flag = False
    if len(ocs_attention) > 0:
        attention_flag = True
        logging.info("\nOCs flagged for attention:")
        logging.info(
            "{:<25} {:<5} {}".format(
                "name", "cent_flag", "[arcmin] [pmRA %] [pmDE %] [plx %]"
            )
        )
        for oc in ocs_attention:
            fnames, bad_center, d_arcmin, pmra_p, pmde_p, plx_p = oc
            flag_log(logging, bad_center, d_arcmin, pmra_p, pmde_p, plx_p, fnames)
    else:
        logging.info("\nNo OCs flagged for attention")

    return attention_flag


def flag_log(
    logging,
    bad_center: str,
    d_arcmin: float,
    pmra_p: float,
    pmde_p: float,
    plx_p: float,
    fnames: str,
) -> None:
    """
    Logs details about OCs flagged for attention based on center comparison.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    bad_center : str
        String indicating the quality of the center comparison.
    fnames : str
        String of fnames for the cluster.
    """
    txt = ""
    if bad_center[0] == "y":
        txt += "{:.1f} ".format(d_arcmin)
    else:
        txt += "-- "

    if bad_center[1] == "y":
        txt += "{:.1f} {:.1f} ".format(pmra_p, pmde_p)
    else:
        txt += "-- -- "

    if bad_center[2] == "y":
        txt += "{:.1f}".format(plx_p)
    else:
        txt += "--"

    txt = txt.split()
    logging.info(
        "{:<25} {:<5} {:>12} {:>8} {:>8} {:>7}".format(fnames[:24], bad_center, *txt)
    )
