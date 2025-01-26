from difflib import SequenceMatcher

import Levenshtein
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from scipy.spatial.distance import cdist
from ..utils import check_centers, radec2lonlat


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
        ra=df_new[newDB_json["pos"]["RA"]].values * u.deg,  # pyright: ignore
        dec=df_new[newDB_json["pos"]["DEC"]].values * u.deg,  # pyright: ignore
    )
    lb = gc.transform_to("galactic")
    glon_glat = list(zip(*[lb.l, lb.b]))

    # Read GCs DB
    df_gcs = pd.read_csv(GCs_path)
    l_gc, b_gc = df_gcs["GLON"].values * u.deg, df_gcs["GLAT"].values * u.deg  # pyright: ignore

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
        logging.info("\nOCs flagged for attention:")
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
