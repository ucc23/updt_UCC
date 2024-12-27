import sys
from difflib import SequenceMatcher

import Levenshtein
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from scipy.spatial.distance import cdist

from modules import aux
from modules.HARDCODED import GCs_cat, dbs_folder


def run(logging, pars_dict, df_UCC, df_new, json_pars, new_DB_fnames, db_matches):
    """
    1. Checks for duplicate entries between the new database and the UCC.
    2. Checks for nearby GCs.
    3. Checks for OCs very close to each other within the new database.
    4. Checks for OCs very close to each other between the new database and the UCC.
    5. Checks for instances of 'vdBergh-Hagen' and 'vdBergh'.
    6. Checks positions and flags for attention if required.
    """
    new_DB = pars_dict["new_DB"]

    # Duplicate check between entries in the new DB and the UCC
    logging.info(f"\nChecking for entries in {new_DB} that must be combined")
    dup_flag = dups_check_newDB_UCC(logging, new_DB, df_UCC, new_DB_fnames, db_matches)
    if dup_flag:
        raise ValueError("Resolve the above issues before moving on.")
    else:
        logging.info("No issues found")

    # Check for GCs
    logging.info("\nClose GC check")
    glon, glat, gc_flag = GCs_check(logging, pars_dict, df_new)
    if gc_flag:
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Check for OCs very close to each other (possible duplicates)
    logging.info("\nProbable inner duplicates check")
    inner_flag = close_OC_check(logging, df_new, pars_dict)
    if inner_flag:
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Check for OCs very close to each other (possible duplicates)
    logging.info("\nProbable UCC duplicates check")
    dups_flag = close_OC_UCC_check(
        logging, df_UCC, new_DB_fnames, db_matches, pars_dict, glon, glat
    )
    if dups_flag:
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    logging.info("\nPossible vdBergh-Hagen/vdBergh check")
    vdb_flag = vdberg_check(logging, df_new, pars_dict)
    if vdb_flag:
        if input("Move on? (y/n): ").lower() != "y":
            sys.exit()

    # Check positions and flag for attention if required
    attention_flag = positions_check(
        logging, df_UCC, pars_dict, df_new, json_pars, new_DB_fnames, db_matches
    )
    if attention_flag is True:
        if input("\nMove on? (y/n): ").lower() != "y":
            sys.exit()


def dups_check_newDB_UCC(logging, new_DB, df_UCC, new_DB_fnames, db_matches):
    """ """

    def list_duplicates(seq):
        """ """
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


def GCs_check(logging, pars_dict, df_new):
    """
    Check for nearby GCs for a new database
    """
    # Equatorial to galactic
    RA, DEC = pars_dict["RA"], pars_dict["DEC"]
    ra, dec = df_new[RA].values, df_new[DEC].values
    gc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    lb = gc.transform_to("galactic")
    glon, glat = np.array(lb.l), np.array(lb.b)

    search_rad, ID = (pars_dict["search_rad"], pars_dict["ID"])

    # Read GCs DB
    df_gcs = pd.read_csv(dbs_folder + GCs_cat)
    l_gc, b_gc = df_gcs["GLON"].values, df_gcs["GLAT"].values

    gc_all, GCs_found = [], 0
    for idx, row in df_new.iterrows():
        l_new, b_new = glon[idx], glat[idx]

        d_arcmin = (
            angular_separation(l_new * u.deg, b_new * u.deg, l_gc * u.deg, b_gc * u.deg)
            .to("deg")
            .value
            * 60
        )
        j1 = np.argmin(d_arcmin)

        if d_arcmin[j1] < search_rad:
            GCs_found += 1
            gc_all.append([idx, row[ID], df_gcs["Name"][j1], d_arcmin[j1]])

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

    return glon, glat, gc_flag


def close_OC_check(logging, df_new, pars_dict):
    """
    Looks for OCs in the new DB that are close to other OCs in the new DB (RA, DEC)
    whose names are somewhat similar (Levenshtein distance).
    """
    cID, RA, DEC, rad_dup, leven_rad = (
        pars_dict["ID"],
        pars_dict["RA"],
        pars_dict["DEC"],
        pars_dict["rad_dup"],
        pars_dict["leven_rad"],
    )
    x, y = df_new[RA].values, df_new[DEC].values
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

        cl_name = df_new[cID][i].strip()
        if cl_name in dups_list:
            # print(cl_name, "continue")
            continue

        N_inner_dups, dups, dist, L_ratios = 0, [], [], []
        for j in idxs[msk]:
            dup_name = df_new[cID][j].strip()

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
    logging, df_UCC, new_DB_fnames, db_matches, pars_dict, glon, glat
):
    """
    Looks for OCs in the new DB that are close to OCs in the UCC (GLON, GLAT) but
    with different names.
    """
    rad_dup = pars_dict["rad_dup"]

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


def vdberg_check(logging, df_new, pars_dict):
    """
    Check for instances of 'vdBergh-Hagen' and 'vdBergh'

    Per CDS recommendation:

    * VDBergh-Hagen --> VDBH
    * VDBergh       --> VDB
    """
    names_lst = ["vdBergh-Hagen", "vdBergh", "van den Bergh–Hagen", "van den Bergh"]
    names_lst = [_.lower().replace("-", "").replace(" ", "") for _ in names_lst]

    cID = pars_dict["ID"]
    vds_found = 0
    for i, new_cl in enumerate(df_new[cID]):
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
    df_new: "pd.DataFrame", json_pars: dict, new_DB_fnames: list, db_matches: list
) -> dict:
    """
    Prepare information from a new database matched with the Unified Cluster Catalog
    (UCC).

    Args:
        df_new (pd.DataFrame): DataFrame containing information about the new database.
        json_pars (dict): Dictionary containing parameters specifying column names for
        position (RA, Dec, etc.).
        new_DB_fnames (list): List of lists, where each inner list contains file names
        for each cluster in the new database.
        db_matches (list): List of indices representing matches in the UCC, or None if
        no match exists.

    Returns:
        dict: A dictionary containing:
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
    # Extract names of (ra, dec, plx, pmRA, pmDE) columns
    cols = []
    for v in json_pars["pos"].split(","):
        if str(v) == "None":
            v = None
        cols.append(v)
    # Remove Rv column
    ra_c, dec_c, plx_c, pmra_c, pmde_c = cols[:-1]

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
        ra_n, dec_n = row_n[ra_c], row_n[dec_c]
        if plx_c is not None:
            plx_n = row_n[plx_c]
        if pmra_c is not None:
            pmra_n = row_n[pmra_c]
        if pmde_c is not None:
            pmde_n = row_n[pmde_c]

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

    lon_n, lat_n = aux.radec2lonlat(new_db_info["RA_ICRS"], new_db_info["DE_ICRS"])
    new_db_info["GLON"] = list(np.round(lon_n, 4))
    new_db_info["GLAT"] = list(np.round(lat_n, 4))

    return new_db_info


def flag_log(logging, df_UCC, new_db_info, bad_center, fnames, i, j):
    """ """
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


def positions_check(
    logging, df_UCC, pars_dict, df_new, json_pars, new_DB_fnames, db_matches
):
    """ """
    new_db_info = prep_newDB(df_new, json_pars, new_DB_fnames, db_matches)

    ocs_attention = []
    rad_dup = pars_dict["rad_dup"]
    for i, fnames in enumerate(new_db_info["fnames"]):
        j = new_db_info["UCC_idx"][i]
        # If the OC is already present in the UCC
        if j is not None:
            bad_center = aux.check_centers(
                (df_UCC["GLON_m"][j], df_UCC["GLAT_m"][j]),
                (df_UCC["pmRA_m"][j], df_UCC["pmDE_m"][j]),
                df_UCC["Plx_m"][j],
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
