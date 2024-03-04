import csv
import Levenshtein
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import angular_separation
from difflib import SequenceMatcher
from modules import logger
from modules import read_ini_file
from modules import UCC_new_match
from HARDCODED import dbs_folder, all_DBs_json, UCC_folder, GCs_cat


def main():
    """ """
    logging = logger.main()
    pars_dict = read_ini_file.main()
    new_DB = pars_dict["new_DB"]

    logging.info(f"Running 'new_DB_check' script on {new_DB}")
    df_UCC, df_new, _, new_DB_fnames, db_matches = UCC_new_match.main(
        logging, dbs_folder, all_DBs_json, UCC_folder
    )

    # Duplicate check between entries in the new DB and the UCC
    logging.info(f"Checking for entries in {new_DB} that must be combined")
    dup_flag = dups_check_newDB_UCC(logging, new_DB, df_UCC, new_DB_fnames, db_matches)
    if dup_flag:
        print("Resolve the above issues before moving on.")
        return

    # Equatorial to galactic
    RA, DEC = pars_dict["RA"], pars_dict["DEC"]
    ra, dec = df_new[RA].values, df_new[DEC].values
    gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    lb = gc.transform_to("galactic")
    glon, glat = lb.l.value, lb.b.value

    # Check for GCs
    logging.info("\n*Close CG check")
    GCs_check(logging, pars_dict, df_new, glon, glat)

    # Check for OCs very close to each other (possible duplicates)
    logging.info("\n*Possible inner duplicates check")
    close_OC_check(logging, df_new, pars_dict)

    # Check for OCs very close to each other (possible duplicates)
    logging.info("\n*Possible UCC duplicates check")
    close_OC_UCC_check(
        logging, df_UCC, df_new, new_DB_fnames, db_matches, pars_dict, glon, glat
    )

    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    logging.info("\n*Possible vdBergh-Hagen/vdBergh check")
    vdberg_check(logging, df_new, pars_dict)

    # Check for semi-colon present in name column
    logging.info("\n*Possible bad characters in names")
    name_chars_check(logging, df_new, pars_dict)

    # Replace empty positions with 'nans'
    logging.info("\n*Empty entries replace finished")
    empty_nan_replace(logging, dbs_folder, new_DB, df_new)

    logging.info("\nFinished\n")


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
        logging.info(f"WARNING! Entries in {new_DB} that must be combined")
        print("")
        for didx in dup_idxs:
            for i, db_idx in enumerate(db_matches):
                if db_idx == didx:
                    logging.info(
                        f"  UCC {didx}: {df_UCC['fnames'][didx]}: {i} {new_DB_fnames[i]}"
                    )
            print("")
        return True

    return False


def GCs_check(logging, pars_dict, df_new, glon, glat):
    """
    Check for nearby GCs for a new database
    """
    search_rad, ID = (pars_dict["search_rad"], pars_dict["ID"])

    # Read GCs DB
    df_gcs = pd.read_csv(dbs_folder + GCs_cat)
    l_gc, b_gc = df_gcs["GLON"].values, df_gcs["GLAT"].values

    logging.info("Dist [arcmin], OC, GC")
    logging.info("---------------------")

    GCs_found = 0
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
            logging.info(
                f"{idx} {row[ID]} --> "
                + f"{df_gcs['Name'][j1].strip()}, d={round(d_arcmin[j1], 2)}"
            )

    if GCs_found == 0:
        logging.info("No probable GCs found")


def close_OC_check(logging, df_new, pars_dict):
    """
    Looks for OCs in the new DB that are close to other OCs in the new DB (GLON, GLAT)
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
    # Find the distances to all clusters, for all clusters
    dist = cdist(coords, coords)
    # Change distance to itself from 0 to inf
    msk = dist == 0.0
    dist[msk] = np.inf

    idxs = np.arange(0, len(df_new))
    dups_list, dups_found = [], 0
    for i, cl_d in enumerate(dist):
        msk = cl_d < rad_dup
        if msk.sum() == 0:
            continue

        cl_name = df_new[cID][i].strip()
        if cl_name in dups_list:
            # print(cl_name, "continue")
            continue

        dups_found += 1

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
            logging.info(
                f"{i} {cl_name} (N={N_inner_dups}) --> "
                + f"{';'.join(dups)}, d={';'.join(dist)}, L={';'.join(L_ratios)}"
            )

    if dups_found == 0:
        logging.info("No probable duplicates found")


def close_OC_UCC_check(
    logging, df_UCC, df_new, new_DB_fnames, db_matches, pars_dict, glon, glat
):
    """
    Looks for OCs in the new DB that are close to OCs in the UCC (GLON, GLAT) but
    with different names.
    """
    # RA, DEC = pars_dict["RA"], pars_dict["DEC"],
    rad_dup = pars_dict["rad_dup"]

    coords_new = np.array([glon, glat]).T
    coords_UCC = np.array([df_UCC["GLON"], df_UCC["GLAT"]]).T
    # Find the distances to all clusters, for all clusters
    dist = cdist(coords_new, coords_UCC)

    idxs = np.arange(0, len(df_UCC))
    dups_list, dups_found = [], 0
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

        dups_found += 1

        dups, dist = [], []
        for j in idxs[msk]:
            dup_name = df_UCC["fnames"][j]
            dups_list.append(dup_name)
            dups.append(dup_name)
            dist.append(str(round(cl_d[j], 1)))
        logging.info(
            f"{i} {cl_name} (N={msk.sum()}) --> "
            + f"{'|'.join(dups)}, d={'|'.join(dist)}"
        )

    if dups_found == 0:
        logging.info("No probable duplicates found")


def name_chars_check(logging, df_new, pars_dict):
    """ """
    ID = pars_dict["ID"]
    badchars_found = 0
    for new_cl in df_new[ID]:
        if ";" in new_cl or "_" in new_cl:
            badchars_found += 1
            logging.info(f"{new_cl}: bad char found")
    if badchars_found == 0:
        logging.info("No bad-chars found in name(s) column")


def vdberg_check(logging, df_new, pars_dict):
    """
    Check for instances of 'vdBergh-Hagen' and 'vdBergh'
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
                logging.info(f"Possible VDB(H): {i} {new_cl}, {round(sm_ratio, 2)}")
                break
    if vds_found == 0:
        logging.info("No probable vdBergh-Hagen/vdBergh OCs found")


def empty_nan_replace(logging, dbs_folder, new_DB, df_new):
    """
    Replace possible empty entries in columns
    """
    df_new.to_csv(
        dbs_folder + new_DB + ".csv",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


if __name__ == "__main__":
    main()
