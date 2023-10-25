
import csv
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


def main():
    """
    """
    logging = logger.main()
    pars_dict = read_ini_file.main()
    new_DB, dbs_folder = pars_dict['new_DB'], pars_dict['dbs_folder']

    logging.info(f"Running 'new_DB_check' script on {new_DB}")
    df_UCC, df_new, _, new_DB_fnames, db_matches = UCC_new_match.main(logging)

    # Check for GCs
    logging.info('\n*Close CG check')
    GCs_check(logging, pars_dict, df_new)

    # Check for OCs very close to each other (possible duplicates)
    logging.info('\n*Possible inner duplicates check')
    close_OC_check(logging, df_new, pars_dict)

    # Check for OCs very close to each other (possible duplicates)
    logging.info('\n*Possible UCC duplicates check')
    close_OC_UCC_check(
        logging, df_UCC, df_new, new_DB_fnames, db_matches, pars_dict)

    # Check for 'vdBergh-Hagen', 'vdBergh' OCs
    logging.info('\n*Possible vdBergh-Hagen/vdBergh check')
    vdberg_check(logging, df_new, pars_dict)

    # Check for semi-colon present in name column
    logging.info('\n*Possible bad characters in names')
    name_chars_check(logging, df_new, pars_dict)

    # Replace empty positions with 'nans'
    logging.info('\n*Empty entries replace finished')
    empty_nan_replace(logging, dbs_folder, new_DB, df_new)

    logging.info('\nFinished\n')


def GCs_check(logging, pars_dict, df_new):
    """
    Check for nearby GCs for a new database
    """
    dbs_folder, GCs_cat, search_rad, cID, clon, clat, coords = \
        pars_dict['dbs_folder'], pars_dict['GCs_cat'], \
        pars_dict['search_rad'], pars_dict['cID'], \
        pars_dict['clon'], pars_dict['clat'], pars_dict['coords']

    if coords == 'equatorial':
        ra, dec = df_new[clon].values, df_new[clat].values
        gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        lb = gc.transform_to('galactic')
        glon, glat = lb.l.value, lb.b.value
    else:
        glon, glat = df_new[clon].values, df_new[clat].values

    # Read GCs DB
    df_gcs = pd.read_csv(dbs_folder + GCs_cat)
    l_gc, b_gc = df_gcs['GLON'].values, df_gcs['GLAT'].values

    logging.info("Dist [arcmin], OC, GC")
    logging.info('---------------------')

    GCs_found = 0
    for idx, row in df_new.iterrows():
        l_new, b_new = glon[idx], glat[idx]

        d_arcmin = angular_separation(
            l_new*u.deg, b_new*u.deg, l_gc*u.deg,
            b_gc*u.deg).to('deg').value * 60
        j1 = np.argmin(d_arcmin)

        if d_arcmin[j1] < search_rad:
            GCs_found += 1
            logging.info(
                f"{idx} {row[cID]} --> "
                + f"{df_gcs['Name'][j1].strip()}, d={round(d_arcmin[j1], 2)}")

    if GCs_found == 0:
        logging.info("No probable GCs found")


def close_OC_check(logging, df_new, pars_dict):
    """
    """
    cID, clon, clat, rad_dup = pars_dict['cID'], pars_dict['clon'], \
        pars_dict['clat'], pars_dict['rad_dup']
    x, y = df_new[clon].values, df_new[clat].values
    coords = np.array([x, y]).T
    # Find the distances to all clusters, for all clusters
    dist = cdist(coords, coords)
    # Change distance to itself from 0 to inf
    msk = dist == 0.
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

        dups, dist = [], []
        for j in idxs[msk]:
            dup_name = df_new[cID][j].strip()
            dups_list.append(dup_name)

            dups.append(dup_name)
            dist.append(str(round(cl_d[j], 1)))
        logging.info(f"{i} {cl_name} (N={msk.sum()}) --> "
                     + f"{';'.join(dups)}, d={';'.join(dist)}")

    if dups_found == 0:
        logging.info("No probable duplicates found")


def close_OC_UCC_check(
    logging, df_UCC, df_new, new_DB_fnames, db_matches, pars_dict
):
    """
    """
    clon, clat, rad_dup = pars_dict['clon'], \
        pars_dict['clat'], pars_dict['rad_dup']

    if pars_dict['coords'] == 'equatorial':
        ra, dec = df_new[clon].values, df_new[clat].values
        gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        lb = gc.transform_to('galactic')
        glon, glat = lb.l.value, lb.b.value
    else:
        glon, glat = df_new[clon].values, df_new[clat].values

    coords_new = np.array([glon, glat]).T
    coords_UCC = np.array([df_UCC['GLON'], df_UCC['GLAT']]).T    
    # Find the distances to all clusters, for all clusters
    dist = cdist(coords_new, coords_UCC)

    idxs = np.arange(0, len(df_UCC))
    dups_list, dups_found = [], 0
    for i, cl_d in enumerate(dist):

        msk = cl_d < rad_dup
        if msk.sum() == 0:
            continue

        cl_name = ','.join(new_DB_fnames[i])
        if cl_name in dups_list:
            continue

        if db_matches[i] is not None:
            # cl_name found in UCC (df_UCC['fnames'][db_matches[i]])
            continue

        dups_found += 1

        dups, dist = [], []
        for j in idxs[msk]:
            dup_name = df_UCC['fnames'][j]
            dups_list.append(dup_name)

            dups.append(dup_name)
            dist.append(str(round(cl_d[j], 1)))
        logging.info(f"{i} {cl_name} (N={msk.sum()}) --> "
                     + f"{'|'.join(dups)}, d={'|'.join(dist)}")

    if dups_found == 0:
        logging.info("No probable duplicates found")


def name_chars_check(logging, df_new, pars_dict):
    """
    """
    cID = pars_dict['cID']
    badchars_found = 0
    for new_cl in df_new[cID]:
        if ';' in new_cl or '_' in new_cl or '-' in new_cl:
            badchars_found += 1
            logging.info(f"{new_cl}: bad char found")
    if badchars_found == 0:
        logging.info("No bad-chars found in name(s) column")


def vdberg_check(logging, df_new, pars_dict):
    """
    Check for instances of 'vdBergh-Hagen' and 'vdBergh'
    """
    names_lst = [
        'vdBergh-Hagen', 'vdBergh', 'van den Bergh–Hagen', 'van den Bergh']
    names_lst = [
        _.lower().replace('-', '').replace(' ', '') for _ in names_lst]

    cID = pars_dict['cID']
    vds_found = 0
    for i, new_cl in enumerate(df_new[cID]):
        new_cl = new_cl.lower().strip().replace(' ', '').replace(
            '-', '').replace('_', '')
        for name_check in names_lst:
            sm_ratio = SequenceMatcher(None, new_cl, name_check).ratio()
            if sm_ratio > 0.5:
                vds_found += 1
                logging.info(
                    f"Possible VDB(H): {i} {new_cl}, {round(sm_ratio, 2)}")
                break
    if vds_found == 0:
        logging.info("No probable vdBergh-Hagen/vdBergh OCs found")


def empty_nan_replace(logging, dbs_folder, new_DB, df_new):
    """
    Replace possible empty entries in columns
    """
    df_new.to_csv(dbs_folder + new_DB + '.csv', na_rep='nan', index=False,
                  quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()
