
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


def main():
    """
    """
    logging = logger.main()
    pars_dict = read_ini_file.main()
    new_DB, dbs_folder = pars_dict['new_DB'], pars_dict['dbs_folder']

    logging.info(f"Running 'new_DB_check' script on {new_DB}")
    df_new = pd.read_csv(dbs_folder + new_DB + '.csv')

    # Check for GCs
    logging.info('\n*Close CG check')
    GCs_check(logging, pars_dict, df_new)

    # Check for OCs very close to each other (possible duplicates)
    logging.info('\n*Possible duplicates check')
    close_OC_check(logging, df_new, pars_dict)

    # Checkfor 'vdBergh-Hagen', 'vdBergh' OCs
    logging.info('\n*Possible vdBergh-Hagen/vdBergh check')
    vdberg_check(logging, df_new, pars_dict)

    # Checkfor semi-colon present in name column
    logging.info('\n*Possible semi-colon in names')
    semicolon_check(logging, df_new, pars_dict)

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
        df_new[clon], df_new[clat] = lb.l.value, lb.b.value

    # Read GCs DB
    df_gcs = pd.read_csv(dbs_folder + GCs_cat)
    l_gc, b_gc = df_gcs['GLON'].values, df_gcs['GLAT'].values

    logging.info("Dist [arcmin], OC, GC")
    logging.info('---------------------')

    GCs_found = 0
    for idx, row in df_new.iterrows():
        l_new, b_new = row[clon], row[clat]

        d_arcmin = angular_separation(
            l_new*u.deg, b_new*u.deg, l_gc*u.deg,
            b_gc*u.deg).to('deg').value * 60
        j1 = np.argmin(d_arcmin)

        if d_arcmin[j1] < search_rad:
            GCs_found += 1
            logging.info(
                f"{round(d_arcmin[j1], 2)}, {row[cID]}, "
                + f"{df_gcs['Name'][j1].strip()}")

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
    dups_found = 0
    for i, cl_d in enumerate(dist):

        msk = cl_d < rad_dup
        if msk.sum() == 0:
            continue
        dups_found += 1

        cl_d = [str(round(_, 1)) for _ in cl_d[msk]]
        cl_d = ';'.join(cl_d)
        dups = [df_new[cID][j].strip() for j in idxs[msk]]
        dups = ';'.join(dups)
        logging.info(f"N={msk.sum()} prob dups for {df_new[cID][i].strip()}: "
                     + f"{dups} | d={cl_d}")

    if dups_found == 0:
        logging.info("No probable duplicates found")


def semicolon_check(logging, df_new, pars_dict):
    """
    """
    cID = pars_dict['cID']
    semics_found = 0
    for new_cl in df_new[cID]:
        if ';' in new_cl:
            semics_found += 1
            logging.info(f"Semi-colon found: {new_cl}")
    if semics_found == 0:
        logging.info("No semi-colons found in name(s) column")


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
    for new_cl in df_new[cID]:
        new_cl = new_cl.lower().strip().replace(' ', '').replace(
            '-', '').replace('_', '')
        for name_check in names_lst:
            sm_ratio = SequenceMatcher(None, new_cl, name_check).ratio()
            if sm_ratio > 0.5:
                vds_found += 1
                logging.info(
                    f"Possible VDB(H): {new_cl}, {round(sm_ratio, 2)}")
                break
    if vds_found == 0:
        logging.info("No probable vdBergh-Hagen/vdBergh OCs found")


def empty_nan_replace(logging, dbs_folder, new_DB, df_new):
    """
    Replace possible empty entries in columns
    """
    df = df_new.replace(r'^\s*$', np.nan, regex=True)
    df_diff = df.compare(df_new)
    if df_diff.empty is True:
        logging.info(f'No empty entries found in {new_DB}')
        return

    # Something changed, replace old DB
    df.to_csv(dbs_folder + new_DB, na_rep='nan', index=False,
              quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()
