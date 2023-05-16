
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial.distance import cdist
from string import ascii_lowercase


"""
This module contains all the necessary functions to generate the extended
combined DB when a new DB is added, as well as the shared functions used by
the initial combined DBs generation.
"""


def get_fnames_new_DB(df_new, json_pars, sep) -> list:
    """
    Extract and standardize all names in new catalogue
    """
    names_all = df_new[json_pars['names']]
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
        if ' ' in name or '_' in name:
            if '_' in name:
                n2 = name.split('_')[1]
            else:
                n2 = name.split(' ')[1]
            n2 = int(n2)
            if n2 < 10:
                n2 = '000' + str(n2)
            elif n2 < 100:
                n2 = '00' + str(n2)
            elif n2 < 1000:
                n2 = '0' + str(n2)
            else:
                n2 = str(n2)
            name = "FSR_" + n2

    if name.startswith("ESO"):
        if name[:4] not in ('ESO_', 'ESO '):
            # This is a LOKTIN17 ESO cluster
            name = 'ESO_' + name[3:]

        if ' ' in name[4:]:
            n1, n2 = name[4:].split(' ')
        elif '_' in name[4:]:
            n1, n2 = name[4:].split('_')
        elif '' in name[4:]:
            n1, n2 = name[4:].split('-')

        n1 = int(n1)
        if n1 < 10:
            n1 = '00' + str(n1)
        elif n1 < 100:
            n1 = '0' + str(n1)
        else:
            n1 = str(n1)
        n2 = int(n2)
        if n2 < 10:
            n2 = '0' + str(n2)
        else:
            n2 = str(n2)
        name = "ESO_" + n1 + '_' + n2

    if 'UBC' in name and 'UBC ' not in name and 'UBC_' not in name:
        name = name.replace('UBC', 'UBC ')
    if 'UBC_' in name:
        name = name.replace('UBC_', 'UBC ')

    if 'UFMG' in name and 'UFMG ' not in name and 'UFMG_' not in name:
        name = name.replace('UFMG', 'UFMG ')

    if 'LISC' in name and 'LISC ' not in name and 'LISC_' not in name\
            and 'LISC-' not in name:
        name = name.replace('LISC', 'LISC ')

    if 'OC-' in name:
        name = name.replace('OC-', 'OC ')

    # Removes duplicates such as "NGC_2516" and "NGC 2516" 
    name = name.replace('_', ' ')

    return name


def rm_chars_from_name(name):
    """
    """
    # We replace '+' with 'p' to avoid duplicating names for clusters
    # like 'Juchert J0644.8-0925' and 'Juchert_J0644.8+0925'
    name = name.lower().replace('_', '').replace(' ', '').replace(
        '-', '').replace('.', '').replace("'", '').replace('+', 'p')
    return name


def rm_name_dups(names_l):
    """
    Remove duplicates of the kind: Berkeley 102, Berkeley102,
    Berkeley_102; keeping only the name with the space
    """
    for i, n in enumerate(names_l):
        n2 = n.replace(' ', '')
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n
        n2 = n.replace(' ', '_')
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n
    return ';'.join(list(dict.fromkeys(names_l)))


def get_matches_new_DB(df_comb, new_DB_fnames):
    """
    Get cluster matches for the new DB being added to the combined DB
    """
    def match_fname(new_cl):
        for name_new in new_cl:
            for j, old_cl in enumerate(df_comb['fnames']):
                for name_old in old_cl.split(';'):
                    if name_new == name_old:
                        return j
        return None

    db_matches = []
    for i, new_cl in enumerate(new_DB_fnames):
        # Check if this new fname is already in the old DBs list of fnames
        db_matches.append(match_fname(new_cl))

    return db_matches


def combine_new_DB(
    DB_new_ID, df_comb, df_new, json_pars, DBs_fnames, db_matches, sep
):
    """
    """
    cols = []
    for v in json_pars['pos'].split(','):
        if str(v) == 'None':
            v = None
        cols.append(v)
    # Remove Rv column
    ra_c, dec_c, plx_c, pmra_c, pmde_c = cols[:-1]

    new_db_dict = {
        'DB': [], 'DB_i': [], 'ID': [], 'RA_ICRS': [], 'DE_ICRS': [],
        'GLON': [], 'GLAT': [], 'plx': [], 'pmRA': [], 'pmDE': [],
        'UCC_ID': [], 'fnames': [], 'dups_fnames': []}
    idx_rm_comb_db = []
    for i, new_cl in enumerate(DBs_fnames):

        row_n = df_new.iloc[i]

        new_names = row_n[json_pars['names']].split(sep)
        new_names = [_.strip() for _ in new_names]
        new_names = ';'.join(new_names)

        plx_m, pmra_m, pmde_m = np.nan, np.nan, np.nan
        ra_m, dec_m = row_n[ra_c], row_n[dec_c]
        if plx_c is not None:
            plx_m = row_n[plx_c]
        if pmra_c is not None:
            pmra_m = row_n[pmra_c]
        if pmde_c is not None:
            pmde_m = row_n[pmde_c]

        # Index of the match for this new cluster in the old DB (if any)
        db_match_j = db_matches[i]

        # If the cluster is already present in the combined DB
        if db_match_j is not None:
            # Store indexes in old DB of clusters present in new DB
            idx_rm_comb_db.append(db_match_j)

            # Identify row in old DB where this match is
            row = df_comb.iloc[db_match_j]

            # Combine old data with that of the new matched cluster
            ra_m = np.nanmedian([row['RA_ICRS'], ra_m])
            dec_m = np.nanmedian([row['DE_ICRS'], dec_m])
            if not np.isnan([row['plx'], plx_m]).all():
                plx_m = np.nanmedian([row['plx'], plx_m])
            if not np.isnan([row['pmRA'], pmra_m]).all():
                pmra_m = np.nanmedian([row['pmRA'], pmra_m])
            if not np.isnan([row['pmDE'], pmde_m]).all():
                pmde_m = np.nanmedian([row['pmDE'], pmde_m])

            DB_ID = row['DB'] + ';' + DB_new_ID
            DB_i = row['DB_i'] + ';' + str(i)

            ID = row['ID'] + ';' + new_names
            UCC_ID = row['UCC_ID']
            fnames = row['fnames'] + ';' + ';'.join(new_cl)
            dups_fnames = row['dups_fnames']
        else:
            DB_ID = DB_new_ID
            DB_i = str(i)
            ID = new_names
            fnames = ';'.join(new_cl)
            # These values will be assigned later on for these new clusters
            UCC_ID = np.nan
            dups_fnames = np.nan

        new_db_dict['DB'].append(DB_ID)
        new_db_dict['DB_i'].append(DB_i)
        # Remove duplicates
        if ';' in ID:
            ID = ';'.join(list(dict.fromkeys(ID.split(';'))))
        new_db_dict['ID'].append(ID)
        lon, lat = radec2lonlat(ra_m, dec_m)
        new_db_dict['RA_ICRS'].append(round(ra_m, 4))
        new_db_dict['DE_ICRS'].append(round(dec_m, 4))
        new_db_dict['GLON'].append(lon)
        new_db_dict['GLAT'].append(lat)
        new_db_dict['plx'].append(plx_m)
        new_db_dict['pmRA'].append(pmra_m)
        new_db_dict['pmDE'].append(pmde_m)
        new_db_dict['UCC_ID'].append(UCC_ID)
        # Remove duplicates
        if ';' in fnames:
            fnames = ';'.join(list(dict.fromkeys(fnames.split(';'))))
        new_db_dict['fnames'].append(fnames)
        new_db_dict['dups_fnames'].append(dups_fnames)

    # Remove duplicates of the kind: Berkeley 102, Berkeley102,
    # Berkeley_102; keeping only the name with the space
    for q, names in enumerate(new_db_dict['ID']):
        names_l = names.split(';')
        names = rm_name_dups(names_l)

    return new_db_dict, idx_rm_comb_db


def radec2lonlat(ra, dec):
    gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    lb = gc.transform_to('galactic')
    lon, lat = lb.l.value, lb.b.value
    return np.round(lon, 4), np.round(lat, 4)


def assign_UCC_ids(glon, glat, ucc_ids_old):
    """
    Format: UCC GXXX.X+YY.Y
    """
    ll = trunc(np.array([glon, glat]).T)
    lon, lat = str(ll[0]), str(ll[1])

    if ll[0] < 10:
        lon = '00' + lon
    elif ll[0] < 100:
        lon = '0' + lon

    if ll[1] >= 10:
        lat = '+' + lat
    elif ll[1] < 10 and ll[1] > 0:
        lat = '+0' + lat
    elif ll[1] == 0:
        lat = '+0' + lat.replace('-', '')
    elif ll[1] < 0 and ll[1] >= -10:
        lat = '-0' + lat[1:]
    elif ll[1] < -10:
        pass

    ucc_id = 'UCC G' + lon + lat

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
    return np.trunc(values*10**decs)/(10**decs)


def QXY_fold(UCC_ID):
    """
    """
    # UCC_ID = cl['UCC_ID']
    lonlat = UCC_ID.split('G')[1]
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
        Qfold += 'P'
    else:
        Qfold += "N"

    return Qfold


def dups_identify(df, N_dups, noXY):
    """
    Find the closest clusters to all clusters

    noXY: flag indicating whether to use (x, y) coordinates
    """
    x, y = df['GLON'], df['GLAT']
    pmRA, pmDE, plx = df['pmRA'], df['pmDE'], df['plx']
    coords = np.array([x, y]).T
    # Find the distances to all clusters, for all clusters
    dist = cdist(coords, coords)
    # Change distance to itself from 0 to inf
    msk = dist == 0.
    dist[msk] = np.inf

    dups_fnames = []
    for i, cl in enumerate(dist):
        idx = np.argsort(cl)[:N_dups]

        dups_fname = []
        for j in idx:
            if duplicate_find(x, y, pmRA, pmDE, plx, i, j, noXY):
                # Store just the first fname
                dups_fname.append(df['fnames'][j].split(';')[0])

        if dups_fname:
            # print(i, df['DB'][i], df['fnames'][i], dups_fname)
            dups_fname = ";".join(dups_fname)
        else:
            dups_fname = 'nan'

        dups_fnames.append(dups_fname)

    return dups_fnames


def duplicate_find(x, y, pmRA, pmDE, plx, i, j, noXY=False):
    """
    Identify a cluster as a duplicate following an arbitrary definition
    that depends on the parallax

    The 'noXY' is required because the results obtained with noXY=False are
    used to reject duplicates when running the 'fastMP' script. The noXY=True
    results are only used for showing in the clusters' entries in the UCC
    site.
    """
    #  If the 'noXY' flag is True, the coordinates distance is not used
    if noXY is False:
        # Define 'duplicate regions' for different parallax brackets
        if np.isnan(plx[i]):
            plx_r, rad, pm_r = np.nan, 5, 0.5
        elif plx[i] >= 4:
            rad, plx_r, pm_r = 15, 0.5, 1
        elif 3 <= plx[i] and plx[i] < 4:
            rad, plx_r, pm_r = 10, 0.25, 0.5
        elif 2 <= plx[i] and plx[i] < 3:
            rad, plx_r, pm_r = 5, 0.15, 0.25
        elif 1 <= plx[i] and plx[i] < 2:
            rad, plx_r, pm_r = 2.5, 0.1, 0.15
        elif plx[i] < 1:
            rad, plx_r, pm_r = 1, 0.05, 0.1

        # Angular distance in arcmin
        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) * 60
        # If the coordinates distance is larger than the defined radius,
        # mark as no duplicate
        if d > rad:
            return False
    else:
        # Define larger 'duplicate regions' for this run
        if np.isnan(plx[i]):
            plx_r, pm_r = np.nan, 0.5
        elif plx[i] >= 4:
            plx_r, pm_r = 0.5, 1
        elif 3 <= plx[i] and plx[i] < 4:
            plx_r, pm_r = 0.25, 0.5
        elif 2 <= plx[i] and plx[i] < 3:
            plx_r, pm_r = 0.2, 0.35
        elif 1 <= plx[i] and plx[i] < 2:
            plx_r, pm_r = 0.15, 0.2
        elif plx[i] < 1:
            plx_r, pm_r = 0.1, 0.15

    # PMs distance
    pm_d = np.sqrt((pmRA[i]-pmRA[j])**2 + (pmDE[i]-pmDE[j])**2)
    # Parallax distance
    plx_d = abs(plx[i] - plx[j])

    if not np.isnan(plx_d) and not np.isnan(pm_d):
        if pm_d < pm_r and plx_d < plx_r:
            return True
    elif not np.isnan(plx_d) and np.isnan(pm_d):
        if plx_d < plx_r:
            return True
    elif np.isnan(plx_d) and not np.isnan(pm_d):
        if pm_d < pm_r:
            return True
    elif np.isnan(plx_d) and np.isnan(pm_d):
        # If the coordinates distance is within the duplicates range and
        # neither PMs or Plx distances could be obtained, also mark as
        # possible duplicate
        if noXY is False:
            return True

    return False
