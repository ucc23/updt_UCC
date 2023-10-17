
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial.distance import cdist
from string import ascii_lowercase
from . import duplicate_probs


"""
This module contains all the necessary functions to generate the extended
combined DB when a new DB is added, as well as the shared functions used by
the initial combined DBs generation.
"""


def get_fnames_new_DB(df_new, json_pars, sep) -> list:
    """
    Extract and standardize all names in the new catalogue
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
    if 'NGC 1976' in names_l:
        print(names_l)
    for i, n in enumerate(names_l):
        n2 = n.replace(' ', '')
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n
        n2 = n.replace(' ', '_')
        if n2 in names_l:
            j = names_l.index(n2)
            names_l[j] = n

    if 'NGC 1976' in names_l:
        print(';'.join(list(dict.fromkeys(names_l))))

    return ';'.join(list(dict.fromkeys(names_l)))


def get_matches_new_DB(df_UCC, new_DB_fnames):
    """
    Get cluster matches for the new DB being added to the combined DB
    """
    def match_fname(new_cl):
        for name_new in new_cl:
            for j, old_cl in enumerate(df_UCC['fnames']):
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
    logging, new_OCs_info, new_DB_ID, df_UCC, df_new, json_pars, new_DB_fnames,
    db_matches, sep
):
    """
    """
    # Extract names of (ra, dec, plx, pmRA, pmDE) columns
    cols = []
    for v in json_pars['pos'].split(','):
        if str(v) == 'None':
            v = None
        cols.append(v)
    # Remove Rv column
    ra_c, dec_c, plx_c, pmra_c, pmde_c = cols[:-1]

    new_db_dict = {_: [] for _ in df_UCC.keys()}
    idx_rm_comb_db = []
    for i, new_cl in enumerate(new_DB_fnames):

        row_n = df_new.iloc[i]

        new_names = row_n[json_pars['names']].split(sep)
        new_names = [_.strip() for _ in new_names]
        new_names = ';'.join(new_names)

        # Coordinates for this cluster in the new DB
        plx_n, pmra_n, pmde_n = np.nan, np.nan, np.nan
        ra_n, dec_n = row_n[ra_c], row_n[dec_c]
        if plx_c is not None:
            plx_n = row_n[plx_c]
        if pmra_c is not None:
            pmra_n = row_n[pmra_c]
        if pmde_c is not None:
            pmde_n = row_n[pmde_c]

        # Index of the match for this new cluster in the UCC (if any)
        db_match_j = db_matches[i]

        # If the cluster is already present in the UCC
        if db_match_j is not None:
            # Store indexes in UCC of clusters present in new DB
            idx_rm_comb_db.append(db_match_j)

            # Identify row in UCC where this match is located
            row = df_UCC.iloc[db_match_j]

            # Read the flag that informs if the data for this OC should be
            # combined (thus changing the values in the 5D space) or not
            fname0 = new_cl[0]
            if fname0 in new_OCs_info['fnames'][i]:
                combine_flag = new_OCs_info['process_f'][i]
            else:
                # The 'fnames' should be aligned. Here just in case
                print("ERROR")
                return

            if combine_flag:
                # Combine UCC data with that of the new matched cluster
                ra_n = np.nanmedian([row['RA_ICRS'], ra_n])
                dec_n = np.nanmedian([row['DE_ICRS'], dec_n])
                if not np.isnan([row['plx'], plx_n]).all():
                    plx_n = np.nanmedian([row['plx'], plx_n])
                if not np.isnan([row['pmRA'], pmra_n]).all():
                    pmra_n = np.nanmedian([row['pmRA'], pmra_n])
                if not np.isnan([row['pmDE'], pmde_n]).all():
                    pmde_n = np.nanmedian([row['pmDE'], pmde_n])
            else:
                # Use UCC data, do not combine with new DB data
                ra_n = row['RA_ICRS']
                dec_n = row['DE_ICRS']
                plx_n = row['plx']
                pmra_n = row['pmRA']
                pmde_n = row['pmDE']

            # Add new DB information
            DB_ID = row['DB'] + ';' + new_DB_ID
            DB_i = row['DB_i'] + ';' + str(i)
            # Add name in new DB
            ID = row['ID'] + ';' + new_names
            # Add fnames in new DB
            fnames = row['fnames'] + ';' + ';'.join(new_cl)

            # Copy values from the UCC for these columns
            UCC_ID = row['UCC_ID']
            quad = row['quad']
            dups_fnames = row['dups_fnames']
            dups_probs = row['dups_probs']
            r_50 = row['r_50']
            N_50 = row['N_50']
            N_fixed = row['N_fixed']
            N_membs = row['N_membs']
            fixed_cent = row['fixed_cent']
            cent_flags = row['cent_flags']
            C1 = row['C1']
            C2 = row['C2']
            C3 = row['C3']
            GLON_m = row['GLON_m']
            GLAT_m = row['GLAT_m']
            RA_ICRS_m = row['RA_ICRS_m']
            DE_ICRS_m = row['DE_ICRS_m']
            plx_m = row['plx_m']
            pmRA_m = row['pmRA_m']
            pmDE_m = row['pmDE_m']
            Rv_m = row['Rv_m']
            N_Rv = row['N_Rv']
            dups_fnames_m = row['dups_fnames_m']
            dups_probs_m = row['dups_probs_m']

        else:
            # The cluster is not present in the UCC
            logging.info(f"New OC found: {i}, {new_names}")
            idx_rm_comb_db.append(None)
            DB_ID = new_DB_ID
            DB_i = str(i)
            ID = new_names
            fnames = ';'.join(new_cl)

            # These values will be assigned later on for these new clusters
            UCC_ID = 'nan'
            quad = 'nan'
            dups_fnames = 'nan'
            dups_probs = 'nan'

            # These values will be assigned later when fastMP is run
            r_50 = np.nan
            N_50 = 0
            N_fixed = 0
            N_membs = 0
            fixed_cent = 'nan'
            cent_flags = 'nan'
            C1 = np.nan
            C2 = np.nan
            C3 = 'nan'
            GLON_m = np.nan
            GLAT_m = np.nan
            RA_ICRS_m = np.nan
            DE_ICRS_m = np.nan
            plx_m = np.nan
            pmRA_m = np.nan
            pmDE_m = np.nan
            Rv_m = np.nan
            N_Rv = 0
            dups_fnames_m = 'nan'
            dups_probs_m = 'nan'

        new_db_dict['DB'].append(DB_ID)
        new_db_dict['DB_i'].append(DB_i)
        # Remove duplicates
        if ';' in ID:
            ID = ';'.join(list(dict.fromkeys(ID.split(';'))))
        new_db_dict['ID'].append(ID)
        lon_n, lat_n = radec2lonlat(ra_n, dec_n)
        new_db_dict['RA_ICRS'].append(round(ra_n, 4))
        new_db_dict['DE_ICRS'].append(round(dec_n, 4))
        new_db_dict['GLON'].append(round(lon_n, 4))
        new_db_dict['GLAT'].append(round(lat_n, 4))
        new_db_dict['plx'].append(round(plx_n, 4))
        new_db_dict['pmRA'].append(round(pmra_n, 4))
        new_db_dict['pmDE'].append(round(pmde_n, 4))
        # Remove duplicates
        if ';' in fnames:
            fnames = ';'.join(list(dict.fromkeys(fnames.split(';'))))
        new_db_dict['fnames'].append(fnames)
        new_db_dict['UCC_ID'].append(UCC_ID)
        new_db_dict['quad'].append(quad)
        new_db_dict['dups_fnames'].append(dups_fnames)
        new_db_dict['dups_probs'].append(dups_probs)
        new_db_dict['r_50'].append(r_50)
        new_db_dict['N_50'].append(N_50)
        new_db_dict['N_fixed'].append(N_fixed)
        new_db_dict['N_membs'].append(N_membs)
        new_db_dict['fixed_cent'].append(fixed_cent)
        new_db_dict['cent_flags'].append(cent_flags)
        new_db_dict['C1'].append(C1)
        new_db_dict['C2'].append(C2)
        new_db_dict['C3'].append(C3)
        new_db_dict['GLON_m'].append(GLON_m)
        new_db_dict['GLAT_m'].append(GLAT_m)
        new_db_dict['RA_ICRS_m'].append(RA_ICRS_m)
        new_db_dict['DE_ICRS_m'].append(DE_ICRS_m)
        new_db_dict['plx_m'].append(plx_m)
        new_db_dict['pmRA_m'].append(pmRA_m)
        new_db_dict['pmDE_m'].append(pmDE_m)
        new_db_dict['Rv_m'].append(Rv_m)
        new_db_dict['N_Rv'].append(N_Rv)
        new_db_dict['dups_fnames_m'].append(dups_fnames_m)
        new_db_dict['dups_probs_m'].append(dups_probs_m)

    # Remove duplicates of the kind: Berkeley 102, Berkeley102,
    # Berkeley_102; keeping only the name with the space
    for q, names in enumerate(new_db_dict['ID']):
        names_l = names.split(';')
        new_db_dict['ID'][q] = rm_name_dups(names_l)

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


def dups_identify(df, prob_cut=0.5, Nmax=3):
    """
    Find the closest clusters to all clusters
    """
    x, y = df['GLON'], df['GLAT']
    coords = np.array([x, y]).T
    # Find the distances to all clusters, for all clusters
    dist = cdist(coords, coords)
    pmRA, pmDE, plx = df['pmRA'], df['pmDE'], df['plx']

    dups_fnames, dups_probs = [], []
    for i, dists_i in enumerate(dist):

        # Only process relatively close clusters
        rad_max = Nmax * duplicate_probs.max_coords_rad(plx[i])
        msk_rad = dists_i <= rad_max
        idx_j = np.arange(0, len(dists_i))
        dists_i_msk = dists_i[msk_rad]
        j_msk = idx_j[msk_rad]

        dups_fname_i, dups_prob_i = [], []
        for k, d_ij in enumerate(dists_i_msk):
            j = j_msk[k]
            # Skip itself
            if j == i:
                continue

            # Fetch duplicated flag and probability for the i,j clusters
            dup_prob = duplicate_probs.run(x, y, pmRA, pmDE, plx, i, j)
            if dup_prob >= prob_cut:
                # print(df['fnames'][i], df['fnames'][j], dup_prob)
                # Store just the first fname
                dups_fname_i.append(df['fnames'][j].split(';')[0])
                dups_prob_i.append(str(dup_prob))

        if dups_fname_i:
            dups_fname_i = ";".join(dups_fname_i)
            dups_prob_i = ";".join(dups_prob_i)
        else:
            dups_fname_i, dups_prob_i = 'nan', 'nan'

        dups_fnames.append(dups_fname_i)
        dups_probs.append(dups_prob_i)

    return dups_fnames, dups_probs


def check_cents_diff(xy_c_o, vpd_c_o, plx_c_o, xy_c_n, vpd_c_n, plx_c_n):
    """
    """
    # cents_diff_close = False

    bad_center_xy, bad_center_pm, bad_center_plx = '0', '0', '0'

    # 5 arcmin maximum
    d_arcmin = np.sqrt((xy_c_o[0]-xy_c_n[0])**2+(xy_c_o[1]-xy_c_n[1])**2) * 60
    if d_arcmin > 5:
        bad_center_xy = '1'

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
            bad_center_pm = '1'

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
            bad_center_plx = '1'

    bad_center = bad_center_xy + bad_center_pm + bad_center_plx

    return bad_center
