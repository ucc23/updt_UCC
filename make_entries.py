
import numpy as np
import json
import pandas as pd
from astropy.coordinates import angular_separation
from scipy import spatial
from add_new_DB import new_DB
from modules import ucc_plots, ucc_entry, fastMP_process


# Date of the latest version of the catalogue
UCC_cat_date_new = "20230508"


def main(entries_path="../ucc/_clusters/", N_membs_min=25):
    """
    """
    print("Reading databases...")
    with open('databases/all_dbs.json') as f:
        DBs_used = json.load(f)
    DBs_data = {}
    for k, v in DBs_used.items():
        DBs_data[k] = pd.read_csv("databases/" + k + '.csv')

    # Read latest UCC catalogue
    UCC_data = pd.read_csv('UCC_cat_' + UCC_cat_date_new + '.csv')

    # Load notebook template
    with open("notebook.txt", "r") as f:
        ntbk_str = f.readlines()

    for i, row in UCC_data.iterrows():

        # Only generate new entries foe those clusters in the recently added
        # database
        if new_DB not in row['DB']:
            continue

        fname0 = row['fnames'].split(';')[0]

        # if 'berkeley102' not in row['fnames']:
        #     continue
        print(row['DB'], fname0)

        Qfold = fastMP_process.QXY_fold(row)
        # Folder name where datafile is stored
        files_path = "../" + Qfold + "/datafiles/"
        # Folder names where files will be stored
        notb_path = "../" + Qfold + "/notebooks/"
        plots_path = "../" + Qfold + "/plots/"

        # Load data file with members for this cluster
        df_cl = pd.read_csv(files_path + fname0 + '.csv.gz')

        # Basic cluster data from its members
        df_membs, df_field, Nmemb, lon_c, lat_c, ra_c, dec_c, plx_c, pmRA_c,\
            pmDE_c, RV_c, probs_final = extract_cl_data(df_cl, N_membs_min)

        abcd, abcd_v = get_classif(
            df_membs, df_field, (lon_c, lat_c), (pmRA_c, pmDE_c), plx_c)
        # abcd_v = UCC_value(abcd)
        abcd_c = UCC_color(abcd)

        # xy_c, vpd_c, plx_c: literature values
        bad_center = check_centers(
            (row['GLON'], row['GLAT']), (row['pmRA'], row['pmDE']),
            row['plx'], (lon_c, lat_c), (pmRA_c, pmDE_c), plx_c)

        # Make catalogue entry
        DBs, DBs_i = row['DB'].split(';'), row['DB_i'].split(';')
        fpars_table = fpars_in_lit(DBs_used, DBs_data, DBs, DBs_i)
        posit_table = positions_in_lit(DBs_used, DBs_data, DBs, DBs_i)
        close_table = close_cat_cluster(UCC_data, UCC_data['fnames'], row)

        # All names for this cluster
        cl_names = row['ID'].split(';')
        ucc_entry.make_entry(
            entries_path, cl_names, fname0, row['UCC_ID'],
            abcd_v, abcd_c, Nmemb, lon_c, lat_c, ra_c, dec_c, plx_c, pmRA_c,
            pmDE_c, RV_c, fpars_table, posit_table, close_table)

        # Make notebook
        make_notebook(notb_path, ntbk_str, row['fnames'])

        # Make plot
        ucc_plots.make_plot(plots_path, row['fnames'], df_cl, N_membs_min)
        breakpoint()
        

def extract_cl_data(df_cl, N_membs_min, prob_min=0.5):
    """
    """
    probs_final = df_cl['probs']
    msk = probs_final > prob_min
    if msk.sum() < N_membs_min:
        idx = np.argsort(probs_final)[::-1][:N_membs_min]
        msk = np.full(len(probs_final), False)
        msk[idx] = True

    df_membs, df_field = df_cl[msk], df_cl[~msk]
    Nmembs = len(df_membs)

    lon, lat = np.nanmedian(df_membs['GLON']), np.nanmedian(df_membs['GLAT'])
    ra, dec = np.nanmedian(
        df_membs['RA_ICRS']), np.nanmedian(df_membs['DE_ICRS'])
    plx = np.nanmedian(df_membs['Plx'])
    pmRA, pmDE = np.nanmedian(df_membs['pmRA']), np.nanmedian(df_membs['pmDE'])
    RV = np.nan
    if not np.isnan(df_membs['RV'].values).all():
        RV = np.nanmedian(df_membs['RV'])
    lon, lat = round(lon, 3), round(lat, 3)
    ra, dec = round(ra, 3), round(dec, 3)
    plx = round(plx, 3)
    pmRA, pmDE = round(pmRA, 3), round(pmDE, 3)
    RV = round(RV, 3)

    return df_membs, df_field, Nmembs, lon, lat, ra, dec, plx, pmRA, pmDE,\
        RV, probs_final


def get_classif(df_membs, df_field, xy_c, vpd_c, plx_c, rad_max=2):
    """
    """
    lon_f, lat_f, pmRA_f, pmDE_f, plx_f = df_field['GLON'], df_field['GLAT'],\
        df_field['pmRA'], df_field['pmDE'], df_field['Plx']

    lon_m, lat_m, pmRA_m, pmDE_m, plx_m, = df_membs['GLON'],\
        df_membs['GLAT'], df_membs['pmRA'], df_membs['pmDE'],\
        df_membs['Plx']

    # Median distances to centers for members
    xy = np.array([lon_m, lat_m]).T
    xy_dists = spatial.distance.cdist(xy, np.array([xy_c])).T[0]
    xy_50 = np.nanmedian(xy_dists)
    pm = np.array([pmRA_m, pmDE_m]).T
    pm_dists = spatial.distance.cdist(pm, np.array([vpd_c])).T[0]
    pm_50 = np.nanmedian(pm_dists)
    plx_dists = abs(plx_m - plx_c)
    plx_50 = np.nanmedian(plx_dists)
    # Count member stars within median distances
    N_memb_xy = (xy_dists < xy_50).sum()
    N_memb_pm = (pm_dists < pm_50).sum()
    N_memb_plx = (plx_dists < plx_50).sum()

    # Median distances to centers for field stars
    xy = np.array([lon_f, lat_f]).T
    xy_dists_f = spatial.distance.cdist(xy, np.array([xy_c])).T[0]
    pm = np.array([pmRA_f, pmDE_f]).T
    pm_dists_f = spatial.distance.cdist(pm, np.array([vpd_c])).T[0]
    plx_dists_f = abs(plx_f - plx_c)
    # Count field stars within median distances
    N_field_xy = (xy_dists_f < xy_50).sum()
    N_field_pm = (pm_dists_f < pm_50).sum()
    N_field_plx = (plx_dists_f < plx_50).sum()

    def ABCD_classif(Nm, Nf):
        """Obtain 'ABCD' classification"""
        if Nm == 0:
            return "D"
        if Nf == 0:
            return "A"
        N_ratio = Nm / Nf

        if N_ratio >= 1:
            cl = "A"
        elif N_ratio < 1 and N_ratio >= 0.5:
            cl = "B"
        elif N_ratio < 0.5 and N_ratio > 0.1:
            cl = "C"
        else:
            cl = "D"
        return cl, min(N_ratio, 1)

    c_xy, ratio_xy = ABCD_classif(N_memb_xy, N_field_xy)
    c_pm, ratio_pm = ABCD_classif(N_memb_pm, N_field_pm)
    c_plx, ratio_plx = ABCD_classif(N_memb_plx, N_field_plx)
    classif = c_xy + c_pm + c_plx
    classif_v = round((ratio_xy + ratio_pm + ratio_plx) / 3., 2)

    return classif, classif_v


# def UCC_value(abcd_c):
#     """
#     """
#     class_2_num = {'A': 1., 'B': 0.5, 'C': 0.25, 'D': 0.1}
#     c1, c2, c3 = abcd_c
#     abcd_v = (class_2_num[c1] + class_2_num[c2] + class_2_num[c3]) / 3
#     return round(abcd_v, 2)


def UCC_color(abcd):
    """
    """
    abcd_c = ""
    line = r"""<span style="color: {}; font-weight: bold;">{}</span>"""
    cc = {'A': 'green', 'B': '#FFC300', 'C': 'red', 'D': 'purple'}
    for letter in abcd:
        abcd_c += line.format(cc[letter], letter) + '\n'
    return abcd_c


def check_centers(xy_c, vpd_c, plx_c, xy_c_f, vpd_c_f, plx_c_f):
    """
    xy_c, vpd_c, plx_c: literature values
    xy_c_f, vpd_c_f, plx_c_f: fastMP values
    """
    bad_center_xy, bad_center_pm, bad_center_plx = '0', '0', '0'

    # 5 arcmin maximum
    d_arcmin = angular_separation(xy_c_f[0], xy_c_f[1], xy_c[0], xy_c[1]) * 60
    if d_arcmin > 5:
        # print("d_arcmin: {:.1f}".format(d_arcmin))
        # print(xy_c, xy_c_f)
        bad_center_xy = '1'

    # Relative difference
    if vpd_c is not None:
        pm_max = []
        for vpd_c_i in abs(np.array(vpd_c)):
            if vpd_c_i > 10:
                pm_max.append(20)
            elif vpd_c_i > 1:
                pm_max.append(25)
            elif vpd_c_i > 0.1:
                pm_max.append(35)
            elif vpd_c_i > 0.01:
                pm_max.append(50)
            else:
                pm_max.append(70)
        pmra_p = 100 * abs(vpd_c_f[0] - vpd_c[0]) / (vpd_c[0] + 0.001)
        pmde_p = 100 * abs(vpd_c_f[1] - vpd_c[1]) / (vpd_c[1] + 0.001)
        if pmra_p > pm_max[0] or pmde_p > pm_max[1]:
            # print("pm: {:.2f} {:.2f}".format(pmra_p, pmde_p))
            # print(vpd_c, vpd_c_f)
            bad_center_pm = '1'

    # Relative difference
    if plx_c is not None:
        if plx_c > 0.2:
            plx_max = 25
        elif plx_c > 0.1:
            plx_max = 30
        elif plx_c > 0.05:
            plx_max = 35
        elif plx_c > 0.01:
            plx_max = 50
        else:
            plx_max = 70
        plx_p = 100 * abs(plx_c_f - plx_c) / (plx_c + 0.001)
        if abs(plx_p) > plx_max:
            # print("plx: {:.2f}".format(plx_p))
            # print(plx_c, plx_c_f)
            bad_center_plx = '1'

    bad_center = bad_center_xy + bad_center_pm + bad_center_plx

    return bad_center


def fpars_in_lit(DBs_used, DBs_data, DBs, DBs_i):
    """
    """

    # Select DBs with parameters and re-arrange them by year
    DBs_w_pars, DBs_i_w_pars = [], []
    for i, db in enumerate(DBs):
        # If this database contains any estimated fundamental parameters
        pars_f = (np.array(DBs_used[db]['pars'].split(',')) != 'None').any()
        if pars_f:
            DBs_w_pars.append(db)
            DBs_i_w_pars.append(DBs_i[i])
    # Extract years
    DBs_years = [int(_.split('_')[0][-2:]) for _ in DBs_w_pars]
    # Sort
    sort_idxs = np.argsort(DBs_years)
    # Re-arrange
    DBs_w_pars = np.array(DBs_w_pars)[sort_idxs]
    DBs_i_w_pars = np.array(DBs_i_w_pars)[sort_idxs]

    txt = ''
    for i, db in enumerate(DBs_w_pars):

        # # If this database contains any estimated fundamental parameters
        # pars_f = (np.array(DBs_used[db]['pars'].split(',')) != 'None').any()
        # if not pars_f:
        #     continue

        txt += '| '
        # Full 'db' database
        df = DBs_data[db]
        # Add reference
        txt += DBs_used[db]['ref'] + ' | `'
        # Add non-nan parameters
        for par in DBs_used[db]['pars'].split(','):
            # Read parameter value from DB as string
            if par != 'None':
                par_v = str(df[par][int(DBs_i_w_pars[i])])
                # Remove empty spaces if any
                par_v = par_v.replace(' ', '')
                try:
                    if par_v != '' and par_v != 'nan':
                        par_v = float(df[par][int(DBs_i_w_pars[i])])
                    else:
                        par_v = np.nan
                    if not np.isnan(par_v):
                        txt += par + '=' + str(round(par_v, 2)) + ', '
                except:
                    # DBs like SANTOS21 list more than 1 value for
                    # parameter
                    txt += par + '=' + par_v + ', '
            else:
                txt += '---=---, '
        # Close row
        txt = txt[:-2] + '` |\n'

    # Remove final new line
    if txt != '':
        table = txt[:-1]
    else:
        table = ''

    return table


def positions_in_lit(DBs_used, DBs_data, DBs, DBs_i):
    """
    """
    # Re-arrange DBs by year
    DBs_years = [int(_.split('_')[0][-2:]) for _ in DBs]
    # Sort
    sort_idxs = np.argsort(DBs_years)
    # Re-arrange
    DBs_sort = np.array(DBs)[sort_idxs]
    DBs_i_sort = np.array(DBs_i)[sort_idxs]

    txt = ''
    for i, db in enumerate(DBs_sort):
        txt += '|'
        # Full 'db' database
        df = DBs_data[db]
        # Add reference
        txt += DBs_used[db]['ref'] + ' | '

        # Add positions
        for c in DBs_used[db]['pos'].split(','):
            if c != 'None':
                # Read position as string
                pos_v = str(df[c][int(DBs_i_sort[i])])
                # Remove empty spaces if any
                pos_v = pos_v.replace(' ', '')
                if pos_v != '' and pos_v != 'nan':
                    txt += str(round(float(pos_v), 3)) + ' | '
                else:
                    txt += '--' + ' | '
            else:
                txt += '--' + ' | '
        # Close row
        txt = txt[:-2] + ' |\n'

    # Remove final new line
    if txt != '':
        table = txt[:-1]
    else:
        table = ''

    return table


def close_cat_cluster(fMP_data, fMP_fnames, row):
    """
    """
    close_table = ''
    if str(row['dups_fnames']) == 'nan':
        return close_table

    fnames0 = [_.split(';')[0] for _ in fMP_fnames]

    dups_fnames = row['dups_fnames'].split(';')

    for i, fname in enumerate(dups_fnames):
        close_table += '|'
        j = fnames0.index(fname)
        name = fMP_data['ID'][j].split(';')[0]
        close_table += f"[{name}](https://ucc.ar/_clusters/{fname}/) | "

        ra, dec, plx, pmRA, pmDE = [round(fMP_data[_][j], 3) for _ in (
            'RA_ICRS', 'DE_ICRS', 'plx', 'pmRA', 'pmDE')]
        close_table += f"{ra} | "
        close_table += f"{dec} | "
        close_table += f"{plx} | "
        close_table += f"{pmRA} | "
        close_table += f"{pmDE} |\n"
        # close_table += f"{Rv} |\n"
    # Remove final new line
    close_table = close_table[:-1]

    return close_table


def make_notebook(notb_path, ntbk_str, fnames):
    """
    """
    clname = fnames.split(';')[0]
    cl_str = r""""source": ["cluster = \"{}\""],""".format(clname)
    ntbk_str[30] = '      ' + cl_str + '\n'
    with open(notb_path + clname + ".ipynb", "w") as f:
        contents = "".join(ntbk_str)
        f.write(contents)


if __name__ == '__main__':
    main()
