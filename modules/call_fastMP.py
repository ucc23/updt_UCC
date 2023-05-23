
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import spatial
import csv


def run(
    fastMP, G3Q, frames_path, frames_data, df_UCC, df_gcs, UCC_cat, out_path,
    clusters_list, N_cl_extra=10, max_mag=20
):
    """
    N_cl_extra: number of extra clusters in frame to detect
    max_mag: maximum magnitude to retrieve
    """

    # Create output folders if not present
    for quad in ('1', '2', '3', '4'):
        for s in ('P', 'N'):
            Qfold = 'Q' + quad + s
            Path(out_path + Qfold + '/datafiles/').mkdir(
                parents=True, exist_ok=True)

    # Parameters used to search for close-by clusters
    xys = np.array([
        df_UCC['GLON'].values, df_UCC['GLAT'].values]).T
    tree = spatial.cKDTree(xys)
    close_cl_idx = tree.query(xys, k=N_cl_extra + 1)

    index_all, r50_all, N_fixed_all, N_survived_all, fixed_centers_all,\
        cent_flags_all, C1_all, C2_all, quad_all, membs_cents_all =\
        [[] for _ in range(10)]
    for index, cl in clusters_list.iterrows():

        # if 'berkeley29' not in cl['fnames']:
        #     continue

        index_all.append(index)

        print(f"*** {index} Processing {cl['ID']} with fastMP...")
        print(cl['GLON'], cl['GLAT'], cl['pmRA'], cl['pmDE'], cl['plx'])

        # Get close clusters coords
        centers_ex = get_close_cls(
            index, df_UCC, close_cl_idx, cl['dups_fnames'], df_gcs)

        # Generate frame
        data = get_frame(G3Q, max_mag, frames_path, frames_data, cl)
        # fname0 = cl['fnames'].split(';')[0]
        # Store full file
        # data.to_csv(out_path + fname0 + "_full.csv", index=False)
        # Read from file
        # data = pd.read_csv(out_path + fname0 + "_full.csv")

        # Extract center coordinates
        xy_c, vpd_c, plx_c = (cl['GLON'], cl['GLAT']), None, None
        if not np.isnan(cl['pmRA']):
            vpd_c = (cl['pmRA'], cl['pmDE'])
        if not np.isnan(cl['plx']):
            plx_c = cl['plx']

        fix_N_clust = False
        fixed_centers = False
        if vpd_c is None and plx_c is None:
            fixed_centers = True

        # Generate input data array for fastMP
        X = np.array([
            data['GLON'].values, data['GLAT'].values, data['pmRA'].values,
            data['pmDE'].values, data['Plx'].values, data['e_pmRA'].values,
            data['e_pmDE'].values, data['e_Plx'].values])

        # Process with fastMP
        while True:
            print("Fixed centers?:", fixed_centers)
            probs_all, N_survived = fastMP(
                xy_c=xy_c, vpd_c=vpd_c, plx_c=plx_c, centers_ex=centers_ex,
                fixed_centers=fixed_centers).fit(X)

            bad_center = check_centers(
                *X[:5, :], xy_c, vpd_c, plx_c, probs_all)

            if bad_center == '000' or fixed_centers is True:
                break
            else:
                # print("Re-run with fixed_centers = True")
                fixed_centers = True

        fixed_centers_all.append(fixed_centers)
        N_fixed_all.append(fix_N_clust)
        N_survived_all.append(int(N_survived))

        bad_center = check_centers(*X[:5, :], xy_c, vpd_c, plx_c, probs_all)
        cent_flags_all.append(bad_center)
        print("Nsurv={}, (P>0.5)={}, cents={}".format(
              N_survived, (probs_all > 0.5).sum(), bad_center))

        df_comb, df_membs, df_field, r_50, xy_c, vpd_c, plx_c =\
            split_membs_field(data, probs_all)
        r50_all.append(r_50)

        C1, C2 = get_classif(df_membs, df_field, xy_c, vpd_c, plx_c)
        C1_all.append(C1)
        C2_all.append(C2)

        N_50, lon, lat, ra, dec, plx, pmRA, pmDE, RV, N_Rv = extract_cl_data(
            df_membs)
        membs_cents_all.append([
            N_50, lon, lat, ra, dec, plx, pmRA, pmDE, RV, N_Rv])

        # Write member stars for cluster and some field
        save_cl_datafile(cl, df_comb, out_path)

        print(f"*** Cluster {cl['ID']} processed with fastMP\n")

    membs_cents_all = np.array(membs_cents_all).T
    # Load again in case it was updates while the script run
    df_UCC = pd.read_csv(UCC_cat)
    # Update these values for all the processed clusters
    for i, idx in enumerate(index_all):
        df_UCC.at[idx, 'r_50'] = r50_all[i]
        df_UCC.at[idx, 'N_fixed'] = N_fixed_all[i]
        df_UCC.at[idx, 'N_membs'] = int(N_survived_all[i])
        df_UCC.at[idx, 'fixed_cent'] = fixed_centers_all[i]
        df_UCC.at[idx, 'cent_flags'] = cent_flags_all[i]
        df_UCC.at[idx, 'C1'] = C1_all[i]
        df_UCC.at[idx, 'C2'] = C2_all[i]
        df_UCC.at[idx, 'N_50'] = membs_cents_all[0][i]
        df_UCC.at[idx, 'GLON_m'] = membs_cents_all[1][i]
        df_UCC.at[idx, 'GLAT_m'] = membs_cents_all[2][i]
        df_UCC.at[idx, 'RA_ICRS_m'] = membs_cents_all[3][i]
        df_UCC.at[idx, 'DE_ICRS_m'] = membs_cents_all[4][i]
        df_UCC.at[idx, 'plx_m'] = membs_cents_all[5][i]
        df_UCC.at[idx, 'pmRA_m'] = membs_cents_all[6][i]
        df_UCC.at[idx, 'pmDE_m'] = membs_cents_all[7][i]
        df_UCC.at[idx, 'Rv_m'] = membs_cents_all[8][i]
        df_UCC.at[idx, 'N_Rv'] = membs_cents_all[9][i]

    df_UCC.to_csv(UCC_cat, na_rep='nan', index=False,
                  quoting=csv.QUOTE_NONNUMERIC)


def read_input(frames_ranges, UCC_cat, GCs_cat):
    """
    Read input file with the list of clusters to process
    """
    frames_data = pd.read_csv(frames_ranges)
    df_UCC = pd.read_csv(UCC_cat)
    df_gcs = pd.read_csv(GCs_cat)
    return frames_data, df_UCC, df_gcs


def get_frame(G3Q, max_mag, frames_path, frames_data, cl):
    """
    """
    if not np.isnan(cl['plx']):
        c_plx = cl['plx']
    else:
        c_plx = None

    if c_plx is None:
        box_s_eq = 1
    else:
        if c_plx > 3:
            box_s_eq = min(50, 20 * np.log(.5*c_plx))
        # if c_plx > 15:
        #     box_s_eq = 30
        # elif c_plx > 4:
        #     box_s_eq = 20
        elif c_plx > 2:
            box_s_eq = 6
        elif c_plx > 1.5:
            box_s_eq = 5
        elif c_plx > 1:
            box_s_eq = 4
        elif c_plx > .75:
            box_s_eq = 3
        elif c_plx > .5:
            box_s_eq = 2
        elif c_plx > .25:
            box_s_eq = 1.5
        elif c_plx > .1:
            box_s_eq = 1
        else:
            box_s_eq = .5

    if 'Ryu' in cl['ID']:
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
            plx_p = .7
        else:
            plx_p = .6
        plx_min = c_plx - plx_p

    data = G3Q.run(frames_path, frames_data, cl['RA_ICRS'], cl['DE_ICRS'],
                   box_s_eq, plx_min, max_mag)

    return data


def get_close_cls(idx, database_full, close_cl_idx, dups, df_gcs):
    """
    Get data on the closest clusters to the one being processed

    idx: Index to the cluster in the full list
    """
    # Indexes to the closest clusters in XY
    ex_cls_idx = close_cl_idx[1][idx][1:]

    duplicate_cls = []
    if str(dups) != 'nan':
        duplicate_cls = dups.split(';')

    centers_ex = []
    for i in ex_cls_idx:

        # Check if this close cluster is identified as a probable duplicate
        # of this cluster. If it is, do not add it to the list of extra
        # clusters in the frame
        skip_cl = False
        if duplicate_cls:
            for dup_fname_i in database_full['fnames'][i].split(';'):
                if dup_fname_i in duplicate_cls:
                    # print("skip", database_full['fnames'][i])
                    skip_cl = True
                    break
            if skip_cl:
                continue

        ex_cl_dict = {
            'xy': [database_full['GLON'][i], database_full['GLAT'][i]]}

        # Only use clusters with defined centers in PMs and Plx, otherwise
        # non-bonafide clusters disrupt the process for established clusters
        # like NGC 2516
        if np.isnan(database_full['pmRA'][i])\
                or np.isnan(database_full['plx'][i]):
            continue

        if not np.isnan(database_full['pmRA'][i]):
            ex_cl_dict['pms'] = [
                database_full['pmRA'][i], database_full['pmDE'][i]]
        if not np.isnan(database_full['plx'][i]):
            ex_cl_dict['plx'] = [database_full['plx'][i]]

        # print(database_full['ID'][i], ex_cl_dict)

        centers_ex.append(ex_cl_dict)

    # Add closest GC
    x, y = database_full['GLON'][idx], database_full['GLAT'][idx]
    gc_i = np.argmin((x-df_gcs['GLON'])**2 + (y-df_gcs['GLAT'])**2)
    ex_cl_dict = {'xy': [df_gcs['GLON'][gc_i], df_gcs['GLAT'][gc_i]],
                  'pms': [df_gcs['pmRA'][gc_i], df_gcs['pmDE'][gc_i]],
                  'plx': [df_gcs['plx'][gc_i]]}
    # print(df_gcs['Name'][gc_i], ex_cl_dict)
    centers_ex.append(ex_cl_dict)

    return centers_ex


def check_centers(
    lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, probs_all, N_membs_min=25
):
    """
    """
    # Select high-quality members
    msk = probs_all > 0.5
    if msk.sum() < N_membs_min:
        idx = np.argsort(probs_all)[::-1][:N_membs_min]
        msk = np.full(len(probs_all), False)
        msk[idx] = True

    # Centers of selected members
    xy_c_f = np.nanmedian([lon[msk], lat[msk]], 1)
    vpd_c_f = np.nanmedian([pmRA[msk], pmDE[msk]], 1)
    plx_c_f = np.nanmedian(plx[msk])

    bad_center_xy, bad_center_pm, bad_center_plx = '0', '0', '0'

    # 5 arcmin maximum
    # d_arcmin = angular_separation(
    #     xy_c_f[0]*u.deg, xy_c_f[1]*u.deg, xy_c[0]*u.deg,
    #     xy_c[1]*u.deg).to('deg').value * 60
    d_arcmin = np.sqrt((xy_c_f[0]-xy_c[0])**2+(xy_c_f[1]-xy_c[1])**2) * 60
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


def split_membs_field(data, probs_all, prob_min=0.5, N_membs_min=25):
    """
    """
    # This first filter removes stars beyond 2 times the 95th percentile
    # of the most likely members

    # Select most likely members
    msk_membs = probs_all >= 0.5
    if msk_membs.sum() < N_membs_min:
        # Select the 'N_membs_min' stars with the largest probabilities
        idx = np.argsort(probs_all)[::-1][:N_membs_min]
        msk_membs = np.full(len(probs_all), False)
        msk_membs[idx] = True

    # Find xy filter
    xy = np.array([data['GLON'].values, data['GLAT'].values]).T
    xy_c = np.nanmedian(xy[msk_membs], 0)
    xy_dists = spatial.distance.cdist(xy, np.array([xy_c])).T[0]
    x_dist = abs(xy[:, 0] - xy_c[0])
    y_dist = abs(xy[:, 1] - xy_c[1])
    # 2x95th percentile XY mask
    xy_95 = np.percentile(xy_dists[msk_membs], 95)
    xy_rad = xy_95 * 2
    msk_rad = (x_dist <= xy_rad) & (y_dist <= xy_rad)

    # Add a minimum probability mask to ensure that all stars with P>prob_min
    # are included
    msk_pmin = probs_all >= prob_min

    # Combine masks with logical OR
    msk = msk_rad | msk_pmin

    # Generate filtered combined dataframe
    data['probs'] = np.round(probs_all, 5)
    # This dataframe contains both members and a selected portion of
    # field stars
    df_comb = data.loc[msk]
    df_comb.reset_index(drop=True, inplace=True)

    # Split into members and field, now using the filtered dataframe
    msk_membs = df_comb['probs'] > 0.5
    if msk_membs.sum() < N_membs_min:
        idx = np.argsort(df_comb['probs'].values)[::-1][:N_membs_min]
        msk_membs = np.full(len(df_comb['probs']), False)
        msk_membs[idx] = True
    df_membs, df_field = df_comb[msk_membs], df_comb[~msk_membs]

    # XY center and distances
    xy = np.array([df_membs['GLON'].values, df_membs['GLAT'].values]).T
    xy_c = np.nanmedian(xy, 0)
    xy_dists = spatial.distance.cdist(xy, np.array([xy_c])).T[0]

    # Radius that contains half the members
    r50_idx = np.argsort(xy_dists)[int(len(df_membs)/2)]
    r_50 = xy_dists[r50_idx]
    # To arcmin
    r_50 = round(r_50 * 60, 1)

    # PMs and Plx centers
    vpd_c = np.nanmedian(np.array([
        df_membs['pmRA'].values, df_membs['pmDE'].values]).T, 0)
    plx_c = np.nanmedian(df_membs['Plx'].values)

    return df_comb, df_membs, df_field, r_50, xy_c, vpd_c, plx_c


def get_classif(df_membs, df_field, xy_c, vpd_c, plx_c, rad_max=2):
    """
    """
    lon_f, lat_f, pmRA_f, pmDE_f, plx_f = df_field['GLON'].values,\
        df_field['GLAT'].values, df_field['pmRA'].values,\
        df_field['pmDE'].values, df_field['Plx'].values

    lon_m, lat_m, pmRA_m, pmDE_m, plx_m, = df_membs['GLON'].values,\
        df_membs['GLAT'].values, df_membs['pmRA'].values,\
        df_membs['pmDE'].values, df_membs['Plx'].values

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

    def ABCD_classif(Nm, Nf, ratio_max=10):
        """Obtain 'ABCD' classification"""
        if Nm == 0:
            return "D", 0
        if Nf == 0:
            return "A", ratio_max
        N_ratio = Nm / Nf

        if N_ratio >= 1:
            cl = "A"
        elif N_ratio < 1 and N_ratio >= 0.5:
            cl = "B"
        elif N_ratio < 0.5 and N_ratio > 0.1:
            cl = "C"
        else:
            cl = "D"
        return cl, min(N_ratio, ratio_max)

    c_xy, ratio_xy = ABCD_classif(N_memb_xy, N_field_xy)
    c_pm, ratio_pm = ABCD_classif(N_memb_pm, N_field_pm)
    c_plx, ratio_plx = ABCD_classif(N_memb_plx, N_field_plx)

    C1 = c_xy + c_pm + c_plx
    C2 = round((ratio_xy + ratio_pm + ratio_plx), 2)

    return C1, C2


def extract_cl_data(df_membs):
    """
    """
    N_50 = (df_membs['probs'] >= 0.5).sum()
    lon, lat = np.nanmedian(df_membs['GLON']), np.nanmedian(df_membs['GLAT'])
    ra, dec = np.nanmedian(
        df_membs['RA_ICRS']), np.nanmedian(df_membs['DE_ICRS'])
    plx = np.nanmedian(df_membs['Plx'])
    pmRA, pmDE = np.nanmedian(df_membs['pmRA']), np.nanmedian(df_membs['pmDE'])
    RV, N_Rv = np.nan, 0
    if not np.isnan(df_membs['RV'].values).all():
        RV = np.nanmedian(df_membs['RV'])
        N_Rv = len(df_membs['RV']) - np.isnan(df_membs['RV'].values).sum()
    lon, lat = round(lon, 3), round(lat, 3)
    ra, dec = round(ra, 3), round(dec, 3)
    plx = round(plx, 3)
    pmRA, pmDE = round(pmRA, 3), round(pmDE, 3)
    RV = round(RV, 3)

    return N_50, lon, lat, ra, dec, plx, pmRA, pmDE, RV, N_Rv


def save_cl_datafile(cl, df_comb, out_path):
    """
    """
    fname0 = cl['fnames'].split(';')[0]
    quad = cl['quad']

    # Order by probabilities
    df_comb = df_comb.sort_values('probs', ascending=False)

    df_comb.to_csv(out_path + quad + "/datafiles/" + fname0 + ".csv.gz",
                   index=False, compression='gzip')
    # # There's a ~10% reduction in size using parquet
    # df_comb.to_parquet(
    #     out_path + Qfold + "/datafiles/" + fname0 + ".parquet.gz",
    #     index=False, compression='gzip')

