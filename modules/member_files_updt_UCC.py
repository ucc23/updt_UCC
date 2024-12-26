import os

import asteca
import numpy as np
import pandas as pd
from scipy import spatial

from modules import GDR3_query as G3Q
from modules import aux, classif
from modules.HARDCODED import GCs_cat, dbs_folder, temp_fold


def run(logging, pars_dict, df_UCC):
    """
    Updates the Unified Cluster Catalogue (UCC) with new open clusters (OCs).
    This function performs the following steps:

    1. Reads extra input data including frame ranges, globular clusters (GCs) catalog,
       and manual parameters.
    2. Constructs a KD-tree for efficient spatial queries on the UCC.
    3. Processes each new OC by:
        a. Checking if it is a new OC that should be processed.
        b. Generating a frame for the OC.
        c. Applying manual parameters if available.
        d. Identifying close clusters.
        e. Requesting data for the OC.
        f. Processing the OC with the `fastMP` method.
        g. Splitting the data into members and field stars.
        h. Extracting cluster data and updating the UCC.
        i. Save members .parquet file in the proper Q folder

    Args:
        logging (logging.Logger): Logger instance for logging messages.
        pars_dict (dict): Dictionary containing parameters for the new database.
        df_UCC (pd.DataFrame): DataFrame containing the current UCC.

    Returns:
        pd.DataFrame: Updated UCC DataFrame with new OCs processed.
    """

    # Read extra input data
    frames_data = pd.read_csv(pars_dict["frames_ranges"])
    df_gcs = pd.read_csv(dbs_folder + GCs_cat)
    # Read OCs manual parameters
    manual_pars = pd.read_csv(pars_dict["manual_pars_f"])

    # Parameters used to search for close-by clusters
    xys = np.array([df_UCC["GLON"].values, df_UCC["GLAT"].values]).T
    tree = spatial.cKDTree(xys)

    # For each new OC
    for index, new_cl in df_UCC.iterrows():
        # Check if this is a new OC that should be processed
        if str(new_cl["C3"]) != "nan":
            continue

        logging.info(f"*** {index} Processing {new_cl['fnames']}")
        df_membs, df_UCC = process_new_OC(
            logging,
            df_UCC,
            pars_dict["frames_path"],
            pars_dict["max_mag"],
            pars_dict["verbose"],
            frames_data,
            df_gcs,
            manual_pars,
            tree,
            index,
            new_cl,
        )

        # Write member stars for cluster and some field
        save_cl_datafile(logging, new_cl, df_membs)

        logging.info(f"*** Cluster {new_cl['ID']} processed with fastMP\n")

    return df_UCC


def process_new_OC(
    logging,
    df_UCC,
    frames_path,
    max_mag,
    verbose,
    frames_data,
    df_gcs,
    manual_pars,
    tree,
    UCC_idx,
    new_cl,
):
    """ """
    # Identify position in the UCC
    fname0 = str(new_cl["fnames"]).split(";")[0]

    # Generate frame
    box_s, plx_min = get_frame(new_cl)

    #
    fix_N_clust = np.nan
    for _, row_manual_p in manual_pars.iterrows():
        if fname0 == row_manual_p["fname"]:
            if row_manual_p["Nmembs"] != "nan":
                fix_N_clust = int(row_manual_p["Nmembs"])
                logging.info(f"Manual N_membs applied: {fix_N_clust}")

            if row_manual_p["box_s"] != "nan":
                box_s = float(row_manual_p["box_s"])
                logging.info(f"Manual box size applied: {box_s}")

    # Get close clusters coords
    centers_ex = get_close_cls(
        new_cl["GLON"],
        new_cl["GLAT"],
        tree,
        box_s,
        UCC_idx,
        df_UCC,
        new_cl["dups_fnames"],
        df_gcs,
    )
    if len(centers_ex) > 0:
        logging.info("WARNING: there are clusters close by:")
        logging.info(
            f"*{new_cl['ID']}: {(new_cl['GLON'], new_cl['GLAT'])}"
            + f" {new_cl['pmRA'], new_cl['pmDE']} ({new_cl['Plx']})"
        )
        for gc in centers_ex:
            logging.info(gc)

    # Request data
    data = G3Q.run(
        logging,
        frames_path,
        frames_data,
        box_s,
        plx_min,
        max_mag,
        new_cl["RA_ICRS"],
        new_cl["DE_ICRS"],
        verbose,
    )

    # Extract center coordinates
    xy_c, vpd_c, plx_c = (new_cl["GLON"], new_cl["GLAT"]), None, None
    if not np.isnan(new_cl["pmRA"]):
        vpd_c = (new_cl["pmRA"], new_cl["pmDE"])
    if not np.isnan(new_cl["Plx"]):
        plx_c = new_cl["Plx"]

    fixed_centers = False
    if vpd_c is None and plx_c is None:
        fixed_centers = True

    # Process with fastMP
    while True:
        logging.info(f"Fixed centers?: {fixed_centers}")
        probs_all = run_fastMP(
            data, (new_cl["RA_ICRS"], new_cl["DE_ICRS"]), vpd_c, plx_c, fixed_centers
        )

        xy_c_m, vpd_c_m, plx_c_m = extract_centers(data, probs_all)
        cent_flags = aux.check_centers(xy_c_m, vpd_c_m, plx_c_m, xy_c, vpd_c, plx_c)

        if cent_flags == "nnn" or fixed_centers is True:
            break
        else:
            # Re-run with fixed centers
            fixed_centers = True

    xy_c_m, vpd_c_m, plx_c_m = extract_centers(data, probs_all)
    cent_flags = aux.check_centers(xy_c_m, vpd_c_m, plx_c_m, xy_c, vpd_c, plx_c)
    logging.info("\nP>0.5={}, cents={}".format((probs_all > 0.5).sum(), cent_flags))

    df_membs, df_field = split_membs_field(data, probs_all)
    C1, C2, C3 = classif.get_classif(df_membs, df_field)
    lon, lat, ra, dec, plx, pmRA, pmDE, Rv, N_Rv, N_50, r_50 = extract_cl_data(df_membs)
    logging.info(f"*{new_cl['ID']}: {(lon, lat)} {pmRA, pmDE} ({plx})")

    # Update UCC
    df_UCC.at[UCC_idx, "N_fixed"] = fix_N_clust
    df_UCC.at[UCC_idx, "fixed_cent"] = fixed_centers
    df_UCC.at[UCC_idx, "cent_flags"] = cent_flags
    df_UCC.at[UCC_idx, "C1"] = C1
    df_UCC.at[UCC_idx, "C2"] = C2
    df_UCC.at[UCC_idx, "C3"] = C3
    df_UCC.at[UCC_idx, "GLON_m"] = lon
    df_UCC.at[UCC_idx, "GLAT_m"] = lat
    df_UCC.at[UCC_idx, "RA_ICRS_m"] = ra
    df_UCC.at[UCC_idx, "DE_ICRS_m"] = dec
    df_UCC.at[UCC_idx, "Plx_m"] = plx
    df_UCC.at[UCC_idx, "pmRA_m"] = pmRA
    df_UCC.at[UCC_idx, "pmDE_m"] = pmDE
    df_UCC.at[UCC_idx, "Rv_m"] = Rv
    df_UCC.at[UCC_idx, "N_Rv"] = N_Rv
    df_UCC.at[UCC_idx, "N_50"] = N_50
    df_UCC.at[UCC_idx, "r_50"] = r_50
    logging.info(f"UCC entry for {new_cl['ID']} updated")

    return df_membs, df_UCC


def get_frame(cl):
    """ """
    if not np.isnan(cl["Plx"]):
        c_plx = cl["Plx"]
    else:
        c_plx = None

    if c_plx is None:
        box_s_eq = 0.5
    else:
        if c_plx > 10:
            box_s_eq = 25
        elif c_plx > 8:
            box_s_eq = 20
        elif c_plx > 6:
            box_s_eq = 15
        elif c_plx > 5:
            box_s_eq = 10
        elif c_plx > 4:
            box_s_eq = 7.5
        elif c_plx > 2:
            box_s_eq = 5
        elif c_plx > 1.5:
            box_s_eq = 3
        elif c_plx > 1:
            box_s_eq = 2
        elif c_plx > 0.75:
            box_s_eq = 1.5
        elif c_plx > 0.5:
            box_s_eq = 1
        elif c_plx > 0.25:
            box_s_eq = 0.75
        elif c_plx > 0.1:
            box_s_eq = 0.5
        else:
            box_s_eq = 0.25  # 15 arcmin

    if "Ryu" in cl["ID"]:
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
            plx_p = 0.7
        else:
            plx_p = 0.6
        plx_min = c_plx - plx_p

    return box_s_eq, plx_min


def get_close_cls(x, y, tree, box_s, idx, df_UCC, dups_fnames, df_gcs):
    """
    Get data on the closest clusters to the one being processed

    idx: Index to the cluster in the full list
    """

    # Radius that contains the entire frame
    rad = np.sqrt(2 * (box_s / 2) ** 2)
    # Indexes to the closest clusters in XY
    ex_cls_idx = tree.query_ball_point([x, y], rad)
    # Remove self cluster
    del ex_cls_idx[ex_cls_idx.index(idx)]

    duplicate_cls = []
    if str(dups_fnames) != "nan":
        duplicate_cls = dups_fnames.split(";")

    centers_ex = []
    for i in ex_cls_idx:
        # Check if this close cluster is identified as a probable duplicate
        # of this cluster. If it is, do not add it to the list of extra
        # clusters in the frame
        skip_cl = False
        if duplicate_cls:
            for dup_fname_i in df_UCC["fnames"][i].split(";"):
                if dup_fname_i in duplicate_cls:
                    # print("skip", df_UCC['fnames'][i])
                    skip_cl = True
                    break
            if skip_cl:
                continue

        # If the cluster does not contain PM or Plx information, check its
        # distance in (lon, lat) with the main cluster. If the distance locates
        # this cluster within 0.75 of the frame's radius (i.e.: within the
        # expected region of the main cluster), don't store it for removal.
        #
        # This prevents clusters with no PM|Plx data from disrupting
        # neighboring clusters (e.g.: NGC 2516 disrupted by FSR 1479) and
        # at the same time removes more distant clusters that disrupt the
        # number of members estimation process in fastMP
        if np.isnan(df_UCC["pmRA"][i]) or np.isnan(df_UCC["Plx"][i]):
            xy_dist = np.sqrt(
                (x - df_UCC["GLON"][i]) ** 2 + (y - df_UCC["GLAT"][i]) ** 2
            )
            if xy_dist < 0.75 * rad:
                # print(df_UCC['ID'][i], df_UCC['GLON'][i], df_UCC['GLAT'][i], xy_dist)
                continue

        # if np.isnan(df_UCC['pmRA'][i]) or np.isnan(df_UCC['plx'][i]):
        #     continue

        ex_cl_dict = f'{df_UCC["ID"][i]}: {(df_UCC["GLON"][i], df_UCC["GLAT"][i])}'
        if not np.isnan(df_UCC["pmRA"][i]):
            ex_cl_dict += f' {df_UCC["pmRA"][i], df_UCC["pmDE"][i]}'
        if not np.isnan(df_UCC["Plx"][i]):
            ex_cl_dict += f' ({df_UCC["Plx"][i]})'

        # print(df_UCC['ID'][i], ex_cl_dict)
        centers_ex.append(ex_cl_dict)

    # Add closest GC
    x, y = df_UCC["GLON"][idx], df_UCC["GLAT"][idx]
    gc_d = np.sqrt((x - df_gcs["GLON"]) ** 2 + (y - df_gcs["GLAT"]) ** 2).values
    for gc_i in range(len(df_gcs)):
        if gc_d[gc_i] < rad:
            ex_cl_dict = (
                f'{df_gcs["ID"][gc_i]}: {(df_gcs["GLON"][gc_i], df_gcs["GLAT"][gc_i])}'
                + f' {df_gcs["pmRA"][gc_i], df_gcs["pmDE"][gc_i]}'
                + f' ({df_gcs["Plx"][gc_i]})'
            )
            # print(df_gcs['Name'][gc_i], ex_cl_dict)
            centers_ex.append(ex_cl_dict)

    return centers_ex


def run_fastMP(field_df, radec_c, pms_c, plx_c, fixed_centers):
    """ """
    my_field = asteca.cluster(
        obs_df=field_df,
        ra="RA_ICRS",
        dec="DE_ICRS",
        pmra="pmRA",
        pmde="pmDE",
        plx="Plx",
        e_pmra="e_pmRA",
        e_pmde="e_pmDE",
        e_plx="e_Plx",
    )

    # Estimate the cluster's center coordinates
    my_field.get_center(radec_c=radec_c)
    if fixed_centers:
        my_field.radec_c = radec_c
        if pms_c is not None:
            my_field.pms_c = pms_c
        if plx_c is not None:
            my_field.plx_c = plx_c

    # Estimate the number of cluster members
    my_field.get_nmembers()

    # Define a ``membership`` object
    memb = asteca.membership(my_field)

    # Run ``fastmp`` method
    probs_fastmp = memb.fastmp()

    return probs_fastmp


def extract_centers(data, probs_all, N_membs_min=25):
    """Extract centers as medians of highest probability members"""

    # Select high-quality members
    msk = probs_all > 0.5
    # Use at least 'N_membs_min' stars
    if msk.sum() < N_membs_min:
        idx = np.argsort(probs_all)[::-1][:N_membs_min]
        msk = np.full(len(probs_all), False)
        msk[idx] = True

    # Centers of selected members
    xy_c_m = np.nanmedian([data["GLON"].values[msk], data["GLAT"].values[msk]], 1)
    vpd_c_m = np.nanmedian([data["pmRA"].values[msk], data["pmDE"].values[msk]], 1)
    plx_c_m = np.nanmedian(data["Plx"].values[msk])

    return xy_c_m, vpd_c_m, plx_c_m


def split_membs_field(
    data, probs_all, prob_cut=0.5, N_membs_min=25, perc_cut=95, N_perc=2
):
    """ """
    # This first filter removes stars beyond 2 times the 95th percentile
    # of the most likely members

    # Select most likely members
    msk_membs = probs_all >= prob_cut
    if msk_membs.sum() < N_membs_min:
        # Select the 'N_membs_min' stars with the largest probabilities
        idx = np.argsort(probs_all)[::-1][:N_membs_min]
        msk_membs = np.full(len(probs_all), False)
        msk_membs[idx] = True

    # Find xy filter
    xy = np.array([data["GLON"].values, data["GLAT"].values]).T
    xy_c = np.nanmedian(xy[msk_membs], 0)
    xy_dists = spatial.distance.cdist(xy, np.array([xy_c])).T[0]
    x_dist = abs(xy[:, 0] - xy_c[0])
    y_dist = abs(xy[:, 1] - xy_c[1])
    # 2x95th percentile XY mask
    xy_95 = np.percentile(xy_dists[msk_membs], perc_cut)
    xy_rad = xy_95 * N_perc
    msk_rad = (x_dist <= xy_rad) & (y_dist <= xy_rad)

    # Add a minimum probability mask to ensure that all stars with P>prob_min
    # are included
    msk_pmin = probs_all >= prob_cut

    # Combine masks with logical OR
    msk = msk_rad | msk_pmin

    # Generate filtered combined dataframe
    data["probs"] = np.round(probs_all, 5)
    # This dataframe contains both members and a selected portion of
    # field stars
    df_comb = data.loc[msk]
    df_comb.reset_index(drop=True, inplace=True)

    # Split into members and field, now using the filtered dataframe
    msk_membs = df_comb["probs"] > prob_cut
    if msk_membs.sum() < N_membs_min:
        idx = np.argsort(df_comb["probs"].values)[::-1][:N_membs_min]
        msk_membs = np.full(len(df_comb["probs"]), False)
        msk_membs[idx] = True
    df_membs, df_field = df_comb[msk_membs], df_comb[~msk_membs]

    return df_membs, df_field


def extract_cl_data(df_membs, prob_cut=0.5):
    """ """
    N_50 = (df_membs["probs"] >= prob_cut).sum()
    lon, lat = np.nanmedian(df_membs["GLON"]), np.nanmedian(df_membs["GLAT"])
    ra, dec = np.nanmedian(df_membs["RA_ICRS"]), np.nanmedian(df_membs["DE_ICRS"])
    plx = np.nanmedian(df_membs["Plx"])
    pmRA, pmDE = np.nanmedian(df_membs["pmRA"]), np.nanmedian(df_membs["pmDE"])
    RV, N_Rv = np.nan, 0
    if not np.isnan(df_membs["RV"].values).all():
        RV = np.nanmedian(df_membs["RV"])
        N_Rv = len(df_membs["RV"]) - np.isnan(df_membs["RV"].values).sum()
    lon, lat = round(lon, 3), round(lat, 3)
    ra, dec = round(ra, 3), round(dec, 3)
    plx = round(plx, 3)
    pmRA, pmDE = round(pmRA, 3), round(pmDE, 3)
    RV = round(RV, 3)

    # Radius that contains half the members
    xy = np.array([df_membs["GLON"].values, df_membs["GLAT"].values]).T
    xy_dists = spatial.distance.cdist(xy, np.array([[lon, lat]])).T[0]
    r50_idx = np.argsort(xy_dists)[int(len(df_membs) / 2)]
    r_50 = xy_dists[r50_idx]
    # To arcmin
    r_50 = round(r_50 * 60, 1)

    return lon, lat, ra, dec, plx, pmRA, pmDE, RV, N_Rv, N_50, r_50


def save_cl_datafile(logging, cl, df_comb):
    """ """
    fname0 = cl["fnames"].split(";")[0]
    quad = cl["quad"]

    out_path = temp_fold + quad + "/datafiles/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Order by probabilities
    df_comb = df_comb.sort_values("probs", ascending=False)

    out_fname = out_path + fname0 + ".parquet"
    df_comb.to_parquet(out_fname, index=False)
    logging.info(f"Saved file to: {out_fname}")
