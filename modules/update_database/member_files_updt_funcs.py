import asteca
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from ..utils import check_centers
from .classification import get_classif
from .gaia_query_frames import run as query_run


def process_new_OC(
    logging,
    box_s,
    plx_min: float,
    frames_path: str,
    max_mag: float,
    frames_data: pd.DataFrame,
    manual_pars: pd.DataFrame,
    new_cl: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Processes a new OC, performs data extraction, runs fastMP, and updates the UCC.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    frames_path : str
        Path to the directory containing Gaia data frames.
    max_mag : float
        Maximum magnitude for Gaia data retrieval.
    frames_data : pd.DataFrame
        DataFrame containing information about Gaia data frames.
    df_gcs : pd.DataFrame
        DataFrame of globular clusters.
    manual_pars : pd.DataFrame
        DataFrame with manual parameters for specific clusters.
    tree : KDTree
        KDTree for nearest neighbor searches in the UCC.
    UCC_idx : int
        Index of the new cluster in the UCC.
    new_cl : pd.Series
        Series representing the new cluster.

    Returns
    -------
    tuple
        A tuple containing:
        - gaia_frame: DataFrame centered on cluster.
        - probs_all: Array with probabilities
    """
    fname0 = str(new_cl["fnames"]).split(";")[0]

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

    # Request data
    gaia_frame = query_run(
        logging,
        frames_path,
        frames_data,
        box_s,
        plx_min,
        max_mag,
        float(new_cl["RA_ICRS"]),
        float(new_cl["DE_ICRS"]),
    )

    # Extract center coordinates from UCC
    lonlat_c = (float(new_cl["GLON"]), float(new_cl["GLAT"]))
    vpd_c = (np.nan, np.nan)
    if new_cl["pmRA"] != "nan":
        vpd_c = (float(new_cl["pmRA"]), float(new_cl["pmDE"]))
    plx_c = np.nan
    if new_cl["Plx"] != "nan":
        plx_c = float(new_cl["Plx"])

    # If the cluster has no PM or Plx center values assigned, run fastMP with fixed
    # (lon, lat) centers
    fixed_centers = False
    if np.isnan(vpd_c[0]) and np.isnan(plx_c):
        fixed_centers = True

    # Process with fastMP
    while True:
        # logging.info(f"Fixed centers?: {fixed_centers}")
        probs_all = run_fastMP(
            logging,
            gaia_frame,
            (float(new_cl["RA_ICRS"]), float(new_cl["DE_ICRS"])),
            vpd_c,
            plx_c,
            fixed_centers,
        )

        xy_c_m, vpd_c_m, plx_c_m = extract_centers(gaia_frame, probs_all)
        cent_flags = check_centers(xy_c_m, vpd_c_m, plx_c_m, lonlat_c, vpd_c, plx_c)
        if cent_flags == "nnn" or fixed_centers is True:
            break
        else:
            # Re-run with fixed centers
            fixed_centers = True

    xy_c_m, vpd_c_m, plx_c_m = extract_centers(gaia_frame, probs_all)
    cent_flags = check_centers(xy_c_m, vpd_c_m, plx_c_m, lonlat_c, vpd_c, plx_c)
    logging.info("P>0.5={}, cents={}".format((probs_all > 0.5).sum(), cent_flags))

    return gaia_frame, probs_all


def get_frame_limits(cl: pd.Series) -> tuple[float, float]:
    """
    Determines the frame size and minimum parallax for data retrieval based on cluster
    properties.

    Parameters
    ----------
    cl : pd.Series
        Series representing the cluster.

    Returns
    -------
    tuple
        A tuple containing:
        - box_s_eq (float): Size of the box to query (in degrees).
        - plx_min (float): Minimum parallax value for data retrieval.
    """
    if cl["Plx"] != "nan":
        c_plx = float(cl["Plx"])
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


def get_close_cls(
    logging,
    df_UCC,
    idx: int,
    new_cl,
    tree: KDTree,
    box_s: float,
    df_gcs: pd.DataFrame,
) -> list[str]:
    """
    Identifies clusters and globular clusters (GCs) close to the specified coordinates.

    Parameters
    ----------
    x : float
        Galactic longitude of the cluster.
    y : float
        Galactic latitude of the cluster.
    tree : KDTree
        KDTree for nearest neighbor searches in the UCC.
    box_s : float
        Size of the box to query (in degrees).
    idx : int
        Index of the cluster in the UCC.
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    df_gcs : pd.DataFrame
        DataFrame of globular clusters.

    Returns
    -------
    list
        A list of strings, each representing a nearby cluster or GC with its
        coordinates and properties.
    """
    x, y = float(new_cl["GLON"]), float(new_cl["GLAT"])

    # Radius that contains the entire frame
    rad = np.sqrt(2 * (box_s / 2) ** 2)
    # Indexes to the closest clusters in XY
    ex_cls_idx = list(tree.query_ball_point([x, y], rad))
    # Remove self cluster
    del ex_cls_idx[ex_cls_idx.index(idx)]

    # duplicate_cls = []
    # if str(dups_fnames) != "nan":
    #     duplicate_cls = dups_fnames.split(";")

    centers_ex = []
    for i in ex_cls_idx:
        # DEPRECATED: 27/01/25
        # # Check if this close cluster is identified as a probable duplicate
        # # of this cluster. If it is, do not add it to the list of extra
        # # clusters in the frame
        # skip_cl = False
        # if duplicate_cls:
        #     for dup_fname_i in str(df_UCC["fnames"][i]).split(";"):
        #         if dup_fname_i in duplicate_cls:
        #             skip_cl = True
        #             break
        #     if skip_cl:
        #         continue

        # If the cluster does not contain PM or Plx information, check its
        # distance in (lon, lat) with the main cluster. If the distance locates
        # this cluster within 0.75 of the frame's radius (i.e.: within the
        # expected region of the main cluster), don't store it for removal.
        #
        # This prevents clusters with no PM|Plx data from disrupting
        # neighboring clusters (e.g.: NGC 2516 disrupted by FSR 1479) and
        # at the same time removes more distant clusters that disrupt the
        # number of members estimation process in fastMP
        if (df_UCC["pmRA"][i] == "nan") or (df_UCC["pmRA"][i] == "nan"):
            xy_dist = np.sqrt(
                (x - float(df_UCC["GLON"][i])) ** 2
                + (y - float(df_UCC["GLAT"][i])) ** 2
            )
            if xy_dist < 0.75 * rad:
                continue

        ex_cl_dict = (
            f"{df_UCC['ID'][i]}: ({df_UCC['GLON'][i]:.4f}, {df_UCC['GLAT'][i]:.4f})"
        )
        if df_UCC["pmRA"][i] != "nan":
            ex_cl_dict += f", ({df_UCC['pmRA'][i]:.4f}, {df_UCC['pmDE'][i]:.4f})"
        if df_UCC["Plx"][i] != "nan":
            ex_cl_dict += f", {df_UCC['Plx'][i]:.4f}"

        centers_ex.append(ex_cl_dict)

    # Add closest GC
    glon, glat = df_UCC["GLON"][idx], df_UCC["GLAT"][idx]
    gc_d = np.sqrt(
        (glon - df_gcs["GLON"].values) ** 2 + (glat - df_gcs["GLAT"].values) ** 2
    )
    for i, gc_di in enumerate(gc_d):
        if gc_di < rad:
            ex_cl_dict = (
                f"{df_gcs['Name'][i]}: ({df_gcs['GLON'][i]:.4f}, {df_gcs['GLAT'][i]:.4f})"
                + f", ({df_gcs['pmRA'][i]:.4f}, {df_gcs['pmDE'][i]:.4f})"
                + f", {df_gcs['plx'][i]:.4f}"
            )
            centers_ex.append(ex_cl_dict)

    if len(centers_ex) > 0:
        logging.info("WARNING: there are clusters close by:")
        logging.info(
            f"*{new_cl['ID']}: {(new_cl['GLON'], new_cl['GLAT'])}"
            + f", {new_cl['pmRA'], new_cl['pmDE']}, {new_cl['Plx']} <-- Processed"
        )
        for clust in centers_ex:
            logging.info("  " + clust)

    return centers_ex


def run_fastMP(
    logging,
    field_df: pd.DataFrame,
    radec_c: tuple[float, float],
    pms_c: tuple[float, float],
    plx_c: float,
    fixed_centers: bool,
) -> np.ndarray:
    """
    Runs the fastMP algorithm to estimate membership probabilities.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    field_df : pd.DataFrame
        DataFrame of the field stars.
    radec_c : tuple
        Center coordinates (RA, Dec) for fastMP.
    pms_c : tuple
        Center proper motion (pmRA, pmDE) for fastMP.
    plx_c : float
        Center parallax for fastMP.
    fixed_centers : bool
        Boolean indicating whether to use fixed centers.

    Returns
    -------
    np.ndarray
        Array of membership probabilities.
    """
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
        verbose=0,
    )

    # Estimate the cluster's center coordinates
    my_field.get_center(radec_c=radec_c)
    if fixed_centers:
        my_field.radec_c = radec_c
        if not np.isnan(pms_c[0]):
            my_field.pms_c = pms_c
        if not np.isnan(plx_c):
            my_field.plx_c = plx_c
    logging.info(
        f"Center used: ({my_field.radec_c[0]:.4f}, {my_field.radec_c[1]:.4f}), "
        + f"({my_field.pms_c[0]:.4f}, {my_field.pms_c[1]:.4f}), {my_field.plx_c:.4f}"
    )

    # Estimate the number of cluster members
    my_field.get_nmembers()

    # Define a ``membership`` object
    memb = asteca.membership(my_field, verbose=0)

    # Run ``fastmp`` method
    probs_fastmp = memb.fastmp(fixed_centers=fixed_centers)

    return probs_fastmp


def extract_centers(
    data: pd.DataFrame,
    probs_all: np.ndarray,
    N_membs_min: int = 25,
    prob_cut: float = 0.5,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    Extracts the cluster center coordinates, proper motion, and parallax from
    high-probability members.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing cluster data.
    probs_all : np.ndarray
        Array of membership probabilities.
    N_membs_min : int, optional
        Minimum number of members to use for center estimation. Default is 25.
    prob_cut : float, optional
        Probability value to select members. Default is 0.5.

    Returns
    -------
    tuple
        A tuple containing:
        - xy_c_m: Center coordinates (lon, lat).
        - vpd_c_m: Center proper motion (pmRA, pmDE).
        - plx_c_m: Center parallax.
    """

    # Select high-quality members
    msk = probs_all > prob_cut
    # Use at least 'N_membs_min' stars
    if msk.sum() < N_membs_min:
        idx = np.argsort(probs_all)[::-1][:N_membs_min]
        msk = np.full(len(probs_all), False)
        msk[idx] = True

    # Centers of selected members
    xy_c_m = np.nanmedian([np.array(data["GLON"])[msk], np.array(data["GLAT"])[msk]], 1)
    vpd_c_m = np.nanmedian(
        [np.array(data["pmRA"])[msk], np.array(data["pmDE"])[msk]], 1
    )
    plx_c_m = np.nanmedian(np.array(data["Plx"])[msk])

    # pyright issue due to: https://github.com/numpy/numpy/issues/28076
    return xy_c_m, vpd_c_m, plx_c_m  # pyright: ignore


def split_membs_field(
    data: pd.DataFrame,
    probs_all: np.ndarray,
    prob_cut: float = 0.5,
    N_membs_min: int = 25,
    perc_cut: int = 95,
    N_perc: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into member and field star DataFrames based on membership
    probabilities.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing cluster data.
    probs_all : np.ndarray
        Array of membership probabilities.
    prob_cut : float, optional
        Probability threshold for considering a star a member. Default is 0.5.
    N_membs_min : int, optional
        Minimum number of members to use for filtering. Default is 25.
    perc_cut : int, optional
        Percentile to use for distance-based filtering. Default is 95.
    N_perc : int, optional
        Number of times the percentile distance to use for filtering. Default is 2.

    Returns
    -------
    tuple
        A tuple containing:
        - df_membs: DataFrame of cluster members.
        - df_field: DataFrame of field stars.
    """
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
    xy_dists = cdist(xy, np.array([xy_c])).T[0]
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


def extract_cl_data(
    df_membs: pd.DataFrame, df_field: pd.DataFrame, prob_cut: float = 0.5
) -> dict:
    """
    Extracts cluster parameters from the member DataFrame.

    Parameters
    ----------
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    df_field : pd.DataFrame
        DataFrame of field stars.
    prob_cut : float, optional
        Probability threshold for calculating N_50. Default is 0.5.

    Returns
    -------
    dict
        A dictionary containing:
        - lon: Median galactic longitude.
        - lat: Median galactic latitude.
        - ra: Median right ascension.
        - dec: Median declination.
        - plx: Median parallax.
        - pmRA: Median proper motion in RA.
        - pmDE: Median proper motion in DE.
        - Rv: Median radial velocity.
        - N_Rv: Number of stars with RV measurements.
        - N_50: Number of stars with membership probability above prob_cut.
        - r_50: Radius containing half the members.
    """
    N_50 = int((df_membs["probs"] >= prob_cut).sum())
    lon, lat = np.nanmedian(df_membs["GLON"]), np.nanmedian(df_membs["GLAT"])
    ra, dec = np.nanmedian(df_membs["RA_ICRS"]), np.nanmedian(df_membs["DE_ICRS"])
    plx = np.nanmedian(df_membs["Plx"])
    pmRA, pmDE = np.nanmedian(df_membs["pmRA"]), np.nanmedian(df_membs["pmDE"])
    Rv, N_Rv = np.nan, 0
    if not np.isnan(df_membs["RV"].values).all():
        Rv = np.nanmedian(df_membs["RV"])
        N_Rv = int(len(df_membs["RV"]) - np.isnan(df_membs["RV"].values).sum())
    lon, lat = round(lon, 3), round(lat, 3)
    ra, dec = round(ra, 3), round(dec, 3)
    plx = round(plx, 3)
    pmRA, pmDE = round(pmRA, 3), round(pmDE, 3)
    Rv = round(Rv, 3)

    # Radius that contains half the members
    xy = np.array([df_membs["GLON"].values, df_membs["GLAT"].values]).T
    xy_dists = cdist(xy, np.array([[lon, lat]])).T[0]
    r50_idx = np.argsort(xy_dists)[int(len(df_membs) / 2)]
    r_50 = xy_dists[r50_idx]
    # To arcmin
    r_50 = float(round(r_50 * 60.0, 1))

    # Classification data
    C1, C2, C3 = get_classif(df_membs, df_field)

    df_UCC_updt = {
        "C1": C1,
        "C2": C2,
        "C3": C3,
        "GLON_m": lon,
        "GLAT_m": lat,
        "RA_ICRS_m": ra,
        "DE_ICRS_m": dec,
        "Plx_m": plx,
        "pmRA_m": pmRA,
        "pmDE_m": pmDE,
        "Rv_m": Rv,
        "N_Rv": N_Rv,
        "N_50": N_50,
        "r_50": r_50,
    }
    return df_UCC_updt


def save_cl_datafile(
    logging,
    temp_fold: str,
    members_folder: str,
    cl: pd.Series,
    df_membs: pd.DataFrame,
) -> None:
    """
    Saves the cluster member data to a Parquet file.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    temp_fold : str
        Path to the temporary folder.
    cl : pd.Series
        Series representing the cluster.
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    """
    fname0 = str(cl["fnames"]).split(";")[0]
    quad = cl["quad"] + "/"

    # Order by probabilities
    df_membs = df_membs.sort_values("probs", ascending=False)

    out_fname = temp_fold + quad + members_folder + fname0 + ".parquet"
    df_membs.to_parquet(out_fname, index=False)
    logging.info(f"Saved file to: {out_fname}")
