import asteca
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from ..utils import check_centers
from .classification import get_classif
from .gaia_query_frames import run as query_run


def get_frame_limits(cl_ID: str, plx: float) -> tuple[float, float]:
    """
    Determines the frame size and minimum parallax for data retrieval based on cluster
    properties.

    Parameters
    ----------
    cl_ID : str
        Cluster name(s)
    plx: float | str
        Parallax value of the cluster

    Returns
    -------
    tuple
        A tuple containing:
        - box_s_eq (float): Size of the box to query (in degrees).
        - plx_min (float): Minimum parallax value for data retrieval.
    """
    if np.isnan(plx):
        c_plx = None
    else:
        c_plx = float(plx)

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

    # If the cluster is Ryu, use a fixed box size of 10 arcmin
    if "Ryu" in cl_ID:
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
    glon: float,
    glat: float,
    pmra: float,
    pmde: float,
    plx: float,
    tree: KDTree,
    box_s: float,
    df_gcs: pd.DataFrame,
) -> list[str]:
    """
    Identifies clusters and globular clusters (GCs) close to the specified coordinates.

    Parameters
    ----------
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    idx : int
        Index of the cluster in the UCC.
    glon : float
        Galactic longitude of the cluster.
    glat : float
        Galactic latitude of the cluster.
    pmra: float
        RA proper motion value of the cluster
    pmde: float
        DE proper motion value of the cluster
    plx: float | str
        Parallax value of the cluster
    tree : KDTree
        KDTree for nearest neighbor searches in the UCC.
    box_s : float
        Size of the box to query (in degrees).
    df_gcs : pd.DataFrame
        DataFrame of globular clusters.

    Returns
    -------
    list
        A list of strings, each representing a nearby cluster or GC with its
        coordinates and properties.
    """
    # Radius that contains the entire frame
    rad = np.sqrt(2 * (box_s / 2) ** 2)
    # Indexes to the closest clusters in XY
    ex_cls_idx = list(tree.query_ball_point([glon, glat], rad))
    # Remove self cluster
    del ex_cls_idx[ex_cls_idx.index(idx)]

    centers_ex = []
    for i in ex_cls_idx:
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
                (glon - float(df_UCC["GLON"][i])) ** 2
                + (glat - float(df_UCC["GLAT"][i])) ** 2
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
    ucc_glon, ucc_glat = df_UCC["GLON"][idx], df_UCC["GLAT"][idx]
    gc_d = np.sqrt(
        (ucc_glon - df_gcs["GLON"].values) ** 2
        + (ucc_glat - df_gcs["GLAT"].values) ** 2
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
        logging.info(f"  WARNING: close OCs to {(glon, glat)}, {pmra, pmde}, {plx}")
        for clust in centers_ex:
            logging.info("    " + clust)

    return centers_ex


def get_gaia_frame(
    logging,
    box_s,
    plx_min: float,
    frames_path: str,
    max_mag: float,
    frames_data: pd.DataFrame,
    manual_pars: pd.DataFrame,
    fnames: str,
    ra_icrs: float,
    de_icrs: float,
) -> pd.DataFrame:
    """
    Request Gaia data for a given position.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    box_s : float
        Size of the square frame to request (in degrees)
    plx_min : float
        Minimum parallax value
    frames_path : str
        Path to the directory containing Gaia data frames.
    max_mag : float
        Maximum magnitude for Gaia data retrieval.
    frames_data : pd.DataFrame
        DataFrame containing information about Gaia data frames.
    manual_pars : pd.DataFrame
        DataFrame with manual parameters for specific clusters.
    fnames : str
        Names associated to this cluster
    ra_icrs : float
        Right Ascension in ICRS coordinates.
    de_icrs : float
        Declination in ICRS coordinates.

    Returns
    -------
    - gaia_frame: DataFrame centered on cluster.
    """
    fname0 = str(fnames).split(";")[0]

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
        ra_icrs,
        de_icrs,
    )

    return gaia_frame


def get_fastMP_membs(
    logging,
    ra_icrs: float,
    de_icrs: float,
    glon: float,
    glat: float,
    pmra: float,
    pmde: float,
    plx: float,
    gaia_frame: pd.DataFrame,
) -> np.ndarray:
    """
    Runs fastMP for a given Gaia data frame.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    new_cl : pd.Series
        Series representing the new cluster.
    gaia_frame : pd.DataFrame
        Square frame with Gaia data to process

    Returns
    -------
    - probs_all: Array with probabilities
    """

    # Extract center coordinates from UCC
    lonlat_c = (glon, glat)
    vpd_c = (np.nan, np.nan)
    if not np.isnan(pmra):
        vpd_c = (pmra, pmde)
    plx_c = np.nan
    if not np.isnan(plx):
        plx_c = float(plx)

    # If the cluster has no PM or Plx center values assigned, run fastMP with fixed
    # (lon, lat) centers
    fixed_centers = False
    if np.isnan(vpd_c[0]) and np.isnan(plx_c):
        fixed_centers = True

    my_field = asteca.Cluster(
        ra=gaia_frame["RA_ICRS"],
        dec=gaia_frame["DE_ICRS"],
        pmra=gaia_frame["pmRA"],
        pmde=gaia_frame["pmDE"],
        plx=gaia_frame["Plx"],
        e_pmra=gaia_frame["e_pmRA"],
        e_pmde=gaia_frame["e_pmDE"],
        e_plx=gaia_frame["e_Plx"],
        N_clust_max=1500,
        verbose=0,
    )
    logging.info(f"  Setting N_clust_max={my_field.N_clust_max}")
    # Set radius as 10% of the frame's length, perhaps used in the number of members
    # estimation
    my_field.radius = (
        np.mean([np.ptp(gaia_frame["GLON"]), np.ptp(gaia_frame["GLAT"])]) * 0.1
    )

    # Process with fastMP
    while True:
        # logging.info(f"Fixed centers?: {fixed_centers}")
        probs_all = run_fastMP(
            logging,
            my_field,
            (ra_icrs, de_icrs),
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
    logging.info("  P>0.5={}, cents={}".format((probs_all > 0.5).sum(), cent_flags))

    return probs_all


def run_fastMP(
    logging,
    my_field: asteca.Cluster,
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
    my_field : asteca.Cluster
        Cluster object
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
    # Estimate the cluster's center coordinates
    my_field.get_center(radec_c=radec_c)
    if fixed_centers:
        my_field.radec_c = radec_c
        if not np.isnan(pms_c[0]):
            my_field.pms_c = pms_c
        if not np.isnan(plx_c):
            my_field.plx_c = plx_c
    logging.info(
        f"  Center used: ({my_field.radec_c[0]:.4f}, {my_field.radec_c[1]:.4f}), "
        + f"({my_field.pms_c[0]:.4f}, {my_field.pms_c[1]:.4f}), {my_field.plx_c:.4f}"
    )

    # Estimate the number of cluster members
    my_field.get_nmembers()

    if my_field.N_cluster == 1500:
        logging.info(
            f"  WARNING: N_cluster=1500, using radius={my_field.radius:.2f}"
        )
        my_field.get_nmembers("density")
        if my_field.N_cluster == 1500:
            my_field.N_cluster = 25
            logging.info("  WARNING: N_cluster=1500, setting value at: 25")
        else:
            logging.info(f"  N_cluster={my_field.N_cluster}")
    else:
        logging.info(f"  N_cluster={my_field.N_cluster}")

    # Define a ``membership`` object
    memb = asteca.Membership(my_field, verbose=0)

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

    # Store data used to update the UCC
    dict_UCC_updt = {
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
    return dict_UCC_updt


def updt_UCC_new_cl_data(df_UCC_updt, UCC_idx, dict_UCC_updt):
    """ """
    # Add UCC idx for this new entry
    df_UCC_updt["UCC_idx"].append(UCC_idx)

    # Update 'df_UCC_updt' with 'dict_UCC_updt' values
    for key, val in dict_UCC_updt.items():
        df_UCC_updt[key].append(val)

    return df_UCC_updt


def save_cl_datafile(
    logging,
    temp_fold: str,
    members_folder: str,
    fnames: str,
    quad: str,
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
    members_folder : str
        Path to the temporary members folder.
    fnames : str
        Names associated with the cluster.
    quad : str
        Quadrant of the cluster.
    """
    fname0 = str(fnames).split(";")[0]
    quad = quad + "/"

    # Order by probabilities
    df_membs = df_membs.sort_values("probs", ascending=False)

    out_fname = temp_fold + quad + members_folder + fname0 + ".parquet"
    df_membs.to_parquet(out_fname, index=False)
    logging.info(f"  Saved file to: {out_fname}")
