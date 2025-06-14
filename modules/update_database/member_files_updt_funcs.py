import asteca
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# from modules.update_database.possible_duplicates_funcs import dprob
from ..utils import check_centers
from .classification import get_classif


def get_frame_limits(cl_ID: str, fname0: str, plx: float) -> tuple[float, float]:
    """
    Determines the frame size and minimum parallax for data retrieval based on cluster
    properties.

    Parameters
    ----------
    cl_ID : str
        Cluster name(s)
    fname0 : str
        Cluster file name
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

    # # Attempt to extract manual value for this cluster
    # try:
    #     idx = list(manual_pars["fname"]).index(fname0)
    #     manual_box_s = float(manual_pars["box_s"].values[idx])
    #     if not np.isnan(manual_box_s):
    #         box_s_eq = manual_box_s
    #         logging.info(f"Manual box size applied: {box_s_eq}")
    # except (ValueError, KeyError):
    #     pass

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
    gaia_frame,
    fname0,
    glon_c: float,
    glat_c: float,
    plx_c: float,
    df_gcs: pd.DataFrame,
) -> None:
    """
    Identifies clusters and globular clusters (GCs) close to the specified coordinates.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    gaia_frame : pd.DataFrame
        Square frame with Gaia data to process
    fname0 : str
        Cluster file name
    glon_c : float
        Galactic longitude of the cluster.
    glat_c : float
        Galactic latitude of the cluster.
    pmra_c : float
        Proper motion in right ascension of the cluster.
    pmde_c : float
        Proper motion in declination of the cluster.
    plx_c : float | str
        Parallax value of the cluster
    df_gcs : pd.DataFrame
        DataFrame of globular clusters.

    """
    # Frame limits
    l_min, l_max = gaia_frame["GLON"].min(), gaia_frame["GLON"].max()
    b_min, b_max = gaia_frame["GLAT"].min(), gaia_frame["GLAT"].max()
    plx_min = np.nanmin(gaia_frame["Plx"])

    # Find OCs in frame. Use coordinates estimated using members
    msk = (
        (df_UCC["GLON_m"] > l_min)
        & (df_UCC["GLON_m"] < l_max)
        & (df_UCC["GLAT_m"] > b_min)
        & (df_UCC["GLAT_m"] < b_max)
        & (df_UCC["Plx_m"] > plx_min)
    )
    in_frame = df_UCC[
        ["ID", "fnames", "GLON_m", "GLAT_m", "Plx_m", "pmRA_m", "pmDE_m"]
    ][msk]
    # Remove this OC from the dataframe
    msk = in_frame["fnames"].str.split(";").str[0] != fname0
    # Drop columns
    in_frame = in_frame[msk].drop("fnames", axis=1)
    # Assign type of object
    in_frame["Type"] = ["o"] * len(in_frame)
    # Rename columns to match GCs
    in_frame.rename(
        columns={
            "ID": "Name",
            "GLON_m": "GLON",
            "GLAT_m": "GLAT",
            "Plx_m": "plx",
            "pmRA_m": "pmRA",
            "pmDE_m": "pmDE",
        },
        inplace=True,
    )

    # Find GCs in frame
    msk = (
        (df_gcs["GLON"] > l_min)
        & (df_gcs["GLON"] < l_max)
        & (df_gcs["GLAT"] > b_min)
        & (df_gcs["GLAT"] < b_max)
        & (df_gcs["plx"] > plx_min)
    )
    in_frame_gcs = df_gcs[["Name", "GLON", "GLAT", "plx", "pmRA", "pmDE"]][msk]
    in_frame_gcs["Name"] = in_frame_gcs["Name"].str.strip()  # For prettier printing
    in_frame_gcs["Type"] = ["g"] * len(in_frame_gcs)

    # Combine DataFrames
    in_frame_all = pd.concat(
        [pd.DataFrame(in_frame_gcs), in_frame], axis=0, ignore_index=True
    )

    # # Insert row at the top with the cluster under analysis
    # new_row = {
    #     "Name": fname0,
    #     "GLON": glon_c,
    #     "GLAT": glat_c,
    #     "plx": plx_c,
    #     "pmRA": pmra_c,
    #     "pmDE": pmde_c,
    # }
    # in_frame_all = pd.concat([pd.DataFrame([new_row]), in_frame_all], ignore_index=True)

    # # Estimate duplicate probabilities between the analyzed cluster and those in frame
    # rm_idx, j = [0], 1
    # for _, row in in_frame_all[1:].iterrows():
    #     dup_prob = dprob(
    #         np.array(in_frame_all["GLON"]),
    #         np.array(in_frame_all["GLAT"]),
    #         np.array(in_frame_all["pmRA"]),
    #         np.array(in_frame_all["pmDE"]),
    #         np.array(in_frame_all["plx"]),
    #         0,  # Compare OCs in frame with self
    #         j,
    #     )
    #     j += 1
    #     # Remove OCs with a large duplicate probability of 90%
    #     if dup_prob > 0.9:
    #         rm_idx.append(j)
    # # Drop OCs that are identified as duplicates (and self)
    # in_frame_all = in_frame_all.drop(index=rm_idx)

    # Print info to screen
    if len(in_frame_all) > 0:
        logging.info(
            f"  WARNING: {len(in_frame_all)} extra OCs/GCs in frame: "
            + f"[{glon_c:.3f}, {glat_c:.3f}], {plx_c:.3f}"
        )
        for row in in_frame_all.to_string(index=False).split("\n")[:11]:
            logging.info("  " + row)
        if len(in_frame_all) > 10:
            logging.info(f"  ({len(in_frame_all) - 10} more)")

    # return in_frame_all


def get_fastMP_membs(
    logging,
    run_mode: str,
    manual_pars: pd.DataFrame,
    fname0: str,
    ra_c: float,
    de_c: float,
    glon_c: float,
    glat_c: float,
    pmra_c: float,
    pmde_c: float,
    plx_c: float,
    gaia_frame: pd.DataFrame,
    N_clust_max=1500,
) -> np.ndarray:
    """
    Runs fastMP for a given Gaia data frame.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    run_mode : str
        Mode of the run, e.g., 'update' or 'new'.
    manual_pars : pd.DataFrame
        Manual parameters
    fname0 : str
        Cluster file name
    ra_c, ..., plx_c : float
        Center values
    gaia_frame : pd.DataFrame
        Square frame with Gaia data to process

    Returns
    -------
    - probs_all: Array with probabilities
    """

    N_clust_manual = -1
    if run_mode == "manual":
        # Attempt to extract manual value for this cluster
        try:
            idx = list(manual_pars["fname"]).index(fname0)
            N_clust_manual = int(manual_pars["Nmembs"].values[idx])
        except ValueError:
            pass

    # Extract center coordinates from UCC
    lonlat_c = (glon_c, glat_c)
    radec_c = (ra_c, de_c)
    pms_c = (np.nan, np.nan)
    if not np.isnan(pmra_c):
        pms_c = (pmra_c, pmde_c)

    # If the cluster has no PM or Plx center values assigned, run fastMP with fixed
    # (lon, lat) centers
    fixed_centers = False
    if np.isnan(pms_c[0]) and np.isnan(plx_c):
        fixed_centers = True

    my_field = asteca.Cluster(
        ra=np.array(gaia_frame["RA_ICRS"]),
        dec=np.array(gaia_frame["DE_ICRS"]),
        pmra=np.array(gaia_frame["pmRA"]),
        pmde=np.array(gaia_frame["pmDE"]),
        plx=np.array(gaia_frame["Plx"]),
        e_pmra=np.array(gaia_frame["e_pmRA"]),
        e_pmde=np.array(gaia_frame["e_pmDE"]),
        e_plx=np.array(gaia_frame["e_Plx"]),
        N_clust_max=N_clust_max,
        verbose=0,
    )
    logging.info(f"  N_clust_max={my_field.N_clust_max}")
    # Set radius as 10% of the frame's length, perhaps used in the number of members
    # estimation
    my_field.radius = float(
        np.mean([np.ptp(gaia_frame["GLON"]), np.ptp(gaia_frame["GLAT"])]) * 0.1
    )

    # Process with fastMP
    while True:
        set_cents(
            logging,
            my_field,
            fixed_centers,
            radec_c,
            pms_c,
            plx_c,
        )

        set_Nmembs(logging, my_field, N_clust_max, N_clust_manual)

        probs_all = run_fastMP(
            my_field,
            fixed_centers,
        )

        xy_c_m, vpd_c_m, plx_c_m = extract_centers(gaia_frame, probs_all)
        cent_flags = check_centers(xy_c_m, vpd_c_m, plx_c_m, lonlat_c, pms_c, plx_c)[0]
        if cent_flags == "nnn" or fixed_centers is True:
            break
        else:
            # Re-run with fixed centers
            fixed_centers = True

    xy_c_m, vpd_c_m, plx_c_m = extract_centers(gaia_frame, probs_all)
    cent_flags = check_centers(xy_c_m, vpd_c_m, plx_c_m, lonlat_c, pms_c, plx_c)[0]
    logging.info("  P>0.5={}, cents={}".format((probs_all > 0.5).sum(), cent_flags))

    return probs_all


def set_cents(
    logging,
    my_field: asteca.Cluster,
    fixed_centers: bool,
    radec_c: tuple,
    pms_c: tuple,
    plx_c: float,
) -> None:
    """
    Estimate the cluster's center coordinates

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    my_field : asteca.Cluster
        Cluster object
    fixed_centers : bool
        Boolean indicating whether to use fixed centers.
    radec_c : tuple
        Center coordinates (RA, Dec) for fastMP.
    pms_c : tuple
        Center proper motion (pmRA, pmDE) for fastMP.
    plx_c : float
        Center parallax for fastMP.
    """
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


def set_Nmembs(
    logging, my_field: asteca.Cluster, N_clust_max: int, N_clust_manual: int
) -> None:
    """
    Estimate the number of cluster members

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information
    my_field : asteca.Cluster
        Cluster object
    N_clust_max : int
        Maximum number of cluster members to estimate
    N_clust_manual : int
        Manual number of cluster members
    """

    if N_clust_manual > 0:
        my_field.N_cluster = N_clust_manual
        logging.info(f"  Manual number of members applied: {N_clust_manual}")
        return

    # Use default ASteCA method
    my_field.get_nmembers()

    if my_field.N_cluster == N_clust_max:
        # If the default method results in the max number, try the 'density' method
        logging.info(
            f"  WARNING: estimated N_cluster={N_clust_max}, "
            + f"using radius={my_field.radius:.2f}"
        )
        my_field.get_nmembers("density")

        # If this method also estimates the max value, set the minimum
        if my_field.N_cluster == N_clust_max:
            my_field.N_cluster = my_field.N_clust_min
            logging.info(
                f"  WARNING: estimated N_cluster={N_clust_max}, "
                + f"setting N_cluster={my_field.N_clust_min}"
            )
        else:
            logging.info(f"  Setting N_cluster={my_field.N_cluster}")
    else:
        logging.info(f"  Setting N_cluster={my_field.N_cluster}")


def run_fastMP(
    my_field: asteca.Cluster,
    fixed_centers: bool,
) -> np.ndarray:
    """
    Runs the fastMP algorithm to estimate membership probabilities.

    Parameters
    ----------
    my_field : asteca.Cluster
        Cluster object
    fixed_centers : bool
        Boolean indicating whether to use fixed centers.

    Returns
    -------
    np.ndarray
        Array of membership probabilities.
    """
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
