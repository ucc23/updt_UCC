import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ..HARDCODED import (
    gaia_max_mag,
    local_asteca_path,
    members_folder,
    path_gaia_frames,
    temp_fold,
)
from ..utils import check_centers, radec2lonlat
from .classification import get_classif
from .gaia_query_frames import query_run

# Local version
sys.path.append(local_asteca_path)
import asteca


def get_fastMP_membs(
    logging,
    manual_pars: pd.DataFrame,
    df_UCC: pd.DataFrame,
    df_GCs: pd.DataFrame,
    gaia_frames_data,
    UCC_idx: int,
    ra_c: float,
    dec_c: float,
    glon_c: float,
    glat_c: float,
    pmra_c: float,
    pmde_c: float,
    plx_c: float,
    fname0: str,
    df_UCC_updt: dict,
) -> dict:
    """ """
    # If this is a 'manual' run, check if this OC is present in the file
    N_clust, N_clust_max, Nbox, frame_limit = np.nan, np.nan, np.nan, np.nan
    if manual_pars.empty is False:
        row = manual_pars[manual_pars["fname"].str.contains(fname0)].iloc[0]
        if row.empty is False:
            _, N_clust, N_clust_max, Nbox, frame_limit = row.to_numpy()
    if isinstance(frame_limit, float):
        frame_limit = ""

    # Obtain the full Gaia frame
    gaia_frame = get_gaia_frame(
        logging, gaia_frames_data, fname0, ra_c, dec_c, plx_c, Nbox, frame_limit
    )

    my_field = set_centers(gaia_frame, ra_c, dec_c, pmra_c, pmde_c, plx_c)
    logging.info(
        f"  Center used: ({my_field.radec_c[0]:.4f}, {my_field.radec_c[1]:.4f}), "
        + f"({my_field.pms_c[0]:.4f}, {my_field.pms_c[1]:.4f}), {my_field.plx_c:.4f}"
    )

    N_clust, N_clust_max = get_Nmembs(logging, N_clust, N_clust_max, my_field)

    # Only check if the number of members larger than the minimum value
    if my_field.N_cluster > my_field.N_clust_min:
        check_close_cls(
            logging,
            df_UCC,
            gaia_frame,
            fname0,
            glon_c,
            glat_c,
            pmra_c,
            pmde_c,
            plx_c,
            df_GCs,
        )

    # Define membership object
    memb = asteca.Membership(my_field, verbose=0)

    # Run fastMP
    probs_fastmp = memb.fastmp()
    logging.warning(f"probs_all>0.5={(probs_fastmp > 0.5).sum()}")

    # Check initial versus members centers
    xy_c_m, vpd_c_m, plx_c_m = extract_centers(my_field, probs_fastmp)
    cent_flags = check_centers(
        xy_c_m, vpd_c_m, plx_c_m, (glon_c, glat_c), (pmra_c, pmde_c), plx_c
    )[0]
    # "nnn" --> Centers are in agreement
    if cent_flags[0] != "n":
        logging.warning(f"  Centers flag: {cent_flags}")

    # Split into members and field stars according to the probability values
    # assigned
    df_field, df_membs = extract_members(gaia_frame, probs_fastmp)

    # Write selected member stars to file
    save_cl_datafile(logging, temp_fold, members_folder, fname0, df_membs)

    # Store data from this new entry used to update the UCC
    df_UCC_updt = updt_UCC_new_cl_data(
        df_UCC_updt, UCC_idx, df_field, df_membs, N_clust, N_clust_max
    )

    return df_UCC_updt


def get_gaia_frame(
    logging,
    gaia_frames_data,
    fname0,
    ra_c,
    dec_c,
    plx_c,
    Nbox: float,
    frame_limit: str,
    N_min_stars: int = 100,
    box_length_add: float = 0.5,
) -> pd.DataFrame:
    """ """
    # Extract possible manual frame limits
    frame_lims = []
    if frame_limit != "":
        for fm in frame_limit.split(","):
            vals = fm.split("_")
            if vals[0] not in ("b", "t", "l", "r"):
                raise ValueError(
                    f"Unknown frame limit '{vals[0]}', must be one of: b, t, l, r"
                )
            frame_lims.append([vals[0], float(vals[1])])

    # Make sure a minimum number of stars is present in the frame
    extra_length = 0.0
    while True:
        # Get frame limits
        box_s, plx_min = get_frame_limits(fname0, plx_c, extra_length)
        if not np.isnan(Nbox):
            box_s = box_s * Nbox

        # Request Gaia frame
        gaia_frame = query_run(
            logging,
            path_gaia_frames,
            gaia_frames_data,
            box_s,
            plx_min,
            gaia_max_mag,
            ra_c,
            dec_c,
            frame_lims,
        )

        if len(gaia_frame) < N_min_stars:
            extra_length += box_length_add
        else:
            break

    return gaia_frame


def get_frame_limits(
    fname: str, plx: float, extra_length: float
) -> tuple[float, float]:
    """
    Determines the frame size and minimum parallax for data retrieval based on cluster
    properties.

    Parameters
    ----------
    fname : str
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
    if fname.startswith("ryu"):
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

    box_s_eq += extra_length

    return box_s_eq, plx_min


def set_centers(
    gaia_frame: pd.DataFrame,
    ra_c: float,
    de_c: float,
    pmra_c_in: float,
    pmde_c_in: float,
    plx_c_in: float,
    max_dist_arcmin: int = 1,
) -> asteca.Cluster:
    """
    Estimate the cluster's center coordinates

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
    """
    my_field = asteca.Cluster(
        ra=np.array(gaia_frame["RA_ICRS"]),
        dec=np.array(gaia_frame["DE_ICRS"]),
        pmra=np.array(gaia_frame["pmRA"]),
        pmde=np.array(gaia_frame["pmDE"]),
        plx=np.array(gaia_frame["Plx"]),
        e_pmra=np.array(gaia_frame["e_pmRA"]),
        e_pmde=np.array(gaia_frame["e_pmDE"]),
        e_plx=np.array(gaia_frame["e_Plx"]),
        verbose=0,
    )

    radec_c = (ra_c, de_c)

    pms_c, plx_c = None, None
    if not np.isnan(pmra_c_in):
        pms_c = (pmra_c_in, pmde_c_in)
    if not np.isnan(plx_c_in):
        plx_c = plx_c_in

    my_field.get_center(radec_c=radec_c, pms_c=pms_c, plx_c=plx_c)

    # If no PMs and plx are given initially or the 'max_dist_arcmin' is exceeded,
    # use the initial (ra, dec) values to avoid wandering off the actual cluster
    d_arcmin = np.linalg.norm(np.array(my_field.radec_c) - np.array(radec_c)) * 60
    if (np.isnan(pmra_c_in) and np.isnan(plx_c_in)) or (d_arcmin > max_dist_arcmin):
        my_field.radec_c = radec_c

    # If PMs or plx are given initially, re-write using initial values
    if not np.isnan(pmra_c_in):
        my_field.pms_c = (pmra_c_in, pmde_c_in)
    if not np.isnan(plx_c_in):
        my_field.plx_c = plx_c_in

    return my_field


def get_Nmembs(
    logging,
    N_clust: float,
    N_clust_max: float,
    my_field: asteca.Cluster,
) -> tuple[int | float, int | float]:
    """
    Estimate the number of cluster members

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information
    N_clust: float
        Manual value for the fixed number of members
    N_clust_max: float
        Manual value for the maximum number of members
    my_field : asteca.Cluster
        ASteCA Cluster object
    """

    # If 'N_clust' was given, use it
    if not np.isnan(N_clust):
        my_field.N_cluster = int(N_clust)
        logging.info(f"  Using manual N_cluster={int(N_clust)}")
        return N_clust, N_clust_max
    # Else, if 'N_clust_max' was given use it to cap the maximum number of members
    elif not np.isnan(N_clust_max):
        my_field.N_clust_max = int(N_clust_max)
        logging.info(f"  Using manual N_clust_max={int(N_clust_max)}")

    # Use default ASteCA method
    my_field.get_nmembers()

    if my_field.N_cluster > my_field.N_clust_max:
        if not np.isnan(N_clust_max):
            my_field.N_cluster = my_field.N_clust_max
            txt = f", using N_cluster={N_clust_max}"
        else:
            txt = f", using N_cluster={my_field.N_clust_min}"
            my_field.N_cluster = my_field.N_clust_min
        logging.info(f"  WARNING: {my_field.N_cluster} > {my_field.N_clust_max}" + txt)
    else:
        logging.info(f"  Estimated N_cluster={my_field.N_cluster}")

    return N_clust, N_clust_max


def check_close_cls(
    logging,
    df_UCC,
    gaia_frame,
    fname,
    glon_c: float,
    glat_c: float,
    pmra_c: float,
    pmde_c: float,
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
    fname : str
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
    msk = in_frame["fnames"].str.split(";").str[0] != fname
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
    in_frame_gcs["Name"] = in_frame_gcs["Name"].str.strip()
    in_frame_gcs["Type"] = ["g"] * len(in_frame_gcs)

    # Combine DataFrames
    in_frame_all = pd.concat(
        [pd.DataFrame(in_frame_gcs), in_frame], axis=0, ignore_index=True
    )

    # Insert row at the top with the cluster under analysis
    new_row = {
        "Name": fname,
        "GLON": glon_c,
        "GLAT": glat_c,
        "plx": plx_c,
        "pmRA": pmra_c,
        "pmDE": pmde_c,
    }
    in_frame_all = pd.concat([pd.DataFrame([new_row]), in_frame_all], ignore_index=True)

    # Fetch duplicate probability
    dups_prob_i = []
    for j in range(1, len(in_frame_all)):
        dups_prob_i.append(
            dprob(
                np.array(in_frame_all["GLON"]),
                np.array(in_frame_all["GLAT"]),
                np.array(in_frame_all["pmRA"]),
                np.array(in_frame_all["pmDE"]),
                np.array(in_frame_all["plx"]),
                0,
                j,
            )
        )

    # Remove first row in dataframe
    in_frame_all = in_frame_all.drop(0, axis=0).reset_index(drop=True)
    # Add probabilities
    in_frame_all["P_d"] = dups_prob_i
    # Order by probability column
    in_frame_all = in_frame_all.sort_values("P_d", ascending=False).reset_index(
        drop=True
    )

    in_frame_all["Name"] = [_.split(";")[0] for _ in in_frame_all["Name"]]

    # Print info to screen
    in_frame_all = in_frame_all[
        ["Name", "P_d", "GLON", "GLAT", "plx", "pmRA", "pmDE", "Type"]
    ]
    if len(in_frame_all) > 0:
        logging.info(
            f"  WARNING: {len(in_frame_all)} extra OCs/GCs in frame: "
            + f"[{glon_c:.3f}, {glat_c:.3f}], {plx_c:.3f}"
        )
        for row in in_frame_all.to_string(index=False).split("\n")[:11]:
            logging.info("  " + row)
        if len(in_frame_all) > 10:
            logging.info(f"  ({len(in_frame_all) - 10} more)")


def extract_centers(
    my_field: asteca.Cluster,
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

    glon, glat = radec2lonlat(my_field.ra, my_field.dec)

    # Centers of selected members
    xy_c_m = np.nanmedian([np.array(glon)[msk], np.array(glat)[msk]], 1)
    vpd_c_m = np.nanmedian([my_field.pmra[msk], my_field.pmde[msk]], 1)
    plx_c_m = np.nanmedian(my_field.plx[msk])

    # pyright issue due to: https://github.com/numpy/numpy/issues/28076
    return xy_c_m, vpd_c_m, plx_c_m  # pyright: ignore


def extract_members(
    data: pd.DataFrame,
    probs_all: np.ndarray,
    prob_cut: float = 0.5,
    N_membs_min: int = 25,
    # perc_cut: int = 95,
    # N_perc: int = 2,
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
    tuple[pd.DataFrame, pd.DataFrame]
        DataFrames of field stars and cluster members
    """
    # Stars with probabilities greater than zero
    N_p_g_0 = (probs_all > 0.0).sum()

    # This should never happen but check anyhow
    if N_p_g_0 == 0:
        raise ValueError(
            "No stars with P>0.0, cannot select members. "
            + "Check the input data and parameters."
        )

    # Add probabilities to dataframe
    data["probs"] = np.round(probs_all, 3)

    if (probs_all >= prob_cut).sum() >= N_membs_min:
        # Apply prob_cut
        msk_membs = probs_all >= prob_cut
    else:
        # Select 'N_membs_min' maximum number of stars with P>0
        N_membs_min = min(N_membs_min, N_p_g_0)
        idx = np.argsort(probs_all)[::-1][:N_membs_min]
        # Select indexes
        msk_membs = np.full(len(probs_all), False)
        msk_membs[idx] = True

    return pd.DataFrame(data[~msk_membs]), pd.DataFrame(data[msk_membs])


def updt_UCC_new_cl_data(
    df_UCC_updt: dict,
    UCC_idx: int,
    df_field: pd.DataFrame,
    df_membs: pd.DataFrame,
    N_clust: int | float,
    N_clust_max: int | float,
    prob_cut: float = 0.5,
) -> dict:
    """
    Extracts cluster parameters from the member DataFrame.

    Parameters
    ----------
    df_field : pd.DataFrame
        DataFrame of field stars.
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    prob_cut : float, optional
        Probability threshold for calculating N_50. Default is 0.5.

    Returns
    -------
    pd.DataFrame
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
    C3 = get_classif(df_membs, df_field)

    # Temp dict used to update the UCC
    dict_UCC_updt = {
        "UCC_idx": UCC_idx,
        "N_clust": N_clust,
        "N_clust_max": N_clust_max,
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

    # Update 'df_UCC_updt' with 'dict_UCC_updt' values
    for key, val in dict_UCC_updt.items():
        df_UCC_updt[key].append(val)

    return df_UCC_updt


def save_cl_datafile(
    logging,
    temp_fold: str,
    members_folder: str,
    fname0: str,
    df_membs: pd.DataFrame,
) -> None:
    """
    Saves the cluster member data to a parquet file.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    temp_fold : str
        Path to the temporary folder.
    members_folder : str
        Path to the temporary members folder.
    fnames : str
        Name associated with the cluster file.
    """

    # Order by probabilities
    df_membs = df_membs.sort_values("probs", ascending=False)

    out_fname = temp_fold + members_folder + fname0 + ".parquet"
    df_membs.to_parquet(out_fname, index=False)
    logging.info(f"  Saved file to: {out_fname} (N={len(df_membs)})")


def dprob(
    x: np.ndarray,
    y: np.ndarray,
    pmRA: np.ndarray,
    pmDE: np.ndarray,
    plx: np.ndarray,
    i: int,
    j: int,
    Nmax: int = 2,
) -> float:
    """
    Calculate the probability of being duplicates for the i,j clusters

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates (e.g., GLON).
    y : np.ndarray
        Array of y-coordinates (e.g., GLAT).
    pmRA : np.ndarray
        Array of proper motion in RA.
    pmDE : np.ndarray
        Array of proper motion in DE.
    plx : np.ndarray
        Array of parallax values.
    i : int
        Index of the first cluster.
    j : int
        Index of the second cluster.
    Nmax : int, optional
        maximum number of times allowed for the two objects to be apart
        in any of the dimensions. If this happens for any of the dimensions,
        return a probability of zero. Default is 2.

    Returns
    -------
    float
        The probability of the two clusters being duplicates.
    """

    # Define reference parallax
    if np.isnan(plx[i]) and np.isnan(plx[j]):
        plx_ref = np.nan
    elif np.isnan(plx[i]):
        plx_ref = plx[j]
    elif np.isnan(plx[j]):
        plx_ref = plx[i]
    else:
        plx_ref = (plx[i] + plx[j]) * 0.5

    # Arbitrary 'duplicate regions' for different parallax brackets
    if np.isnan(plx_ref):
        rad, plx_r, pm_r = 2.5, np.nan, 0.2
    elif plx_ref >= 4:
        rad, plx_r, pm_r = 20, 0.5, 1
    elif 3 <= plx_ref < 4:
        rad, plx_r, pm_r = 15, 0.25, 0.75
    elif 2 <= plx_ref < 3:
        rad, plx_r, pm_r = 10, 0.2, 0.5
    elif 1.5 <= plx_ref < 2:
        rad, plx_r, pm_r = 7.5, 0.15, 0.35
    elif 1 <= plx_ref < 1.5:
        rad, plx_r, pm_r = 5, 0.1, 0.25
    elif 0.5 <= plx_ref < 1:
        rad, plx_r, pm_r = 2.5, 0.075, 0.2
    elif plx_ref < 0.5:
        rad, plx_r, pm_r = 2, 0.05, 0.15
    elif plx_ref < 0.25:
        rad, plx_r, pm_r = 1.5, 0.025, 0.1
    else:
        raise ValueError(
            "Could not define 'rad, plx_r, pm_r' values, plx_ref is out of bounds"
        )

    # Angular distance in arcmin
    d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) * 60
    # PMs distance
    pm_d = np.sqrt((pmRA[i] - pmRA[j]) ** 2 + (pmDE[i] - pmDE[j]) ** 2)
    # Parallax distance
    plx_d = abs(plx[i] - plx[j])

    # If *any* distance is *very* far away, return no duplicate disregarding
    # the rest with 0 probability
    if d > Nmax * rad or pm_d > Nmax * pm_r or plx_d > Nmax * plx_r:
        return 0

    d_prob = lin_relation(d, rad)
    pms_prob = lin_relation(pm_d, pm_r)
    plx_prob = lin_relation(plx_d, plx_r)

    # Combined probability
    prob = round(np.nanmean((d_prob, pms_prob, plx_prob)), 2)

    # pyright issue due to: https://github.com/numpy/numpy/issues/28076
    return prob  # pyright: ignore


def lin_relation(dist: float, d_max: float) -> float:
    """
    Calculates a linear probability based on distance.

    d_min=0 is fixed
    Linear relation for: (0, d_max), (1, d_min)

    Parameters
    ----------
    dist : float
        The distance between two objects.
    d_max : float
        The maximum distance for considering two objects related.

    Returns
    -------
    float
        A probability value between 0 and 1, where 1 indicates maximum proximity
        and 0 indicates the maximum distance `d_max`.
    """
    # m, h = (d_min - d_max), d_max
    # prob = (dist - h) / m
    # m, h = -d_max, d_max
    p = (dist - d_max) / -d_max
    if p < 0:  # np.isnan(p) or
        return 0
    return p
