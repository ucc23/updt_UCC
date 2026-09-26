import sys
import warnings

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord

from ..utils import plx_to_pc, radec2lonlat
from ..variables import (
    N_membs_min,
    gaia_max_mag,
    local_asteca_path,
    path_gaia_frames,
    prob_cut,
    temp_members_folder,
)
from .gaia_query_frames import query_run

# Local version
sys.path.append(local_asteca_path)
import asteca

print(f"ASteCA version: {asteca.__version__}")


def get_fastMP_membs(
    logging,
    df_GCs: pd.DataFrame,
    gaia_frames_data,
    df_UCC_m: pd.DataFrame,
    fname0,
    ra_c,
    dec_c,
    glon_c,
    glat_c,
    pmra_c,
    pmde_c,
    plx_c,
    N_clust,
    N_clust_max,
    box_size,
    frame_limit,
    rad_arcmin: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieves Gaia data for a specified cluster and processes it using the fastMP
    algorithm
    """
    # Obtain the full Gaia frame
    gaia_frame = get_gaia_frame(
        logging, gaia_frames_data, fname0, ra_c, dec_c, plx_c, box_size, frame_limit
    )
    # gaia_frame.to_csv("temp_clust.csv", index=False)
    # breakpoint()

    my_field = set_centers(gaia_frame, ra_c, dec_c, pmra_c, pmde_c, plx_c)
    logging.info(
        f"  Center used: ({my_field.radec_c[0]:.4f}, {my_field.radec_c[1]:.4f}), "
        + f"({my_field.pms_c[0]:.4f}, {my_field.pms_c[1]:.4f}), {my_field.plx_c:.4f}"
    )

    get_Nmembs(logging, N_clust, N_clust_max, my_field)

    # Only check if the number of members is larger than the minimum value
    if my_field.N_cluster > my_field.N_clust_min:
        check_close_cls(
            logging,
            df_UCC_m,
            gaia_frame,
            fname0,
            glon_c,
            glat_c,
            pmra_c,
            pmde_c,
            plx_c,
            df_GCs,
        )

    # Run fastMP
    my_field.membership.fastmp()
    probs_fastmp = my_field.probs
    logging.info(f"probs_all>=0.5={(probs_fastmp >= 0.5).sum()}")

    # Check initial versus members centers
    center_check(glon_c, glat_c, my_field, probs_fastmp)

    # Split into members and field stars according to the probability values
    # assigned
    df_field, df_membs = extract_members(
        gaia_frame,
        probs_fastmp,
        radec_cent=my_field.radec_c,
        N_membs=my_field.N_cluster,
        rad_arcmin=rad_arcmin,
    )

    return df_field, df_membs


def get_gaia_frame(
    logging,
    gaia_frames_data,
    fname0,
    ra_c,
    dec_c,
    plx_c,
    box_size: float = np.nan,
    frame_limit: str = "",
    N_min_stars: int = 100,
    box_length_add: float = 0.5,
) -> pd.DataFrame:
    """
    Retrieves a Gaia frame for a specified cluster, ensuring a minimum number of stars
    """
    # Extract possible manual frame limits
    frame_lims = []
    if frame_limit != "":
        for fm in frame_limit.split(","):
            vals = fm.split("_")
            if vals[0] not in (
                "b",
                "t",
                "l",
                "r",
                "plxl",
                "plxr",
                "pmb",
                "pmt",
                "pml",
                "pmr",
            ):
                raise ValueError(f"Unknown frame limit '{vals[0]}'")
            frame_lims.append([vals[0], float(vals[1])])

    # Make sure a minimum number of stars is present in the frame
    extra_length = 0.0
    while True:
        # Get frame limits
        box_s, plx_min = get_frame_limits(fname0, plx_c, extra_length)
        if not np.isnan(box_size):
            box_s = box_size

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
) -> None:
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
        return

    # If 'N_clust_max' was given use it to cap the maximum number of members
    if not np.isnan(N_clust_max):
        my_field.N_clust_max = int(N_clust_max)
        logging.info(f"  Using manual N_clust_max={int(N_clust_max)}")

    # Use default ASteCA method
    my_field.get_nmembers()
    logging.info(f"  Using N_clust={int(my_field.N_cluster)}")


def check_close_cls(
    logging,
    df_UCC_m,
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
        (df_UCC_m["GLON_m"] > l_min)
        & (df_UCC_m["GLON_m"] < l_max)
        & (df_UCC_m["GLAT_m"] > b_min)
        & (df_UCC_m["GLAT_m"] < b_max)
        & (df_UCC_m["Plx_m"] > plx_min)
    )
    in_frame = df_UCC_m[msk].copy()
    # Assign type of object
    in_frame["Type"] = ["o"] * len(in_frame)
    # Rename columns to match GCs
    in_frame.rename(
        columns={
            "fname": "Name",
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
    # in_frame_gcs["Name"] = in_frame_gcs["Name"].str.strip()
    in_frame_gcs["Name"] = [_.strip() for _ in in_frame_gcs["Name"]]
    in_frame_gcs["Type"] = ["g"] * len(in_frame_gcs)

    # Combine DataFrames
    frames = []
    df1 = pd.DataFrame(in_frame_gcs)
    if not df1.empty and not df1.isna().all().all():
        frames.append(df1)
    if not in_frame.empty and not in_frame.isna().all().all():
        frames.append(in_frame)

    new_row = pd.DataFrame(
        [
            {
                "Name": fname,
                "GLON": glon_c,
                "GLAT": glat_c,
                "plx": plx_c,
                "pmRA": pmra_c,
                "pmDE": pmde_c,
                "Type": "o",
            }
        ]
    )
    if frames:
        frames_concat = pd.concat(frames, ignore_index=True)
        # Insert row at the top with the cluster under analysis
        in_frame_all = pd.concat([new_row, frames_concat], ignore_index=True)
    else:
        in_frame_all = new_row

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
    # Remove any row['Name']==fname
    in_frame_all = in_frame_all[in_frame_all["Name"] != fname].reset_index(drop=True)

    # Print info to screen
    in_frame_all = in_frame_all[
        ["Name", "P_d", "GLON", "GLAT", "plx", "pmRA", "pmDE", "Type"]
    ]
    if len(in_frame_all) > 0:
        logging.info(
            f"  WARNING: {len(in_frame_all)} OCs/GCs in frame: "
            + f"[{glon_c:.3f}, {glat_c:.3f}], {plx_c:.3f}"
        )
        for row in in_frame_all.to_string(index=False).split("\n")[:11]:
            logging.info("  " + row)
        if len(in_frame_all) > 10:
            logging.info(f"  ({len(in_frame_all) - 10} more)")


def center_check(
    glon_c: float,
    glat_c: float,
    my_field: asteca.Cluster,
    probs_all: np.ndarray,
    prob_cut: float = 0.5,
    rad_max: float = 15,
) -> None:
    """
    Extracts the cluster center coordinates, proper motion, and parallax from
    high-probability members.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing cluster data.
    probs_all : np.ndarray
        Array of membership probabilities.
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
    lonlat_c_m = np.nanmedian([np.array(glon)[msk], np.array(glat)[msk]], 1)
    # vpd_c_m = np.nanmedian([my_field.pmra[msk], my_field.pmde[msk]], 1)
    # plx_c_m = np.nanmedian(my_field.plx[msk])

    # cent_flags = check_centers(
    #     xy_c_m, vpd_c_m, plx_c_m, (glon_c, glat_c), (pmra_c, pmde_c), plx_c
    # )[0]
    # # "nnn" --> Centers are in agreement
    # if cent_flags[0] != "n":
    d_arcmin = (
        np.sqrt((glon_c - lonlat_c_m[0]) ** 2 + (glat_c - lonlat_c_m[1]) ** 2) * 60
    )
    # Store information on the OCs that require attention
    if d_arcmin > rad_max:
        warnings.warn(f"\nWARNING: Distance between centers {d_arcmin:.1f} [arcmin]")

    # pyright issue due to: https://github.com/numpy/numpy/issues/28076
    # return lonlat_c_m, vpd_c_m, plx_c_m  # pyright: ignore


def dist_cent_arcmin(data, radec_cent):
    """
    Compute the angular distance in arcminutes between each star in the DataFrame
    """
    ra_cent, dec_cent = radec_cent

    ra = np.deg2rad(data["RA_ICRS"].to_numpy(dtype=float))
    dec = np.deg2rad(data["DE_ICRS"].to_numpy(dtype=float))
    ra_cent = np.deg2rad(ra_cent)
    dec_cent = np.deg2rad(dec_cent)

    # Angular separation using the haversine formula
    dra = ra - ra_cent
    ddec = dec - dec_cent
    a = (
        np.sin(ddec / 2.0) ** 2
        + np.cos(dec_cent) * np.cos(dec) * np.sin(dra / 2.0) ** 2
    )
    sep_rad = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    sep_arcmin = np.rad2deg(sep_rad) * 60.0

    return sep_arcmin


def extract_members(
    data: pd.DataFrame,
    probs_all: np.ndarray,
    radec_cent: tuple[float, float],
    N_membs: int,
    rad_arcmin: None | float = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into field and member-star DataFrames.

    Select cluster members using a radius-constrained probability ranking when
    `rad_arcmin` is provided; otherwise, use a probability-threshold selection with
    fallbacks to the highest non-zero probabilities or the stars closest to the
    cluster center.

    if rad_arcmin is provided
    │
    ├── N_inside >= N_membs
    │   └── select N_membs stars with highest P inside radius
    │
    ├── N_inside >= N_membs_min
    │   └── select all stars inside radius
    │
    └── N_inside < N_membs_min
        └── select N_membs_min stars closest to center
    else
    │
    ├── N(P >= prob_cut) >= N_membs_min
    │   └── select all stars with P >= prob_cut
    │
    ├── N(P > 0) >= N_membs_min
    │   └── select N_membs_min stars with highest P > 0
    │
    ├── N(P > 0) > 0
    │   └── select all stars with P > 0
    │
    └── no stars with P > 0
        └── select N_membs_min stars closest to center

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing cluster data.
    probs_all : np.ndarray
        Membership probabilities.
    radec_cent : tuple[float, float]
        Cluster center as (RA, DEC), in degrees.
    N_membs : int
        Number of members to select within `rad_arcmin`.
    rad_arcmin : float or None, optional
        Maximum angular distance from the center, in arcminutes.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Field stars and cluster members.
    """
    # Add probabilities to dataframe
    data["probs"] = np.round(probs_all, 3)

    # Initialize membership mask (all False)
    msk_membs = np.zeros(len(probs_all), dtype=bool)

    if rad_arcmin is not None:
        # The cluster's radius was given, use it to select members
        sep_arcmin = dist_cent_arcmin(data, radec_cent)

        # Candidate stars inside requested radius
        idx_inside = np.flatnonzero(sep_arcmin <= rad_arcmin)
        N_in_rad = len(idx_inside)

        if N_in_rad >= N_membs:
            # Select N_membs stars inside the radius with the  largest probabilities
            idx_sorted = idx_inside[np.argsort(probs_all[idx_inside])[::-1]]
            idx_selected = idx_sorted[:N_membs]
        else:
            if N_in_rad >= N_membs_min:
                warnings.warn(
                    f"Not enough stars inside radius ({N_in_rad} < {N_membs}), "
                    + "using all stars inside radius"
                )
                idx_selected = idx_inside
            else:
                # Select N_membs_min closest stars to center
                warnings.warn(
                    f"Not enough stars inside radius ({N_in_rad} < {N_membs_min}), "
                    + f"using {N_membs_min} closest stars to center"
                )
                idx_sorted = np.argsort(sep_arcmin)
                idx_selected = idx_sorted[:N_membs_min]

    else:
        # If no radius is given, use the standard probability-cut method

        if (probs_all >= prob_cut).sum() >= N_membs_min:
            # Use default probability threshold for membership
            idx_selected = np.flatnonzero(probs_all >= prob_cut)
        else:
            # Stars with membership probabilities larger than 0
            msk_probs_g_0 = probs_all > 0.0
            N_p_g_0 = msk_probs_g_0.sum()

            if N_p_g_0 >= N_membs_min:
                warnings.warn(
                    f"Not enough stars with P>{prob_cut}, using the {N_membs_min} "
                    + "stars with the largest P>0"
                )
                # Select N_membs_min stars with largest P>0
                idx_p_g_0 = np.flatnonzero(msk_probs_g_0)
                idx_sorted = idx_p_g_0[np.argsort(probs_all[idx_p_g_0])[::-1]]
                idx_selected = idx_sorted[:N_membs_min]
            elif N_p_g_0 > 0:
                warnings.warn(
                    f"Not enough stars with P>{prob_cut}, using all {N_p_g_0} "
                    + "stars with P>0"
                )
                idx_selected = np.flatnonzero(msk_probs_g_0)
            else:
                # If no stars have P>0, select the N_membs_min stars closest to the center
                warnings.warn(
                    f"No stars with P>0, using the {N_membs_min} closest "
                    + "stars to the center"
                )
                sep_arcmin = dist_cent_arcmin(data, radec_cent)
                idx_sorted = np.argsort(sep_arcmin)
                idx_selected = idx_sorted[:N_membs_min]

    msk_membs[idx_selected] = True

    # Return field stars and cluster members
    return data[~msk_membs], data[msk_membs]


def get_new_cl_data(
    df_membs: pd.DataFrame,
    prob_cut: float = 0.5,
    N_digits: int = 5,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    float,
]:
    """
    Extracts cluster parameters from the member DataFrame.

    df_membs : pd.DataFrame
        DataFrame of cluster members.
    prob_cut : float, optional
        Probability threshold for calculating N_membs. Default is 0.5.

    """
    # Number of estimated members (P>50%)
    N_membs = int((df_membs["probs"] >= prob_cut).sum())

    # Center values
    c_lon, c_lat = np.nanmedian(df_membs["GLON"]), np.nanmedian(df_membs["GLAT"])
    c_ra, c_dec = np.nanmedian(df_membs["RA_ICRS"]), np.nanmedian(df_membs["DE_ICRS"])
    c_plx = np.nanmedian(df_membs["Plx"])
    c_pmRA, c_pmDE = np.nanmedian(df_membs["pmRA"]), np.nanmedian(df_membs["pmDE"])
    c_Rv = df_membs["RV"].median() if df_membs["RV"].count() else np.nan
    N_Rv = df_membs["RV"].count()

    # TODO: fix this
    e_plx = np.nanmedian(df_membs["e_Plx"])
    e_pmRA, e_pmDE = np.nanmedian(df_membs["e_pmRA"]), np.nanmedian(df_membs["e_pmDE"])
    e_Rv = df_membs["e_RV"].median() if df_membs["e_RV"].count() else np.nan

    # Galactocentric coordinates
    X_GC, Y_GC, Z_GC, R_GC = gc_values(c_ra, c_dec, c_plx)

    # Radius that contains half the members
    xy_dists = np.sqrt(
        (np.array(df_membs["GLON"]) - c_lon) ** 2
        + (np.array(df_membs["GLAT"]) - c_lat) ** 2
    )
    # This is equivalent to the median of xy_dists
    # r_50 = xy_dists[int(np.argsort(xy_dists)[int(len(df_membs) / 2)])]
    # Store in arcmin
    r_50 = round(float(np.median(xy_dists)) * 60.0, 1)

    r_core, dens_core = core_values(df_membs)

    # Positive longitude
    if c_lon < 0:
        c_lon += 360.0
    # Round all values
    vals = [
        c_lon,
        c_lat,
        c_ra,
        c_dec,
        c_plx,
        e_plx,
        c_pmRA,
        c_pmDE,
        e_pmRA,
        e_pmDE,
        c_Rv,
        e_Rv,
    ]
    (
        c_lon,
        c_lat,
        c_ra,
        c_dec,
        c_plx,
        e_plx,
        c_pmRA,
        c_pmDE,
        e_pmRA,
        e_pmDE,
        c_Rv,
        e_Rv,
    ) = [round(v, N_digits) for v in vals]

    return (
        c_lon,
        c_lat,
        c_ra,
        c_dec,
        c_plx,
        e_plx,
        c_pmRA,
        c_pmDE,
        e_pmRA,
        e_pmDE,
        c_Rv,
        e_Rv,
        N_Rv,
        X_GC,
        Y_GC,
        Z_GC,
        R_GC,
        N_membs,
        r_50,
        r_core,
        dens_core,
    )


def updt_UCC_new_cl_data(
    idx: int,
    df_UCC_C_updt: pd.DataFrame,
    C1,
    C2,
    C3,
    c_lon,
    c_lat,
    c_ra,
    c_dec,
    c_plx,
    e_plx,
    c_pmRA,
    c_pmDE,
    e_pmRA,
    e_pmDE,
    c_Rv,
    e_Rv,
    N_Rv,
    X_GC,
    Y_GC,
    Z_GC,
    R_GC,
    N_membs,
    r_50,
    r_core,
    dens_core,
) -> pd.DataFrame:
    """
    Updates the UCC DataFrame with new cluster data.
    """
    # Temp dict used to update the UCC
    dict_updt = {
        "make_plots": "y",  # plots are required for this entry. Used by D script
        "process": "n",  # 'n' indicates this entry was processed
        "bad_oc": "n",  # Default value, will be updated later
        "C1": C1,
        "C2": C2,
        "C3": C3,
        "RA_ICRS_m": c_ra,
        "DE_ICRS_m": c_dec,
        "GLON_m": c_lon,
        "GLAT_m": c_lat,
        "Plx_m": c_plx,
        "e_Plx_m": e_plx,
        "pmRA_m": c_pmRA,
        "e_pmRA_m": e_pmRA,
        "pmDE_m": c_pmDE,
        "e_pmDE_m": e_pmDE,
        "Rv_m": c_Rv,
        "e_Rv_m": e_Rv,
        "N_Rv": N_Rv,
        "X_GC": X_GC,
        "Y_GC": Y_GC,
        "Z_GC": Z_GC,
        "R_GC": R_GC,
        "N_membs": N_membs,
        "r_50": r_50,
        "r_core_pc": r_core,
        "dens_core_pc2": dens_core,
    }
    # Update 'df_UCC_updt' with 'dict_updt' values
    for key, val in dict_updt.items():
        df_UCC_C_updt.loc[idx, key] = val

    return df_UCC_C_updt


def core_values(df_membs):
    """
    Estimates the core radius and core density of a cluster based on its members.
    """
    x, y = df_membs["GLON"].values, df_membs["GLAT"].values
    # Center estimation
    if len(df_membs) < 100:
        # For low count, use medians
        cent_lon, cent_lat = np.nanmedian(x), np.nanmedian(y)
    else:
        Nbins = int(max(min(np.sqrt(len(df_membs)), 100), 5))
        H, xedges, yedges = np.histogram2d(x, y, bins=Nbins)
        # index of maximum density
        i, j = np.unravel_index(np.argmax(H), H.shape)
        # bin boundaries
        x0, x1 = xedges[i], xedges[i + 1]
        y0, y1 = yedges[j], yedges[j + 1]
        cent_lon, cent_lat = (x0 + x1) / 2, (y0 + y1) / 2

    # Distances with cos(lat) correction
    cos_lat = np.cos(np.deg2rad(cent_lat))
    dlon = (x - cent_lon) * cos_lat
    dlat = y - cent_lat
    dists_deg = np.sqrt(dlon**2 + dlat**2)

    # RDP in degrees
    num_bins = int(max(min(25, np.sqrt(len(df_membs))), 5))
    counts, bin_edges = np.histogram(dists_deg, bins=num_bins)

    # To parsec
    dist_pc = 1000 / np.clip(np.nanmedian(df_membs["Plx"]), 0.01, 50)
    bin_edges_pc = dist_pc * np.tan(np.deg2rad(bin_edges))
    bin_centers_pc = (bin_edges_pc[:-1] + bin_edges_pc[1:]) / 2
    annulus_areas = np.pi * (bin_edges_pc[1:] ** 2 - bin_edges_pc[:-1] ** 2)
    densities_pc = counts / annulus_areas

    # Density at r_core, estimated as half the peak density within the first 3 bins
    half_density_pc = densities_pc[:3].max() * 0.5

    # Find first bin where density drops at or below target
    below = np.where(densities_pc <= half_density_pc)[0]
    r_c = None
    if len(below) > 0:
        idx = below[0]
        if idx > 0:
            # Linear interpolation between the bin just above and just below target
            d0, d1 = densities_pc[idx - 1], densities_pc[idx]
            r0, r1 = bin_centers_pc[idx - 1], bin_centers_pc[idx]
            if d0 != d1:  # avoid division by zero
                r_c = r0 + (half_density_pc - d0) * (r1 - r0) / (d1 - d0)
        else:
            r_c = bin_centers_pc[0]  # peak is already below half-max
    if r_c is None or r_c <= 0:
        # The 10% value comes from Tarricq et al 2022 (Structural parameters of 389
        # local open clusters) Fig 7: R_c/R_t ~ 0.08
        r_c = 0.1 * bin_centers_pc.max()

    # Final core density estimation
    dists_pc = dist_pc * np.tan(np.deg2rad(dists_deg))
    r_c = np.clip(r_c, 0.01, 10)
    N_core = (dists_pc <= r_c).sum()
    dens_core = np.clip(N_core / (np.pi * r_c**2), 0, 250)

    return round(r_c, 2), round(dens_core, 2)


def gc_values(ra, dec, plx, max_xyz=20):
    """
    PZPO:

    Fig 8 shows several values for the PZPO:
    https://ui.adsabs.harvard.edu/abs/2025AJ....169..211D/abstract
    Global parallax zero point offset (selected by me): -0.02
    """

    d_pc = plx_to_pc(plx)

    coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=d_pc * u.pc,
        frame="icrs",
    )

    gc = Galactocentric()  # galcen_distance=R_sun)
    XYZ = coords.transform_to(gc)
    X_GC = np.clip(XYZ.x.to(u.kpc).value, -max_xyz, max_xyz)
    Y_GC = np.clip(XYZ.y.to(u.kpc).value, -max_xyz, max_xyz)
    Z_GC = np.clip(XYZ.z.to(u.kpc).value, -max_xyz, max_xyz)
    R_GC = np.sqrt(X_GC**2 + Y_GC**2 + Z_GC**2)

    return X_GC, Y_GC, Z_GC, R_GC


def save_cl_datafile(
    logging,
    fname0: str,
    df_membs: pd.DataFrame,
) -> None:
    """
    Saves the cluster member data to a parquet file.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    fname0 : str
        Main name associated with the cluster file.
    """

    # Order by probabilities
    df_membs = df_membs.sort_values("probs", ascending=False)

    out_fname = temp_members_folder + fname0 + ".parquet"
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
