import warnings

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.special import loggamma
from scipy.stats import gaussian_kde


def get_classif(df_membs: pd.DataFrame, df_field: pd.DataFrame) -> str:
    """
    Calculates classification metrics C1, C2, and C3 for a cluster.

    Parameters
    ----------
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    df_field : pd.DataFrame
        DataFrame of field stars.

    Returns
    -------
    str
        C3: Combined classification string (e.g., "AA", "BC").
    """
    C1 = lkl_phot(df_membs, df_field)
    C2 = dens_ratio(df_membs, df_field)

    def ABCD_classif(CC):
        """Obtain 'ABCD' classification"""
        if CC >= 0.75:
            return "A"
        elif CC >= 0.5:
            return "B"
        elif CC >= 0.25:
            return "C"
        else:
            return "D"

    C3 = ABCD_classif(C1) + ABCD_classif(C2)

    # return round(C1, 2), round(C2, 2), C3
    return C3


def lkl_phot(
    df_membs: pd.DataFrame, df_field: pd.DataFrame, max_mag_perc: int = 90
) -> float:
    """
    Calculates the photometric likelihood ratio (C1) for cluster classification.

    Parameters
    ----------
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    df_field : pd.DataFrame
        DataFrame of field stars.
    max_mag_perc : int, optional
        Percentile of maximum magnitude to consider. Default is 90.

    Returns
    -------
    float
        The C1 classification metric.
    """
    x_cl, y_cl = df_membs["Gmag"].values, df_membs["BP-RP"].values
    msk_nan = np.isnan(x_cl) | np.isnan(y_cl)
    x_cl, y_cl = x_cl[~msk_nan], y_cl[~msk_nan]

    #
    max_mag = np.percentile(x_cl, max_mag_perc)
    msk_mag = x_cl <= max_mag
    x_cl, y_cl = x_cl[msk_mag], y_cl[msk_mag]

    x_fl, y_fl = np.array(df_field["Gmag"]), np.array(df_field["BP-RP"])
    msk_mag = x_fl <= max_mag
    x_fl, y_fl = x_fl[msk_mag], y_fl[msk_mag]
    msk_nan = np.isnan(x_fl) | np.isnan(y_fl)
    x_fl, y_fl = x_fl[~msk_nan], y_fl[~msk_nan]

    N_membs, N_field = len(x_cl), len(x_fl)

    if N_field <= N_membs:
        # Not enough field stars
        warnings.warn(
            "Not enough field stars to obtain C1 class, will result in 'D' class"
        )
        return np.nan
    elif N_field < 1.5 * N_membs:
        runs = 10
    elif N_field < 2 * N_membs:
        runs = 25
    elif N_field < 5 * N_membs:
        runs = 75
    else:
        runs = 100

    idx = np.arange(0, N_field)
    prep_clust_cl = prep_data(x_cl, y_cl)
    lkl_cl_max = tremmel([x_cl, y_cl], prep_clust_cl)

    pv_cl, pv_fr = [], []
    for _ in range(runs):
        # Sample two field regions
        msk1 = np.random.choice(idx, N_membs, replace=False)
        field1 = [x_fl[msk1], y_fl[msk1]]
        msk2 = np.random.choice(idx, N_membs, replace=False)
        field2 = [x_fl[msk2], y_fl[msk2]]

        prep_clust_f1 = prep_data(field1[0], field1[1])
        lkl_fl_max = tremmel(field1, prep_clust_f1)

        lkl_cl = tremmel(field1, prep_clust_cl)
        lkl_fl = tremmel(field2, prep_clust_f1)

        pv_cl.append(lkl_cl - lkl_cl_max)
        pv_fr.append(lkl_fl - lkl_fl_max)

    C_lkl = KDEoverlap(pv_cl, pv_fr)
    return C_lkl


def prep_data(mag: np.ndarray, col: np.ndarray) -> list:
    """
    Prepares data for photometric likelihood calculation by creating histograms and
    identifying bins with stars.

    Parameters
    ----------
    mag : np.ndarray
        Array of magnitudes.
    col : np.ndarray
        Array of colors.

    Returns
    -------
    list
        A list containing:
        - bin_edges: Bin edges for each dimension.
        - cl_histo_f_z: Flattened histogram of observed cluster with empty bins removed.
        - cl_z_idx: Indices of bins where stars were observed in the cluster.
    """

    # Obtain bin edges for each dimension, defining a grid.
    bin_edges = bin_edges_f(mag, col)

    # Histogram for observed cluster.
    cl_histo = np.histogramdd([mag, col], bins=bin_edges)[0]

    # Flatten N-dimensional histograms.
    cl_histo_f = np.array(cl_histo).ravel()

    # Index of bins where stars were observed
    cl_z_idx = cl_histo_f != 0

    # Remove all bins where n_i=0 (no observed stars)
    cl_histo_f_z = cl_histo_f[cl_z_idx]

    return [bin_edges, cl_histo_f_z, cl_z_idx]


def bin_edges_f(
    mag: np.ndarray, col: np.ndarray, min_bins: int = 2, max_bins: int = 50
) -> list[np.ndarray]:
    """
    Calculates bin edges for magnitude and color, ensuring a minimum and maximum number
    of bins.

    Parameters
    ----------
    mag : np.ndarray
        Array of magnitudes.
    col : np.ndarray
        Array of colors.
    min_bins : int, optional
        Minimum number of bins per dimension. Default is 2.
    max_bins : int, optional
        Maximum number of bins per dimension. Default is 50.

    Returns
    -------
    list
        A list of arrays, where each array contains the bin edges for a dimension.
    """
    bin_edges = []
    b_num = int(round(max(2, (max(mag) - min(mag)) / 1.0)))
    bin_edges.append(np.histogram(mag, bins=b_num)[1])
    b_num = int(round(max(2, (max(col) - min(col)) / 0.5)))
    bin_edges.append(np.histogram(col, bins=b_num)[1])

    # Impose a minimum of 'min_bins' cells per dimension. The number of bins
    # is the number of edges minus 1.
    for i, be in enumerate(bin_edges):
        N_bins = len(be) - 1
        if N_bins < min_bins:
            bin_edges[i] = np.linspace(be[0], be[-1], min_bins + 1)

    # Impose a maximum of 'max_bins' cells per dimension.
    for i, be in enumerate(bin_edges):
        N_bins = len(be) - 1
        if N_bins > max_bins:
            bin_edges[i] = np.linspace(be[0], be[-1], max_bins)

    return bin_edges


def tremmel(field: list[np.ndarray], prep_clust: list) -> float:
    """
    Calculates the Tremmel et al. (2013) likelihood ratio for a synthetic cluster
    as defined in Eq 10 with v_{i,j}=1.

    Parameters
    ----------
    field : list
        List of arrays representing the synthetic cluster data (e.g.,
        [magnitudes, colors]).
    prep_clust : list
        Prepared data for the observed cluster, including bin edges,
        flattened histogram, and indices of non-empty bins.

    Returns
    -------
    float
        The Tremmel likelihood ratio.
    """
    # Observed cluster's data.
    bin_edges, cl_histo_f_z, cl_z_idx = prep_clust

    # Histogram of the synthetic cluster, using the bin edges calculated
    # with the observed cluster.
    syn_histo = np.histogramdd(field, bins=bin_edges)[0]

    # Flatten N-dimensional histogram.
    syn_histo_f = syn_histo.ravel()
    # Remove all bins where n_i = 0 (no observed stars).
    syn_histo_f_z = syn_histo_f[cl_z_idx]

    SumLogGamma = np.sum(
        loggamma(cl_histo_f_z + syn_histo_f_z + 0.5) - loggamma(syn_histo_f_z + 0.5)
    )

    # M = field.shape[0]
    # ln(2) ~ 0.693
    tremmel_lkl = SumLogGamma - 0.693 * len(field[0])

    return tremmel_lkl


def KDEoverlap(p_vals_cl: list[float], p_vals_fr: list[float]) -> float:
    """
    Calculates the overlap between two kernel density estimations (KDEs).

    Parameters
    ----------
    p_vals_cl : list
        List of p-values for the cluster.
    p_vals_fr : list
        List of p-values for the field.

    Returns
    -------
    float
        The overlap probability for the cluster.
    """
    if (np.median(p_vals_cl) - np.median(p_vals_fr)) > 2 * np.std(p_vals_cl):
        return 0.0

    def y_pts(pt: float) -> float:
        y_pt = min(kcl(pt), kfr(pt))
        return y_pt

    kcl, kfr = gaussian_kde(p_vals_cl), gaussian_kde(p_vals_fr)

    all_pvals = np.concatenate([p_vals_cl, p_vals_fr])
    pmin, pmax = all_pvals.min(), all_pvals.max()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlap = quad(y_pts, pmin, pmax)[0]

    # Probability value for the cluster.
    prob_cl = 1.0 - overlap

    return prob_cl


def dens_ratio(
    df_membs: pd.DataFrame,
    df_field: pd.DataFrame,
    perc: int = 95,
    N_neigh: int = 10,
    N_max: int = 1000,
    norm_v: float = 5,
) -> float:
    """
    Calculates the density ratio (C2) for cluster classification.

    Parameters
    ----------
    df_membs : pd.DataFrame
        DataFrame of cluster members.
    df_field : pd.DataFrame
        DataFrame of field stars.
    perc : int, optional
        Percentile for radius calculation. Default is 95.
    N_neigh : int, optional
        Number of nearest neighbors for distance calculation. Default is 10.
    N_max : int, optional
        Maximum number of field stars to consider. Default is 1000.
    norm_v : int, optional
        Normalization value for the density ratio. Default is 5.

    Returns
    -------
    float
        The C2 density ratio.
    """
    # Obtain the median distance to the 'N_neigh' closest neighbors in 5D
    # for each member
    arr = np.array(df_membs[["GLON", "GLAT", "pmRA", "pmDE", "Plx"]])
    tree = KDTree(arr)
    dists = np.array(tree.query(arr, min(arr.shape[0], N_neigh))[0])
    med_d_membs = np.median(dists[:, 1:])

    # Radius that contains 'perc' of the members for the coordinates
    xy = np.array([df_membs["GLON"].values, df_membs["GLAT"].values]).T
    xy_c = np.nanmedian(xy, 0)
    xy_dists = cdist(xy, np.array([xy_c])).T[0]
    rad = np.percentile(xy_dists, perc)

    # Select field stars within the above radius from the member's center
    xy = np.array([df_field["GLON"].values, df_field["GLAT"].values]).T
    xy_dists = cdist(xy, np.array([xy_c])).T[0]
    msk = np.arange(0, len(xy_dists))[xy_dists < rad]
    if len(msk) > N_max:
        step = max(1, int(len(msk) / N_max))
        msk = msk[::step]

    # Median distance to the 'N_neigh' closest neighbours in 5D for field stars
    arr = np.array(df_field[["GLON", "GLAT", "pmRA", "pmDE", "Plx"]])
    if len(df_field) > 10:
        arr = arr[msk, :]
    tree = KDTree(arr)
    dists = np.array(tree.query(arr, min(arr.shape[0], N_neigh))[0])
    med_d_field = np.median(dists[:, 1:])

    d_ratio = min(float(med_d_field / med_d_membs), norm_v) / norm_v

    return d_ratio
