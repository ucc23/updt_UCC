import warnings

import numpy as np
from scipy import spatial
from scipy.integrate import quad
from scipy.special import loggamma
from scipy.stats import gaussian_kde


def get_classif(df_membs, df_field):
    """ """
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

    return round(C1, 2), round(C2, 2), C3


def lkl_phot(df_membs, df_field, max_mag_perc=90):
    """ """
    x_cl, y_cl = df_membs["Gmag"].values, df_membs["BP-RP"].values
    msk_nan = np.isnan(x_cl) | np.isnan(y_cl)
    x_cl, y_cl = x_cl[~msk_nan], y_cl[~msk_nan]

    #
    max_mag = np.percentile(x_cl, max_mag_perc)
    msk_mag = x_cl <= max_mag
    x_cl, y_cl = x_cl[msk_mag], y_cl[msk_mag]

    x_fl, y_fl = df_field["Gmag"].values, df_field["BP-RP"].values
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


def prep_data(mag, col):
    """ """

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


def bin_edges_f(mag, col, min_bins=2, max_bins=50):
    """
    Obtain bin edges for each photometric dimension using the cluster region
    diagram. The 'bin_edges' list will contain all magnitudes first, and then
    all colors (in the same order in which they are read).
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


def tremmel(field, prep_clust):
    """
    Poisson likelihood ratio as defined in Tremmel et al (2013), Eq 10 with
    v_{i,j}=1.
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


def KDEoverlap(p_vals_cl, p_vals_fr):
    """
    Calculate overlap between the two KDEs
    """
    if (np.median(p_vals_cl) - np.median(p_vals_fr)) > 2 * np.std(p_vals_cl):
        return 0.0

    def y_pts(pt):
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


def dens_ratio(df_membs, df_field, perc=95, N_neigh=10, N_max=1000, norm_v=5):
    """ """
    # Obtain the median distance to the 'N_neigh' closest neighbors in 5D
    # for each member
    arr = df_membs[["GLON", "GLAT", "pmRA", "pmDE", "Plx"]].values
    tree = spatial.KDTree(arr)
    dists = tree.query(arr, min(arr.shape[0], N_neigh))[0]
    med_d_membs = np.median(dists[:, 1:])

    # Radius that contains 'perc' of the members for the coordinates
    xy = np.array([df_membs["GLON"].values, df_membs["GLAT"].values]).T
    xy_c = np.nanmedian(xy, 0)
    xy_dists = spatial.distance.cdist(xy, np.array([xy_c])).T[0]
    rad = np.percentile(xy_dists, perc)

    # Select field stars within the above radius from the member's center
    xy = np.array([df_field["GLON"].values, df_field["GLAT"].values]).T
    xy_dists = spatial.distance.cdist(xy, np.array([xy_c])).T[0]
    msk = np.arange(0, len(xy_dists))[xy_dists < rad]
    if len(msk) > N_max:
        step = max(1, int(len(msk) / N_max))
        msk = msk[::step]

    # Median distance to the 'N_neigh' closest neighbours in 5D for field stars
    arr = df_field[["GLON", "GLAT", "pmRA", "pmDE", "Plx"]].values
    if len(df_field) > 10:
        arr = arr[msk, :]
    tree = spatial.KDTree(arr)
    dists = tree.query(arr, min(arr.shape[0], N_neigh))[0]
    med_d_field = np.median(dists[:, 1:])

    d_ratio = min(med_d_field / med_d_membs, norm_v) / norm_v

    return d_ratio
