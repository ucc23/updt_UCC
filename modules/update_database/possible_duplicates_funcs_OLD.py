import numpy as np
from scipy.spatial.distance import cdist


def duplicate_probs(
    fnames: list[str],
    glon: np.ndarray,
    glat: np.ndarray,
    plx: np.ndarray,
    pmRA: np.ndarray,
    pmDE: np.ndarray,
    prob_cut: float,
    Nmax: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Identifies potential duplicate clusters based on proximity and similarity of
    parameters.

    Parameters
    ----------
    fnames :list[str]
        List of cluster names (filenames).
    glon : np.ndarray
        Array of x-coordinates (GLON).
    glat : np.ndarray
        Array of y-coordinates (GLAT).
    plx : np.ndarray
        Array of parallax values.
    pmRA : np.ndarray
        Array of proper motion in RA.
    pmDE : np.ndarray
        Array of proper motion in DE.
    prob_cut : float
        Probability threshold for considering a cluster a duplicate.
    Nmax : int, optional
        Maximum number of standard deviations for considering clusters close in
        a dimension. Default is 3.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - dups_fnames: List of semicolon-separated filenames of probable duplicates for
          each cluster.
        - dups_probs: List of semicolon-separated probabilities of being duplicates for
          each cluster.
    """
    # Find the (glon, glat) distances to all clusters, for all clusters
    coords = np.array([glon, glat]).T
    gal_dist = cdist(coords, coords)

    dups_fnames, dups_probs = [], []
    for i, dists_i in enumerate(gal_dist):
        # Only process relatively close clusters
        rad_max = Nmax * max_coords_rad(plx[i])
        msk_rad = dists_i <= rad_max
        idx_j = np.arange(0, dists_i.size)
        j_msk = idx_j[msk_rad]

        dups_fname_i, dups_prob_i = [], []
        for j in j_msk:
            # Skip itself
            if j == i:
                continue

            # Fetch duplicate probability for the i,j clusters
            dup_prob = dprob(glon, glat, pmRA, pmDE, plx, i, j)
            if dup_prob >= prob_cut:
                # Store just the first fname
                dups_fname_i.append(fnames[j].split(";")[0])
                dups_prob_i.append(dup_prob)

        if dups_fname_i:
            # Store list of probable duplicates respecting a given order
            dups_fname_i, dups_prob_i = sort_by_float_desc_then_alpha(
                dups_fname_i, dups_prob_i
            )
            dups_fname_i = ";".join(dups_fname_i)
            dups_prob_i = ";".join(dups_prob_i)
        else:
            dups_fname_i, dups_prob_i = "nan", "nan"

        dups_fnames.append(dups_fname_i)
        dups_probs.append(dups_prob_i)

    return dups_fnames, dups_probs


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


def max_coords_rad(plx_i: float) -> float:
    """
    Defines a maximum coordinate radius based on parallax.

    Parameters
    ----------
    plx_i : float
        Parallax value.

    Returns
    -------
    float
        Maximum radius in degrees (parallax dependent).
    """
    if np.isnan(plx_i):
        rad = 5
    elif plx_i >= 4:
        rad = 20
    elif 3 <= plx_i < 4:
        rad = 15
    elif 2 <= plx_i < 3:
        rad = 10
    elif 1.5 <= plx_i < 2:
        rad = 7.5
    elif 1 <= plx_i < 1.5:
        rad = 5
    elif 0.5 <= plx_i < 1:
        rad = 2.5
    elif plx_i < 0.5:
        rad = 1.5
    else:
        raise ValueError("Could not define 'rad' values, plx_i is out of bounds")

    rad = rad / 60  # To degrees
    return rad


def sort_by_float_desc_then_alpha(
    strings: list[str], floats: list[float]
) -> tuple[list[str], list[str]]:
    """
    Sorts two parallel lists: one of strings and one of floats. The lists are sorted
    by the float values in descending order, and by the strings in ascending
    alphabetical order if float values are the same. Returns two lists:
    one with the sorted strings and another with the sorted floats as strings.

    Parameters
    ----------
    strings : list
        A list of strings to be sorted alphabetically when float
        values are equal.
    floats : list
        A list of float values to be sorted in descending order.

    Returns
    -------
    tuple
        A tuple containing two lists:
            - A list of strings sorted by the criteria.
            - A list of floats as strings, sorted by the criteria.
    """
    # Combine strings and floats into a list of tuples
    data = list(zip(strings, floats))

    # Sort by float in descending order and string alphabetically in ascending order
    sorted_data = sorted(data, key=lambda x: (-x[1], x[0].lower()))

    # Unzip the sorted data back into two separate lists
    sorted_strings, sorted_floats = zip(*sorted_data)

    return list(sorted_strings), [str(f) for f in sorted_floats]
