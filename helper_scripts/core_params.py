import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    df_C = pd.read_csv("../data/UCC_cat_C.csv")

    df_C2 = pd.read_csv("UCC_cat_C_new.csv")

    msk = df_C["UTI"] > 0.0

    plt.subplot(121)
    plt.scatter(
        df_C["r_core_pc"][msk], df_C["dens_core_pc2"][msk], label="Original", alpha=0.2
    )
    plt.scatter(
        df_C2["r_core_pc"][msk], df_C2["dens_core_pc2"][msk], label="New", alpha=0.2
    )
    plt.xlabel("Core radius (pc)")
    plt.ylabel("Core density (stars/pc^2)")
    plt.legend()
    plt.subplot(122)
    plt.scatter(df_C["r_core_pc"][msk], df_C["N_membs"][msk], label="Old", alpha=0.2)
    plt.scatter(df_C2["r_core_pc"][msk], df_C2["N_membs"][msk], label="New", alpha=0.2)
    plt.xlabel("Core radius (pc)")
    plt.ylabel("N membs")
    plt.legend()
    plt.show()

    plt.subplot(121)
    plt.scatter(
        df_C["r_core_pc"][msk],
        df_C["r_core_pc"][msk] - df_C2["r_core_pc"][msk],
        alpha=0.2,
    )
    plt.xlabel("Core radius (pc)")
    plt.ylabel("Delta")
    plt.legend()
    plt.subplot(122)
    plt.scatter(
        df_C["dens_core_pc2"][msk],
        df_C["dens_core_pc2"][msk] - df_C2["dens_core_pc2"][msk],
        alpha=0.2,
    )
    plt.xlabel("Core density (pc)")
    plt.ylabel("Delta")
    plt.legend()
    plt.show()

    plt.subplot(121)
    msk = df_C["r_core_pc"] < 15
    plt.hist(df_C["r_core_pc"][msk], alpha=0.2, label="Original", bins=50)
    plt.hist(df_C2["r_core_pc"], alpha=0.2, label="New", bins=50)
    plt.xlabel("Core radius")
    plt.subplot(122)
    msk = df_C["dens_core_pc2"] < 15
    plt.hist(df_C["dens_core_pc2"][msk], alpha=0.2, label="Original", bins=50)
    msk = df_C2["dens_core_pc2"] < 15
    plt.hist(df_C2["dens_core_pc2"][msk], alpha=0.2, label="New", bins=50)
    plt.xlabel("Core density")
    plt.legend()
    plt.show()

    breakpoint()

    all_membs = pd.read_parquet("../data/zenodo/UCC_members.parquet")

    for row in df_C.itertuples():
        fname = row.fname
        # if fname != "hyades":
        #     continue
        df_membs = all_membs[all_membs["name"] == fname]

        r_core, dens_core = core_values(df_membs)  # , df_C["r_core_pc"][row.Index])
        print(f"{fname},{r_core},{dens_core}")
        df_C.loc[int(row.Index), "r_core_pc"] = r_core
        df_C.loc[int(row.Index), "dens_core_pc2"] = dens_core

    df_C.to_csv(
        "UCC_cat_C_new.csv", na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC
    )


def core_values(df_membs):
    """ """
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


def core_values_new(df_membs, rc_old):

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
    dlon = (x - cent_lon) * np.cos(np.deg2rad(y))
    dlat = y - cent_lat
    dists_deg = np.sqrt(dlon**2 + dlat**2)

    # Estimate core radius r_c
    if len(df_membs) <= 50:
        # For small objects estimate the total radius as the (average) range,
        # and the core radius as 10% of that, The 10% value comes from Tarricq et al
        # 2022 (Structural parameters of 389 local open clusters) Fig 7: R_c/R_t
        r_t = 0.5 * 0.5 * ((x.max() - x.min()) + (y.max() - y.min()))
        r_c_deg = 0.1 * r_t
    else:
        # RDP (degrees)
        num_bins = max(10, int(np.sqrt(len(df_membs))))
        counts, bin_edges = np.histogram(dists_deg, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        annulus_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
        densities = counts / annulus_areas

        if densities[0] < densities[1]:
            # Discard bin before the first peak density
            bin_centers, densities = bin_centers[1:], densities[1:]

        # Density at r_core, estimated as half the peak density
        half_density = densities.max() * 0.5
        # Find bins where density drops below target
        dens_below = np.where(densities <= half_density)[0]
        if len(dens_below) == 0:
            # No values below half density, use the max radius
            r_c_deg = bin_centers[-1]
        else:
            # If there are bins below half density, use the first one
            r_c_deg = bin_centers[dens_below[0]]

    N_core = (dists_deg <= r_c_deg).sum()
    # To parsec
    dist_pc = 1000 / np.nanmedian(np.clip(df_membs["Plx"], 0.1, 100))
    r_c = np.clip(dist_pc * np.tan(np.deg2rad(r_c_deg)), 0.01, 10)
    dens_core = np.clip(N_core / (np.pi * r_c**2), 0, 250)

    return round(r_c, 2), round(dens_core, 2)


if __name__ == "__main__":
    main()
