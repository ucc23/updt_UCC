import numpy as np
import pandas as pd

def main():
    """
    Compare center coordinates of clusters in catalog B with median positions of
    their members in the parquet members file.
    """
    # --- read data
    df_B = pd.read_csv("../data/UCC_cat_B.csv")
    df_Z = pd.read_parquet("../data/zenodo/UCC_members.parquet")

    # --- extract primary name
    df_B["fname"] = df_B["fnames"].str.split(";").str[0]

    # --- exclude unwanted prefixes
    mask_valid = ~df_B["fname"].str.startswith(("hsc", "theia", "cwnu"))
    # mask_valid = np.array([True] * len(df_B))  # disable filtering

    # --- compute cluster statistics only once
    g = df_Z.groupby("name")
    stats = g.agg(
        RA_Z=("RA_ICRS", "median"),
        DE_Z=("DE_ICRS", "median"),
        GLON_min=("GLON", "min"),
        GLON_max=("GLON", "max"),
        n=("GLON", "size"),
    )


    def circular_span(x):
        x = np.sort(x.to_numpy())
        gaps = np.diff(np.r_[x, x[0] + 360])
        return 360 - gaps.max()


    stats["rad"] = g["GLON"].apply(circular_span)
    # keep only clusters with >2 members
    stats = stats[stats["n"] > 2]


    # --- merge stats into catalog B
    df = df_B.merge(
        stats[["RA_Z", "DE_Z", "rad"]], left_on="fname", right_index=True, how="left"
    )

    # --- compute spherical angular distance (deg)
    ra1 = np.deg2rad(df["RA_ICRS"])
    dec1 = np.deg2rad(df["DE_ICRS"])
    ra2 = np.deg2rad(df["RA_Z"])
    dec2 = np.deg2rad(df["DE_Z"])
    d_ra = ra2 - ra1
    d_dec = dec2 - dec1
    a = np.sin(d_dec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2) ** 2

    df["dist"] = np.rad2deg(2 * np.arcsin(np.sqrt(a)))

    df["d_rad"] = df["dist"] / df["rad"]

    # --- filter valid rows and large separations
    res = df[mask_valid & (df["dist"] > 0.5) & (df["d_rad"] > 0.25)].sort_values(
    # res = df[(df["dist"] > 0.5) & (df["d_rad"] > 0.25)].sort_values(
        "d_rad", ascending=False
    )

    # --- report
    for i, r in res.iterrows():
        print(
            f"Row {i:<7}: {r.fname[:14]:<15} | "
            f"dist={r.dist:.2f} deg | "
            f"GLON span={r.rad:.2f} deg | "
            f"d_norm={r.d_rad:.2f}"
            f"   {r.RA_ICRS:.2f}, {r.DE_ICRS:.2f}, {r.RA_Z:.2f}, {r.DE_Z:.2f}"
        )

if __name__ == "__main__":
    main()