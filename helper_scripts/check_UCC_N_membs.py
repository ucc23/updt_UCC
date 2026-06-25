import sys

import pandas as pd


def check_N_clust() -> None:
    """Check that the number of elements per unique 'name' group in df_members_new
    matched the N_clust column in df_UCC_C"""

    # Load your dataframes here
    # df_UCC_C = pd.read_csv("../data/UCC_cat_C.csv")
    # df_members = pd.read_parquet("../data/zenodo/UCC_members.parquet")

    df_UCC_C = pd.read_csv("../temp_updt/UCC_cat_C.csv")
    df_members = pd.read_parquet("../temp_updt/data/zenodo/UCC_members.parquet")

    # Group by 'name' and count unique 'Source'
    member_counts = df_members.groupby("name")["Source"].nunique().reset_index()
    member_counts.rename(columns={"Source": "N_clust_actual"}, inplace=True)

    # Merge with df_UCC_C to compare with 'N_clust'
    merged = pd.merge(
        df_UCC_C,
        member_counts,
        left_on="fname",
        right_on="name",
        how="left",
    )

    # Check for mismatches
    mismatches = merged[merged["N_membs"] != merged["N_clust_actual"]]

    if not mismatches.empty:
        # Count and remove small clusters
        small = mismatches[mismatches["N_membs"] < 25]
        small_flag = (small["N_clust_actual"] == 25).all()
        if small_flag is False:
            n_small = small.sum()
            print(f"Not all {n_small} entries with N_membs<25 have 25 members\n")
            breakpoint()
            sys.exit(1)

        mismatches = mismatches[mismatches["N_membs"] >= 25]
        if not mismatches.empty:
            batch_size = 100
            for start in range(0, len(mismatches), batch_size):
                batch = mismatches.iloc[start : start + batch_size]

                for row in batch.itertuples():
                    print(
                        f"  Cluster '{row.fname}': "
                        f"(C cat)={row.N_membs} vs (members)={row.N_clust_actual}"
                    )

                if start + batch_size < len(mismatches):
                    ans = (
                        input(
                            f"\nDisplayed {start + len(batch)}/{len(mismatches)} mismatches "
                            "(N_membs>=25). Show next 100? [y/N]: "
                        )
                        .strip()
                        .lower()
                    )
                    if ans != "y":
                        break


if __name__ == "__main__":
    check_N_clust()
