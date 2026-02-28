import csv

import numpy as np
import pandas as pd

df = pd.read_csv("../data/databases/RICHER2021.csv")

cmmts = {"Clusters": [], "Comments": []}
for i, row in df.iterrows():
    Nwd, Nt1, Nt4 = row["Nwd"], row["Nt1"], row["Nt4"]
    if np.isnan(Nwd):
        Nwd = 0
    if Nwd > 0 or Nt1 > 0 or Nt4 > 0:
        print(row["Cl"], ":", Nwd, Nt1, Nt4)
        cmmts["Clusters"].append(row["Cl"])
        s1 = "s" if Nt1 != 1 else ""
        s2 = "s" if Nt4 != 1 else ""
        txt = f"The expected number of WDs is {Nwd}, {Nt1} WD candidate{s1} found, and {Nt4} WD candidate{s2} found in the wide search."
        cmmts["Comments"].append(txt)

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/RICHER2021.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()


df = pd.read_csv("../data/databases/MORALES2013.csv")

mtype = {
    "EC1": "deeply embedded cluster",
    "EC2": "partially embedded cluster",
    "OC0": "emerging exposed cluster",
    "OC1": "totally exposed cluster still physically associated with gas",
    "OC2": "totally exposed cluster without correlation with ATLASGAL emission",
}

mflag = {
    "emb": "cluster fully embedded",
    "p-emb": "cluster partially embedded",
    "surr": "possibly associated submm emission surrounding the cluster",
    "few": "one or a few submm emission within the cluster area",
    "few*": "one or a few submm emission within the cluster area, but the submm emission is likely associated to the cluster",
    "exp": "exposed cluster, without submm emission",
    "exp*": "exposed cluster, with submm emission not associated with the cluster",
    #
    "bub-cen": "presence of an IR bubble",
    "bub-cen-trig": "presence of an IR bubble and possible YSOs",
    "bub-edge": "the cluster appears at the edge of an IR bubble",
    "pah": "presence of emission related to PAH or warm dust",
}

cmmts = {}
for i, row in df.iterrows():
    aa = df.at[i, "Mtype"]
    bb = df.at[i, "Mflag"]
    cc = ""
    if "." in bb:
        bb, cc = bb.split(".")
        cc = f", {mflag[cc]}"
    cmmts[df.at[i, "Name"]] = (
        f"Classified as morphological type '{aa}' ({mtype[aa]}). Morphological flag: {mflag[bb]}{cc}."
    )

cluster_df = pd.DataFrame(list(cmmts.items()), columns=["Cluster", "Description"])
cluster_df.to_csv("MORALES2013.csv", index=False, quoting=csv.QUOTE_ALL)

breakpoint()


df = pd.read_csv("data/databases/LIU2025_2.csv")
df2 = pd.read_csv("data/databases/LIU2025_1.csv")

cols_check = ["logt", "DM", "E(B-V)"]

df_combined = df.merge(
    df2,
    on="Cluster",
    how="outer",
    suffixes=("_left", "_right"),
)

for col in cols_check:
    left = f"{col}_left"
    right = f"{col}_right"

    # Merge preferring left, then right
    df_combined[col] = df_combined[left].combine_first(df_combined[right])

    # Drop rows where both were NaN (optional, per column)
    df_combined = df_combined.dropna(subset=[col], how="all")

# Optional: remove original suffixed columns
df_combined = df_combined.drop(
    columns=[f"{c}_{side}" for c in cols_check for side in ("left", "right")]
)
df_combined.to_csv("LIU2025.csv", index=False, quoting=csv.QUOTE_ALL)
breakpoint()

# # Find duplicated entries in 'Cluster' column of df and merge them into a single entry, combining the values in each column into a single one separated by a comma
# cols_equal_check = ["logt", "DM", "E(B-V)"]

# def agg_equal_or_join(series):
#     vals = series.dropna().unique()
#     if len(vals) == 1:
#         return vals[0]
#     return ", ".join(series.astype(str))

# def agg_join_all(series):
#     return ", ".join(series.astype(str))

# dup_mask = df.duplicated(subset="Cluster", keep=False)

# df_dup = df[dup_mask]
# df_unique = df[~dup_mask]

# agg_dict = {}

# for col in df.columns:
#     if col == "Cluster":
#         continue
#     elif col in cols_equal_check:
#         agg_dict[col] = agg_equal_or_join
#     elif col == "Type":
#         agg_dict[col] = agg_join_all  # preserve both values
#     else:
#         agg_dict[col] = lambda x: ", ".join(x.astype(str).unique())

# df_merged = (
#     df_dup
#     .groupby("Cluster", as_index=False)
#     .agg(agg_dict)
# )

# df_final = (
#     pd.concat([df_unique, df_merged], ignore_index=True)
#     .sort_values("Cluster")
#     .reset_index(drop=True)
# )
# df_final.to_csv("LIU2025_2.csv.csv", index=False)
# breakpoint()


# # Group by column "Pair"
# grouped = df.groupby("Pair")
# type_dict = {
#     "PBC": "primordial binary cluster",
#     "TBC": "tidal capture (resonant trapping binary)",
#     "HEP": "hyperbolic encounter pair",
# }
# clust_dict = {}
# # For each group, print the "BCO" column values
# for pair, group in grouped:
#     for i, cluster in enumerate(group["Cluster"].values):
#         # print(f'"{cluster}": "Classified as {type_dict[group['Type'].values[i]]} {pair}, along with {group['Cluster'].values[1-i]}.",')

#         txt0 = f'Classified as {type_dict[group["Type"].values[i]]} {pair} along with {group["Cluster"].values[1 - i]}.'
#         if cluster in clust_dict:
#             txt = f', and as {type_dict[group["Type"].values[i]]} {pair} along with {group["Cluster"].values[1 - i]}.'
#             clust_dict[cluster] = clust_dict[cluster][:-1] + txt
#         else:
#             clust_dict[cluster] = txt0

# grouped = df2.groupby("Group")
# for pair, group in grouped:
#     clusters = group["Cluster"].tolist()

#     for cluster in clusters:
#         others = [c for c in clusters if c != cluster]

#         if len(others) == 1:
#             others_str = others[0]
#         elif len(others) == 2:
#             others_str = " and ".join(others)
#         else:
#             others_str = ", ".join(others[:-1]) + " and " + others[-1]

#         # print(
#         #     f'"{cluster}": "Part of multiple system {pair}, along with {others_str}.",'
#         # )
#         txt0 = f'Part of multiple system {pair} along with {others_str}.'
#         if cluster in clust_dict:
#             txt = f', and of multiple system {pair} along with {others_str}.'
#             # print(f"Cluster {cluster} is in both dataframes.")
#             clust_dict[cluster] = clust_dict[cluster][:-1] + txt
#         else:
#             clust_dict[cluster] = txt0

# cluster_df = pd.DataFrame(list(clust_dict.items()), columns=["Cluster", "Description"])
# cluster_df.to_csv("LIU2025.csv", index=False, quoting=csv.QUOTE_ALL)


## PALMA2025

# # apply strip to columns "Pair, Cluster"
# df2["Group"] = df2["Group"].str.strip()
# df2["Cluster"] = df2["Cluster"].str.strip()
# df2["Ref"] = df2["Ref"].str.strip()

# # save to new csv
# df2.to_csv("PALMA2025_2_s.csv", index=False)
# breakpoint()

# # Check if the 'Cluster' columns in df andf df2 share any values
# shared_clusters = set(df["Cluster"]).intersection(set(df2["Cluster"]))
# # for every cluster in shared_clusters, check that the columns 'logt,DM,E(B-V)' are equal in both dataframes
# for cluster in shared_clusters:
#     print(cluster)
#     df_cluster = df[df["Cluster"] == cluster]
#     df2_cluster = df2[df2["Cluster"] == cluster]

#     for col in ["logt", "DM", "E(B-V)"]:
#         if not df_cluster[col].values[0] == df2_cluster[col].values[0]:
#             print(
#                 f"Cluster {cluster} has different values for column {col} in the two dataframes."
#             )


# # Group by column "Pair"
# grouped = df.groupby("Pair")
# bco_dict = {
#     "B": "genetic pair",
#     "C": "tidal capture (resonant trapping pair)",
#     "O": "optical pair",
#     "Oa": "optical pair (not dynamically associated)"
# }
# # For each group, print the "BCO" column values
# for pair, group in grouped:
#     for i, cluster in enumerate(group['Cluster'].values):
#         print(f'"{cluster}": "Classified as {bco_dict[group['BCO'].values[i]]} {pair}, along with {group['Cluster'].values[1-i]}.",')


# grouped = df2.groupby("Group")
# for pair, group in grouped:
#     clusters = group["Cluster"].tolist()

#     for cluster in clusters:
#         others = [c for c in clusters if c != cluster]

#         if len(others) == 1:
#             others_str = others[0]
#         elif len(others) == 2:
#             others_str = " and ".join(others)
#         else:
#             others_str = ", ".join(others[:-1]) + " and " + others[-1]

#         print(
#             f'"{cluster}": "Part of multiple system {pair}, along with {others_str}.",'
#         )
