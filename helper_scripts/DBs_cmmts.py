import csv

import numpy as np
import pandas as pd

df = pd.read_csv("/home/gabriel/Github/UCC/updt_UCC/temp_updt/todo/BICA2019_1.csv")

# cmmt_dict = {
#     "REF250": "Present study",
#     "REF039": "2MASS website",
#     "REF1622": "Discoveries by our group members and communicated in the present work (2002 to 2017)",
#     "REF298": "Web updates in DAML02, some were later removed (2003 to 2010)",
#     "REF913": "The Preliminary Amateur Open Cluster Catalog, Version 08/03/2003",
#     "REF947": "Analysis of Dolidze objects by M. Kronberger reported in DAML02",
#     "REF949": "Spitzer website",
#     "REF526": "Private communication by S. Ortolani to E. Bica",
#     "REF891": "List of clusters and alike reported by B. Alessi",
#     "REF1051": "Original NGC and IC catalogues at Vizier: NGC 2000.0, Sky Publishing, ed. Sinnott 1988 (1997yCat.7118....0S)",
#     "REF1070": "Asterisms and cluster alikes by B. Alessi reported in DAML02",
#     "REF963": "Asterisms reported by amateur astronomers in the web",
#     "REF1023": "Particular objects in SIMBAD",
#     "REF7000": "Asterisms and clusters by L. Ferrero",
# }

refs_csv = pd.read_csv("/home/gabriel/Descargas/bica2019_refs.csv")
# Remove final ; from Code column
refs_csv["Code"] = refs_csv["Code"].str.rstrip(";")
# Remove initial ; from Code Bibcode
refs_csv["Bibcode"] = refs_csv["Bibcode"].str.lstrip(";")
# strip columns "Code","Ref"
refs_csv["Code"] = refs_csv["Code"].str.strip()
refs_csv["Ref"] = refs_csv["Ref"].str.strip()
# Turn into dictionary using Code as key and Ref and Bibcode as values of a tuple
refs_csv = dict(zip(refs_csv["Code"], zip(refs_csv["Ref"], refs_csv["Bibcode"])))
# refs_csv = dict(zip(refs_csv["Code"], refs_csv["Ref"]))

class_dict = {
    "Assoc": "Association",
    "EC": "Embedded Cluster",
    "ECC": "Embedded Cluster Candidate",
    "OC": "Open Cluster",
    "OCC": "Open Cluster Candidate",
    "EGr": "Embedded Group",
    "Ast": "Asterism",
    "DGAL": "Local Group Dwarf Galaxiy",
    "NGAL": "Local Group Normal Galaxy",
    "GC": "Globular Cluster",
    "GCC": "Globular Cluster Candidate",
    "OVD": "Overdenstiy",
    "KAs": "Kinematical Association",
    "KGr": "Kinematical Group",
    "MHC": "Magellanic Halos's Cluster",
    "MGr": "Moving Group",
    "MC clusts": "Magellanic Cloud cluster",
    "POCR": "Open Cluster Remnant Candidate",
    "lPOCR": "loose Open Cluster Remnant Candidate",
    "cPOCR": "compact Open Cluster Remnant Candidate",
    "UltrF": "Ultra-faint",
}

# Strip 'Name' column
df["Name"] = df["Name"].str.strip()

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    txt = ""
    c1 = row['Class1']
    c2 = row['Class2']
    txt = ""
    if str(c2) != "nan":
        txt += f"Classified as {class_dict[c1]} and {class_dict[c2]}."
    else:
        txt += f"Classified as {class_dict[c1]}."

    if str(row["Code"]) != "nan":
        refs = [_.strip() for _ in row["Code"].split(",")]
        if len(refs) > 1:
            txt += " References: "
            for ref in refs:
                if ref in refs_csv:
                    if refs_csv[ref][1].startswith("--"):
                        txt += refs_csv[ref][0] + ", "
                    else:
                        txt += refs_csv[ref][1] + ", "
            txt = txt[:-2] +  "."
        else:
            txt += " Reference: "
            if refs[0] in refs_csv:
                if refs_csv[refs[0]][1].startswith("--"):
                    txt += refs_csv[refs[0]][0] + "."
                else:
                    txt += refs_csv[refs[0]][1] + "."



    if txt != "":
        cmmts["Cluster"].append(row["Name"])
        cmmts["Comment"].append(txt.strip())

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "/home/gabriel/Github/UCC/updt_UCC/temp_updt/todo/BICA2019_2.csv",
    index=False,
    quoting=csv.QUOTE_ALL,
)
breakpoint()


df = pd.read_csv("../temp_updt/data/databases/CAMARGO2015.csv")

types = {
    "OC": "Open cluster",
    "OCC": "Open cluster candidate",
    "EC": " Embedded cluster",
    "ECC": "Embedded cluster candidate",
}

aved_txt = "Cross-identified with Avedisova (2002ARep...46..193A, Cat. V/112) SFRs or candidates within 5' of the central coordinates."

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    txt = f"{types[row['Type']]}; {row['Com']}"
    cmmts["Cluster"].append(row["Name"])
    cmmts["Comment"].append(txt)

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/CAMARGO2015.csv", index=False, quoting=csv.QUOTE_ALL
)
breakpoint()


df = pd.read_csv("/home/gabriel/Descargas/KRONBERGER2006.csv")

# Find duplicated entries in 'Cluster' column of df and merge them into a single entry,
# combining the values in each column into a single one separated by a comma


def agg_join_all(series):
    return "".join(series.astype(str))


dup_mask = df.duplicated(subset="Cluster", keep=False)

df_dup = df[dup_mask]

agg_dict = {}
for col in df.columns:
    if col == "Comment":
        agg_dict[col] = agg_join_all

df_merged = df_dup.groupby("Cluster", as_index=False).agg(agg_dict)

# Capitalize the first letter of each merged comment
df_merged["Comment"] = df_merged["Comment"].str.replace(
    r"^([a-z])", lambda m: m.group(1).upper(), regex=True
)

df_merged.to_csv(
    "../data/databases/cmmts/KRONBERGER2006.csv", index=False, quoting=csv.QUOTE_ALL
)
breakpoint()


df = pd.read_csv("../temp_updt/data/databases/LI2026.csv")

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    txt = f"Tidal & Core radii: r_t={row['Rt']:.0f}+/-{row['e_Rt']:.1f} [pc], r_c={row['Rcore']}+/-{row['e_Rcore']} [pc]; stars within r_t: N~{int(row['N'])}"
    cmmts["Cluster"].append(row["Name"])
    cmmts["Comment"].append(txt)

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/LI2026.csv", index=False, quoting=csv.QUOTE_ALL
)
breakpoint()


df = pd.read_csv("../temp_updt/data/databases/HU2021.csv")

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    txt = f"Ellipticities (core, all): e_core={row['ecore']}, e_all={row['eall']}."
    cmmts["Cluster"].append(row["Cluster"])
    cmmts["Comment"].append(txt)

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/HU2021.csv", index=False, quoting=csv.QUOTE_ALL
)
breakpoint()


df = pd.read_csv("/home/gabriel/Descargas/HUNT2023.csv")

type_dict = {
    "o": "Classified as open cluster",
    "m": "Classified as moving group",
    "g": "ERROR",
    "d": "Too distant to classify",
    "r": "Rejected:",
}
clss_dict = {
    "FP": "false positive",
    "FP?": "false positive?",
    "TP": "true positive",
    "TP?": "true positive?",
    "": "",
}

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    if row["Type"] == "g":
        continue

    # if float(row['CMDCl50']) > 0.75 and row["Name"].startswith("HSC") and row['N'] > 250:
    #     print(f"{row['Name']} {row['CMDCl50']:.2f} {row['N']}")
    # continue

    cmmts["Cluster"].append(row["Name"].strip())
    cmd_human = clss_dict[row["CMDClHuman"].strip()]

    ss = ""
    if cmd_human != "":
        ss = "es"
    txt = f" CMD class{ss}: {row['CMDCl50']:.2f} (50th percentile)"

    if cmd_human != "":
        txt += f", {cmd_human} (human-assigned)."
    else:
        txt += "."
    if row["Type"] == "r":
        cmmt = f"Rejected: {row['Note'].strip()}"
        cmmts["Comment"].append(cmmt + ".")
    else:
        cmmt = type_dict[row["Type"]]
        cmmts["Comment"].append(cmmt + "." + txt)

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/HUNT2023.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()


df = pd.read_csv("/home/gabriel/Descargas/BICA2019_cmmts.csv")

cmmt_dict = {
    "REF250": "Present study",
    "REF039": "2MASS website",
    "REF1622": "Discoveries by our group members and communicated in the present work (2002 to 2017)",
    "REF298": "Web updates in DAML02, some were later removed (2003 to 2010)",
    "REF913": "The Preliminary Amateur Open Cluster Catalog, Version 08/03/2003",
    "REF947": "Analysis of Dolidze objects by M. Kronberger reported in DAML02",
    "REF949": "Spitzer website",
    "REF526": "Private communication by S. Ortolani to E. Bica",
    "REF891": "List of clusters and alike reported by B. Alessi",
    "REF1051": "Original NGC and IC catalogues at Vizier: NGC 2000.0, Sky Publishing, ed. Sinnott 1988 (1997yCat.7118....0S)",
    "REF1070": "Asterisms and cluster alikes by B. Alessi reported in DAML02",
    "REF963": "Asterisms reported by amateur astronomers in the web",
    "REF1023": "Particular objects in SIMBAD",
    "REF7000": "Asterisms and clusters by L. Ferrero",
}


# Strip 'Name' column
df["Name"] = df["Name"].str.strip()

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    refs = [_.strip() for _ in row["Code"].split(",")]
    txt = ""
    for ref in refs:
        if ref in cmmt_dict:
            txt += cmmt_dict[ref] + ". "
    if txt != "":
        cmmts["Cluster"].append(row["Name"])
        cmmts["Comment"].append(txt.strip())

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/BICA2019.csv", index=False, quoting=csv.QUOTE_ALL
)
breakpoint()


df = pd.read_csv("../data/databases/YAN2026.csv")

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    txt = f"Number of WDs: expected from single-star evolution N={row['expect_wd_number']}, with probability of formation through binary evolution >=0.5 N={row['prob_wd_number']}."
    cmmts["Cluster"].append(row["Name"])
    cmmts["Comment"].append(txt)

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/YAN2026.csv", index=False, quoting=csv.QUOTE_ALL
)
breakpoint()


df = pd.read_csv("../data/databases/cmmts/RAIN2021.csv")

# Strip strings in the 'Cluster' column
df["Cluster"] = df["Cluster"].str.strip()

df = df.groupby("Cluster", as_index=False)["Comment"].agg(
    lambda x: "".join(x.dropna().astype(str))
)


df.to_csv("../data/databases/cmmts/RAIN2021_1.csv", index=False, quoting=csv.QUOTE_ALL)

breakpoint()


df = pd.read_csv("../data/databases/cmmts/AHUMADA2007.csv")
ref_df = pd.read_csv("../data/databases/cmmts/refs.csv")

df = df.groupby("Cluster", as_index=False)["Comment"].agg(
    lambda x: "".join(x.dropna().astype(str))
)


# Mapping: integer reference -> BibCode
ref_map = dict(zip(ref_df["Ref"], ref_df["BibCode"]))


# Replacement function
def replace_refs(text):
    if pd.isna(text):
        return text
    return re.sub(
        r"\((\d+)\)",
        lambda m: f"({ref_map.get(int(m.group(1)), m.group(1))})",
        text,
    )


# Apply to Comment column
df["Comment"] = df["Comment"].apply(replace_refs)

df.to_csv(
    "../data/databases/cmmts/AHUMADA2007_1.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()


df = pd.read_csv("../data/databases/DUTRA2001.csv")

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    cmmts["Cluster"].append(row["Object"].split(",")[0])
    cmmts["Comment"].append(row["Remarks"].capitalize())

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/DUTRA2001.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()


df = pd.read_csv("../data/databases/DUTRA2003.csv")

type_dict = {
    "IRC": "Classified as infrared cluster (IRC).",
    "IRCC": "Classified as cluster candidate (IRCC).",
    "IRGr": "Classified as stellar group (IRGr).",
    "IROC": "Classified as open cluster (IROC).",
}

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    cmmts["Cluster"].append(row["Seq"].split(",")[0])
    cmmts["Comment"].append(type_dict[row["Type"]])

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/DUTRA2003.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()


df = pd.read_csv("../data/databases/BICA2003.csv")

type_dict = {
    "IRC": "Classified as infrared cluster (IRC).",
    "IRCC": "Classified as cluster candidate (IRCC).",
    "IRGr": "Classified as stellar group (IRGr).",
    "IROC": "Classified as open cluster (IROC).",
}

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    cmmts["Cluster"].append(row["Name"].split(",")[0])
    cmmts["Comment"].append(type_dict[row["Class"]])

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/BICA2003.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()


df = pd.read_csv("../data/databases/BICA2003_1.csv")

type_dict = {
    "IRC": "Classified as infrared cluster (IRC).",
    "IRCC": "Classified as cluster candidate (IRCC).",
    "IRGr": "Classified as stellar group (IRGr).",
    "IROC": "Classified as open cluster (IROC).",
}

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    cmmts["Cluster"].append(row["BDS2003"].split(",")[0])
    cmmts["Comment"].append(type_dict[row["Class"]])

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/BICA2003_1.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()


df = pd.read_csv("../data/databases/MALHOTRA2026.csv")

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    Mlow, Mhi = row["M*low"], row["M*hi"]
    cmmts["Cluster"].append(row["Name"])
    txt = f"Lowest/Highest stellar mass in the catalogue with a mass-ratio estimate: {Mlow}/{Mhi} Msun"
    cmmts["Comment"].append(txt)

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/MALHOTRA2026.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()


df = pd.read_csv("../data/databases/RICHER2021.csv")

cmmts = {"Cluster": [], "Comment": []}
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


df = pd.read_csv("/home/gabriel/Descargas/HUNT2024.csv")

type_dict = {
    "o": "Classified as open cluster",
    "m": "Classified as moving group",
    "g": "ERROR",
    "d": "Too distant to classify",
    "r": "Rejected:",
}
clss_dict = {
    "FP": "false positive",
    "FP?": "false positive?",
    "TP": "true positive",
    "TP?": "true positive?",
    "": "",
}

cmmts = {"Cluster": [], "Comment": []}
for i, row in df.iterrows():
    if row["Type"] == "g":
        continue

    # if float(row['CMDCl50']) > 0.75 and row["Name"].startswith("HSC") and row['N'] > 250:
    #     print(f"{row['Name']} {row['CMDCl50']:.2f} {row['N']}")
    # continue

    cmmts["Cluster"].append(row["Name"].strip())
    cmd_human = clss_dict[row["CMDClHuman"].strip()]

    ss = ""
    if cmd_human != "":
        ss = "es"
    txt = f" CMD class{ss}: {row['CMDCl50']:.2f} (50th percentile)"

    if cmd_human != "":
        txt += f", {cmd_human} (human-assigned)."
    else:
        txt += "."
    if row["Type"] == "r":
        cmmt = f"Rejected: {row['Note'].strip()}"
        cmmts["Comment"].append(cmmt + ".")
    else:
        cmmt = type_dict[row["Type"]]
        cmmts["Comment"].append(cmmt + "." + txt)

cluster_df = pd.DataFrame(cmmts)
cluster_df.to_csv(
    "../data/databases/cmmts/HUNT2024_2.csv", index=False, quoting=csv.QUOTE_ALL
)

breakpoint()
