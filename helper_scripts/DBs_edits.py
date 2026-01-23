import csv

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df1 = pd.read_csv("data/databases/ALMEIDA2023_integ.csv")
df2 = pd.read_csv("data/databases/ALMEIDA2023_detailed.csv")

# Drop these columns from df2
drop_cols = [
    "RA_ICRS",
    "DE_ICRS",
    "dist",
    "e_dist",
    "age",
    "e_age",
    "FeH",
    "e_FeH",
    "Av",
    "e_Av",
    "Nc",
    "bin_frac",
    "average_ratio",
    "str_average_ratio",
    "segr_ratio",
    "segr_ratio_std",
    "mass_seg",
    "mass_seg_pval",
]
df2 = df2.drop(columns=drop_cols)

df_comb = pd.merge(
    df1, df2, on="Cluster", how="outer", suffixes=("_integ", "_detailed")
)

# # Compare the differences in the columns 'RA_ICRS,DE_ICRS,dist,e_dist,age,e_age,FeH,e_FeH,Av,e_Av,Nc', adding the _left and _right suffixes
# for col in [
#     "RA_ICRS",
#     "DE_ICRS",
#     "dist",
#     "e_dist",
#     "age",
#     "e_age",
#     "FeH",
#     "e_FeH",
#     "Av",
#     "e_Av",
#     "Nc",
#     "bin_frac","average_ratio","str_average_ratio","segr_ratio","segr_ratio_std","mass_seg","mass_seg_pval"
# ]:
#     col_left = f"{col}_left"
#     col_right = f"{col}_right"
#     print(col, np.max(df_comb[col_left] - df_comb[col_right]))


df_comb.to_csv(
    "data/databases/ALMEIDA2023.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df1 = pd.read_csv("data/databases/DIAS2014.csv")
df2 = pd.read_csv("data/databases/DIAS2014_1.csv")

# Print which elements in 'Cluster' column in df2 are not in 'Cluster' column in df1
missing_clusters = set(df2["Cluster"]) - set(df1["Cluster"])
print("Missing clusters in DIAS2014 compared to DIAS2014_1:")
for cluster in missing_clusters:
    print(cluster)

df_comb = pd.merge(df1, df2, on="Cluster", how="outer", suffixes=("_left", "_right"))

for cluster in df2["Cluster"]:
    row = df_comb[df_comb["Cluster"] == cluster]
    ra1 = row["RAJ2000"].values[0]
    dec1 = row["DEJ2000"].values[0]
    if np.isnan(ra1):
        df_comb.loc[row.index[0], "RAJ2000"] = row["_RA"].values[0]
        df_comb.loc[row.index[0], "DEJ2000"] = row["_DE"].values[0]

df_comb.to_csv(
    "data/databases/DIAS2014_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


# VANDENBERGH2006
df = pd.read_csv("/home/gabriel/Github/UCC/updt_UCC/data/databases/LOKTIN2017_2.csv")

# In the 'Name' column, change the entries 'NXXX' where XXX is a number to 'NGC XXX'
txt = "VDB"
Nt = len(txt)
df["Name"] = df["Name"].apply(
    lambda x: f"{txt} {x[Nt:]}" if x.startswith(f"{txt}") and x[Nt].isdigit() else x
)
df.to_csv(
    "/home/gabriel/Github/UCC/updt_UCC/data/databases/LOKTIN2017_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df = pd.read_csv("data/databases/MONTEIRO2019.csv")
df1 = pd.read_csv("data/databases/MONTEIRO2019_1.csv")

df_comb = pd.merge(df, df1, on="Name", how="outer", suffixes=("_left", "_right"))

# For each name in the 'Cluster' column, add an underscore between to separate the
# letters from the numbers
new_names = []
for name in df_comb["Name"]:
    split_index = next(
        (i for i, c in enumerate(name) if c.isdigit()), len(name)
    )  # Find index of first digit
    new_name = name[:split_index] + "_" + name[split_index:]
    new_names.append(new_name)
df_comb["Name"] = new_names

df2 = pd.read_csv("data/databases/MONTEIRO2019_2.csv")
# Add df2 rows to df_comb, filling the missing columns with 'nan' values
df_comb = pd.concat([df_comb, df2], ignore_index=True, sort=False)
df3 = pd.read_csv("data/databases/MONTEIRO2019_3.csv")
df_comb = pd.concat([df_comb, df3], ignore_index=True, sort=False)

df_comb.to_csv(
    "data/databases/MONTEIRO2019_4.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df = pd.read_csv("data/databases/ZHANG2024.csv")
df1 = pd.read_csv("data/databases/ZHANG2024_1.csv")

# Combine both dfs grouping by 'Cluster' columns. If an antry is missing, fill columns with "nan" values
df_combined = pd.merge(df, df1, on="Cluster", how="outer", suffixes=("_left", "_right"))
# Drop 'RA_ICRS_right', 'DE_ICRS_right' columns
df_combined = df_combined.drop(columns=["RA_ICRS_right", "DE_ICRS_right"])
# Rename 'RA_ICRS_left' to 'RA_ICRS' and 'DE_ICRS_left' to 'DE_ICRS'
df_combined = df_combined.rename(
    columns={"RA_ICRS_left": "RA_ICRS", "DE_ICRS_left": "DE_ICRS"}
)
# Convert columns "RA_ICRS","DE_ICRS" in 'h:m:s, d:m:s' into degrees using astropy
df_combined["RA_ICRS"] = np.round(
    Angle(df_combined["RA_ICRS"].astype(str), unit=u.hourangle).degree, 5
)  # pyright: ignore
df_combined["DE_ICRS"] = np.round(
    Angle(df_combined["DE_ICRS"].astype(str), unit=u.deg).degree, 5
)  # pyright: ignore

# For each name in the 'Cluster' column, add an underscore between to separate the letters from the numbers
new_names = []
for name in df_combined["Cluster"]:
    split_index = next(
        (i for i, c in enumerate(name) if c.isdigit()), len(name)
    )  # Find index of first digit
    new_name = name[:split_index] + "_" + name[split_index:]
    new_names.append(new_name)
df_combined["Cluster"] = new_names

df_combined.to_csv(
    "data/databases/ZHANG2024_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)


breakpoint()


breakpoint()


df = pd.read_csv("data/databases/NIZOVKINA2025.csv")

# Group rows with equivalent values in the 'Cluster' column
grouped = df.groupby("Cluster")
new_rows = []
for cluster_name, group in grouped:
    print(f"Cluster: {cluster_name}")
    # print(group)
    # Combine all rows for this group so that a single row remains
    combined_row = group.iloc[0].copy()
    for col in df.columns:
        if col != "Cluster":
            # combined_row[col] = ';'.join(group[col].astype(str).unique())
            combined_row[col] = ";".join(group[col].astype(str))
    print("Combined Row:")
    print(combined_row)
    new_rows.append(combined_row)

# Turn new_rows list into a dataframe
df_combined = pd.DataFrame(new_rows)


# Keep only unique floats in column 'RA_ICRS' and 'DE_ICRS'
def unique_floats(entry):
    floats = entry.split(";")
    unique = sorted(set(floats), key=lambda x: floats.index(x))
    # return ';'.join(unique)
    return float(unique[0])


df_combined["RA_ICRS"] = df_combined["RA_ICRS"].apply(unique_floats)
df_combined["DE_ICRS"] = df_combined["DE_ICRS"].apply(unique_floats)


df_combined.to_csv(
    "data/databases/NIZOVKINA2025_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
breakpoint()


df_old = pd.read_csv("../data/UCC_cat_B.csv")
df_new = pd.read_csv("UCC_cat_B.csv")

# COmpare the values in columns 'RA_ICRS, DE_ICRS'

plt.subplot(121)
plt.scatter(df_old["RA_ICRS"], df_old["RA_ICRS"] - df_new["RA_ICRS"])
plt.subplot(122)
plt.scatter(df_old["DE_ICRS"], df_old["DE_ICRS"] - df_new["DE_ICRS"])
plt.show()


breakpoint()

df = pd.read_csv("../data/UCC_cat_B.csv")
fnames0 = [_.split(";")[0] for _ in df["fnames"]]
idx = np.argsort(fnames0)
df = df.reindex(idx)
df.to_csv(
    "../data/UCC_cat_B_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df_MWSC = pd.read_csv("data/databases/MWSC_notes.csv")
df_MWSC_names = [_.split(",")[1] for _ in df_MWSC["Name"]]
# new_name = []
# for i, name in enumerate(df['Name']):
#     name = name.strip()
#     if not name.startswith('MWSC_'):
#         new_name.append(name + ', MWSC_' + str(df['MWSC'][i]))
#     else:
#         new_name.append(name)
# df['Name'] = new_name
# df = df.drop(columns=["MWSC"])
# df.to_csv(
#     "data/databases/MWSC_notes.csv",
#     na_rep="nan",
#     index=False,
#     quoting=csv.QUOTE_NONNUMERIC,
# )


db_name = "JUST2023"

df = pd.read_csv(f"data/databases/{db_name}.csv")

# Replace all spaces in entries in column 'Name' with a SINGLE underscore
# df["Name"] = df["Name"].str.replace(r"\s+", "_", regex=True)

# # Remove "psFile","atlas","BDA" columns
# df = df.drop(columns=["map","cmd","stars","MWSC"])

# # Convert columns "RA_ICRS","DE_ICRS" in 'h:m:s, d:m:s' into degrees using astropy
# from astropy.coordinates import Angle
# import astropy.units as u
# df["RA_deg"] = np.round(Angle(df["RA_ICRS"].astype(str), unit=u.hourangle).degree, 5)
# df["DE_deg"] = np.round(Angle(df["DE_ICRS"].astype(str), unit=u.deg).degree, 5)


# new_name = []
# for i, name in enumerate(df['Name']):
#     if not name.startswith('MWSC_'):
#         new_name.append(name + ', MWSC_' + str(df['MWSC'][i]))
#     else:
#         new_name.append(name)
# df['Name'] = new_name
# df = df.drop(columns=["MWSC"])


new_name, type = [], []
for i, name in enumerate(df["Name"]):
    try:
        i = df_MWSC_names.index(name)
        new_name.append(df_MWSC["Name"][i])
        type.append(df_MWSC["Type"][i])
    except:
        print(name)
        new_name.append(name)
        type.append("")

df["Name"] = new_name
df["Type"] = type


# Remove rows with 'Type' column matching either of
df = df[~df["Type"].isin(["a", "g", "m", "n", "r", "s"])]

txt = "" if db_name.endswith("_2") else "_2"
new_df_name = f"data/databases/{db_name}{txt}.csv"
df.to_csv(
    new_df_name,
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
