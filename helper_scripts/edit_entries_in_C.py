import csv

import pandas as pd

"""
This script edits entries in the UCC_cat_C.csv file based on the provided 'edit_lst'.
"""

C_path = "../data/UCC_cat_C.csv"

# Format: ("fname", "column1=value1;column2=value2;...")
# E.g: ("feigelson1", "N_clust_max=125;frame_limit=plxl_9;box_size=15")
edit_lst = [
    ("platais8", "N_clust_max=400"),
    ("gulliver42", "N_clust_max=150"),
    ("bdsb96", "N_clust_max=175"),
    ("dc5", "N_clust_max=250"),
    ("ubc565", "N_clust_max=300"),
    ("dbsb101", "N_clust_max=250"),
    ("westerlund2", "N_clust_max=200"),
    ("collinder106", "N_clust_max=200;frame_limit=plxl_0.55,plxr_0.8"),
    ("ascc45", "N_clust_max=100"),
    ("juchert11", "N_clust_max=100"),
    ("king14", "N_clust_max=300"),
    ("fsr0344", "N_clust_max=150"),
    ("rsg8", "N_clust_max=300;frame_limit=plxr_2.17"),
    ("ngc3603", "N_clust_max=400"),
    ("berkeley92", "N_clust_max=200"),
    ("ruprecht78", "N_clust_max=200"),
    ("fsr0366", "N_clust_max=150"),
    ("saurer2", "N_clust_max=200"),
    ("fsr0716", "N_clust_max=275"),
    ("pismis5", "N_clust_max=250"),
]


df_C = pd.read_csv(C_path)

fname0 = df_C["fname"].tolist()

for cluster in edit_lst:
    fname, new_data = cluster
    idx = fname0.index(fname)
    for new_data_col in new_data.split(";"):
        col, val = new_data_col.split("=")
        if col in ("N_clust", "N_clust_max"):
            # Round to nearest 10
            val = ((int(val) + 5) // 10) * 10
        if col == "box_size":
            val = float(val)
        df_C.loc[idx, col] = val
    df_C.loc[idx, "process"] = "y"
    print(f"Updated {fname}: {new_data}")

df_C.to_csv(C_path, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC)
