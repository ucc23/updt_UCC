import csv

import pandas as pd

"""
This script edits entries in the UCC_cat_C.csv file based on the provided 'edit_lst'.
"""

C_path = "../data/UCC_cat_C.csv"

# Format: ("fname", "column1=value1;column2=value2;...")
# E.g: ("feigelson1", "N_clust_max=125;frame_limit=plxl_9;box_size=15")
edit_lst = [
    ("saurer1", "N_clust=75"),
    ("fsr1212", "N_clust=80"),
    ("vdbh37", "N_clust_max=175"),
]

# Check for duplicate fnames in edit_lst before processing
fnames_in_edit_lst = [cluster[0] for cluster in edit_lst]
if len(fnames_in_edit_lst) != len(set(fnames_in_edit_lst)):
    seen = set()
    dupes = set()
    for fname in fnames_in_edit_lst:
        if fname in seen:
            dupes.add(fname)
        seen.add(fname)
    raise ValueError(f"Duplicate fname(s) found in edit_lst: {sorted(dupes)}")


df_C = pd.read_csv(C_path)
fname0 = df_C["fname"].tolist()

for cluster in edit_lst:
    fname, new_data = cluster
    idx = fname0.index(fname)
    for new_data_col in new_data.split(";"):
        col, val = new_data_col.split("=")
        if col not in (
            "N_clust",
            "N_clust_max",
            "box_size",
            "frame_limit",
            "plot_used",
        ):
            raise ValueError(f"{col} not recognized")
        if col in ("N_clust", "N_clust_max"):
            # Round to nearest 5
            val = round(int(val) / 5) * 5
        if col == "box_size":
            val = float(val)
        df_C.loc[idx, col] = val
    df_C.loc[idx, "process"] = "y"
    print(f"Updated {fname}: {new_data}")

df_C.to_csv(C_path, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC)
