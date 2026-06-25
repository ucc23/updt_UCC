import csv

import pandas as pd

C_path = "../data/UCC_cat_C.csv"


edit_lst = [
    ("alessij06016p3531", "frame_limit=plxl_0.5"),
    ("bochum1", "N_clust=100"),
    ("fof5", "N_clust=250"),
    ("alessi116", "N_clust=150"),
    ("ic1442", "N_clust=100"),
    ("theia1079", "N_clust=75"),
    ("hsc1943", "N_clust=40"),
    ("fof373", "N_clust=150"),
    ("upk655", "N_clust=100"),
    ("ubc1325", "N_clust=70"),
    ("hogg22", "N_clust=75"),
    ("collinder471", "N_clust=75"),
    ("ubc1116", "N_clust=60"),
    ("sigmaorionis", "N_clust=150"),
    ("cepob5", "N_clust=150"),
    ("alessi65", "N_clust=80"),
    ("ascc72", "N_clust=100"),
    ("sai15", "N_clust=90"),
    ("ubc1514", "N_clust=125"),
    ("ubc44", "N_clust=100"),
    ("majaess65", "N_clust=100"),
    ("bdsb88", "N_clust=50"),
]


df_C = pd.read_csv(C_path)

fname0 = df_C["fname"].tolist()

for cluster in edit_lst:
    fname, new_data = cluster
    idx = fname0.index(fname)
    col, val = new_data.split("=")
    if col in ("N_clust", "N_clust_max"):
        val = int(val)
    df_C.loc[idx, col] = val
    df_C.loc[idx, "process"] = "y"
    print(f"Updated {fname}: {col}={val}")

df_C.to_csv(C_path, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC)
