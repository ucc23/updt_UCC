import csv

import pandas as pd

C_path = "../data/UCC_cat_C.csv"


edit_lst = [
    # ("alessij06016p3531", "frame_limit=plxl_0.5"),
    # ("bochum1", "N_clust=100"),
    # ("fof5", "N_clust=250"),
    # ("alessi116", "N_clust=150"),
    # ("ic1442", "N_clust=100"),
    # ("theia1079", "N_clust=75"),
    # ("hsc1943", "N_clust=40"),
    # ("fof373", "N_clust=150"),
    # ("upk655", "N_clust=100"),
    # ("ubc1325", "N_clust=70"),
    # ("hogg22", "N_clust=75"),
    # ("collinder471", "N_clust=75"),
    # ("ubc1116", "N_clust=60"),
    # ("sigmaorionis", "N_clust=150"),
    # ("cepob5", "N_clust=150"),
    # ("alessi65", "N_clust=80"),
    # ("ascc72", "N_clust=100"),
    # ("sai15", "N_clust=90"),
    # ("ubc1514", "N_clust=125"),
    # ("ubc44", "N_clust=100"),
    # ("majaess65", "N_clust=100"),
    # ("bdsb88", "N_clust=50"),
    #
    # ("trumpler28a", "N_clust=187"),
    # ("ubc59", "N_clust=63"),
    # ("ngc2327", "N_clust=682"),
    # ("alessij06016p3531", "N_clust=85"),
    # ("ubc1574", "N_clust=59"),
    # ("upk402", "N_clust=96"),
    # ("ubc1459", "N_clust=62"),
    # ("cwwdl1713", "N_clust=75"),
    # ("ubc1129", "N_clust=67"),
    # ("ubc7", "N_clust=230"),
    # ("fof1624", "N_clust=493"),
    # ("dbsb101", "N_clust=339"),
    # ("fof2238", "N_clust=185"),
    # ("fof597", "N_clust=235"),
    # ("fof321", "N_clust=333"),
    # ("czernik43", "N_clust=331"),
    # ("ubc277", "N_clust=196"),
    # ("fof1620", "N_clust=269"),
    # ("sai106", "N_clust=206"),
    # ("fof1994", "N_clust=319"),
    #
    # ("collinder394", "frame_limit=r_15.2"),
    # ("ngc6716", "frame_limit=l_15.2"),
    # ("fof145", "N_clust=500"),
    # ("mamajek4", "N_clust=200"),
    # ("fof866", "N_clust=200"),
    # ("ngc1582", "N_clust=150"),
    # ("ngc2374", "N_clust=150"),
    # ("vdbh23", "N_clust=200"),
    # ("teutsch265", "N_clust=150"),
    #
    ("mcm25", "N_clust=50"),
    ("ubc254", "N_clust=100"),
    ("fof145", "N_clust=600"),
    ("ngc6716", "frame_limit=l_15.2,plxl_1.1"),
]


df_C = pd.read_csv(C_path)

fname0 = df_C["fname"].tolist()

for cluster in edit_lst:
    fname, new_data = cluster
    idx = fname0.index(fname)
    col, val = new_data.split("=")
    if col in ("N_clust", "N_clust_max"):
        # val = int(val)
        # Round to nearest 10
        val = ((int(val) + 5) // 10) * 10
    df_C.loc[idx, col] = val
    df_C.loc[idx, "process"] = "y"
    print(f"Updated {fname}: {col}={val}")

df_C.to_csv(C_path, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC)
