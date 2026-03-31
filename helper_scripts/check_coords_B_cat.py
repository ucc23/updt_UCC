import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """ """ ""
    df_B = pd.read_csv("../data/UCC_cat_B.csv")

    # check_coords(df_B)

    # "dist","av","diff_ext","age","met","mass","bi_frac","blue_str"
    check_params(df_B)  # , "age")  # "dist")


def check_params(
    df_B,
    param_col="dist",
    median_perc=0.25,
    skip_ocs=("cwnu"),  # ("hsc", "theia", ),
):
    """
    Check consistency of parameter values across databases.

    Conflict definition:
        abs(value - median(vals)) > median_perc * median(vals)
    """

    results = []
    for row in df_B.itertuples(index=False):
        name = row.fnames.split(";")[0]

        if name.startswith(skip_ocs):
            continue
        if ";" not in row.DB:
            continue

        cl_dbs = row.DB.split(";")
        raw_vals = getattr(row, param_col).split(";")

        vals, dbs = [], []
        for db, v in zip(cl_dbs, raw_vals):
            v = v.replace("*", "").strip()
            if v.lower() == "nan" or v == "":
                continue

            try:
                vals.append(float(v))
                dbs.append(db)
            except ValueError:
                continue

        if len(vals) < 2:
            continue

        vals = np.asarray(vals)
        med_val = np.median(vals)
        # relative threshold
        thr = median_perc * med_val
        diffs = np.abs(vals - med_val)
        bad = np.where(diffs > thr)[0]
        if bad.size:
            max_diff = diffs.max()

            if bad.size == 1:
                offender = bad[0]
                msg = (
                    f"{name:<15} "
                    f"(Δ={max_diff:.3g}, "
                    f"{100 * max_diff / med_val:.1f}%): "
                    f"suspect {dbs[offender]}"
                )
            else:
                # DB farthest from the median
                offender = bad[np.argmax(diffs[bad])]
                msg = (
                    f"{name:<15} "
                    f"(Δ={max_diff:.3g}, "
                    f"{100 * max_diff / med_val:.1f}%): "
                    f"suspect {dbs[offender]} "
                    f"[{bad.size} conflicts]"
                )

            results.append((bad.size, max_diff, msg))

    # sort by largest deviation then number of conflicts
    results.sort(key=lambda x: (x[1], x[0]), reverse=True)
    for _, _, msg in results:
        print(msg)


def check_coords(df_B):
    """ """
    json_data_path = "../data/databases_info.json"
    with open(json_data_path, "r") as f:
        databases_info = json.load(f)

    all_dbs = {}
    cat_folder = Path("../data/databases/")
    for db in cat_folder.glob("*.csv"):
        # print(db)
        db_df = pd.read_csv(db)
        pos = databases_info[db.stem]["pos"]
        if pos:
            ra = db_df[pos["RA"]].values.round(4)
            dec = db_df[pos["DEC"]].values.round(4)
            all_dbs[db.stem] = {"RA": ra, "DEC": dec}
        else:
            print(f"Database {db.stem} does not have position information.")

    skip_ocs = ("hsc", "theia", "cwnu")
    knonw_bad_coords = {
        "ascc120": ["BICA2019"],
        "ascc123": ["ALFONSO2024"],
        "ascc72": ["BICA2019"],
        "blanco1": ["ALMEIDA2023", "DIAS2021"],
        "graham1": ["BUKOWIECKI2011"],
        "melotte245": ["DIAS2021"],
        "ngc1333": ["HE2022_1"],
        "saurer1": ["FROEBRICH2007"],
        "melotte186": ["DIAS2019"],
        "berkeley58": ["DIAS2021"],
        "berkeley59": ["DIAS2021"],
        "collinder461": ["ALMEIDA2023"],
        "fsr0839": ["GLUSHKOVA2010"],
        "ic5146": ["LADA2003"],
    }

    print("\n\n")
    print("\n===========================\n")

    diff_lim = 1
    col1, col2 = "RA", "DEC"

    results = []
    for _, row in df_B.iterrows():
        # OCs to skip check
        if row["fnames"].startswith(skip_ocs):
            continue
        if row["fnames"].split(";")[0] in knonw_bad_coords:
            continue

        cl_dbs = row["DB"].split(";")
        cl_db_i = [int(_) for _ in row["DB_i"].split(";")]
        pairs = [(db, i) for db, i in zip(cl_dbs, cl_db_i) if db in all_dbs]
        cl_dbs, cl_db_i = map(list, zip(*pairs)) if pairs else ([], [])

        if len(cl_dbs) > 1:
            cl_col1, cl_col2 = [], []
            for i, cl_db in enumerate(cl_dbs):
                cl_col1.append(all_dbs[cl_db][col1][cl_db_i[i]])
                cl_col2.append(all_dbs[cl_db][col2][cl_db_i[i]])

            if cl_col1:
                conflicts = Counter()
                pairs = []
                max_col_diff = 0

                for i in range(len(cl_col1)):
                    for j in range(i + 1, len(cl_col1)):
                        col1_diff = abs(cl_col1[i] - cl_col1[j])
                        col1_diff = min(col1_diff, 360 - col1_diff)
                        col2_diff = abs(cl_col2[i] - cl_col2[j])

                        max_col_diff = max(max_col_diff, col1_diff, col2_diff)

                        if col1_diff > diff_lim or col2_diff > diff_lim:
                            conflicts[i] += 1
                            conflicts[j] += 1
                            pairs.append((i, j))

                if conflicts:
                    name = row["fnames"].split(";")[0]

                    if len(conflicts) == 2 and all(v == 1 for v in conflicts.values()):
                        i, j = pairs[0]
                        msg = (
                            f"{name:<15} (Δ={max_col_diff:.2f}º): "
                            f"{name}: {cl_dbs[i]} vs {cl_dbs[j]} "
                        )
                    else:
                        offender = max(conflicts, key=conflicts.get)
                        msg = (
                            f"{name:<15} (Δ={max_col_diff:.2f}º): "
                            f"suspect {cl_dbs[offender]} "
                            f"[{conflicts[offender]} conflicts]"
                        )

                    results.append((max_col_diff, msg))

    # --- print sorted results ---
    for _, msg in sorted(results, key=lambda x: x[0], reverse=True):
        print(msg)


if __name__ == "__main__":
    main()
