import json
from collections import Counter
from pathlib import Path

import pandas as pd

df_B = pd.read_csv("../data/UCC_cat_B.csv")

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

deg_diff_lim = 1

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
# for _, row in df_B.iterrows():
#     if row["fnames"].startswith(skip_ocs):
#         continue

#     if row["fnames"].split(";")[0] in knonw_bad_coords.keys():
#         continue

#     cl_dbs = row["DB"].split(";")
#     cl_db_i = [int(_) for _ in row["DB_i"].split(";")]
#     pairs = [(db, i) for db, i in zip(cl_dbs, cl_db_i) if db in all_dbs]
#     cl_dbs, cl_db_i = map(list, zip(*pairs)) if pairs else ([], [])

#     if len(cl_dbs) > 1:
#         cl_ra, cl_dec = [], []
#         for i, cl_db in enumerate(cl_dbs):
#             cl_ra.append(all_dbs[cl_db]["RA"][cl_db_i[i]])
#             cl_dec.append(all_dbs[cl_db]["DEC"][cl_db_i[i]])
#         if cl_ra:
#             conflicts = Counter()
#             pairs = []
#             max_ra_dec_diff = 0

#             for i in range(len(cl_ra)):
#                 for j in range(i + 1, len(cl_ra)):
#                     ra_diff = abs(cl_ra[i] - cl_ra[j])
#                     ra_diff = min(ra_diff, 360 - ra_diff)
#                     dec_diff = abs(cl_dec[i] - cl_dec[j])

#                     # track maxima
#                     max_ra_dec_diff = max(max_ra_dec_diff, ra_diff, dec_diff)

#                     if ra_diff > deg_diff_lim or dec_diff > deg_diff_lim:
#                         conflicts[i] += 1
#                         conflicts[j] += 1
#                         pairs.append((i, j))

#             if conflicts:
#                 name = row["fnames"].split(";")[0]

#                 # Only one conflicting pair
#                 if len(conflicts) == 2 and all(v == 1 for v in conflicts.values()):
#                     i, j = pairs[0]
#                     print(
#                         f"{name}: {cl_dbs[i]} vs {cl_dbs[j]} "
#                         f"(max Δ={max_ra_dec_diff:.2f}º)"
#                     )
#                 else:
#                     offender = max(conflicts, key=conflicts.get)
#                     print(
#                         f"{name}: suspect {cl_dbs[offender]} "
#                         f"(max Δ={max_ra_dec_diff:.2f}º) "
#                         f"[{conflicts[offender]} conflicts]"
#                     )

results = []

print("\n===========================\n")
for _, row in df_B.iterrows():
    if row["fnames"].startswith(skip_ocs):
        continue

    if row["fnames"].split(";")[0] in knonw_bad_coords:
        continue

    cl_dbs = row["DB"].split(";")
    cl_db_i = [int(_) for _ in row["DB_i"].split(";")]
    pairs = [(db, i) for db, i in zip(cl_dbs, cl_db_i) if db in all_dbs]
    cl_dbs, cl_db_i = map(list, zip(*pairs)) if pairs else ([], [])

    if len(cl_dbs) > 1:
        cl_ra, cl_dec = [], []
        for i, cl_db in enumerate(cl_dbs):
            cl_ra.append(all_dbs[cl_db]["RA"][cl_db_i[i]])
            cl_dec.append(all_dbs[cl_db]["DEC"][cl_db_i[i]])

        if cl_ra:
            conflicts = Counter()
            pairs = []
            max_ra_dec_diff = 0

            for i in range(len(cl_ra)):
                for j in range(i + 1, len(cl_ra)):
                    ra_diff = abs(cl_ra[i] - cl_ra[j])
                    ra_diff = min(ra_diff, 360 - ra_diff)
                    dec_diff = abs(cl_dec[i] - cl_dec[j])

                    max_ra_dec_diff = max(max_ra_dec_diff, ra_diff, dec_diff)

                    if ra_diff > deg_diff_lim or dec_diff > deg_diff_lim:
                        conflicts[i] += 1
                        conflicts[j] += 1
                        pairs.append((i, j))

            if conflicts:
                name = row["fnames"].split(";")[0]

                if len(conflicts) == 2 and all(v == 1 for v in conflicts.values()):
                    i, j = pairs[0]
                    msg = (
                        f"{name:<15} (Δ={max_ra_dec_diff:.2f}º): "
                        f"{name}: {cl_dbs[i]} vs {cl_dbs[j]} "
                    )
                else:
                    offender = max(conflicts, key=conflicts.get)
                    msg = (
                        f"{name:<15} (Δ={max_ra_dec_diff:.2f}º): "
                        f"suspect {cl_dbs[offender]} "
                        f"[{conflicts[offender]} conflicts]"
                    )

                results.append((max_ra_dec_diff, msg))

# --- print sorted results ---
for _, msg in sorted(results, key=lambda x: x[0], reverse=True):
    print(msg)
