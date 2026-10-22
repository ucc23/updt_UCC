import csv
import json
import os
import re

import pandas as pd

# Paste the output of the B script here. Two formats are recognized:
#
# 1. Duplicated fnames within the DB (check_new_DB_fnames), category 'duplicate':
#    ' - LISC-III_3323 --> LISC-III 3668 (49)'
#
# 2. Probable GCs (GCs_check), category 'GC':
#    '49     NGC 6558      --> 12     NGC 6558,           d=0.0'
#
# The listed entries are NOT removed from the DB file: they are added to the DB's
# exclusion file, and the B script skips them
input_txt = """
----------------------------------------
Adding CHI2023_1 (83) to the UCC

Found 1 groups of duplicated fnames within the DB:
Group 1: lisciii3668
 - LISC-III_3323 --> LISC-III 3668 (49)
> /home/gabriel/Github/UCC/updt_UCC/modules/B_update_UCC_DB.py(938)check_new_DB_fnames()
-> sys.exit(1)
(Pdb)
"""

# Paths relative to the 'helper_scripts/' folder
root = "../"
dbs_path = "data/databases/"
excl_path = dbs_path + "excluded/"
json_path = "data/databases_info.json"
temp_path = "temp_updt/"

excl_cols = ["row", "name", "category", "comment"]


def main():
    """ """
    db_name = input_txt.split("Adding ")[1].split()[0]
    # Raise error if db_name is empty or not found in the input text
    if not db_name:
        raise ValueError("DB name not found in the input text")

    # Duplicates: ' - LISC-III_3323 --> LISC-III 3668 (49)'
    dups = re.findall(
        r"^\s*-\s*(.*?)\s+-->\s+(.*?)\s+\((\d+)\)\s*$",
        input_txt,
        flags=re.MULTILINE,
    )
    # GCs: '49     NGC 6558     --> 12     NGC 6558,     d=0.0'
    gcs = re.findall(
        r"^(\d+)\s+(.*?)\s+-->\s+\d+\s+(.*?)\s*d=([\d.]+)\s*$",
        input_txt,
        flags=re.MULTILINE,
    )

    new_entries = []
    for desc, c_name, idx in dups:
        new_entries.append([int(idx), "duplicate", f"{desc} → {c_name}"])
    for idx, _, gc_name, dist in gcs:
        gc_name = gc_name.strip().rstrip(",")
        new_entries.append([int(idx), "GC", f"{gc_name} (d={dist}')"])
    if not new_entries:
        raise ValueError("No entries found in the input text")

    # New DBs (not yet moved by the B script) are stored in the temp folder
    db_file = temp_path + dbs_path + db_name + ".csv"
    if os.path.isfile(root + db_file):
        excl_file = temp_path + excl_path + db_name + ".csv"
        json_file = temp_path + json_path
    else:
        db_file = dbs_path + db_name + ".csv"
        excl_file = excl_path + db_name + ".csv"
        json_file = json_path

    with open(root + json_file) as f:
        names_col = json.load(f)[db_name]["names"]
    df_db = pd.read_csv(root + db_file)

    # Load current exclusion file (if any)
    if os.path.isfile(root + excl_file):
        df_excl = pd.read_csv(root + excl_file)
    else:
        df_excl = pd.DataFrame(columns=excl_cols)
    rows_excl = set(df_excl["row"].astype(int))

    rows = []
    for idx, category, comment in new_entries:
        if idx in rows_excl:
            print(f"Row {idx} already excluded, skipping")
            continue
        rows_excl.add(idx)
        name = str(df_db.loc[idx, names_col]).strip()
        rows.append([idx, name, category, comment])

    print(f"DB name: {db_name}, excluded N={len(rows)} rows --> {excl_file}")
    for category, title in (("duplicate", "Removed duplicates"), ("GC", "Removed GCs")):
        lines = [f"- {comment}" for _, _, cat, comment in rows if cat == category]
        if lines:
            print(f"\n{title}:\n")
            print("\n".join(lines))

    if not rows:
        return

    df_excl = pd.concat([df_excl, pd.DataFrame(rows, columns=excl_cols)])
    df_excl["row"] = df_excl["row"].astype(int)
    df_excl = df_excl.sort_values("row")
    os.makedirs(os.path.dirname(root + excl_file), exist_ok=True)
    df_excl.to_csv(root + excl_file, index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    main()
