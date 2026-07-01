import csv
import re

import pandas as pd

input_txt = """
Adding QIN2025 (149) to the UCC

Found 1 groups of duplicated fnames within the DB:
Group 1: oc0470
 - Huluwa_3 --> OC 0470 (47)
> /home/gabriel/Github/UCC/updt_UCC/modules/B_update_UCC_DB.py(938)check_new_DB_fnames()
-> sys.exit(1)
(Pdb) 
"""

dbs_path = "../data/databases/"


def main():
    """ """
    db_name = input_txt.split("Adding ")[1].split()[0]
    # Raise error if db_name is empty or not found in the input text
    if not db_name:
        raise ValueError("DB name not found in the input text")

    # Extract rows to remove and their descriptions
    removed_entries = re.findall(
        r"^\s*-\s+(.*?)\s+\((\d+)\)\s*$",
        input_txt,
        flags=re.MULTILINE,
    )

    idx_to_drop = [int(idx) for _, idx in removed_entries]

    print(f"DB name: {db_name}, removed N={len(idx_to_drop)} rows")
    print("\nRemoved duplicates:\n")
    for desc, _ in removed_entries:
        print(f"- {desc}")

    # Remove entries from DB
    db_path = f"{dbs_path}{db_name}.csv"
    df = pd.read_csv(db_path)
    df.drop(index=idx_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(db_path, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    main()
