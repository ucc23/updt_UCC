import csv
import json

import pandas as pd

df_UCC = pd.read_csv("../zenodo/UCC_cat_241130.csv")
N = 0
for i, fnames in enumerate(df_UCC['fnames']):
    fname0 = fnames.split(';')[0]
    if fname0.startswith('fof'):
        # print(i)
        N += 1
print(f"{N} FoF entries in the UCC\n")


print("Databases with 'FoF' entries:")
# Path to the JSON file and the directory containing the CSV files
json_file_path = "all_dbs.json"  # Replace with the actual JSON file path
out_path = "OUT/"

# Load the JSON file
with open(json_file_path, "r") as json_file:
    json_data = json.load(json_file)


# Iterate through each key in the JSON file
for key, value in json_data.items():
    # Extract the 'name' variable from the 'names' parameter
    name = value["names"]

    # Construct the path to the corresponding CSV file
    csv_file_path = f"{key}.csv"

    # Load the CSV file using pandas
    df = pd.read_csv(csv_file_path)
    df_new = df.copy()

    def fof2lp(df_name):
        N = 0
        for i, _name in enumerate(df_name):
            for fof_id in ("FoF", "fof", "FOF", "Fof"):
                if str(_name).startswith(fof_id):
                    _name = _name.replace(fof_id, "LP")
                    df_new.loc[i, name] = _name
                    N += 1
        return df_new[name], N

    # Update the 'Names' column based on the 'name' variable
    if name in df.columns:
        df_new[name], N = fof2lp(df[name])

        are_identical = df.equals(df_new)
        if not are_identical:
            print(key, N)
            # Save the updated DataFrame back to the CSV file
            df_new.to_csv(
                out_path + csv_file_path,
                index=False,
                na_rep="nan",
                quoting=csv.QUOTE_NONNUMERIC,
            )
            # print(f"Updated and saved file: {csv_file_path}")
