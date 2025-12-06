import csv
import numpy as np
import pandas as pd

df_h24 = pd.read_csv("HUNT2024_types.csv")
h24_fnames = [_.split(',')[0] for _ in df_h24["fname"]]



df_h23 = pd.read_csv("../databases/HUNT2023.csv")
fname_check = []
for name in df_h23['Name']:
    fname = name.lower().replace('_', '').replace('-', '').replace('.', '').replace('+', 'p').replace(' ', '')
    fname_check.append(fname)

not_found = []
for i, fname in enumerate(fname_check):
    fname = fname.split(',')[0]
    try:
        j = h24_fnames.index(fname)
        if df_h24["Type"][j] != "o":
            if fname.startswith("hsc") or fname.startswith("theia"):
                print(fname, df_h24["Type"][j])
    except:
        not_found.append(fname)
        pass

print(f"fnames not found in HUNT2024: {len(not_found)}")
print(np.array(not_found))
breakpoint()

# # Strip spaces from all string elements in the column "Name"
# df_h24['Name'] = df_h24['Name'].str.strip()
# df_h24['AllNames'] = df_h24['AllNames'].str.strip()

# fnames = []
# for _ in df_h24['Name']:
#     fname = _.lower().replace('_', '').replace('-', '').replace('.', '').replace('+', 'p')
#     fnames.append(fname)
# df_h24['fname'] = fnames

# # Drop 'recno' column
# df_h24 = df_h24.drop(columns=['recno'])

# # Move "type" column to the first position"
# type_col = df_h24.pop('Type')
# df_h24.insert(0, 'Type', type_col)

# fname_col = df_h24.pop('fname')
# df_h24.insert(0, 'fname', fname_col)


# fname_col = df_h24.pop('AllNames')
# df_h24.insert(5, 'AllNames', fname_col)

# df_h24.to_csv(
#     "HUNT2024_types.csv",
#     na_rep="nan",
#     index=False,
#     quoting=csv.QUOTE_NONNUMERIC,
# )
