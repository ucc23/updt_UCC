import csv

import numpy as np
import pandas as pd

file_path = "../data/all_names.csv"
df = pd.read_csv(file_path)

# Sort by fname0
fnames0 = [_.split(";")[0] for _ in df["fnames"]]
idx = np.argsort(fnames0)
df = df.reindex(idx)

# Save UCC to CSV file
df.to_csv(
    file_path,
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
