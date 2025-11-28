import pandas as pd

df_membs = pd.read_parquet("../zenodo/UCC_members.parquet")

old_name = "mwsc0192"
new_name = "mwsc192"

msk = df_membs["name"] == old_name
df_membs.loc[msk, "name"] = new_name

df_membs.to_parquet("UCC_members.parquet", index=False)

