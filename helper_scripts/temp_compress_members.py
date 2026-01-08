import pandas as pd

df = pd.read_parquet("UCC_members.parquet")

df[["name", "Source", "probs"]].to_csv("temp.csv.gz", compression="gzip", index=False)
breakpoint()
