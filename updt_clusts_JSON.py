
import pandas as pd
import numpy as np


date = "230702"

df_UCC = pd.read_csv('UCC_cat_' + date + '.csv')

# Update cluster's JSON file (used by 'ucc.ar' seach)
df = pd.DataFrame(df_UCC[[
    'ID', 'fnames', 'UCC_ID', 'RA_ICRS', 'DE_ICRS', 'GLON', 'GLAT']])
df['ID'] = [_.split(';')[0] for _ in df['ID']]

df['RA_ICRS'] = np.round(df['RA_ICRS'].values, 2)
df['DE_ICRS'] = np.round(df['DE_ICRS'].values, 2)
df['GLON'] = np.round(df['GLON'].values, 2)
df['GLAT'] = np.round(df['GLAT'].values, 2)

df.to_json('../ucc/_clusters/clusters.json', orient="records", indent=1)

print("File 'clusters.json' updated")
