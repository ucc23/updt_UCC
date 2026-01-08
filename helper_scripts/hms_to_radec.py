import csv
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

filein = "../temp_updt/data/databases/ALESSI2003.csv"
fileout = filein.replace(".csv", "_2.csv")

df = pd.read_csv(filein)

# Convert columns in 'h m s, d m s' to deg using astropy
ra_col = "alpha2000"
dec_col = "delta2000"
coords = SkyCoord(
    ra=df[ra_col].values,
    dec=df[dec_col].values,
    unit=("hourangle", "deg"),
    frame="icrs",
)
df["RA_ICRS"] = np.round(coords.ra.deg, 5)
df["DE_ICRS"] = np.round(coords.dec.deg, 5)

df.to_csv(
    fileout,
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
