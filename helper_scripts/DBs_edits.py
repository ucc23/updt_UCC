import csv

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


df = pd.read_csv("../temp_updt/data/databases/BIJAVARA2025.csv")
df4 = pd.read_csv("../temp_updt/data/databases/BIJAVARA2025_4.csv")

# # Drop the follosing columns
# drop_cols = ["StarID","GaiaDR3", "Gmag"]
# df = df.drop(columns=drop_cols)

# # Combine rows with duplicated values in the 'Cluster' column. For columns
# # "Age","Rgc","Av" use the first value available, for columns "RAJ2000","DEJ2000" use the median
# cols_first = ["Age", "Rgc", "Av"]
# cols_median = ["RAJ2000", "DEJ2000"]

# agg_dict = {c: "first" for c in cols_first}
# agg_dict.update({c: "median" for c in cols_median})

# df_out = (
#     df.groupby("Cluster", as_index=False)
#       .agg(agg_dict)
# )

# df1 = pd.read_csv("../temp_updt/data/databases/BIJAVARA2025_1.csv")
# df2 = pd.read_csv("../temp_updt/data/databases/BIJAVARA2025_2.csv")
# df3 = pd.read_csv("../temp_updt/data/databases/BIJAVARA2025_3.csv")

# # Merge the three dataframes along 'Cluster' column
# df_merged = pd.merge(df1, df2, on="Cluster", how="outer", suffixes=("_1", "_2"))

df_merged = pd.merge(df, df4, on="Cluster", how="outer", suffixes=("_1", "_2"))

df_merged.to_csv(
    "../temp_updt/data/databases/BIJAVARA2025_1.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()






df = pd.read_csv("../temp_updt/data/databases/CUI2025.csv")

# Convert GLON,GLAT columns to RA,DEC using astropy
coords = SkyCoord(
    l=df["GLON"].values * u.deg, b=df["GLAT"].values * u.deg, frame="galactic"
).icrs
df["RA"] = coords.ra.deg.round(5)
df["DEC"] = coords.dec.deg.round(5)

df.to_csv(
    "../temp_updt/data/databases/CUI2025_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()




df = pd.read_csv("../temp_updt/data/databases/SPINA2022.csv")

# Round all float columns to 5 decimals
float_cols = df.select_dtypes(include=["float"]).columns
df[float_cols] = df[float_cols].round(5)

df.to_csv(
    "../temp_updt/data/databases/SPINA2022.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()






df = pd.read_csv("../temp_updt/data/databases/AHUMADA2007.csv")

coords = SkyCoord(
    ra=df["RAJ2000"].astype(str),
    dec=df["DEJ2000"].astype(str),
    unit=(u.hourangle, u.deg),
    frame='icrs'
)
# Extract decimal degrees
df["RA"] = coords.ra.deg.round(5)
df["DEC"] = coords.dec.deg.round(5)

df.to_csv(
    "../temp_updt/data/databases/AHUMADA2007_1.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()




df = pd.read_csv("../temp_updt/data/databases/BICA2003_1.csv")

# Not in UCC
msk = []
oc_check = ('bdsb39','bdsb50','bdsb122','bdsb49','bdsb78','bdsb59','bdsb60','bdsb42','bdsb32','bdsb160','bdsb41','bdsb45','bdsb75','bdsb43','bdsb55','bdsb76','bdsb36','bdsb14','bdsb101','bdsb94','bdsb70','bdsb25','bdsb31','bdsb20','bdsb54','bdsb66','bdsb89','bdsb35','bdsb48','bdsb74','bdsb13','bdsb33','bdsb56','bdsb17','bdsb51','bdsb163','bdsb105','bdsb62','bdsb40','bdsb85','bdsb87','bdsb92','bdsb58','bdsb5','bdsb61','bdsb80','bdsb84','bdsb4','bdsb1','bdsb63','bdsb9','bdsb161','bdsb67','bdsb64','bdsb158','bdsb95','bdsb162','bdsb150','bdsb38','bdsb103','bdsb57','bdsb83','bdsb28','bdsb2','bdsb71','bdsb86','bdsb159','bdsb52','bdsb79','bdsb27','bdsb26','bdsb34','bdsb53','bdsb16','bdsb65','bdsb72','bdsb29',)
for i, oc in enumerate(df['BDS2003']):
    foc = oc.split(',')[0].lower().strip().replace("_", "")
    if foc in oc_check and df.at[i, "Class"] == "IRGr":
        print(oc, df.at[i, "Class"])
        msk.append(False)
    else:
        msk.append(True)
df = df[msk]


df.to_csv(
    "../temp_updt/data/databases/BICA2003_1_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()




df = pd.read_csv("../temp_updt/data/databases/DUTRA2003_orig.csv")

# # Convert GLON,GLAT columns to RA,DEC using astropy
# coords = SkyCoord(
#     l=df["GLON"].values * u.deg, b=df["GLAT"].values * u.deg, frame="galactic"
# ).icrs
# df["RA"] = coords.ra.deg.round(5)
# df["DEC"] = coords.dec.deg.round(5)

# for i, oc in enumerate(df['Seq']):
#     if df.at[i, "Type"] == 'IRGr':
#         print(oc)

# Not in UCC
msk = []
oc_check = ('dbsb25','dbsb41','dbsb59','dbsb84','dbsb76','dbsb33','dbsb44','dbsb42','dbsb124','dbsb32','dbsb4','dbsb30','dbsb65','dbsb28','dbsb81','dbsb67','dbsb34','dbsb1','dbsb112','dbsb128','dbsb110','dbsb8','dbsb107','dbsb18','dbsb13','dbsb54','dbsb23','dbsb20','dbsb26','dbsb172','dbsb24','dbsb5','dbsb16','dbsb15','dbsb22','dbsb38','dbsb55','dbsb126','dbsb11','dbsb53','dbsb109','dbsb14','dbsb58','dbsb127','dbsb111','dbsb46','dbsb17','dbsb50','dbsb57','dbsb74','dbsb56','dbsb47','dbsb45','dbsb29','dbsb2','dbsb63','dbsb31','dbsb49','dbsb83','dbsb52','dbsb37','dbsb71')
for i, oc in enumerate(df['Seq']):
    foc = oc.split(',')[0].lower().strip().replace("_", "")
    if foc in oc_check and df.at[i, "Type"] == "IRGr":
        print(oc, df.at[i, "Type"])
        msk.append(False)
    else:
        msk.append(True)
df = df[msk]

df.to_csv(
    "../temp_updt/data/databases/DUTRA2003.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()




df = pd.read_csv("../temp_updt/data/databases/BICA2003_orig.csv")

# Convert GLON,GLAT columns to RA,DEC using astropy
coords = SkyCoord(
    l=df["GLON"].values * u.deg, b=df["GLAT"].values * u.deg, frame="galactic"
).icrs
df["RA"] = coords.ra.deg.round(5)
df["DEC"] = coords.dec.deg.round(5)

# Add 'Name' column in 0 index, taken from 'Names' colum split by ',' and keep 0 element
df.insert(
    0,
    "Name",
    df["Names"]
      .str.split(",").str[0]
      .str.strip()
      .str.replace("IR Cluster", "", regex=False)
      .str.replace("Cluster", "", regex=False)
      .str.replace("Stellar Group", "", regex=False)
      .str.strip()
)

df.to_csv(
    "../temp_updt/data/databases/BICA2003.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()





df = pd.read_csv("../temp_updt/data/databases/DUTRA2001_orig.csv")

# Convert GLON,GLAT columns to RA,DEC using astropy
coords = SkyCoord(
    l=df["l"].values * u.deg, b=df["b"].values * u.deg, frame="galactic"
).icrs
df["RA"] = coords.ra.deg.round(5)
df["DEC"] = coords.dec.deg.round(5)

df.to_csv(
    "../temp_updt/data/databases/DUTRA2001.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()



df = pd.read_csv("../temp_updt/data/databases/DUTRA2000_orig.csv")

# Convert GLON,GLAT columns to RA,DEC using astropy
coords = SkyCoord(
    l=df["l"].values * u.deg, b=df["b"].values * u.deg, frame="galactic"
).icrs
df["RA"] = coords.ra.deg.round(5)
df["DEC"] = coords.dec.deg.round(5)

df.to_csv(
    "../temp_updt/data/databases/DUTRA2000.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()





df = pd.read_csv("../temp_updt/data/databases/BICA2003_orig.csv")

# Convert GLON,GLAT columns to RA,DEC using astropy
coords = SkyCoord(
    l=df["GLON"].values * u.deg, b=df["GLAT"].values * u.deg, frame="galactic"
).icrs
df["RA"] = coords.ra.deg.round(5)
df["DEC"] = coords.dec.deg.round(5)

# for i, oc in enumerate(df['Seq']):
#     if df.at[i, "Type"] == 'IRGr':
#         print(oc)
# breakpoint()

# # Not in UCC
# oc_check = ("dbsb37","dbsb76","dbsb124","dbsb53","dbsb22","dbsb45","dbsb74","dbsb41","dbsb107","dbsb81","dbsb112","dbsb23","dbsb65","dbsb17","dbsb1","dbsb55","dbsb30","dbsb24","dbsb13","dbsb67","dbsb4","dbsb5","dbsb20","dbsb18","dbsb50","dbsb15","dbsb57","dbsb33","dbsb71","dbsb2","dbsb42","dbsb128","dbsb83","dbsb11","dbsb32","dbsb84","dbsb38","dbsb126","dbsb25","dbsb49","dbsb110","dbsb56","dbsb34","dbsb127","dbsb26","dbsb44","dbsb28","dbsb111","dbsb59","dbsb54","dbsb16","dbsb52","dbsb172","dbsb31","dbsb29","dbsb109","dbsb63","dbsb14","dbsb8","dbsb47","dbsb46","dbsb58",)
# for i, oc in enumerate(df['Seq']):
#     foc = oc.split(',')[0].lower().strip().replace("_", "")
#     if foc in oc_check:
#         print(oc, df.at[i, "Type"])


df.to_csv(
    "../temp_updt/data/databases/BICA2003.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()







df = pd.read_csv("../temp_updt/data/databases/SWIGGUM2024.csv")

drop_cols = ["ID","AllNames","CST","N","CSTt","Nt","GLON","GLAT","r50","rc","rt","rtot","r50pc","rcpc","rtpc","rtotpc","s_pmRA","e_pmRA","s_pmDE","e_pmDE","s_Plx","e_Plx","dist16","dist50","dist84","Ndist","globalPlx","X","Y","Z","RV","s_RV","e_RV","n_RV","CMDCl2.5","CMDCl16","CMDCl50","CMDCl84","CMDCl97.5","CMDClHuman","logAge16","logAge50","logAge84","AV16","AV50","AV84","diffAV16","diffAV50","diffAV84","MOD16","MOD50","MOD84","minClSize","isMerged","isGMMMemb","NXmatches","XmatchType","_RA.icrs","_DE.icrs"]
# Drop columns
df = df.drop(columns=drop_cols)

df['dist'] = np.sqrt(df['Xhelio']**2 + df['Yhelio']**2 + df['Zhelio']**2)

# Round all float columns to 5 decimals
float_cols = df.select_dtypes(include=["float"]).columns
df[float_cols] = df[float_cols].round(5)

df.to_csv(
    "../temp_updt/data/databases/SWIGGUM2024_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()








df = pd.read_csv("../temp_updt/data/databases/KOUNKEL2019.csv")

# # Transformation constant (km/s to mas/yr at 1 kpc)
# k = 4.74047
# # 1. Compute distance from Heliocentric XYZ (kpc)
# df['dist'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2) /1000
# # 2. Define ICRS positions to get the rotation Jacobian
# c_icrs = SkyCoord(ra=df['_RA.icrs'].values*u.deg, dec=df['_DE.icrs'].values*u.deg, frame='icrs')
# c_gal = c_icrs.galactic
# # 3. Convert velocities (km/s) to proper motions (mas/yr) in Galactic frame
# # mu = v / (k * d)
# mu_l_cosb = df['pmGLON'].values / (k * df['dist'].values)
# mu_b = df['pmGLAT'].values / (k * df['dist'].values)
# # 4. Define full Galactic SkyCoord with proper motions
# c_gal_kin = SkyCoord(l=c_gal.l, b=c_gal.b,
#                      pm_l_cosb=mu_l_cosb * u.mas/u.yr,
#                      pm_b=mu_b * u.mas/u.yr,
#                      frame='galactic')

# # 5. Transform to ICRS (Equatorial)
# c_icrs_kin = c_gal_kin.transform_to('icrs')
# # 6. Assign result
# df['pmRA'] = c_icrs_kin.pm_ra_cosdec.value / np.cos(c_icrs_kin.dec.radian)
# df['pmDEC'] = c_icrs_kin.pm_dec.value
# breakpoint()


df_ucc = pd.read_csv("../data/UCC_cat_B.csv")

ucc_fnames = [_.split(';') for _ in df_ucc.fnames.values]
# ucc_fnames = [_.split(";") for _ in ucc_fnames_v]
# # Flatten list
ucc_fnames = [item for sublist in ucc_fnames for item in sublist]

# Compute distance from Heliocentric XYZ
df['dist'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
df['dist'] = df['dist'].round(2)

# Nnans, Ntheia, Neither = 0, 0, 0
msk = []
for row in df.itertuples():
    theia_name = str(row.Theia).lower().replace('_', '')
    
    if theia_name in ucc_fnames:
        # Ntheia += 1
        msk.append(True)
        # ucc_fname = ucc_fnames[ucc_fnames.index(theia_name)]
        # print(f"Theia '{row.Theia}' in UCC")
        # if str(row.OName) != "nan":
        #     print(f"OName is not nan for Theia '{row.Theia}'")
    else:
        msk.append(False)

    # elif str(row.OName) == "nan":
    #     Nnans += 1
    # # fname = str(row.OName).lower().replace(" ", "").replace("_", "").replace("-", "")
    # # if fname in ucc_fnames and theia_name in ucc_fnames:
    # #     # Check that 'fname, theia_name' correspond to the same index row in df_ucc.fnames
    # #     fi = -1
    # #     for i, ucc_fname in enumerate(ucc_fnames_v):
    # #         if fname in ucc_fname:
    # #             fi = i
    # #             break
    # #     fj = -1
    # #     for i, ucc_fname in enumerate(ucc_fnames_v):
    # #         if theia_name in ucc_fname:
    # #             fj = i
    # #             break
    # #     print(fname, theia_name, "both", fi, fj)
    # #     breakpoint()

    # # elif fname in ucc_fnames:
    # #     # ucc_fname = ucc_fnames[ucc_fnames.index(fname)]
    # #     print(f"OName '{row.OName}' in UCC")

    # else:
    #     Neither += 1
    #     print(f"Neither OName '{row.OName}' nor Theia '{row.Theia}' in UCC")

df = df[msk]


df.to_csv(
    "../temp_updt/data/databases/KOUNKEL2019_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()




df = pd.read_csv("../temp_updt/data/databases/CAMARGO2012.csv")

coords = SkyCoord(
    ra=df["alpha"].astype(str),
    dec=df["delta"].astype(str),
    unit=(u.hourangle, u.deg),
    frame='icrs'
)
# Extract decimal degrees
df["RA"] = coords.ra.deg.round(5)
df["DEC"] = coords.dec.deg.round(5)


# The columns "Age","EBV","R_sun" contain uncertainty data after a '±' car, split tese colums ito data ad ucertaity
for col in ["Age", "A_V", "d_sun"]:
    # Ensure string
    df[col] = df[col].astype(str).str.strip()
    # Split once
    split_cols = df[col].str.split("±", n=1, expand=True)
    # Convert
    value = pd.to_numeric(split_cols[0], errors="coerce")
    error = pd.to_numeric(split_cols[1], errors="coerce")
    # Replace original column with value and insert error column next to it
    df[col] = value
    col_idx = df.columns.get_loc(col)
    df.insert(col_idx + 1, f"e_{col}", error)


# df = df.drop(columns=["alpha", "delta"])


df.to_csv(
    "../temp_updt/data/databases/CAMARGO2012_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()





# df = pd.read_csv("../temp_updt/data/databases/TADROSS2014.csv")
df = pd.read_csv("/home/gabriel/Descargas/TADROSS2014.csv")

# Merge cluster_2 column into Cluster column, using a '_'  to join them
df["Cluster"] = df.apply(
    lambda row: f"{row['Cluster']}_{row['cluster_2']}"
    if pd.notna(row["cluster_2"])
    else row["Cluster"],
    axis=1,
)


# # The columns "h","m","s" and "d","m","s" represent RA and DEC data. Replace them with two columns with degrees data, use astropy
# from astropy.coordinates import Angle
# df["RA"] = np.round(Angle(df["h"].astype(str) + "h" + df["m"].astype(str) + "m" + df["s"].astype(str) + "s", unit=u.hourangle).degree, 5)
# df["DEC"] = np.round(Angle(df["d"].astype(str) + "d" + df["m.1"].astype(str) + "m" + df["s.1"].astype(str) + "s", unit=u.deg).degree, 5)

from astropy.coordinates import SkyCoord
import astropy.units as u
coords = SkyCoord(
    ra=df["h"].astype(str) + "h" + df["m"].astype(str) + "m" + df["s"].astype(str) + "s",
    dec=df["d"].astype(str) + "d" + df["m.1"].astype(str) + "m" + df["s.1"].astype(str) + "s",
    unit=(u.hourangle, u.deg),
    frame='icrs'
)
# Extract decimal degrees
df["RA"] = coords.ra.deg.round(5)
df["DEC"] = coords.dec.deg.round(5)


# The columns "Age","EBV","R_sun" contain uncertainty data after a '±' car, split tese colums ito data ad ucertaity
for col in ["Age", "EBV", "R_sun"]:
    # Ensure string
    df[col] = df[col].astype(str).str.strip()
    # Split once
    split_cols = df[col].str.split("±", n=1, expand=True)
    # Convert
    value = pd.to_numeric(split_cols[0], errors="coerce")
    error = pd.to_numeric(split_cols[1], errors="coerce")
    # Replace original column with value and insert error column next to it
    df[col] = value
    col_idx = df.columns.get_loc(col)
    df.insert(col_idx + 1, f"e_{col}", error)


df = df.drop(columns=["cluster_2", "h","m","s","d","m.1","s.1"])


df.to_csv(
    "../temp_updt/data/databases/TADROSS2014_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df1 = pd.read_csv("../temp_updt/data/databases/SPINA2021.csv")
df2 = pd.read_csv("../temp_updt/data/databases/SPINA2021_1.csv")

drop_cols = [
    "XXYZ",
    "e_XXYZ",
    "YXYZ",
    "e_YXYZ",
    "ZXYZ",
    "e_ZXYZ",
    "ULSR",
    "e_ULSR",
    "VLSR",
    "e_VLSR",
    "WLSR",
    "e_WLSR",
    "RRzphi",
    "e_RRzphi",
    "phiRzphi",
    "e_phiRzphi",
    "zRzphi",
    "e_zRzphi",
    "vRRzphi",
    "e_vRRzphi",
    "vphiRzphi",
    "e_vphiRzphi",
    "vzRzphi",
    "e_vzRzphi",
    "JR",
    "e_JR",
    "LZ",
    "e_LZ",
    "JZ",
    "e_JZ",
    "ecc",
    "e_ecc",
    "zmax",
    "e_zmax",
    "Rperi",
    "e_Rperi",
    "Rap",
    "e_Rap",
    "Energy",
    "e_Energy",
    "Rguid",
    "e_Rguid",
]
df1 = df1.drop(columns=drop_cols)

# Show which entries in 'Cluster' column in df2 are not in 'Cluster' column in df1
missing_clusters = set(df2["Cluster"]) - set(df1["Cluster"])
print("Missing clusters in SPINA2021 compared to SPINA2021_1:")
for cluster in missing_clusters:
    print(cluster)
missing_clusters = set(df1["Cluster"]) - set(df2["Cluster"])
print("Missing clusters in SPINA2021_1 compared to SPINA2021:")
for cluster in missing_clusters:
    print(cluster)


# Combine both dataframes filling empty values with "nan"
df = pd.merge(df1, df2, on="Cluster", how="outer", suffixes=("_left", "_right"))

df.to_csv(
    "../temp_updt/data/databases/SPINA2021_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df = pd.read_csv("../temp_updt/data/databases/RICHER2021.csv")

df = df.drop(columns=["_RA", "_DE"])

df.to_csv(
    "../temp_updt/data/databases/RICHER2021_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df = pd.read_csv("../temp_updt/data/databases/MORALES2013.csv")

# # Convert RAj2000, DECJ2000 columns to degrees
# from astropy.coordinates import Angle
# df["RA"] = np.round(Angle(df["RAJ2000"].astype(str), unit=u.hourangle).degree, 5)
# df["DEC"] = np.round(Angle(df["DEJ2000"].astype(str), unit=u.deg).degree, 5)

# # Merge columns "Name","OName" separated by comma
# df["Name"] = df.apply(lambda row: f"{row['Name']}, {row['OName']}" if pd.notna(row["OName"]) else row["Name"], axis=1
# )

# Remove the word Cluster from the 'Name' column
df["Name"] = df["Name"].str.replace("IR Cluster", "", regex=False).str.strip()
df["Name"] = df["Name"].str.replace("Cluster", "", regex=False).str.strip()

df.to_csv(
    "../temp_updt/data/databases/MORALES2013_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df = pd.read_csv(
    "/home/gabriel/Github/UCC/updt_UCC/temp_updt/data/databases/BUKOWIECKI2012.csv"
)

# Ensure string
df["Mtotal"] = df["Mtotal"].astype(str).str.strip()
# Split once
split_cols = df["Mtotal"].str.split("±", n=1, expand=True)
# Convert
d_value = pd.to_numeric(split_cols[0], errors="coerce")
d_error = pd.to_numeric(split_cols[1], errors="coerce")
# Replace 'd' and insert 'd_error' next to it
df["Mtotal"] = d_value
col_idx = df.columns.get_loc("Mtotal")
df.insert(col_idx + 1, "e_Mtotal", d_error)

df = df.sort_values("Star cluster").reset_index(drop=True)

df.to_csv(
    "../temp_updt/BUKOWIECKI2012.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df1 = pd.read_csv(
    "/home/gabriel/Descargas/table1 (1).csv",
    dtype={"alpha": "string", "delta": "string"},
)
df2 = pd.read_csv("/home/gabriel/Descargas/table2.csv")

# # Compare columns 'Star cluster' in both dataframes to see which elements are missing
# # Ensure consistent formatting (optional but recommended)
# df1["Star cluster"] = df1["Star cluster"].astype(str).str.strip()
# df2["Star cluster"] = df2["Star cluster"].astype(str).str.strip()
# # Convert to sets
# set1 = set(df1["Star cluster"])
# set2 = set(df2["Star cluster"])
# # Elements in df1 but not in df2
# missing_in_df2 = sorted(set1 - set2)
# # Elements in df2 but not in df1
# missing_in_df1 = sorted(set2 - set1)
# print("In df1 but not in df2:")
# print(missing_in_df2)
# print("\nIn df2 but not in df1:")
# print(missing_in_df1)

# The columns 'alpha, delta' in df1 are in the format hhmmss, ±ddmmss, add 'RA', 'DEC' columns in degrees (Use astropy)
from astropy.coordinates import SkyCoord

# Ensure strings and strip spaces
df1["alpha"] = df1["alpha"].astype(str).str.strip()
df1["delta"] = df1["delta"].astype(str).str.strip()


# If format is strictly hhmmss and ±ddmmss (no separators),
# insert separators for safe parsing
def format_ra(ra):
    return f"{ra[0:2]}h{ra[2:4]}m{ra[4:]}s"


def format_dec(dec):
    sign = dec[0]
    return f"{sign}{dec[1:3]}d{dec[3:5]}m{dec[5:]}s"


ra_fmt = df1["alpha"].apply(format_ra)
dec_fmt = df1["delta"].apply(format_dec)
# Create SkyCoord object
coords = SkyCoord(ra=ra_fmt.values, dec=dec_fmt.values, frame="icrs")
# Add degree columns
df1["RA"] = coords.ra.deg.round(5)
df1["DEC"] = coords.dec.deg.round(5)

# Ensure string
df2["d"] = df2["d"].astype(str).str.strip()
# Split once
split_cols = df2["d"].str.split("±", n=1, expand=True)
# Convert
d_value = pd.to_numeric(split_cols[0], errors="coerce")
d_error = pd.to_numeric(split_cols[1], errors="coerce")
# Replace 'd' and insert 'd_error' next to it
df2["d"] = d_value
col_idx = df2.columns.get_loc("d")
df2.insert(col_idx + 1, "d_error", d_error)

# Order both dataframes by the 'Star cluster' column
df1 = df1.sort_values("Star cluster").reset_index(drop=True)
df2 = df2.sort_values("Star cluster").reset_index(drop=True)

df_comb = pd.merge(df1, df2, on="Star cluster", how="outer", suffixes=("_t1", "_t2"))

df_comb.to_csv(
    "../temp_updt/BUKOWIECKI2011.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df1 = pd.read_csv("data/databases/ALMEIDA2023_integ.csv")
df2 = pd.read_csv("data/databases/ALMEIDA2023_detailed.csv")

# Drop these columns from df2
drop_cols = [
    "RA_ICRS",
    "DE_ICRS",
    "dist",
    "e_dist",
    "age",
    "e_age",
    "FeH",
    "e_FeH",
    "Av",
    "e_Av",
    "Nc",
    "bin_frac",
    "average_ratio",
    "str_average_ratio",
    "segr_ratio",
    "segr_ratio_std",
    "mass_seg",
    "mass_seg_pval",
]
df2 = df2.drop(columns=drop_cols)

df_comb = pd.merge(
    df1, df2, on="Cluster", how="outer", suffixes=("_integ", "_detailed")
)

# # Compare the differences in the columns 'RA_ICRS,DE_ICRS,dist,e_dist,age,e_age,FeH,e_FeH,Av,e_Av,Nc', adding the _left and _right suffixes
# for col in [
#     "RA_ICRS",
#     "DE_ICRS",
#     "dist",
#     "e_dist",
#     "age",
#     "e_age",
#     "FeH",
#     "e_FeH",
#     "Av",
#     "e_Av",
#     "Nc",
#     "bin_frac","average_ratio","str_average_ratio","segr_ratio","segr_ratio_std","mass_seg","mass_seg_pval"
# ]:
#     col_left = f"{col}_left"
#     col_right = f"{col}_right"
#     print(col, np.max(df_comb[col_left] - df_comb[col_right]))


df_comb.to_csv(
    "data/databases/ALMEIDA2023.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df1 = pd.read_csv("data/databases/DIAS2014.csv")
df2 = pd.read_csv("data/databases/DIAS2014_1.csv")

# Print which elements in 'Cluster' column in df2 are not in 'Cluster' column in df1
missing_clusters = set(df2["Cluster"]) - set(df1["Cluster"])
print("Missing clusters in DIAS2014 compared to DIAS2014_1:")
for cluster in missing_clusters:
    print(cluster)

df_comb = pd.merge(df1, df2, on="Cluster", how="outer", suffixes=("_left", "_right"))

for cluster in df2["Cluster"]:
    row = df_comb[df_comb["Cluster"] == cluster]
    ra1 = row["RAJ2000"].values[0]
    dec1 = row["DEJ2000"].values[0]
    if np.isnan(ra1):
        df_comb.loc[row.index[0], "RAJ2000"] = row["_RA"].values[0]
        df_comb.loc[row.index[0], "DEJ2000"] = row["_DE"].values[0]

df_comb.to_csv(
    "data/databases/DIAS2014_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


# VANDENBERGH2006
df = pd.read_csv("/home/gabriel/Github/UCC/updt_UCC/data/databases/LOKTIN2017_2.csv")

# In the 'Name' column, change the entries 'NXXX' where XXX is a number to 'NGC XXX'
txt = "VDB"
Nt = len(txt)
df["Name"] = df["Name"].apply(
    lambda x: f"{txt} {x[Nt:]}" if x.startswith(f"{txt}") and x[Nt].isdigit() else x
)
df.to_csv(
    "/home/gabriel/Github/UCC/updt_UCC/data/databases/LOKTIN2017_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df = pd.read_csv("data/databases/MONTEIRO2019.csv")
df1 = pd.read_csv("data/databases/MONTEIRO2019_1.csv")

df_comb = pd.merge(df, df1, on="Name", how="outer", suffixes=("_left", "_right"))

# For each name in the 'Cluster' column, add an underscore between to separate the
# letters from the numbers
new_names = []
for name in df_comb["Name"]:
    split_index = next(
        (i for i, c in enumerate(name) if c.isdigit()), len(name)
    )  # Find index of first digit
    new_name = name[:split_index] + "_" + name[split_index:]
    new_names.append(new_name)
df_comb["Name"] = new_names

df2 = pd.read_csv("data/databases/MONTEIRO2019_2.csv")
# Add df2 rows to df_comb, filling the missing columns with 'nan' values
df_comb = pd.concat([df_comb, df2], ignore_index=True, sort=False)
df3 = pd.read_csv("data/databases/MONTEIRO2019_3.csv")
df_comb = pd.concat([df_comb, df3], ignore_index=True, sort=False)

df_comb.to_csv(
    "data/databases/MONTEIRO2019_4.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df = pd.read_csv("data/databases/ZHANG2024.csv")
df1 = pd.read_csv("data/databases/ZHANG2024_1.csv")

# Combine both dfs grouping by 'Cluster' columns. If an antry is missing, fill columns with "nan" values
df_combined = pd.merge(df, df1, on="Cluster", how="outer", suffixes=("_left", "_right"))
# Drop 'RA_ICRS_right', 'DE_ICRS_right' columns
df_combined = df_combined.drop(columns=["RA_ICRS_right", "DE_ICRS_right"])
# Rename 'RA_ICRS_left' to 'RA_ICRS' and 'DE_ICRS_left' to 'DE_ICRS'
df_combined = df_combined.rename(
    columns={"RA_ICRS_left": "RA_ICRS", "DE_ICRS_left": "DE_ICRS"}
)
# Convert columns "RA_ICRS","DE_ICRS" in 'h:m:s, d:m:s' into degrees using astropy
df_combined["RA_ICRS"] = np.round(
    Angle(df_combined["RA_ICRS"].astype(str), unit=u.hourangle).degree, 5
)  # pyright: ignore
df_combined["DE_ICRS"] = np.round(
    Angle(df_combined["DE_ICRS"].astype(str), unit=u.deg).degree, 5
)  # pyright: ignore

# For each name in the 'Cluster' column, add an underscore between to separate the letters from the numbers
new_names = []
for name in df_combined["Cluster"]:
    split_index = next(
        (i for i, c in enumerate(name) if c.isdigit()), len(name)
    )  # Find index of first digit
    new_name = name[:split_index] + "_" + name[split_index:]
    new_names.append(new_name)
df_combined["Cluster"] = new_names

df_combined.to_csv(
    "data/databases/ZHANG2024_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)


breakpoint()


breakpoint()


df = pd.read_csv("data/databases/NIZOVKINA2025.csv")

# Group rows with equivalent values in the 'Cluster' column
grouped = df.groupby("Cluster")
new_rows = []
for cluster_name, group in grouped:
    print(f"Cluster: {cluster_name}")
    # print(group)
    # Combine all rows for this group so that a single row remains
    combined_row = group.iloc[0].copy()
    for col in df.columns:
        if col != "Cluster":
            # combined_row[col] = ';'.join(group[col].astype(str).unique())
            combined_row[col] = ";".join(group[col].astype(str))
    print("Combined Row:")
    print(combined_row)
    new_rows.append(combined_row)

# Turn new_rows list into a dataframe
df_combined = pd.DataFrame(new_rows)


# Keep only unique floats in column 'RA_ICRS' and 'DE_ICRS'
def unique_floats(entry):
    floats = entry.split(";")
    unique = sorted(set(floats), key=lambda x: floats.index(x))
    # return ';'.join(unique)
    return float(unique[0])


df_combined["RA_ICRS"] = df_combined["RA_ICRS"].apply(unique_floats)
df_combined["DE_ICRS"] = df_combined["DE_ICRS"].apply(unique_floats)


df_combined.to_csv(
    "data/databases/NIZOVKINA2025_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
breakpoint()


df_old = pd.read_csv("../data/UCC_cat_B.csv")
df_new = pd.read_csv("UCC_cat_B.csv")

# COmpare the values in columns 'RA_ICRS, DE_ICRS'

plt.subplot(121)
plt.scatter(df_old["RA_ICRS"], df_old["RA_ICRS"] - df_new["RA_ICRS"])
plt.subplot(122)
plt.scatter(df_old["DE_ICRS"], df_old["DE_ICRS"] - df_new["DE_ICRS"])
plt.show()


breakpoint()

df = pd.read_csv("../data/UCC_cat_B.csv")
fnames0 = [_.split(";")[0] for _ in df["fnames"]]
idx = np.argsort(fnames0)
df = df.reindex(idx)
df.to_csv(
    "../data/UCC_cat_B_2.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

breakpoint()


df_MWSC = pd.read_csv("data/databases/MWSC_notes.csv")
df_MWSC_names = [_.split(",")[1] for _ in df_MWSC["Name"]]
# new_name = []
# for i, name in enumerate(df['Name']):
#     name = name.strip()
#     if not name.startswith('MWSC_'):
#         new_name.append(name + ', MWSC_' + str(df['MWSC'][i]))
#     else:
#         new_name.append(name)
# df['Name'] = new_name
# df = df.drop(columns=["MWSC"])
# df.to_csv(
#     "data/databases/MWSC_notes.csv",
#     na_rep="nan",
#     index=False,
#     quoting=csv.QUOTE_NONNUMERIC,
# )


db_name = "JUST2023"

df = pd.read_csv(f"data/databases/{db_name}.csv")

# Replace all spaces in entries in column 'Name' with a SINGLE underscore
# df["Name"] = df["Name"].str.replace(r"\s+", "_", regex=True)

# # Remove "psFile","atlas","BDA" columns
# df = df.drop(columns=["map","cmd","stars","MWSC"])

# # Convert columns "RA_ICRS","DE_ICRS" in 'h:m:s, d:m:s' into degrees using astropy
# from astropy.coordinates import Angle
# import astropy.units as u
# df["RA_deg"] = np.round(Angle(df["RA_ICRS"].astype(str), unit=u.hourangle).degree, 5)
# df["DE_deg"] = np.round(Angle(df["DE_ICRS"].astype(str), unit=u.deg).degree, 5)


# new_name = []
# for i, name in enumerate(df['Name']):
#     if not name.startswith('MWSC_'):
#         new_name.append(name + ', MWSC_' + str(df['MWSC'][i]))
#     else:
#         new_name.append(name)
# df['Name'] = new_name
# df = df.drop(columns=["MWSC"])


new_name, type = [], []
for i, name in enumerate(df["Name"]):
    try:
        i = df_MWSC_names.index(name)
        new_name.append(df_MWSC["Name"][i])
        type.append(df_MWSC["Type"][i])
    except:
        print(name)
        new_name.append(name)
        type.append("")

df["Name"] = new_name
df["Type"] = type


# Remove rows with 'Type' column matching either of
df = df[~df["Type"].isin(["a", "g", "m", "n", "r", "s"])]

txt = "" if db_name.endswith("_2") else "_2"
new_df_name = f"data/databases/{db_name}{txt}.csv"
df.to_csv(
    new_df_name,
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
