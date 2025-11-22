import csv
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord

# df = pd.read_csv("../databases/JAEHNIG2021.csv")
df = pd.read_csv("JAEHNIG2021_3.csv")

# Trim spaces from all columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Remove 'SimbadName' column
df = df.drop(columns=['SimbadName'])


# newl = []
# for _  in df_C['bad_oc']:
#     if str(_) == 'nan':
#         newl.append('n')
#     else:
#         newl.append(_)
# df_C['bad_oc'] = newl

df.to_csv(
    "JAEHNIG2021_3.csv",
    na_rep="nan",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
breakpoint()



def main(N_digits: int = 5,):
    df_C = pd.read_csv("UCC_cat_C.csv")

    # df_C["Plx_m"] = np.round(df_C["Plx_m"], 4)
    # df_C.to_csv(
    #     "UCC_cat_C_1.csv",
    #     na_rep="nan",
    #     index=False,
    #     quoting=csv.QUOTE_NONNUMERIC,
    # )
    # breakpoint()

    # df_membs_all = pd.read_parquet("zenodo/UCC_members.parquet")
    # plx_all = []
    # for fname in df_C['fname']:
    #     print(fname)
    #     df_membs = df_membs_all[df_membs_all['name'] == fname]

    #     # ra, dec = np.nanmedian(df_membs["RA_ICRS"]), np.nanmedian(df_membs["DE_ICRS"])
    #     plx = np.nanmedian(df_membs["Plx"])

    #     # ra, dec = round(ra, N_digits), round(dec, N_digits)
    #     plx_all.append(round(plx, N_digits))

    plx_all = pd.read_csv("plx_all.csv")['plx_all']
    X_GC, Y_GC, Z_GC, R_GC = gc_values(df_C["RA_ICRS_m"].values, df_C["DE_ICRS_m"].values, plx_all)

    df_C["Plx_m"] = np.round(plx_all, 4)
    df_C["X_GC"] = np.round(X_GC, 4)
    df_C["Y_GC"] = np.round(Y_GC, 4)
    df_C["Z_GC"] = np.round(Z_GC, 4)
    df_C["R_GC"] = np.round(R_GC, 4)

    df_C.to_csv(
        "UCC_cat_C_2.csv",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def plx_to_pc(plx, PZPO=-0.02, min_plx=0.035, max_plx=200):
    """
    Ding et al. (2025), Fig 8 shows several PZPO values
    https://ui.adsabs.harvard.edu/abs/2025AJ....169..211D/abstract

    We use -0.02 as a reasonable value here.
    """
    plx = np.array(plx) * 1.

    # "the zero-point returned by the code should be subtracted from the parallax value"
    # https://gitlab.com/icc-ub/public/gaiadr3_zeropoint
    plx -= PZPO

    # Clip to reasonable values
    plx = np.clip(plx, min_plx, max_plx)
    # Convert to pc
    d_pc = 1000 / plx

    return d_pc


def gc_values(ra, dec, plx, max_xyz=20):
    """
    PZPO:

    Fig 8 shows several values for the PZPO:
    https://ui.adsabs.harvard.edu/abs/2025AJ....169..211D/abstract
    Global parallax zero point offset (selected by me): -0.02
    """

    d_pc = plx_to_pc(plx)

    coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=d_pc * u.pc,
        frame="icrs",
    )

    gc = Galactocentric()  # galcen_distance=R_sun)
    XYZ = coords.transform_to(gc)
    X_GC = np.clip(XYZ.x.to(u.kpc).value, -max_xyz, max_xyz)
    Y_GC = np.clip(XYZ.y.to(u.kpc).value, -max_xyz, max_xyz)
    Z_GC = np.clip(XYZ.z.to(u.kpc).value, -max_xyz, max_xyz)
    R_GC = np.sqrt(X_GC**2 + Y_GC**2 + Z_GC**2)

    return X_GC, Y_GC, Z_GC, R_GC


if __name__ == "__main__":
    main()
