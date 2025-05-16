import csv

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord


def main():
    # Load the CSV file
    df = pd.read_csv("BORISSOVA2011.csv")

    df = func(df)

    # Update the CSV file
    df.to_csv(
        "updated_DB.csv",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def add_RADE_cols_add_str_name_col(df):
    """Used in BORISSOVA 2011"""

    # Read columns named 'RAJ2000' and 'DEJ2000'
    ra_hms = df["RAJ2000"]
    de_hms = df["DEJ2000"]

    # Convert the coordinate values from HMS to degrees using astropy
    ra_deg = []
    de_deg = []
    for ra, de in zip(ra_hms, de_hms):
        coord = SkyCoord(ra, de, unit=(u.hourangle, u.deg))
        ra_deg.append(round(coord.ra.deg, 5))  # Round to 5 decimal places
        de_deg.append(round(coord.dec.deg, 5))  # Round to 5 decimal places

    # Add the new 'RA' and 'DEC' columns before 'RAJ2000' and 'DEJ2000'
    idx = df.columns.get_loc("RAJ2000")
    df.insert(idx, "RA", ra_deg)  # Insert 'RA' before 'RAJ2000'
    df.insert(idx + 1, "DEC", de_deg)

    # Add 'VVV ' to each row in the 'VVV-CL' column
    vvv_num = df["VVV-CL"].astype(str)
    # Add leading zeroes
    zeroes = pd.Series(["0" * (3 - len(_)) for _ in vvv_num])
    df["VVV-CL"] = "VVV " + vvv_num + ", VVV-CL " + zeroes + vvv_num

    return df


def merge_name_cols(df):
    """Used in GLUSHKOVA2010"""

    # Merge columns, skipping NaN values
    df["Name"] = df.apply(
        lambda row: ",".join(filter(pd.notna, [row["Name"], row["OtherName"]])), axis=1
    )
    df = df.drop(columns=["OtherName"])

    return df


def merge_files():
    """Used in BORISSOVA2018"""
    # Load the CSV files into DataFrames
    df1 = pd.read_csv("BORISSOVA2018.csv")
    df2 = pd.read_csv("BORISSOVA2018_1.csv")

    # Merge the DataFrames
    combined_df = pd.merge(df1, df2, on="Name", how="left", suffixes=("", "_drop"))

    # Drop columns that were duplicated and marked with '_drop'
    df = combined_df[[col for col in combined_df.columns if not col.endswith("_drop")]]

    return df


if __name__ == "__main__":
    main()
