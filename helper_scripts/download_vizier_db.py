import csv

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier


def main():
    """
    Main function to download and process a database using NASA/ADS and Vizier data.
    """

    temp_CSV_file = "BICA2019_orig.csv"
    viz_t_url = (
        "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=2019AJ....157...12B"
    )

    # temp_CSV_file = "FROEBRICH2007.csv"
    # viz_t_url = "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=2007MNRAS.374..399F"

    table_url = get_CDS_table(viz_t_url)

    df_all = get_DB_from_Vizier(table_url)

    save_DB_CSV(temp_CSV_file, df_all)


def get_CDS_table(viz_t_url: str) -> list:
    """Obtain the available tables from 'ADS_bibcode'"""

    ADS_bibcode = viz_t_url.split("VizieR?-source=")[-1]

    # Use row_limit=1 to avoid downloading the entire tables
    viz = Vizier(row_limit=1)
    cat = viz.get_catalogs(ADS_bibcode)  # pyright: ignore
    if len(cat) == 0:
        raise ValueError(f"Could not extract data from {ADS_bibcode}")

    # Print info to screen
    print(
        f"\nFull url: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source={ADS_bibcode}\n"
    )
    rows = cat.__str__().split("\n")
    print(rows[0].strip())
    for i, row in enumerate(rows[1:]):
        print(
            f"{i}: " + cat[i].meta["description"] + " : " + row.split("and")[0].strip()
        )

    # Select table(s) to download
    while True:
        tab_idx = input("\nInput index(es) of table(s) to store (-1 for none): ")
        if tab_idx.strip() == "-1":
            table_url = []
            break
        tab_idx = tab_idx.strip().split(" ")
        try:
            tab_idx = np.array([int(_) for _ in tab_idx])
            if (tab_idx >= 0).all() and (tab_idx < len(cat)).all():
                table_url = [cat.keys()[_] for _ in tab_idx]
                break
            else:
                print("Invalid input")
        except ValueError:
            print("Invalid input")

    return table_url


def get_DB_from_Vizier(table_url: list) -> list | None:
    """
    Retrieve a database from Vizier and convert it to a pandas DataFrame.

    Parameters:
    table_url (list): The Vizier catalog's table identifier.

    Returns:
    list of pd.DataFrame: List of DataFrame containing the Vizier database.
    """
    df_all = None
    if len(table_url) > 0:
        # No limit to number of rows or columns
        viz = Vizier(row_limit=-1, columns=["all"])
        df_all = []
        for turl in table_url:
            # Download table
            try:
                cat = viz.get_catalogs(turl)  # pyright: ignore
                # Convert to pandas before storing
                df_all.append(cat.values()[0].to_pandas())
            except Exception as e:
                print(f"Could not extract the data from {turl}\n{e!s}")

    return df_all


def save_DB_CSV(temp_CSV_file, df_all):
    """
    Replace possible empty entries in columns
    """
    for idx, df in enumerate(df_all):
        name_CSV_file = str(temp_CSV_file)
        if idx > 0:
            name_CSV_file = temp_CSV_file.replace(".csv", f"_{idx}.csv")

        # Save and reload to bypass issue:
        # https://github.com/astropy/astropy/issues/17601
        df.to_csv(
            name_CSV_file,
            index=False,
        )
        df = pd.read_csv(name_CSV_file)

        # Remove leading and trailing spaces
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Replace empty strings or whitespace-only strings with NaN
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # Remove columns that are not useful
        for col in ("recno", "Simbad", "SimbadName"):
            if col in df.columns:
                df = df.drop(columns=[col])

        df.to_csv(
            name_CSV_file,
            na_rep="nan",
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )


if __name__ == "__main__":
    main()
