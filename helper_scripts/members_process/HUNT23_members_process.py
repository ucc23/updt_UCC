import os
import os.path
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("../../")
from modules.D_funcs import ucc_plots


def main() -> None:
    """
    Main function to process and validate HUNT23 data and generate plots and parquet
    files.
    """
    df_UCC, hunt23_membs, hunt23_not_ocs = load_data()

    h23_names, h23_fnames = generate_fnames(hunt23_membs, hunt23_not_ocs)

    # # Elements from HUNT23 not in UCC
    # set2 = {name.strip() for item in df_UCC["fnames"] for name in item.split(";")}
    # h23_not_in_ucc = [x for x in h23_fnames if x not in set2]

    # Find missing HUNT23 plots
    alias_to_main = {}
    for item in df_UCC["fnames"]:
        names = item.split(";")
        main = names[0]
        for n in names:
            alias_to_main[n] = main
    base = "../../../plots"
    for i, x in enumerate(h23_fnames):
        main = alias_to_main.get(x)
        if main:
            path = f"{base}/plots_{main[0]}/HUNT23/{main}.webp"
            if not os.path.isfile(path):
                print(f"Generating: /plots_{main[0]}/HUNT23/{main}.webp")

                full_name = h23_names[i]
                msk = hunt23_membs["Name"] == full_name
                df_members = hunt23_membs[msk].copy()

                # Create plot file
                # plot_fpath = Path(f"{base}/plots_{main[0]}/HUNT23/{main}.webp")
                plot_fpath = Path(f"/home/gabriel/Descargas/{main}.webp")
                title = r"Hunt & Reffert (2023)"
                ucc_plots.plot_CMD(
                    plot_fpath, df_members, probs_col="Prob", title=title
                )


def load_data() -> tuple:
    """
    Load the UCC database, HUNT23 members
    """
    # Read UCC
    df_UCC = pd.read_csv("../../data/UCC_cat_B.csv")

    print("Reading HUNT23 members...\n")
    hunt23_membs = pd.read_parquet("HUNT23_members.parquet")
    hunt23_membs["Name"] = hunt23_membs["Name"].str.strip()

    hunt23_not_ocs = pd.read_csv("HUNT23_non_ocs.csv")
    hunt23_not_ocs["Name"] = hunt23_not_ocs["Name"].str.strip()

    return df_UCC, hunt23_membs, hunt23_not_ocs


def generate_fnames(hunt23_membs: pd.DataFrame, hunt23_not_ocs):
    """
    Generate standardized filenames for HUNT23 members and types.
    """
    name_changes = [
        ("ESO_429-429", "ESO_429-02"),
        ("CMa_2", "CMa_02"),
        ("AH03_J0748+26.9", "AH03_J0748-26.9"),
        ("Juchert_J0644.8+0925", "Juchert_J0644.8-0925"),
        ("Teutsch_J0718.0+1642", "Teutsch_J0718.0-1642"),
        ("Teutsch_J0924.3+5313", "Teutsch_J0924.3-5313"),
        ("Teutsch_J1037.3+6034", "Teutsch_J1037.3-6034"),
        ("Teutsch_J1209.3+6120", "Teutsch_J1209.3-6120"),
    ]
    for names in name_changes:
        msk = hunt23_membs["Name"] == names[0]
        hunt23_membs.loc[msk, "Name"] = names[1]

        # msk = df_H23_types["Name"] == names[0]
        # df_H23_types.loc[msk, "Name"] = names[1]

    # Extract names maintaining the order
    h23_names = np.array(list(dict.fromkeys(hunt23_membs["Name"]).keys()))

    # Remove not OCs
    msk = []
    for _ in h23_names:
        if _ not in hunt23_not_ocs["Name"].values:
            msk.append(True)
        else:
            msk.append(False)
    msk = np.array(msk)

    h23_fnames = [
        _.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("+", "p")
        .replace(".", "")
        for _ in h23_names
    ]

    # Removed GCs from the original HUNT23 database
    GCs_removed = (
        "palomar2",
        "ic1276",
        "palomar8",
        "palomar10",
        "palomar11",
        "palomar12",
        "1636283",
        "pismis26",
        "lynga7",
        "hsc2890",
        "hsc134",
        "teutsch182",  # This is not a GC but it was removed
    )
    for i, _ in enumerate(h23_fnames):
        if _ in GCs_removed:
            msk[i] = False

    return h23_names[msk], np.array(h23_fnames)[msk]


if __name__ == "__main__":
    main()
