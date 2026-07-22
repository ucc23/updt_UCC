import sys

import pandas as pd

sys.path.append("../")
from modules.D_funcs import ucc_plots

style_path = "../modules/D_funcs/science2.mplstyle"
title = r"Hunt & Reffert (2023)"
GCs_cat = "../data/globulars.csv"

#
h23_name_changes = {
    "ESO_429-429": "ESO_429-02",
    "AH03_J0748+26.9": "AH03_J0748-26.9",
    "Juchert_J0644.8+0925": "Juchert_J0644.8-0925",
    "Teutsch_J0718.0+1642": "Teutsch_J0718.0-1642",
    "Teutsch_J0924.3+5313": "Teutsch_J0924.3-5313",
    "Teutsch_J1037.3+6034": "Teutsch_J1037.3-6034",
    "Teutsch_J1209.3+6120": "Teutsch_J1209.3-6120",
    "XDOCC_9": "XDOCC_09",
    "XDOCC_6": "XDOCC_06",
    "HSC_134": "Gran 3",
    "HSC_2890": "Gran 4",
    "CMa_2": "CMa_02",
    "BH_90": "VDBH_90",
    "vdBergh_92": "VDB_92",
}


def main() -> None:
    """
    Generate plots for a group of OCs from Hunt & Reffert (2023) whose GLON
    wrapped around the 0/360 boundary
    """
    all_names = pd.read_csv("../data/all_names.csv")
    fnames_dict = {}
    for fnames in all_names["fnames"]:
        fname_s = fnames.split(";")
        for fname in fname_s:
            fnames_dict[fname] = fname_s[0]

    print("Reading HUNT23 members...\n")
    hunt23_membs = pd.read_parquet("members_process/HUNT23_members.parquet")
    hunt23_membs["Name"] = hunt23_membs["Name"].str.strip()
    # Update names with replacements stored in h23_name_changes
    hunt23_membs["Name"] = hunt23_membs["Name"].replace(h23_name_changes)

    # Get canonical names (fnames)
    unique_h23_names = list(set(hunt23_membs["Name"]))
    unique_h23_fnames = get_fnames(unique_h23_names)
    unique_h23_fnames = [x for sublist in unique_h23_fnames for x in sublist]

    cl_process = (
        "Theia_70",
        "NGC_6405",
        "OC_0704",
        "NGC_6723",
        "NGC_6475",
        "HSC_95",
        "HSC_2846",
        "FoF_1624",
        "Theia_67",
        "HSC_2973",
        "HSC_1",
        "HSC_52",
        "HSC_3",
        "HSC_759",
        "Collinder_347",
        "NGC_288",
        "HSC_2976",
        "OCSN_99",
        "Palomar_5",
        "Blanco_1",
        "HSC_2971",
        "Ferrero_1",
        "HSC_2986",
        "Melotte_111",
    )

    for i, clname in enumerate(unique_h23_names):
        if clname not in cl_process:
            continue

        msk = hunt23_membs["Name"] == clname
        df_members = hunt23_membs[msk].copy()

        # Fix GLON coordinates that wrap around the 0/360 boundary
        # if np.ptp(df_members["GLON"]) > 180:
        #     print(clname)

        print(f"Making CMD plot for {clname}...")
        glon = df_members["GLON"].to_numpy(copy=True, dtype=float)
        glon[glon > 180.0] -= 360
        df_members["GLON"] = glon

        fname0 = unique_h23_fnames[i]
        plot_fpath = f"{fname0}.webp"
        ucc_plots.plot_CMD(
            plot_fpath, df_members, probs_col="Prob", title=title, style_path=style_path
        )


def get_fnames(names_all, sep: str = ",") -> list[list[str]]:
    """ """
    fnames = []
    for names in names_all:
        names_l = []
        names_s = str(names).split(sep)
        for name in names_s:
            name = name.strip()
            name = rename_standard(name)
            names_l.append(normalize_name(name))
        fnames.append(names_l)
    return fnames


def rename_standard(all_names: str, sep_in: str = ",", sep_out: str = ";") -> str:
    """
    Standardize the naming of these clusters

    FSR XXX w leading zeros
    FSR XXX w/o leading zeros
    FSR_XXX w leading zeros
    FSR_XXX w/o leading zeros

    --> FSR_XXXX (w leading zeroes)

    ESO XXX-YY w leading zeros
    ESO XXX-YY w/o leading zeros
    ESO_XXX_YY w leading zeros
    ESO_XXX_YY w/o leading zeros
    ESO_XXX-YY w leading zeros
    ESO_XXX-YY w/o leading zeros
    ESOXXX_YY w leading zeros (LOKTIN17)

    --> ESO_XXX_YY (w leading zeroes)

    Parameters
    ----------
    name : str
        Name of the cluster.

    Returns
    -------
    str
        Standardized name of the cluster.
    """
    # For each comma separated name for this OC in the new DB
    oc_names = all_names.split(sep_in)

    new_names_rename = []
    for name in oc_names:
        name = name.strip()

        if name.startswith("FSR"):
            if " " in name or "_" in name:
                if "_" in name:
                    n2 = name.split("_")[1]
                else:
                    n2 = name.split(" ")[1]
                n2 = int(n2)
                if n2 < 10:
                    n2 = "000" + str(n2)
                elif n2 < 100:
                    n2 = "00" + str(n2)
                elif n2 < 1000:
                    n2 = "0" + str(n2)
                else:
                    n2 = str(n2)
                name = "FSR_" + n2

        if name.startswith("ESO"):
            if name[:4] not in ("ESO_", "ESO "):
                # E.g.: LOKTIN17, BOSSINI19
                name = "ESO_" + name[3:]

            if " " in name[4:]:
                n1, n2 = name[4:].split(" ")
            elif "_" in name[4:]:
                n1, n2 = name[4:].split("_")
            elif "-" in name[4:]:
                n1, n2 = name[4:].split("-")
            else:
                # This assumes that all ESo clusters are names as: 'ESO XXX YY'
                n1, n2 = name[4 : 4 + 3], name[4 + 3 :]

            n1 = int(n1)
            if n1 < 10:
                n1 = "00" + str(n1)
            elif n1 < 100:
                n1 = "0" + str(n1)
            else:
                n1 = str(n1)
            n2 = int(n2)
            if n2 < 10:
                n2 = "0" + str(n2)
            else:
                n2 = str(n2)
            name = "ESO_" + n1 + "_" + n2

        if "UBC" in name and "UBC " not in name and "UBC_" not in name:
            name = name.replace("UBC", "UBC ")
        if "UBC_" in name:
            name = name.replace("UBC_", "UBC ")

        if "UFMG" in name and "UFMG " not in name and "UFMG_" not in name:
            name = name.replace("UFMG", "UFMG ")

        if (
            "LISC" in name
            and "LISC " not in name
            and "LISC_" not in name
            and "LISC-" not in name
        ):
            name = name.replace("LISC", "LISC ")

        if "OC-" in name:
            name = name.replace("OC-", "OC ")

        # Use spaces not underscores
        name = name.replace("_", " ")

        new_names_rename.append(name)
    oc_names = sep_out.join(new_names_rename)

    return oc_names


def normalize_name(name: str) -> str:
    """
    Removes special characters from a cluster name and converts it to lowercase.

    Parameters
    ----------
    name : str
        The cluster name.

    Returns
    -------
    str
        The cluster name with special characters removed and converted to lowercase.
    """
    # We replace '+' with 'p' to avoid duplicating names for clusters
    # like 'Juchert J0644.8-0925' and 'Juchert_J0644.8+0925'
    name = (
        name.lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("'", "")
        .replace("+", "p")
    )
    return name


if __name__ == "__main__":
    main()
