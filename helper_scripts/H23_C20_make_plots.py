import argparse
import sys

import pandas as pd

sys.path.append("../")
from modules.D_funcs import ucc_plots

style_path = "../modules/D_funcs/science2.mplstyle"

# Per-source configuration
SOURCES = {
    "H23": {
        "title": r"Hunt & Reffert (2023)",
        "membs_path": "members_process/HUNT23_members.parquet",
        "name_col": "Name",
        "probs_col": "Prob",
        "rename_cols": {},
        "name_changes": {
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
        },
    },
    "C20": {
        "title": r"Cantat-Gaudin et al. (2020)",
        "membs_path": "members_process/CANTAT20_members.parquet",
        "name_col": "Cluster",
        "probs_col": "proba",
        "rename_cols": {"pmRA*": "pmRA"},
        "name_changes": {
            "LP_1624": "FoF_1624",
        },
    },
}


# Default entries to process
cl_process = ["oc0704"]
source = "H23"  # C20
cfg = SOURCES[source]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CMD plots for OCs from a selected members source."
    )
    parser.add_argument(
        "source", choices=SOURCES.keys(), help="Members source to use for plotting"
    )
    parser.add_argument(
        "-c",
        "--clusters",
        nargs="+",
        default=None,
        help="Cluster names to process (overrides the source's default list)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Generate plots for a group of OCs from the selected source. If GLON
    wraps around the 0/360 boundary, fix it.
    """
    print(f"Reading {source} members...\n")
    membs = pd.read_parquet(cfg["membs_path"])
    if cfg["rename_cols"]:
        membs.rename(columns=cfg["rename_cols"], inplace=True)
    membs[cfg["name_col"]] = membs[cfg["name_col"]].str.strip()
    # Update names with replacements stored in name_changes
    membs[cfg["name_col"]] = membs[cfg["name_col"]].replace(cfg["name_changes"])

    # Canonical names (fnames) for the clusters to process
    cl_process_f = {x for subl in get_fnames(cl_process) for x in subl}

    # Canonical names (fnames) for each unique cluster name in the source
    unique_names = list(set(membs[cfg["name_col"]]))
    unique_fnames = get_fnames(unique_names)

    for clname, fnames in zip(unique_names, unique_fnames):
        for fname in fnames:
            if fname not in cl_process_f:
                continue

            msk = membs[cfg["name_col"]] == clname
            df_members = membs[msk].copy()

            print(f"Making CMD plot for {clname}...")

            # Fix GLON coordinates that wrap around the 0/360 boundary
            glon_wrap_fix(df_members)

            plot_fpath = f"{fname}.webp"
            ucc_plots.plot_CMD(
                plot_fpath,
                df_members,
                probs_col=cfg["probs_col"],
                title=cfg["title"],
                style_path=style_path,
            )


def glon_wrap_fix(df_members):
    """
    Fix GLON coordinates that wrap around the 0/360 boundary
    """
    glon = df_members["GLON"].to_numpy(copy=True, dtype=float)

    span = glon.max() - glon.min()

    glon_wrapped = glon.copy()
    glon_wrapped[glon_wrapped > 180.0] -= 360.0
    span_wrapped = glon_wrapped.max() - glon_wrapped.min()

    # Apply the fix only if it produces a smaller span
    if span_wrapped < span:
        glon = glon_wrapped

    df_members["GLON"] = glon

    return df_members


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
