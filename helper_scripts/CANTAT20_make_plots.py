import os
import os.path
import shutil  # noqa
import sys

import pandas as pd

sys.path.append("../")

GCs_cat = "../data/globulars.csv"
C20_membs_path = "members_process/CANTAT20_members.parquet"
# bckp_plots_path = "/media/kingston/new_UCC/UCC_260616/plots"

#
c20_name_changes = {
    "LP_282": "FoF_282",
}


def main(style_path="../modules/D_funcs/science2.mplstyle") -> None:
    """ """
    all_names = pd.read_csv("../data/all_names.csv")
    fnames_dict = {}
    for fnames in all_names["fnames"]:
        fname_s = fnames.split(";")
        for fname in fname_s:
            fnames_dict[fname] = fname_s[0]

    print("Reading CANTAT20 members...\n")
    cantat20_membs = pd.read_parquet(C20_membs_path)
    cantat20_membs["Cluster"] = cantat20_membs["Cluster"].str.strip()
    # Update names with replacements stored in c20_name_changes
    cantat20_membs["Cluster"] = cantat20_membs["Cluster"].replace(c20_name_changes)

    # msk1 = cantat20_membs['Cluster']=="Gulliver_6"
    # msk2 = cantat20_membs['Cluster']=="UBC_17b"
    # # Check how many elements are shared between these two groups in the 'GaiaDR2' column
    # shared_elements = set(cantat20_membs.loc[msk1, "GaiaDR2"]).intersection(set(cantat20_membs.loc[msk2, "GaiaDR2"]))
    # breakpoint()

    # Group by "Cluster" column
    unique_c20_names = list(set(cantat20_membs["Cluster"]))

    breakpoint()

    unique_c20_fnames = get_fnames(unique_c20_names)
    unique_c20_fnames = [x for sublist in unique_c20_fnames for x in sublist]

    # Load GCs data
    df_GCs = pd.read_csv(GCs_cat)
    # Extract GCs fnames
    names_gcs = list(df_GCs["Name"] + ", " + df_GCs["OName"])
    gcs_fnames_lst = []
    for gcs_fname in get_fnames(names_gcs):
        if gcs_fname[1] != "":
            gcs_fnames_lst.append(gcs_fname)
        else:
            gcs_fnames_lst.append([gcs_fname[0]])
    gcs_fnames_lst = list(set([x for sublist in gcs_fnames_lst for x in sublist]))

    for i, c20_fname in enumerate(unique_c20_fnames):
        if c20_fname in fnames_dict:
            fname0 = fnames_dict[c20_fname]
            plot_fpath = f"../../plots/plots_{fname0[0]}/CANTAT20/{fname0}.webp"
            # Check if file already exists
            if not os.path.isfile(plot_fpath):
                # Check if the file exists in the backup plots path, under the old name
                bckp_fname_path = (
                    f"{bckp_plots_path}/plots_{c20_fname[0]}/CANTAT20/{c20_fname}.webp"
                )
                if not os.path.isfile(bckp_fname_path):
                    if fname0 != c20_fname:
                        bckp_fname_path_2 = f"{bckp_plots_path}/plots_{fname0[0]}/CANTAT20/{fname0}.webp"
                        # print(fname0, c20_fname)
                        # breakpoint()
                        if not os.path.isfile(bckp_fname_path_2):
                            print(
                                f"Both webp missing from backup: {fname0}, {c20_fname}"
                            )
                        else:
                            print(
                                f"File {fname0} is present in backup but {c20_fname} is not"
                            )
                    else:
                        # if unique_c20_names[i] in hunt_23_nonmembers_names:
                        #     continue
                        print(f"Missing: {c20_fname}.webp, must generate")

                        # msk = hunt23_membs["Name"] == unique_h23_names[i]
                        # df_members = hunt23_membs[msk]
                        # # Create plot file
                        # # plot_fpath = f"../../plots/plots_{fname0[0]}/HUNT23/{fname0}.webp"
                        # plot_fpath = f"{fname0}.webp"
                        # title = r"Hunt & Reffert (2023)"
                        # ucc_plots.plot_CMD(
                        #     plot_fpath, df_members, probs_col="Prob", title=title, style_path=style_path
                        # )
                        # breakpoint()
                else:
                    print(f"Copy {bckp_fname_path} --> {plot_fpath}")
                    # Copy the file from the backup plots path to the new location
                    # shutil.copy2(bckp_fname_path, plot_fpath)
        else:
            if c20_fname not in gcs_fnames_lst:
                # Should never happen
                print(f"ERROR missing name: {unique_c20_names[i]}, {c20_fname}")


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
