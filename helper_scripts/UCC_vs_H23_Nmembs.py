import pandas as pd

GCs_cat = "../data/globulars.csv"
bckp_plots_path = "/media/kingston/new_UCC/UCC_260616/plots"
bckp_UCC_membs_path = "/media/kingston/new_UCC/UCC_260612/data/zenodo/UCC_members.parquet"

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
    """ """
    ucc_members_old = pd.read_parquet(bckp_UCC_membs_path)
    # Group by 'name' and count unique 'Source'
    ucc_member_counts_old = (
        ucc_members_old.groupby("name")["Source"].nunique().reset_index()
    )
    ucc_member_counts_old.rename(columns={"Source": "N_clust_ucc"}, inplace=True)

    ucc_members = pd.read_parquet("../data/zenodo/UCC_members.parquet")
    # Group by 'name' and count unique 'Source'
    ucc_member_counts = ucc_members.groupby("name")["Source"].nunique().reset_index()
    ucc_member_counts.rename(columns={"Source": "N_clust_ucc"}, inplace=True)

    all_names = pd.read_csv("../data/all_names.csv")
    fnames_dict = {}
    for fnames in all_names["fnames"]:
        fname_s = fnames.split(";")
        for fname in fname_s:
            fnames_dict[fname] = fname_s[0]
    # all_names_fname0_lst = [_.split(';')[0] for _ in all_names['fnames']]

    hunt23_membs = pd.read_parquet("members_process/HUNT23_members.parquet")
    h23_member_counts = hunt23_membs.groupby("Name")["GaiaDR3"].nunique().reset_index()
    h23_member_counts.rename(columns={"GaiaDR3": "N_clust_h23"}, inplace=True)

    h23_member_counts["Name"] = h23_member_counts["Name"].str.strip()
    # Update names with replacements stored in h23_name_changes
    h23_member_counts["Name"] = h23_member_counts["Name"].replace(h23_name_changes)

    unique_h23_fnames = get_fnames(h23_member_counts["Name"])
    unique_h23_fnames = [x for sublist in unique_h23_fnames for x in sublist]

    # Storage for the three separate threshold sections
    low_ucc = []  # N_clust_ucc <= 25
    mid_ucc = []  # 25 < N_clust_ucc <= 100
    high_ucc = []  # N_clust_ucc > 100
    highest_ucc = []  # N_clust_ucc > 500

    # Fast lookups
    ucc_counts = dict(zip(ucc_member_counts["name"], ucc_member_counts["N_clust_ucc"]))
    ucc_counts_old = dict(
        zip(ucc_member_counts_old["name"], ucc_member_counts_old["N_clust_ucc"])
    )
    # Map every alias -> full fnames string
    alias_to_fnames = {}
    for fnames in all_names["fnames"]:
        for alias in fnames.split(";"):
            alias_to_fnames[alias] = fnames

    skip_prefixes = ("cwnu", "hsc", "theia", "oc0")
    for i, h23_fname in enumerate(unique_h23_fnames):
        fname0 = fnames_dict.get(h23_fname)
        if fname0 is None or fname0.startswith(skip_prefixes):
            continue

        N_clust_ucc = ucc_counts[fname0]
        N_clust_h23 = h23_member_counts.iloc[i]["N_clust_h23"]

        # Get the maximum N_clust_ucc_old for all aliases of fname0
        N_clust_ucc_old = max(
            (
                ucc_counts_old.get(fname, 0)
                for fname in alias_to_fnames[fname0].split(";")
            ),
            default=0,
        )

        if N_clust_ucc <= 25:
            thresh = 0.90
            target_list = low_ucc
        elif N_clust_ucc <= 100:
            thresh = 0.75
            target_list = mid_ucc
        elif N_clust_ucc <= 500:
            thresh = 0.5
            target_list = high_ucc
        else:
            thresh = 0.25
            target_list = highest_ucc

        # Ratio against HUNT23
        ratio = (N_clust_h23 - N_clust_ucc) / N_clust_ucc

        # # Ratio against old version
        # ratio = (N_clust_ucc_old - N_clust_ucc) / N_clust_ucc

        if ratio > thresh:
            target_list.append(
                {
                    "i": i,
                    "fname0": fname0,
                    "N_clust_ucc_old": N_clust_ucc_old,
                    "N_clust_ucc": N_clust_ucc,
                    "N_clust_h23": N_clust_h23,
                    "ratio": ratio,
                }
            )

    # Define sections for structured printing
    sections = [
        ("--- N_clust_ucc <= 25 ---", low_ucc),
        ("--- 25 < N_clust_ucc <= 100 ---", mid_ucc),
        ("--- N_clust_ucc > 100 ---", high_ucc),
        ("--- N_clust_ucc > 500 ---", highest_ucc),
    ]

    for title, data_list in sections:
        print(f"\n{title}")
        # Sort entries by ratio in descending order
        sorted_data = sorted(data_list, key=lambda x: x["ratio"], reverse=True)

        for item in sorted_data:
            print(
                f"{item['ratio']:.2f}, '{item['fname0']}', "
                f"(UCC_old)={item['N_clust_ucc_old']}, (UCC)={item['N_clust_ucc']},"
                f" (HUNT23)={item['N_clust_h23']}"
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
