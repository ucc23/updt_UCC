import pandas as pd


def main():
    """ """

    # df_old = pd.read_csv("/home/gabriel/Github/UCC/updt_UCC/data/databases/BICA2019.csv")
    df_full = pd.read_csv(
        "/home/gabriel/Github/UCC/updt_UCC/temp_updt/todo/BICA2019_1.csv"
    )
    all_names = pd.read_csv("/home/gabriel/Github/UCC/updt_UCC/data/all_names.csv")

    all_names_dict = {}
    for fnames in all_names["fnames"]:
        fnames_split = fnames.split(";")
        fname0 = fnames_split[0]
        for fname in fnames_split:
            all_names_dict[fname] = fname0

    # Remove from df_full all entries whose 'Class1' column is not EC, ECC, OC, OCC
    # df_full = df_full[df_full["Class1"].isin(["EC", "ECC", "OC", "OCC"])]
    # df_full = df_full[df_full["Class1"].isin(["OC", "OCC"])]
    df_full = df_full[df_full["Class1"].isin(["EC", "ECC"])]
    print(f"Total entries in df_full after filtering: {len(df_full)}")

    fnames_full = get_fnames(df_full["Name"])

    # Check if any of the names in df_full are not in all_names_dict
    N_not_in_UCC = 0
    for i, fnames in enumerate(fnames_full):
        in_ucc = False
        for fname in fnames:
            if fname in all_names_dict:
                in_ucc = True
        if in_ucc is False:
            print(
                f"{fnames} {df_full['Class1'].iloc[i]} {df_full['Class2'].iloc[i]} not in UCC"
            )
            N_not_in_UCC += 1
    print(f"Total entries not in UCC: {N_not_in_UCC}")
    breakpoint()


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
