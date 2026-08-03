import pandas as pd

GCs_cat = "../data/globulars.csv"
HUNT23_membs_path = "members_process/HUNT23_members.parquet"
# bckp_UCC_membs_path = "/media/kingston/new_UCC/UCC_260612/data/zenodo/UCC_members.parquet"
UCC_membs_path = "../data/zenodo/UCC_members.parquet"
all_names_path = "../data/all_names.csv"

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

use_ratio = "HUNT23"
# use_ratio = "OLD UCC"

# Prefixed of low quality OCs (generally)
skip_prefixes = ("cwnu", "hsc", "theia", "oc0")

# Entries already checked manually
ucc_vs_h23_no_issues = (
    "trumpler10",
    "ngc6404",
    "collinder69",
    "hyades",
    "stock2",
    "ngc2327",
    "ngc6475",
    "ic2395",
    "ngc869",
    "pismis3",
    "ngc6494",
    "pozzo1",
    "collinder463",
    "ngc7419",
    "teutsch268",
    "ngc2548",
    "berkeley17",
    "ubc106",
    "ngc6405",
    "ngc2437",
    "ngc3496",
    "fof2117",
    "berkeley32",
    "ryu519",
    "ubc600",
    "mamajek4",
    "alessi37",
    "fsr0717",
    "ngc2175",
    "ascc65",
    "berkeley93",
    "dbsb11",
    "trumpler15",
    "fsr1419",
    "ruprecht139",
    "ascc19",
    "ngc6604",
    "eso42905",
    "ngc1333",
    "ivanov8",
    "ngc3324",
    "fsr0261",
    "berkeley83",
    "fsr0358",
    "pismis27",
    "berkeley29",
    "fsr1088",
    "fsr0416",
    "fsr0031",
    "ic2944",
    "pfleiderer4",
    "fsr1424",
    "ufmg54",
    "teutsch127",
    "ic2948",
    "mcm58",
    "vdbh67",
    "berkeley36",
    "czernik21",
    "berkeley18",
    "berkeley54",
    "berkeley56",
    "collinder197",
    "eso09205",
    "ubc634",
    "ocsn88",
    "ubc261",
    "lisciii3668",
    "upk214",
    "cwwdl14602",
    "ubc553",
    "pismis8",
    "vdbh205",
    "ubc1099",
    "bica631",
    "iras01546p6319",
    "sai90",
    "ufmg16",
    "feigelson1",
    "negueruela1",
    "sai50",
    "kronberger1",
    "fsr0777",
    "stock8",
    "haffner19",
    "haffner18",
    "cwwdl13389",
    "ubc482",
    "fsr0591",
    "ubc1264ubc517",
    "czernik44",
    "sai106",
    "monob1d",
    "ocsn178",
    "collinder338",
    "kronberger83",
    "czernik10",
    "casado9",
    "vdbh144",
    "berkeley86",
    "berkeley34",
    "riddle6",
    "trumpler14",
    "kronberger73",
    "vdbh222",
    "alessiteutsch7",
    "hxhwl26",
    "rsg4",
    "ubc26",
    "dias1",
    "fsr0219",
    "ascc125",
    "eso09218",
    "ocsn65",
    "ocsn41",
    "ngc6716",
    "collinder394",
    "dbsb19",
    "ubc1264",
    "ubc1330",
    "afgl5085",
    "upk303",
    "ngc2220",
    "ubc1306",
    "mayer3",
    "teutsch23",
    "alessij06016p3531",
    "alessi34",
    "roslund7",
    "collinder419",
    "teutsch157",
    "alessi9",
    "ascc108",
    "ubc279",
    "harvard10",
    "pismis4",
    "coingaia13",
    "ngc2183",
    "mamajek2",
    "fof2059",
    "ngc225",
    "fof282",
    "ngc7429",
    "gulliver10",
    "ascc16",
    "czernik20",
    "l1641s",
    "teutsch39",
    "eso37017",
    "czernik3",
    "ocsn3",
    "fof288",
    "fsr0238",
    "ocsn6",
    "ubc1577",
    "alessi59",
    "ubc281",
    "ubc1455",
    "upk627",
    "ubc343",
    "ubc1569",
    "ubc413",
    "lp34",
    "fsr0321",
    "vvv100",
    "hxhwl42",
    "ascc90",
    "db20017",
    "ubc1263",
    "fof2036",
    "collinder135",
    "bochum6",
    "teutsch7",
    "saurer3",
    "ocsn28",
    "fsr0498",
    "ocsn68",
    "juchert7",
    "pfleiderer3",
    "teutsch62",
    "ufmg87",
    "fsr1173",
)


def main() -> None:
    """ """
    print(f"Using ratio type: {use_ratio}")

    # ucc_members_old = pd.read_parquet(bckp_UCC_membs_path)
    # # Group by 'name' and count unique 'Source'
    # ucc_member_counts_old = (
    #     ucc_members_old.groupby("name")["Source"].nunique().reset_index()
    # )
    # ucc_member_counts_old.rename(columns={"Source": "N_clust_ucc"}, inplace=True)

    ucc_members = pd.read_parquet(UCC_membs_path)
    # Group by 'name' and count unique 'Source'
    ucc_member_counts = ucc_members.groupby("name")["Source"].nunique().reset_index()
    ucc_member_counts.rename(columns={"Source": "N_clust_ucc"}, inplace=True)

    all_names = pd.read_csv(all_names_path)
    fnames_dict = {}
    for fnames in all_names["fnames"]:
        fname_s = fnames.split(";")
        for fname in fname_s:
            fnames_dict[fname] = fname_s[0]
    # all_names_fname0_lst = [_.split(';')[0] for _ in all_names['fnames']]

    hunt23_membs = pd.read_parquet(HUNT23_membs_path)
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
    # ucc_counts_old = dict(
    #     zip(ucc_member_counts_old["name"], ucc_member_counts_old["N_clust_ucc"])
    # )
    # Map every alias -> full fnames string
    alias_to_fnames = {}
    for fnames in all_names["fnames"]:
        for alias in fnames.split(";"):
            alias_to_fnames[alias] = fnames

    for i, h23_fname in enumerate(unique_h23_fnames):
        fname0 = fnames_dict.get(h23_fname)
        if fname0 is None or fname0.startswith(skip_prefixes):
            continue

        if fname0 in ucc_vs_h23_no_issues:
            continue

        N_clust_ucc = ucc_counts[fname0]

        # N_clust_h23 = h23_member_counts.iloc[i]["N_clust_h23"]

        # Get the maximum N_clust_h23 for all aliases of fname0
        N_clust_h23 = 0
        for fname in alias_to_fnames[fname0].split(";"):
            try:
                j = unique_h23_fnames.index(fname)
                N_clust_h23 = max(N_clust_h23, h23_member_counts.iloc[j]["N_clust_h23"])
            except:
                continue

        # # Get the maximum N_clust_ucc_old for all aliases of fname0
        # N_clust_ucc_old = max(
        #     (
        #         ucc_counts_old.get(fname, 0)
        #         for fname in alias_to_fnames[fname0].split(";")
        #     ),
        #     default=0,
        # )

        if N_clust_ucc <= 25:
            thresh = 4
            target_list = low_ucc
        elif N_clust_ucc <= 100:
            thresh = 2
            target_list = mid_ucc
        elif N_clust_ucc <= 500:
            thresh = 1
            target_list = high_ucc
        else:
            thresh = 0.5
            target_list = highest_ucc

        if use_ratio == "HUNT23":
            # Ratio against HUNT23
            ratio = abs(N_clust_h23 - N_clust_ucc) / min(N_clust_h23, N_clust_ucc)
        # elif use_ratio == "OLD UCC":
        #     # Ratio against old version
        #     ratio = abs(N_clust_ucc_old - N_clust_ucc) / N_clust_ucc
        else:
            raise ValueError(f"Unknown ratio type: {use_ratio}")

        if ratio > thresh:
            sign = 1
            if N_clust_h23 < N_clust_ucc:
                sign = -1
            target_list.append(
                {
                    "i": i,
                    "fname0": fname0,
                    # "N_clust_ucc_old": N_clust_ucc_old,
                    "N_clust_ucc": N_clust_ucc,
                    "N_clust_h23": N_clust_h23,
                    "ratio": sign * ratio,
                }
            )

    # Define sections for structured printing
    sections = [
        ("--- N_clust_ucc > 500 ---", highest_ucc),
        ("--- N_clust_ucc > 100 ---", high_ucc),
        ("--- 25 < N_clust_ucc <= 100 ---", mid_ucc),
        ("--- N_clust_ucc <= 25 ---", low_ucc),
    ]

    for title, data_list in sections:
        print(f"\n{title}")
        # Sort entries by ratio in descending order
        sorted_data = sorted(data_list, key=lambda x: x["ratio"], reverse=True)

        for item in sorted_data:
            print(
                f"{item['ratio']:.2f}, '{item['fname0']}', "
                # f"(UCC_old)={item['N_clust_ucc_old']}, (UCC)={item['N_clust_ucc']},"
                f"(UCC)={item['N_clust_ucc']},"
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
