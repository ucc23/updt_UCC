import pandas as pd


def run(
    logging,
    df_UCC: pd.DataFrame,
    df_new: pd.DataFrame,
    json_pars: dict,
    pars_dict: dict,
    show_entries: bool = False,
) -> tuple[list[str], list[str | None], int]:
    """
    Standardizes names in a new database and matches them against an existing UCC
    database.

    This function processes entries in a new database, standardizes their names,
    and attempts to match them with entries in an existing UCC database. It also
    provides logging information about the number of matches found and optionally
    displays new OCs that weren't matched.

    Args:
        logging: logger object
        df_UCC (pd.DataFrame): DataFrame containing the existing UCC database entries
        df_new (pd.DataFrame): DataFrame containing the new database entries
        json_pars (dict): Parameters for database combination operations
        pars_dict (dict): Dictionary containing parameter mappings including "ID" field
        show_entries (bool, optional): If True, prints unmatched OCs. Defaults to False

    Returns:
        Tuple[List[str], List[Optional[str]], int]: A tuple containing:
            - List of standardized filenames from the new database
            - List of matching entries where None indicates no match found
            - Integer with the number of new OCs in the DB

    Logs:
        - Info about standardization process
        - Number of matches found
        - Number of new OCs found
    """
    logging.info(f"\nStandardize names in {pars_dict['new_DB']}")
    new_DB_fnames = get_fnames_new_DB(df_new, json_pars)
    db_matches = get_matches_new_DB(df_UCC, new_DB_fnames)
    N_matches = sum(match is not None for match in db_matches)
    logging.info(f"Found {N_matches} matches in {pars_dict['new_DB']}")
    N_new = len(df_new) - N_matches
    logging.info(f"Found {N_new} new OCs in {pars_dict['new_DB']}")

    if show_entries:
        for i, oc_new_db in enumerate(df_new[pars_dict["ID"]].values):
            if db_matches[i] is None:
                print(f"  {i}: {oc_new_db.strip()}")

    return new_DB_fnames, db_matches, N_new


def get_fnames_new_DB(df_new, json_pars, sep=",") -> list:
    """
    Extract and standardize all names in the new catalogue
    """
    names_all = df_new[json_pars["names"]]
    new_DB_fnames = []
    for i, names in enumerate(names_all):
        names_l = []
        names_s = names.split(sep)
        for name in names_s:
            name = name.strip()
            name = rename_standard(name)
            names_l.append(rm_chars_from_name(name))
        new_DB_fnames.append(names_l)

    return new_DB_fnames


def rename_standard(name):
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
    """
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

    # Removes duplicates such as "NGC_2516" and "NGC 2516"
    name = name.replace("_", " ")

    return name


def rm_chars_from_name(name):
    """ """
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


def get_matches_new_DB(df_UCC, new_DB_fnames):
    """
    Get cluster matches for the new DB being added to the combined DB
    """

    def match_fname(new_cl):
        for name_new in new_cl:
            for j, old_cl in enumerate(df_UCC["fnames"]):
                for name_old in old_cl.split(";"):
                    if name_new == name_old:
                        return j
        return None

    db_matches = []
    for i, new_cl in enumerate(new_DB_fnames):
        # Check if this new fname is already in the old DBs list of fnames
        db_matches.append(match_fname(new_cl))

    return db_matches
