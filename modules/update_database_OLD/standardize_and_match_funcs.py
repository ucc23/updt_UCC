import pandas as pd

from ..utils import rename_standard


def get_fnames_new_DB(
    logging, df_new: pd.DataFrame, newDB_json: dict, sep: str = ","
) -> list[list[str]]:
    """
    Extract and standardize all names in the new catalogue

    Parameters
    ----------
    df_new : pd.DataFrame
        DataFrame of the new catalogue.
    newDB_json : dict
        Dictionary with the parameters of the new catalogue.
    sep : str, optional
        Separator used to split the names in the new catalogue. Default is ",".

    Returns
    -------
    list
        List of lists, where each inner list contains the standardized names for
        each cluster in the new catalogue.
    """
    names_all = df_new[newDB_json["names"]]
    new_DB_fnames = []
    for names in names_all:
        names_l = []
        names_s = str(names).split(sep)
        for name in names_s:
            name = name.strip()
            name = rename_standard(name)
            names_l.append(rm_chars_from_name(name))
        new_DB_fnames.append(names_l)

    # Check that no fname is repeated within entries
    seen, duplicates = {}, {}
    for i, sublist in enumerate(new_DB_fnames):
        # use set(sublist) so duplicates inside the same sublist are ignored
        for item in set(sublist):
            if item in seen and seen[item] != i:
                duplicates.setdefault(item, {seen[item]}).add(i)
            else:
                seen[item] = i
    if duplicates:
        for name, idxs in duplicates.items():
            logging.info(f"Duplicate '{name}' found in entries: {sorted(idxs)}")
        raise ValueError("\nDuplicate fnames found in new DB entries")

    return new_DB_fnames


def rm_chars_from_name(name: str) -> str:
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


def get_matches_new_DB(
    df_UCC: pd.DataFrame, new_DB_fnames: list[list[str]]
) -> list[int | None]:
    """
    Get cluster matches for the new DB being added to the UCC

    Parameters
    ----------
    df_UCC : pd.DataFrame
        DataFrame of the UCC.
    new_DB_fnames : list
        List of lists, where each inner list contains the
        standardized names for each cluster in the new catalogue.

    Returns
    -------
    list
        List of indexes into the UCC pointing to each entry in the new DB.
        If an entry in the new DB is not present in the UCC, the corresponding
        index in the list will be None.
    """
    db_matches = []
    # For each fname(s) for each entry in the new DB
    for new_cl_fnames in new_DB_fnames:
        found = None
        # For each fname in this new DB entry
        for new_cl_fname in new_cl_fnames:
            # For each fname(s) for each entry in the UCC
            for j, ucc_cl_fnames in enumerate(df_UCC["fnames"]):
                # Check if the fname in the new DB is included in the UCC
                if new_cl_fname in ucc_cl_fnames.split(";"):
                    # If it is, return the UCC index
                    found = j
                    break
            if found is not None:
                break
        db_matches.append(found)

    return db_matches
