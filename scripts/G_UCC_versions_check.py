import pandas as pd
from HARDCODED import UCC_folder
from modules import UCC_new_match, logger, read_ini_file


def main():
    """ """
    logging = logger.main()
    pars_dict = read_ini_file.main()
    old_UCC_name = pars_dict["old_UCC_name"]

    # Read latest version of the UCC
    UCC_new, _ = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    # Read old version of the UCC
    UCC_old = pd.read_csv(UCC_folder + old_UCC_name)
    old_date = old_UCC_name.split("_")[-1].split(".")[0]
    logging.info(f"UCC version {old_date} loaded (N={len(UCC_old)}) <-- OLD")
    print("")

    fnames_old_all = list(UCC_old["fnames"])
    fnames_new_all = list(UCC_new["fnames"])

    # Check new entries
    for i_new, fnames_new in enumerate(fnames_new_all):
        i_old = None
        try:
            pass
        except ValueError:
            print(f"OC not found in old UCC: {fnames_new}")
            check_new_entries(fnames_old_all, i_new, fnames_new)

    # Check existing entries
    logging.info("old name   | new name   --> column name: differences")
    for i_new, fnames_new in enumerate(fnames_new_all):
        i_old = None
        try:
            i_old = fnames_old_all.index(fnames_new)
        except ValueError:
            pass

        # If 'fnames_new' was found in the old UCC, compare both entries
        # Extract rows
        row_new = UCC_new.iloc[i_new]
        row_old = UCC_old.iloc[i_old]
        # Compare rows using pandas method
        row_compared = row_new.compare(row_old)

        # If rows are not equal
        if row_compared.empty is False:
            # Extract column names with differences
            row_dict = row_compared.to_dict()
            diff_cols = list(row_dict["self"].keys())

            # If the only diffs are in the ID columns, skip check
            if (
                diff_cols == ["DB", "DB_i"]
                or diff_cols == ["DB"]
                or diff_cols == ["DB_i"]
            ):
                continue

            name_old = UCC_old["fnames"][i_old]
            name_new = UCC_new["fnames"][i_new]
            check_rows(name_old, name_new, diff_cols, row_old, row_new)


def check_new_entries(fnames_old_all, i_new, fnames_new):
    """ """
    idxs_old_match = []
    for i_old, fnames_old in enumerate(fnames_old_all):
        for fname_new in fnames_new.split(";"):
            for fname_old in fnames_old.split(";"):
                if fname_new == fname_old:
                    idxs_old_match.append(i_old)
    idxs_old_match = list(set(idxs_old_match))
    if len(idxs_old_match) > 1:
        print(f"ERROR: duplicate fname, new:{i_new}, old:{idxs_old_match}")


def check_rows(name_old: str, name_new: str, diff_cols: list, row_old, row_new) -> None:
    """
    Compares rows from two DataFrames at specified indices and prints details
    of differences in selected columns if differences exceed specified
    thresholds or do not fall under certain exceptions.

    Parameters:
    - name_old: str
        Name(s) of cluster in the old version
    - name_new: str
        Name(s) of cluster in the new version
    - diff_cols : list
        List of column names with differences
    - row_old: pandas.core.series.Series
        Index of the row in UCC_new to compare.
    - row_new: pandas.core.series.Series
        Index of the row in UCC_old to compare.

    Returns:
    - None: Outputs differences directly to the console if they meet specified
      criteria, otherwise returns nothing.
    """
    txt = ""

    # Catch general differences *not* in these columns
    for col in diff_cols:
        if col not in (
            "DB",
            "DB_i",
            "RA_ICRS",
            "DE_ICRS",
            "GLON",
            "GLAT",
            "dups_fnames",
            "dups_probs",
            "dups_fnames_m",
            "dups_probs_m",
        ):
            txt += f"; {col}: {diff_cols}"

    # Catch specific differences in these columns

    # Check (ra, dec)
    if "RA_ICRS" in diff_cols:
        txt = coords_check(txt, "RA_ICRS", row_old, row_new)
    if "DE_ICRS" in diff_cols:
        txt = coords_check(txt, "DE_ICRS", row_old, row_new)

    # Check (lon, lat)
    if "GLON" in diff_cols:
        txt = coords_check(txt, "GLON", row_old, row_new)
    if "GLAT" in diff_cols:
        txt = coords_check(txt, "GLAT", row_old, row_new)

    # Check dups_fnames and dups_probs
    if "dups_fnames" in diff_cols:
        txt = dups_check(txt, "dups_fnames", row_old, row_new)
    if "dups_probs" in diff_cols:
        txt = dups_check(txt, "dups_probs", row_old, row_new)

    # Check dups_fnames_m and dups_probs_m
    if "dups_fnames_m" in diff_cols:
        txt = dups_check(txt, "dups_fnames_m", row_old, row_new)
    if "dups_probs_m" in diff_cols:
        txt = dups_check(txt, "dups_probs_m", row_old, row_new)

    if txt != "":
        print(f"{name_old:<10} | {name_new:<10} --> ", txt)
    return


def coords_check(txt, coord_id, row_old, row_new, deg_diff=0.001):
    """ """
    if abs(row_old[coord_id] - row_new[coord_id]) > deg_diff:
        txt += f"; {coord_id}: " + str(abs(row_old[coord_id] - row_new[coord_id]))
    return txt


def dups_check(txt, dup_id, row_old, row_new):
    """ """
    aa = str(row_old[dup_id]).split(";")
    bb = str(row_new[dup_id]).split(";")
    if len(list(set(aa) - set(bb))) > 0:
        txt += f"; {dup_id}: " + str(row_old[dup_id]) + " | " + str(row_new[dup_id])
    return txt


if __name__ == "__main__":
    main()
