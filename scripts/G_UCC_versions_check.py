import csv
import os

import pandas as pd
from HARDCODED import UCC_folder
from modules import UCC_new_match, logger, read_ini_file

logging = logger.main()


def main():
    """ """
    pars_dict = read_ini_file.main()
    old_UCC_name = pars_dict["old_UCC_name"]

    # Read latest version of the UCC
    UCC_new, _ = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    # Check number of files
    file_checker(UCC_new)

    fnames_checker(UCC_new)

    # Read old version of the UCC
    UCC_old = pd.read_csv(UCC_folder + old_UCC_name)
    old_date = old_UCC_name.split("_")[-1].split(".")[0]
    logging.info(f"\nUCC version {old_date} loaded (N={len(UCC_old)}) <-- OLD")
    print("")

    diff_between_dfs(UCC_old, UCC_new)
    logging.info("File 'UCC_diff.csv' saved\n")

    fnames_old_all = list(UCC_old["fnames"])
    fnames_new_all = list(UCC_new["fnames"])

    # Check new entries
    for i_new, fnames_new in enumerate(fnames_new_all):
        try:
            i_old = fnames_old_all.index(fnames_new)
        except ValueError:
            print(f"OC not found in old UCC: {fnames_new}")
            check_new_entries(fnames_old_all, i_new, fnames_new)

    # Check existing entries
    logging.info("\nold name   | new name   --> column name: differences")
    for i_new, fnames_new in enumerate(fnames_new_all):
        i_old = None
        try:
            i_old = fnames_old_all.index(fnames_new)
        except ValueError:
            pass
        if i_old is None:
            continue

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

            name_old = str(UCC_old["fnames"][i_old])
            name_new = str(UCC_new["fnames"][i_new])
            check_rows(name_old, name_new, diff_cols, row_old, row_new)


def file_checker(df_UCC):
    """ """
    logging.info("\nChecking number of files")
    logging.info("    parquet webp  aladin")

    flag_error = False
    NT_parquet, NT_webp, NT_webp_aladin = 0, 0, 0
    for qnum in range(1, 5):
        for lat in ("P", "N"):
            N_parquet, N_webp, N_webp_aladin = 0, 0, 0
            for ffolder in ("datafiles", "plots"):
                qfold = "../../Q" + str(qnum) + lat + f"/{ffolder}/"
                # Read all files in Q folder
                for file in os.listdir(qfold):
                    if "HUNT23" in file or "CANTAT20" in file:
                        continue
                    elif "aladin" in file:
                        N_webp_aladin += 1
                        NT_webp_aladin += 1
                    elif "parquet" in file:
                        N_parquet += 1
                        NT_parquet += 1
                    elif "webp" in file:
                        N_webp += 1
                        NT_webp += 1

            mark = "V" if (N_parquet == N_webp == N_webp_aladin) else "X"
            logging.info(
                f"{str(qnum) + lat}:   {N_parquet}  {N_webp}  {N_webp_aladin} <-- {mark}"
            )
            if mark == "X":
                flag_error = True

    logging.info(f"Total UCC: {len(df_UCC)}")
    logging.info(
        f"Total parquet/webp/aladin: {NT_parquet}, {NT_webp}, {NT_webp_aladin}"
    )
    if not (NT_parquet == NT_webp == NT_webp_aladin):
        flag_error = True

    if flag_error:
        raise ValueError("The file check was unsuccessful")


def fnames_checker(df_UCC):
    """ """
    fname0_UCC = [_.split(";")[0] for _ in df_UCC["fnames"]]
    NT = len(fname0_UCC)
    N_unique = len(list(set(fname0_UCC)))
    if NT != N_unique:
        raise ValueError("Initial fnames are not unique")


def diff_between_dfs(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Compare two DataFrames, find non-matching rows while preserving order, and
    output these rows interwoven from both DataFrames, with a blank line after each pair.

    Args:
        df1 (pd.DataFrame): First DataFrame to compare.
        df2 (pd.DataFrame): Second DataFrame to compare.

    The function identifies rows unique to each DataFrame, writes them in an interwoven
    pattern to the output file, and adds a blank line after each pair of rows.
    """
    # Convert DataFrames to lists of tuples (rows) for comparison
    rows1 = [[str(_) for _ in row] for row in df1.values]
    rows2 = [[str(_) for _ in row] for row in df2.values]

    # Convert lists to sets for quick comparison
    set1, set2 = set(map(tuple, rows1)), set(map(tuple, rows2))

    # Get non-matching rows in original order
    non_matching1 = [row for row in rows1 if tuple(row) not in set2]
    non_matching2 = [row for row in rows2 if tuple(row) not in set1]

    # Intertwine the rows from both non-matching lists with spacing
    intertwined_lines = []
    max_len = max(len(non_matching1), len(non_matching2))
    for i in range(max_len):
        if i < len(non_matching1):
            intertwined_lines.append(non_matching1[i])
        if i < len(non_matching2):
            intertwined_lines.append(non_matching2[i])

        # Insert a blank row after each pair
        intertwined_lines.append([])

    # Write intertwined lines to the output file
    with open("../UCC_diff.csv", "w", newline="") as out:
        writer = csv.writer(out)
        for row in intertwined_lines:
            writer.writerow(row)


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
            txt += f"; {col}: {row_old[col]} | {row_new[col]}"

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
