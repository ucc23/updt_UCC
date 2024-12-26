import os

import pandas as pd


def run(logging, UCC_old: pd.DataFrame, UCC_new: pd.DataFrame) -> None:
    """Run checks on old and new UCC files to ensure consistency and identify possible
    issues.

    Parameters:
    - logging: Logger instance for recording messages.
    - UCC_old: DataFrame containing the old UCC data.
    - UCC_new: DataFrame containing the new UCC data.

    Returns:
    - None
    """
    # Check number of files
    file_checker(logging, UCC_old, UCC_new)

    fnames_checker(UCC_new)

    fnames_old_all = list(UCC_old["fnames"])
    fnames_new_all = list(UCC_new["fnames"])

    # Check new entries
    for i_new, fnames_new in enumerate(fnames_new_all):
        try:
            i_old = fnames_old_all.index(fnames_new)
        except ValueError:
            logging.info(f"OC not found in old UCC: {fnames_new}")
            check_new_entries(logging, fnames_old_all, i_new, fnames_new)

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
            check_rows(logging, name_old, name_new, diff_cols, row_old, row_new)


def file_checker(logging, UCC_old: pd.DataFrame, UCC_new: pd.DataFrame) -> None:
    """Check the number and types of files in directories for consistency.

    Parameters:
    - logging: Logger instance for recording messages.
    - UCC_old: DataFrame containing the old UCC data.
    - UCC_new: DataFrame containing the new UCC data.

    Returns:
    - None
    """
    logging.info("\nChecking number of files")
    logging.info("    parquet webp  aladin  extra")

    flag_error = False
    NT_parquet, NT_webp, NT_webp_aladin, NT_extra = 0, 0, 0, 0
    for qnum in range(1, 5):
        for lat in ("P", "N"):
            N_parquet, N_webp, N_webp_aladin, N_extra = 0, 0, 0, 0
            for ffolder in ("datafiles", "plots"):
                qfold = "../" + "Q" + str(qnum) + lat + f"/{ffolder}/"
                # Read all files in Q folder
                for file in os.listdir(qfold):
                    if "HUNT23" in file or "CANTAT20" in file:
                        pass
                    elif "aladin" in file:
                        N_webp_aladin += 1
                        NT_webp_aladin += 1
                    elif "parquet" in file:
                        N_parquet += 1
                        NT_parquet += 1
                    elif "webp" in file:
                        N_webp += 1
                        NT_webp += 1
                    else:
                        N_extra += 1
                        NT_extra += 1

            mark = "V" if (N_parquet == N_webp == N_webp_aladin) else "X"
            if N_extra > 0:
                mark = "X"
            logging.info(
                f"{str(qnum) + lat}:   {N_parquet}  {N_webp}  {N_webp_aladin}    {N_extra} <-- {mark}"
            )
            if mark == "X":
                flag_error = True
    logging.info(
        f"Total parquet/webp/aladin/extra: {NT_parquet}, {NT_webp}, {NT_webp_aladin}, {NT_extra}"
    )
    if not (NT_parquet == NT_webp == NT_webp_aladin) or NT_extra > 0:
        flag_error = True
    if flag_error:
        raise ValueError("The file check was unsuccessful")

    logging.info(f"Total old UCC: {len(UCC_new)}")
    logging.info(f"Total new UCC: {len(UCC_old)}\n")


def fnames_checker(df_UCC: pd.DataFrame) -> None:
    """Ensure that filenames in the DataFrame are unique.

    Parameters:
    - df_UCC: DataFrame containing UCC data.

    Returns:
    - None
    """
    fname0_UCC = [_.split(";")[0] for _ in df_UCC["fnames"]]
    NT = len(fname0_UCC)
    N_unique = len(list(set(fname0_UCC)))
    if NT != N_unique:
        raise ValueError("Initial fnames are not unique")


def check_new_entries(
    logging, fnames_old_all: list, i_new: int, fnames_new: str
) -> None:
    """Check for new entries in the UCC that are not present in the old UCC.

    Parameters:
    - logging: Logger instance for recording messages.
    - fnames_old_all: List of filenames in the old UCC.
    - i_new: Index of the new filename being checked.
    - fnames_new: New filename to check.

    Returns:
    - None
    """
    idxs_old_match = []
    for i_old, fnames_old in enumerate(fnames_old_all):
        for fname_new in fnames_new.split(";"):
            for fname_old in fnames_old.split(";"):
                if fname_new == fname_old:
                    idxs_old_match.append(i_old)
    idxs_old_match = list(set(idxs_old_match))
    if len(idxs_old_match) > 1:
        logging.info(f"ERROR: duplicate fname, new:{i_new}, old:{idxs_old_match}")


def check_rows(
    logging, name_old: str, name_new: str, diff_cols: list, row_old, row_new
) -> None:
    """
    Compares rows from two DataFrames at specified indices and prints details
    of differences in selected columns if differences exceed specified
    thresholds or do not fall under certain exceptions.

    Parameters:
    - logging:
        Logger instance for recording messages.
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
        logging.info(f"{name_old:<10} | {name_new:<10} --> ", txt)
    return


def coords_check(
    txt: str,
    coord_id: str,
    row_old: pd.Series,
    row_new: pd.Series,
    deg_diff: float = 0.001,
) -> str:
    """Check differences in coordinate values and append to log message.

    Parameters:
    - txt: Existing log message text.
    - coord_id: Column name of the coordinate.
    - row_old: Series representing the old row.
    - row_new: Series representing the new row.
    - deg_diff: Threshold for coordinate difference.

    Returns:
    - Updated log message text.
    """
    if abs(row_old[coord_id] - row_new[coord_id]) > deg_diff:
        txt += f"; {coord_id}: " + str(abs(row_old[coord_id] - row_new[coord_id]))
    return txt


def dups_check(txt: str, dup_id: str, row_old: pd.Series, row_new: pd.Series) -> str:
    """Check differences in duplicate-related columns and append to log message.

    Parameters:
    - txt: Existing log message text.
    - dup_id: Column name of the duplicate field.
    - row_old: Series representing the old row.
    - row_new: Series representing the new row.

    Returns:
    - Updated log message text.
    """
    aa = str(row_old[dup_id]).split(";")
    bb = str(row_new[dup_id]).split(";")
    if len(list(set(aa) - set(bb))) > 0:
        txt += f"; {dup_id}: " + str(row_old[dup_id]) + " | " + str(row_new[dup_id])
    return txt
