#####################################################################################
#####################################################################################
# check_UCC_versions


# def check_rows(
#     logging, fnames: str, diff_cols: list[str], row_old: pd.Series, row_new: pd.Series
# ) -> None:
#     """
#     Compares rows from two DataFrames at specified indices and prints details
#     of differences in selected columns if differences exceed specified
#     thresholds or do not fall under certain exceptions.

#     Parameters
#     ----------
#     logging : logging.Logger
#         Logger instance for recording messages.
#     fnames : str
#         fname(s) of cluster
#     diff_cols : list
#         List of column names with differences
#     row_old : pd.Series
#         Index of the row in UCC_new to compare.
#     row_new : pd.Series
#         Index of the row in UCC_old to compare.

#     Returns
#     -------
#     None
#         Outputs differences directly to the console if they meet specified
#         criteria, otherwise returns nothing.
#     """
#     txt = ""

#     # Catch general differences *not* in these columns
#     for col in diff_cols:
#         if col not in (
#             "DB",
#             "DB_i",
#             "RA_ICRS",
#             "DE_ICRS",
#             "GLON",
#             "GLAT",
#             "dups_fnames",
#             "dups_probs",
#             "dups_fnames_m",
#             "dups_probs_m",
#         ):
#             txt += f"; {col}: {row_old[col]} | {row_new[col]}"

#     # Catch specific differences in these columns

#     # Check (ra, dec)
#     if "RA_ICRS" in diff_cols:
#         txt = coords_check(txt, "RA_ICRS", row_old, row_new)
#     if "DE_ICRS" in diff_cols:
#         txt = coords_check(txt, "DE_ICRS", row_old, row_new)

#     # Check (lon, lat)
#     if "GLON" in diff_cols:
#         txt = coords_check(txt, "GLON", row_old, row_new)
#     if "GLAT" in diff_cols:
#         txt = coords_check(txt, "GLAT", row_old, row_new)

#     # Check dups_fnames and dups_probs
#     if "dups_fnames" in diff_cols:
#         txt = dups_check(txt, "dups_fnames", row_old, row_new)
#     if "dups_probs" in diff_cols:
#         txt = dups_check(txt, "dups_probs", row_old, row_new)

#     # Check dups_fnames_m and dups_probs_m
#     if "dups_fnames_m" in diff_cols:
#         txt = dups_check(txt, "dups_fnames_m", row_old, row_new)
#     if "dups_probs_m" in diff_cols:
#         txt = dups_check(txt, "dups_probs_m", row_old, row_new)

#     if txt != "":
#         logging.info(f"{fnames:<10} --> {txt}")
#     return


# def coords_check(
#     txt: str,
#     coord_id: str,
#     row_old: pd.Series,
#     row_new: pd.Series,
#     deg_diff: float = 0.001,
# ) -> str:
#     """
#     Check differences in coordinate values and append to log message.

#     Parameters
#     ----------
#     txt : str
#         Existing log message text.
#     coord_id : str
#         Column name of the coordinate.
#     row_old : pd.Series
#         Series representing the old row.
#     row_new : pd.Series
#         Series representing the new row.
#     deg_diff : float, optional
#         Threshold for coordinate difference. Default is 0.001.

#     Returns
#     -------
#     str
#         Updated log message text.
#     """
#     if abs(row_old[coord_id] - row_new[coord_id]) > deg_diff:
#         txt += f"; {coord_id}: " + str(abs(row_old[coord_id] - row_new[coord_id]))
#     return txt


# def dups_check(txt: str, dup_id: str, row_old: pd.Series, row_new: pd.Series) -> str:
#     """
#     Check differences in duplicate-related columns and append to log message.

#     Parameters
#     ----------
#     txt : str
#         Existing log message text.
#     dup_id : str
#         Column name of the duplicate field.
#     row_old : pd.Series
#         Series representing the old row.
#     row_new : pd.Series
#         Series representing the new row.

#     Returns
#     -------
#     str
#         Updated log message text.
#     """
#     aa = str(row_old[dup_id]).split(";")
#     bb = str(row_new[dup_id]).split(";")
#     if len(list(set(aa) - set(bb))) > 0:
#         txt += f"; {dup_id}: " + str(row_old[dup_id]) + " | " + str(row_new[dup_id])
#     return txt


#####################################################################################
#####################################################################################
