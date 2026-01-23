import numpy as np

from ..variables import (
    header_default,
)


def replace_text_between(
    original_text: str,
    replacement_text: str,
    delimiter_a: str,
    delimiter_b: str | None = None,
) -> str:
    """
    Replaces the text between two delimiters in the original text with the replacement
    text.

    Args:
        original_text (str): The original string containing the text to be replaced.
        replacement_text (str): The string to replace the text between the delimiters.
        delimiter_a (str): The starting delimiter marking the beginning of the text
        to be replaced.
        delimiter_b (str, optional): The ending delimiter marking the end of the text
        to be replaced. If None, only the text after delimiter_a will be replaced.

    Returns:
        str: The modified string with the text between the delimiters replaced by
        replacement_text.
    """
    leading_text = original_text.split(delimiter_a)[0]
    if delimiter_b is not None:
        trailing_text = original_text.split(delimiter_b)[1]
        return (
            leading_text + delimiter_a + replacement_text + delimiter_b + trailing_text
        )
    return leading_text + delimiter_a + replacement_text


def count_dups_bad_OCs(dbs_used, df_ucc):
    """"""
    all_DBs = list(dbs_used.keys())

    DBs_dups_badOCs = {}
    for DB_id in all_DBs:
        N_DB, N_DB_dup, N_DB_bad = 0, 0, 0
        for DB_i, Pdup_i, badOC_i in df_ucc[["DB", "P_dup", "bad_oc"]].itertuples(
            index=False, name=None
        ):
            if DB_id in DB_i.split(";"):
                N_DB += 1
                if float(Pdup_i) > 0.5:
                    N_DB_dup += 1
                if badOC_i == "y":
                    N_DB_bad += 1

        p_dup = 100 * N_DB_dup / N_DB
        p_bad = 100 * N_DB_bad / N_DB
        DBs_dups_badOCs[DB_id] = (N_DB, p_dup, p_bad)

    return DBs_dups_badOCs


def updt_articles_table(df_UCC, current_JSON, database_md_in, max_chars_title=50):
    """Update the table with the catalogues used in the UCC"""
    # Count DB occurrences in UCC
    N_in_DB = {_: 0 for _ in current_JSON.keys()}
    for _ in df_UCC["DB"].values:
        for DB in _.split(";"):
            N_in_DB[DB] += 1

    # Invert json by the 'year' key so that larger values are on top
    inv_json = dict(
        sorted(current_JSON.items(), key=lambda item: item[1]["year"], reverse=True)
    )

    # md_table = "\n| Name | N | Name | N |\n"
    md_table = "\n| Title | Author(s) | Year | Vizier | N | CSV |\n"
    md_table += "| ---- | :---: | :--: | :----: | :-: | :-: |\n"
    for DB, DB_data in inv_json.items():
        row = ""
        title = DB_data["title"].replace("'", "").replace('"', "")
        short_title = title[:max_chars_title] + "..."
        ref_url = f"""<a href="{DB_data["SCIX_url"]}" target="_blank" title="{title}">{short_title}</a>"""
        viz_url = f"""<a href="{DB_data["vizier_url"]}" target="_blank"> <img src="/images/vizier.png " alt="Vizier url"></a>"""
        if DB_data["vizier_url"] == "N/A":
            viz_url = "N/A"
        CSV_url = f"""<a href="https://flatgithub.com/ucc23/updt_UCC?filename=data/databases/{DB}.csv" target="_blank">📊</a>"""
        row += f"| {ref_url} | {DB_data['authors']} | {DB_data['year']} | {viz_url} | [{N_in_DB[DB]}](/tables/dbs/{DB}_table) | {CSV_url}"
        md_table += row + "|\n"
    md_table += "\n"

    delimeterA = "<!-- Begin table 1 -->\n"
    delimeterB = "<!-- End table 1 -->\n"
    database_md_updt = replace_text_between(
        database_md_in, md_table, delimeterA, delimeterB
    )

    return database_md_updt


def updt_DBs_tables(dbs_used, df_updt, DBs_dups_badOCs) -> dict:
    """Update the DBs classification table files"""
    header = header_default.format("", "/tables/dbs/DB_link_table/")

    # Count DB occurrences in UCC
    all_DBs = list(dbs_used.keys())

    new_tables_dict = {}
    for DB_id in all_DBs:
        md_table = header.replace("DB_link", DB_id)
        ref_url = f"[{dbs_used[DB_id]['authors']} ({dbs_used[DB_id]['year']})]({dbs_used[DB_id]['SCIX_url']})"
        md_table += "&nbsp;\n" + f"# {ref_url}" + "\n\n"

        #
        N_tot, p_dup, p_bad = DBs_dups_badOCs[DB_id]
        parts = []
        if p_dup > 1:
            parts.append(
                f"{p_dup:.0f}% are probable duplicates ([P<sub>dup</sub>>50%](/faq/#how-is-the-duplicate-probability-estimated))"
            )
        if p_bad > 1:
            parts.append(
                f"{p_bad:.0f}% are classified as [likely non-clusters](/faq/#how-are-objects-flagged-as-likely-not-real)"
            )
        if parts:
            txt = " and ".join(parts)
            md_table += (
                f"This database consists of {N_tot} entries, of which " + txt + ".\n\n"
            )

        msk = []
        for _ in df_updt["DB"].values:
            if DB_id in _.split(";"):
                msk.append(True)
            else:
                msk.append(False)
        msk = np.array(msk)

        new_table = generate_table(df_updt[msk], md_table)
        new_tables_dict[DB_id] = new_table

    return new_tables_dict


def generate_table(df_m, md_table):
    """ """
    md_table += "| Name | RA | DEC | Plx | N50 | r50 | C3 | P<sub>dup</sub> | UTI |\n"
    md_table += "| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\n"

    for i, row in df_m.iterrows():
        for col in (
            "ID_url",
            # "GLON",
            # "GLAT",
            "RA_ICRS",
            "DE_ICRS",
            "Plx_m_round",
            "N_50",
            "r_50",
            "C3_abcd",
            "P_dup",
            "UTI",
        ):
            md_table += "| " + str(row[col]) + " "
        md_table += " |\n"

    # Add script to sort tables
    md_table += (
        "\n\n\n"
        + """<script type="module">
import { enableTableSorting } from '{{ site.baseurl }}/scripts/table-sorting.js';
document.querySelectorAll("table").forEach(table => {
  enableTableSorting(table);
});
</script>"""
    )

    return md_table


def count_OCs_classes(C3, class_order):
    """Count the number of OCs per C3 class"""
    C3_classif, C3_count = np.unique(C3, return_counts=True)
    C3_classif = list(C3_classif)
    OCs_per_class = []
    for c in class_order:
        i = C3_classif.index(c)
        OCs_per_class.append(C3_count[i])
    return OCs_per_class


def ucc_n_total_updt(logging, N_db_UCC, N_cl_UCC, N_members_UCC, database_md):
    """Update the total number of entries and databases in the UCC"""
    delimiter_a = "<!-- ND1 -->"
    delimiter_b = "<!-- ND2 -->"
    replacement_text = str(N_db_UCC)
    database_md_updt = replace_text_between(
        database_md, replacement_text, delimiter_a, delimiter_b
    )

    delimiter_a = "<!-- NT1 -->"
    delimiter_b = "<!-- NT2 -->"
    replacement_text = str(N_cl_UCC)
    database_md_updt = replace_text_between(
        database_md_updt, replacement_text, delimiter_a, delimiter_b
    )

    delimiter_a = "<!-- NM1 -->"
    delimiter_b = "<!-- NM2 -->"
    replacement_text = str(N_members_UCC)
    database_md_updt = replace_text_between(
        database_md_updt, replacement_text, delimiter_a, delimiter_b
    )

    if database_md_updt != database_md:
        logging.info("\nNumber of DBS, OCs and members in the UCC updated")

    return database_md_updt


# def count_shared_membs(df_UCC: pd.DataFrame) -> list:
#     """
#     Categorizes OCs with shared members into five groups based on the number of
#     OCs that share its members. Each group is represented by a mask array.

#     Args:
#         df_UCC (pandas.DataFrame): A DataFrame containing a column named
#         "dups_fnames_m", where each entry is a string of file names separated by ";"
#         or NaN.

#     Returns:
#         list: A list of five numpy boolean arrays. Each array corresponds to a mask
#               identifying rows with a specific range:
#               - Index 0: Rows with exactly 1 OC with shared members
#               - Index 1: Rows with exactly 2 OC with shared members
#               - Index 2: Rows with exactly 3 OC with shared members
#               - Index 3: Rows with exactly 4 OC with shared members
#               - Index 4: Rows with 5 or more OC with shared members
#     """
#     shared_msk = [np.full(len(df_UCC), False) for _ in range(5)]
#     for i, shared_fnames in enumerate(df_UCC["shared_members"]):
#         if str(shared_fnames) == "nan":
#             continue
#         N_dup = len(shared_fnames.split(";"))
#         if N_dup == 1:
#             shared_msk[0][i] = True
#         elif N_dup == 2:
#             shared_msk[1][i] = True
#         elif N_dup == 3:
#             shared_msk[2][i] = True
#         elif N_dup == 4:
#             shared_msk[3][i] = True
#         elif N_dup >= 5:
#             shared_msk[4][i] = True

#     return shared_msk


# def UTI_ranges(df_UCC: pd.DataFrame) -> list:
#     """ """
#     UTI_msk = [list(df_UCC["UTI"] == 0.0)]
#     N_limi = 0.0
#     for N_limf in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
#         UTI_msk.append(list((df_UCC["UTI"] > N_limi) & (df_UCC["UTI"] <= N_limf)))
#         N_limi = N_limf
#     UTI_msk.append(list(df_UCC["UTI"] > N_limi))

#     return UTI_msk


# def P_dup_ranges(df_UCC: pd.DataFrame) -> list:
#     """ """
#     Pdup_msk = [list(df_UCC["P_dup"] == 0.0)]
#     N_limi = 0.0
#     for N_limf in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
#         Pdup_msk.append(list((df_UCC["P_dup"] > N_limi) & (df_UCC["P_dup"] <= N_limf)))
#         N_limi = N_limf
#     Pdup_msk.append(list(df_UCC["P_dup"] > N_limi))

#     return Pdup_msk


# def count_N50membs(df_UCC: pd.DataFrame) -> list:
#     """ """

#     membs_msk = [df_UCC["N_50"] == 0]

#     N_limi = 0
#     for N_limf in (25, 50, 75, 100, 250, 500, 1000, 2000):
#         N50_r = (df_UCC["N_50"] > N_limi) & (df_UCC["N_50"] <= N_limf)
#         membs_msk.append(N50_r)
#         N_limi = N_limf

#     Ninf = df_UCC["N_50"] > 2000
#     membs_msk.append(Ninf)

#     return membs_msk


# def count_fund_pars(df_UCC: pd.DataFrame, current_JSON: dict) -> dict:
#     """ """
#     # Identify which entries in the 'current_JSON' dictionary contain a non empty 'pars' key
#     DB_fp = {}
#     for DB_id, DB_data in current_JSON.items():
#         if len(DB_data["pars"]) == 0:
#             DB_fp[DB_id] = 0
#         else:
#             DB_fp[DB_id] = 1

#     Cfp_arr = []
#     for cl_dbs in df_UCC["DB"]:
#         N_fp = 0
#         for db in cl_dbs.split(";"):
#             N_fp += DB_fp[db]
#         Cfp_arr.append(N_fp)
#     Cfp_arr = np.array(Cfp_arr)

#     Cfp_msk = {"0": Cfp_arr == 0, "1": Cfp_arr == 1}
#     for r in ((2, 5), (6, 10), (11, 15)):
#         Cfp_msk[str(r[0]) + "_" + str(r[1])] = (Cfp_arr >= r[0]) & (Cfp_arr <= r[1])
#     Cfp_msk["15"] = Cfp_arr > 15

#     return Cfp_msk


# def pc_radius(
#     angular_radius_arcmin: np.ndarray, parallax_mas: np.ndarray
# ) -> np.ndarray:
#     """
#     NOT USED YET (24/12/04)

#     Calculate the radius of an object in parsecs given its angular radius (arcmin)
#     and parallax (mas).

#     Parameters:
#     angular_radius_arcmin (np.ndarray): Angular radius in arcminutes.
#     parallax_mas (np.ndarray): Parallax in milliarcseconds.

#     Returns:
#     np.ndarray: Radius in parsecs.
#     """
#     msk = parallax_mas <= 0
#     angular_radius_arcmin[msk] = np.nan
#     parallax_mas[msk] = np.nan

#     # Convert parallax from mas to arcsec and calculate distance
#     distance_pc = 1 / (parallax_mas / 1000)
#     # Convert arcmin to radians
#     angular_radius_rad = (angular_radius_arcmin / 60) * (np.pi / 180)
#     radius_pc = distance_pc * angular_radius_rad  # Radius in parsecs

#     return radius_pc


# def count_N_members_UCC(members_folder):
#     """ """
#     # Initialize total row count
#     N_members_UCC = 0
#     # Process all Q folders
#     for qN in range(1, 5):
#         for lat in ("P", "N"):
#             qfold = f"Q{qN}{lat}/"
#             qpath = f"../{qfold}{members_folder}"

#             # Pre-filter files directly from the directory listing
#             files = (
#                 file
#                 for file in os.listdir(qpath)
#                 if "HUNT23" not in file and "CANTAT20" not in file
#             )

#             for file in files:
#                 # Read Parquet metadata without loading full data
#                 pf = fastparquet.ParquetFile(os.path.join(qpath, file))
#                 N_members_UCC += pf.count()

#     # # Extract the total number of members from the "README.txt" stored in the
#     # # folder 'temp_updt/zenodo/' by the previous script. Run here to fail early if
#     # # something is wrong
#     # temp_zenodo_README = temp_fold + UCC_folder + "README.txt"
#     # with open(temp_zenodo_README, "r") as f:
#     #     dataf = f.read()
#     #     match = re.search(r"combined (\d+) members", dataf)
#     #     if match is None:
#     #         raise ValueError(
#     #             "Could not find the total number of members in the Zenodo README.txt file."
#     #         )
#     #     N_members_UCC = int(match.group(1))

#     return N_members_UCC


# def updt_UTI_main_table(UTI_msk, database_md_in: str):
#     """ """
#     UTI_table = "\n| UTI |  N  | UTI |  N  |\n"
#     UTI_table += "| :--: | :-: | :--: | :-: |\n"
#     UTI_table += f"| == 0.0 | [{sum(UTI_msk[0])}](/tables/UTI0_table) |"
#     for i, msk in enumerate(UTI_msk[1:-1]):
#         fchar = "|\n" if i in (0, 2, 4, 6, 8) else "| "
#         N = f"[{sum(msk)}](/tables/UTI{i + 1}_table)"
#         UTI_table += f" ({0.0 + (i * 0.1):.1f}, {0.1 + (i * 0.1):.1f}] | {N} {fchar}"
#     UTI_table += f"| 0.9 < | [{sum(UTI_msk[-1])}](/tables/UTI10_table) | -- | -- |\n"
#     UTI_table += "\n"

#     delimeterA = "<!-- Begin table 1 -->\n"
#     delimeterB = "<!-- End table 1 -->\n"
#     database_md_updt = replace_text_between(
#         database_md_in, UTI_table, delimeterA, delimeterB
#     )

#     return database_md_updt


# def updt_C3_classif_main_table(class_order, OCs_per_class, database_md_in: str):
#     """ """
#     C3_table = "\n| C3 |  N  | C3 |  N  | C3 |  N  | C3 |  N  |\n"
#     C3_table += "|----| :-: |----| :-: |----| :-: |----| :-: |\n"
#     classes_colors = []
#     for C3 in class_order:
#         col_row = color_C3(C3)
#         classes_colors.append(col_row)

#     idx = -1
#     for r in range(4):
#         row = ""
#         for c in range(4):
#             idx += 1
#             row += "| {} | [{}](/tables/{}_table) ".format(
#                 classes_colors[idx], OCs_per_class[idx], class_order[idx]
#             )
#         C3_table += row + "|\n"
#     C3_table += "\n"

#     delimeterA = "<!-- Begin table 2 -->\n"
#     delimeterB = "<!-- End table 2 -->\n"
#     database_md_updt = replace_text_between(
#         database_md_in, C3_table, delimeterA, delimeterB
#     )

#     return database_md_updt


# def updt_OCs_per_quad_main_table(df_UCC, database_md_in: str):
#     """Update table of OCs per quadrants"""
#     quad_table = "\n| Region  | lon range  | lat range  |   N |\n"
#     quad_table += "|---------|------------|------------| :-: |\n"
#     quad_lines = (
#         "| Q1P: 1st quadrant, positive latitude | [0, 90)    | [0, 90]    |",
#         "| Q1N: 1st quadrant, negative latitude | [0, 90)    | (0, -90]   |",
#         "| Q2P: 2nd quadrant, positive latitude | [90, 180)  | [0, 90]    |",
#         "| Q2N: 2nd quadrant, negative latitude | [90, 180)  | (0, -90]   |",
#         "| Q3P: 3rd quadrant, positive latitude | [180, 270) | [0, 90]    |",
#         "| Q3N: 3rd quadrant, negative latitude | [180, 270) | (0, -90]   |",
#         "| Q4P: 4th quadrant, positive latitude | [270, 360) | [0, 90]    |",
#         "| Q4N: 4th quadrant, negative latitude | [270, 360) | (0, -90]   |",
#     )

#     df = df_UCC["quad"].values
#     i = 0
#     for quad_N in range(1, 5):
#         for quad_s in ("P", "N"):
#             i += 1
#             quad = "Q" + str(quad_N) + quad_s
#             msk = df == quad
#             quad_table += (
#                 quad_lines[i - 1] + f" [{msk.sum()}](/tables/{quad}_table) |\n"
#             )
#     quad_table += "\n"

#     delimeterA = "<!-- Begin table 3 -->\n"
#     delimeterB = "<!-- End table 3 -->\n"
#     database_md_updt = replace_text_between(
#         database_md_in, quad_table, delimeterA, delimeterB
#     )

#     # if database_md_updt != database_md_in:
#     #     logging.info("Table: OCs per quadrant updated")

#     return database_md_updt


# def updt_shared_membs_main_table(shared_msk: list, database_md_in: str) -> str:
#     """
#     Updates a Markdown string with a summary table of duplicate counts, categorized
#     by the number of duplicates.

#     Args:
#         database_md_in (str): The Markdown string to be updated with the duplicates
#         table.
#         dups_msk (list): A list of numpy boolean arrays, where each array masks rows
#                          in `df_UCC` based on the number of duplicates.

#     Returns:
#         str: The updated Markdown string with the duplicates table inserted.
#     """
#     dups_table = "\n| OCs with shared members |   N  |\n"
#     dups_table += "|---------------------| :--: |\n"

#     for i, msk in enumerate(shared_msk):
#         Nde = " N_shared ="
#         if i == 4:
#             Nde = "N_shared >="
#         dups_table += (
#             f"|     {Nde} {i + 1}      | [{msk.sum()}](/tables/Ns{i + 1}_table) |\n"
#         )
#     dups_table += "\n"

#     delimeterA = "<!-- Begin table 4 -->\n"
#     delimeterB = "<!-- End table 4 -->\n"
#     database_md_updt = replace_text_between(
#         database_md_in, dups_table, delimeterA, delimeterB
#     )

#     # if database_md_updt != database_md_in:
#     #     logging.info("Table: shared members updated")

#     return database_md_updt


# def updt_fund_params_main_table(Cfp_msk, database_md_in: str):
#     """ """
#     Cfp_table = "\n| N_fp |  N  | N_fp |  N  |\n"
#     Cfp_table += "| :--: | :-: | :--: | :-: |\n|"

#     for i, (k, v) in enumerate(Cfp_msk.items()):
#         if "_" in k:
#             k1 = "[" + k.replace("_", ", ") + "]"
#         elif i == len(Cfp_msk) - 1:
#             k1 = ">" + k
#         else:
#             k1 = k
#         fchar = "|\n" if i in (1, 3, 5) else "| "
#         Cfp_table += (
#             f" {k1} | [{v.sum()}](/tables/Nfp_{k.replace('_', '')}_table) {fchar}"
#         )
#     Cfp_table += "\n"

#     delimeterA = "<!-- Begin table 3 -->\n"
#     delimeterB = "<!-- End table 3 -->\n"
#     database_md_updt = replace_text_between(
#         database_md_in, Cfp_table, delimeterA, delimeterB
#     )

#     return database_md_updt


# def updt_Pdup_main_table(Pdup_msk, database_md_in: str):
#     """ """
#     Pdup_table = "\n| P_dup |  N  | P_dup |  N  |\n"
#     Pdup_table += "| :--: | :-: | :--: | :-: |\n"
#     Pdup_table += f"| == 0.0 | [{sum(Pdup_msk[0])}](/tables/Pdup0_table) |"
#     for i, msk in enumerate(Pdup_msk[1:-1]):
#         fchar = "|\n" if i in (0, 2, 4, 6, 8) else "| "
#         N = f"[{sum(msk)}](/tables/Pdup{i + 1}_table)"
#         Pdup_table += f" ({0.0 + (i * 0.1):.1f}, {0.1 + (i * 0.1):.1f}] | {N} {fchar}"
#     Pdup_table += f"| 0.9 < | [{sum(Pdup_msk[-1])}](/tables/Pdup10_table) | -- | -- |\n"
#     Pdup_table += "\n"

#     delimeterA = "<!-- Begin table 4 -->\n"
#     delimeterB = "<!-- End table 4 -->\n"
#     database_md_updt = replace_text_between(
#         database_md_in, Pdup_table, delimeterA, delimeterB
#     )

#     return database_md_updt


# def updt_N50_main_table(membs_msk, database_md_in: str):
#     """
#     Updates a Markdown string with a summary table categorized by the number of
#     N_50 members.

#     Args:
#         membs_msk (list): List of boolean masks, where each mask identifies a range
#         of N50 members.
#         database_md_in (str): The Markdown string to be updated with the duplicates
#         table.

#     Returns:
#         str: The updated Markdown string with the table inserted.
#     """

#     dups_table = "\n| N_50 |   N  | N_50 |   N  |\n"
#     dups_table += "| :--: | :--: | :--: | :--: |\n"

#     dups_table += f"| == 0 | [{membs_msk[0].sum()}](/tables/N50_0_table) "

#     N_limi = 0
#     for i, N_limf in enumerate((25, 50, 75, 100, 250, 500, 1000, 2000)):
#         dups_table += f"| ({N_limi}, {N_limf}] | [{membs_msk[i + 1].sum()}](/tables/N50_{N_limf}_table)"
#         if i % 2 == 0:
#             dups_table += " |\n"
#         else:
#             dups_table += " "
#         N_limi = N_limf

#     dups_table += f"| > 2000 | [{membs_msk[-1].sum()}](/tables/N50_inf_table) |\n"
#     dups_table += "\n"

#     delimeterA = "<!-- Begin table 5 -->\n"
#     delimeterB = "<!-- End table 5 -->\n"
#     database_md_updt = replace_text_between(
#         database_md_in, dups_table, delimeterA, delimeterB
#     )

#     # if database_md_updt != database_md_in:
#     #     logging.info("Table: number of members updated")

#     return database_md_updt


# def chunks(data, SIZE=2):
#     """Split dictionary into chunks"""
#     it = iter(data)
#     for i in range(0, len(data), SIZE):
#         yield {k: data[k] for k in islice(it, SIZE)}


# def updt_UTI_tables(df_updt, UTI_msk) -> dict:
#     """Update the duplicates table files"""
#     header = header_default.format("uti_title", "/tables/uti_link_table/")

#     # UTI==0 table
#     md_table = header.replace("uti_title", "UTI==0").replace("uti_link", "UTI0")
#     md_table = generate_table(df_updt[UTI_msk[0]], md_table)
#     new_tables_dict = {"UTI0": md_table}

#     Ni = 0
#     for i, Nf in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
#         title = f"UTI ({Ni}, {Nf}]"
#         Nmembs = f"UTI{i + 1}"
#         md_table = header.replace("uti_title", title).replace("uti_link", Nmembs)
#         md_table = generate_table(df_updt[UTI_msk[i + 1]], md_table)
#         Ni = Nf
#         new_tables_dict[Nmembs] = md_table

#     # N>0.9 table
#     md_table = header.replace("uti_title", "UTI>0.9").replace("uti_link", "UTI10")
#     md_table = generate_table(df_updt[UTI_msk[-1]], md_table)
#     new_tables_dict["UTI10"] = md_table

#     return new_tables_dict


# def updt_N50_tables(df_updt, membs_msk) -> dict:
#     """Update the duplicates table files"""
#     header = header_default.format("nmembs_title", "/tables/nmembs_link_table/")

#     # N==0 table
#     md_table = header.replace("nmembs_title", "N50 members (==0)").replace(
#         "nmembs_link", "N50_0"
#     )
#     md_table = generate_table(df_updt[membs_msk[0]], md_table)
#     new_tables_dict = {"N50_0": md_table}

#     Ni = 0
#     for i, Nf in enumerate((25, 50, 75, 100, 250, 500, 1000, 2000)):
#         title = f"N50 members ({Ni}, {Nf}]"
#         Nmembs = f"N50_{Nf}"
#         md_table = header.replace("nmembs_title", title).replace("nmembs_link", Nmembs)
#         md_table = generate_table(df_updt[membs_msk[i + 1]], md_table)
#         Ni = Nf
#         new_tables_dict[Nmembs] = md_table

#     # N>2000 table
#     md_table = header.replace("nmembs_title", "N50 members (>2000)").replace(
#         "nmembs_link", "N50_inf"
#     )
#     md_table = generate_table(df_updt[membs_msk[-1]], md_table)
#     new_tables_dict["N50_inf"] = md_table

#     return new_tables_dict


# def updt_C3_classif_tables(df_updt, class_order: list) -> dict:
#     """Update the C3 classification table files"""
#     header = header_default.format("C3_title", "/tables/C3_link_table/")

#     new_tables_dict = {}
#     for C3_N in range(16):
#         title = f"{class_order[C3_N]} classification"
#         table_name = class_order[C3_N]
#         md_table = header.replace("C3_title", title).replace("C3_link", table_name)
#         msk = df_updt["C3"] == table_name
#         md_table = generate_table(df_updt[msk], md_table)

#         new_tables_dict[table_name] = md_table

#     return new_tables_dict


# def updt_fund_params_table(df_updt, Cfp_msk: dict) -> dict:
#     """Update the fundamental parameters table files"""
#     header = header_default.format("Nfp_title", "/tables/Nfp_link_table/")

#     new_tables_dict = {}
#     for i, (k, v) in enumerate(Cfp_msk.items()):
#         if "_" in k:
#             k1 = "=[" + k.replace("_", ", ") + "]"
#         elif i == len(Cfp_msk) - 1:
#             k1 = ">" + k
#         else:
#             k1 = "=" + k

#         title = f"N_fp{k1} fundamental parameters"
#         table_name = f"Nfp_{k.replace('_', '')}"
#         md_table = header.replace("Nfp_title", title).replace("Nfp_link", table_name)
#         md_table = generate_table(df_updt[v], md_table)
#         new_tables_dict[table_name] = md_table

#     return new_tables_dict


# def updt_Pdup_tables(df_updt, Pdup_msk) -> dict:
#     """Update the duplicates table files"""
#     header = header_default.format("pdup_title", "/tables/pdup_link_table/")

#     md_table = header.replace("pdup_title", "Pdup==0").replace("pdup_link", "Pdup0")
#     md_table = generate_table(df_updt[Pdup_msk[0]], md_table)
#     new_tables_dict = {"Pdup0": md_table}

#     Ni = 0
#     for i, Nf in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
#         title = f"Pdup ({Ni}, {Nf}]"
#         Nmembs = f"Pdup{i + 1}"
#         md_table = header.replace("pdup_title", title).replace("pdup_link", Nmembs)
#         md_table = generate_table(df_updt[Pdup_msk[i + 1]], md_table)
#         Ni = Nf
#         new_tables_dict[Nmembs] = md_table

#     # N>0.9 table
#     md_table = header.replace("pdup_title", "Pdup>0.9").replace("pdup_link", "Pdup10")
#     md_table = generate_table(df_updt[Pdup_msk[-1]], md_table)
#     new_tables_dict["Pdup10"] = md_table

#     return new_tables_dict


# def updt_shared_membs_tables(df_updt, dups_msk) -> dict:
#     """Update the shared members table files"""
#     header = header_default.format("shared_title", "/tables/shared_link_table/")

#     new_tables_dict = {}
#     for i, shared_N in enumerate(("Ns1", "Ns2", "Ns3", "Ns4", "Ns5")):
#         title = f"{shared_N} shared"
#         md_table = header.replace("shared_title", title).replace(
#             "shared_link", shared_N
#         )
#         msk = dups_msk[i]
#         md_table = generate_table(df_updt[msk], md_table)

#         new_tables_dict[shared_N] = md_table

#     return new_tables_dict


# def updt_OCs_per_quad_tables(df_updt):
#     """Update the per-quadrant table files"""
#     header = (
#         """---\nlayout: page\ntitle: quad_title\n"""
#         + """permalink: /tables/quad_link_table/\n---\n\n"""
#     )

#     title_dict = {
#         1: "1st",
#         2: "2nd",
#         3: "3rd",
#         4: "4th",
#         "P": "positive",
#         "N": "negative",
#     }

#     new_tables_dict = {}
#     for quad_N in range(1, 5):
#         for quad_s in ("P", "N"):
#             quad = "Q" + str(quad_N) + quad_s

#             title = f"{title_dict[quad_N]} quadrant, {title_dict[quad_s]} latitude"
#             md_table = header.replace("quad_title", title).replace("quad_link", quad)
#             msk = df_updt["quad"] == quad
#             md_table = generate_table(df_updt[msk], md_table)
#             new_tables_dict[quad] = md_table

#     return new_tables_dict
