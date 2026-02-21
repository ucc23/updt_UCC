import numpy as np

from ..variables import header_default


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
    md_table = "\n| Title | Author(s) | Year | Data | N | CSV |\n"
    md_table += "| ---- | :---: | :--: | :----: | :-: | :-: |\n"
    for DB, DB_data in inv_json.items():
        row = ""
        title = DB_data["title"].replace("'", "").replace('"', "")
        short_title = title[:max_chars_title] + "..."
        ref_url = f"""<a href="{DB_data["SCIX_url"]}" target="_blank" title="{title}">{short_title}</a>"""
        data_url = DB_data["data_url"]
        if data_url == "N/A":
            pass
        elif "vizier" in data_url:
            data_url = f"""<a href="{data_url}" target="_blank"> <img src="/images/vizier.png" alt="Vizier url"></a>"""
        elif "github" in data_url:
            data_url = f"""<a href="{data_url}" target="_blank"> <img src="/images/github.png" alt="Github url"></a>"""
        elif "zenodo" in data_url:
            data_url = f"""<a href="{data_url}" target="_blank"> <img src="/images/zenodo.png" alt="Zenodo url"></a>"""
        elif "china-vo" in data_url:
            data_url = f"""<a href="{data_url}" target="_blank"> <img src="/images/chinavo.png" alt="ChinaVO url"></a>"""
        else:
            data_url = f"""<a href="{data_url}" target="_blank"> 🔗</a>"""
        CSV_url = f"""<a href="https://flatgithub.com/ucc23/updt_UCC?filename=data/databases/{DB}.csv" target="_blank">📊</a>"""
        row += f"| {ref_url} | {DB_data['authors']} | {DB_data['year']} | {data_url} | [{N_in_DB[DB]}](/tables/dbs/{DB}_table) | {CSV_url}"
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


def ucc_n_total_updt(logging, N_db_UCC, N_cl_UCC, N_fpars, N_members_UCC, database_md):
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

    delimiter_a = "<!-- NP1 -->"
    delimiter_b = "<!-- NP2 -->"
    replacement_text = str(N_fpars)
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
