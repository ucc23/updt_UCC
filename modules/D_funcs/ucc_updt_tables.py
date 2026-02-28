import os
from pathlib import Path

import numpy as np

from ..variables import (
    cmmts_tables_folder,
    header_default,
    root_ucc_path,
    table_sort_js,
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
    # Pre-split DB strings once, outside any loop
    df = df_ucc[["DB", "P_dup", "bad_oc"]].copy()
    db_lists = df["DB"].str.split(";")
    is_dup = (df["P_dup"].astype(float) > 0.5).to_numpy()
    is_bad = (df["bad_oc"] == "y").to_numpy()

    # Single pass over rows, fan out to each DB in the split list
    counts = {db: 0 for db in dbs_used}
    n_dups = {db: 0 for db in dbs_used}
    n_bads = {db: 0 for db in dbs_used}

    for dbs, dup, bad in zip(db_lists, is_dup, is_bad):
        for db in dbs:
            if db in counts:
                counts[db] += 1
                if dup:
                    n_dups[db] += 1
                if bad:
                    n_bads[db] += 1

    return {
        db: (
            counts[db],
            100 * n_dups[db] / counts[db] if counts[db] else 0,
            100 * n_bads[db] / counts[db] if counts[db] else 0,
        )
        for db in dbs_used
    }


def count_OCs_in_tables(
    df_UCC, current_JSON, temp_cmmts_tables_path
) -> tuple[dict, dict]:
    """ """
    # Count DB occurrences in UCC
    N_in_DB = {_: 0 for _ in current_JSON.keys()}
    for _ in df_UCC["DB"].values:
        for DB in _.split(";"):
            N_in_DB[DB] += 1

    # Count comments in tables (read from temp folder if updated/generated, otherwise
    # read from original folder)
    N_cmmts_dict = {}
    for DB in current_JSON.keys():
        if "comments" in current_JSON[DB]["data_cmmts"]:
            # Read from temp folder if the table was updated/generated, otherwise read
            # from the original folder
            if os.path.exists(f"{temp_cmmts_tables_path}/{DB}_table.md"):
                with open(f"{temp_cmmts_tables_path}/{DB}_table.md", "r") as f:
                    comments = f.readlines()
            else:
                with open(
                    f"{root_ucc_path}{cmmts_tables_folder}{DB}_table.md", "r"
                ) as f:
                    comments = f.readlines()
            # Very crude way of reading the number of rows from the table md file
            N_cmmts = len(
                [_ for _ in comments if _.startswith('| <a href="{{ site.baseurl }}')]
            )
            N_cmmts_dict[DB] = N_cmmts
        else:
            N_cmmts_dict[DB] = 0

    return N_in_DB, N_cmmts_dict


def updt_articles_table(
    current_JSON, database_md_in, N_in_DB, N_cmmts_dict, max_chars_title=50
):
    """Update the table with the catalogues used in the UCC"""

    # Invert json by the 'year' key so that larger values are on top
    inv_json = dict(
        sorted(current_JSON.items(), key=lambda item: item[1]["year"], reverse=True)
    )

    md_table = (
        "\n| Title | Author(s) | Year | Data | N<sub>C</sub> | N<sub>D</sub> | CSV |\n"
    )
    md_table += "| ---- | :---: | :--: | :----: | :-: | :-: | :-: |\n"
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

        cmmts_url = "0"
        if N_cmmts_dict[DB] > 0:
            cmmts_url = f"[{N_cmmts_dict[DB]}](/tables/cmmts/{DB}_table)"

        dbs_url, CSV_url = "0", "N/A"
        if N_in_DB[DB] > 0:
            dbs_url = f"[{N_in_DB[DB]}](/tables/dbs/{DB}_table)"
            CSV_url = f"""<a href="https://flatgithub.com/ucc23/updt_UCC?filename=data/databases/{DB}.csv" target="_blank">📊</a>"""

        row += f"| {ref_url} | {DB_data['authors']} | {DB_data['year']} | {data_url} | {cmmts_url} | {dbs_url} | {CSV_url}"
        md_table += row + "|\n"
    md_table += "\n"

    delimeterA = "<!-- Begin table 1 -->\n"
    delimeterB = "<!-- End table 1 -->\n"
    database_md_updt = replace_text_between(
        database_md_in, md_table, delimeterA, delimeterB
    )

    return database_md_updt


def updt_DBs_tables(dbs_used, df_updt, cmmts_JSONS_lst, DBs_dups_badOCs) -> dict:
    """Update the DBs classification table files"""
    header_db = header_default.format("", "/tables/dbs/DB_link_table/")
    header_cmmt = header_default.format("", "/tables/cmmts/DB_link_table/")

    fnames_vec = df_updt["fnames"].values

    new_tables_dict = {"data": {}, "comments": {}}
    for DB_id, vals in dbs_used.items():
        ref_url = f"[{vals['authors']} ({vals['year']})]({vals['SCIX_url']})"

        if "comments" in vals["data_cmmts"]:
            # Original dictionary
            clusters_dict = cmmts_JSONS_lst[DB_id]["clusters"]

            # Find clusters in comments list that are in UCC
            # Flatten all cluster sets into a single master set for O(1) lookups
            master_set = set().union(
                *[set(fname.split(";")) for fname in clusters_dict.keys()]
            )
            # Use a list comprehension with .isdisjoint()
            # (faster as it short-circuits on the first match)
            msk = [not set(val.split(";")).isdisjoint(master_set) for val in fnames_vec]

            # Extract 'Name, fname' columns with mask applied
            names_cols = df_updt[np.array(msk)][["ID_url", "fname", "fnames"]]

            # Filter fnames not in comments
            names_cols = names_cols[
                names_cols["fnames"].apply(
                    lambda val: not set(val.split(";")).isdisjoint(master_set)
                )
            ].reset_index(drop=True)

            # Build flat lookup: individual fname -> comment
            fname_to_cmmt = {}
            for fname, cmmt in clusters_dict.items():
                fname_to_cmmt[fname] = cmmt

            # Add 'cmmt' column
            names_cols["cmmt"] = names_cols["fnames"].apply(
                lambda val: next(
                    (fname_to_cmmt[f] for f in val.split(";") if f in fname_to_cmmt),
                    None,
                )
            )

            new_table = generate_table(header_cmmt, DB_id, ref_url, names_cols, "")
            new_tables_dict["comments"][DB_id] = new_table

        if "data" in vals["data_cmmts"]:
            msk = []
            for _ in df_updt["DB"].values:
                if DB_id in _.split(";"):
                    msk.append(True)
                else:
                    msk.append(False)

            N_tot, p_dup, p_bad = DBs_dups_badOCs[DB_id]
            parts = [
                f"{p_dup:.0f}% are probable duplicates "
                "([P<sub>dup</sub>>50%](/faq/#how-is-the-duplicate-probability-estimated))"
                if p_dup > 1
                else None,
                f"{p_bad:.0f}% are classified as "
                "[likely non-clusters](/faq/#how-are-objects-flagged-as-likely-not-real) "
                "(names colored red)"
                if p_bad > 1
                else None,
            ]
            parts = [p for p in parts if p]
            table_note = (
                f"This database consists of {N_tot} entries, of which "
                f"{' and '.join(parts)}.\n\n"
                if parts
                else ""
            )

            new_table = generate_table(
                header_db, DB_id, ref_url, df_updt[np.array(msk)], table_note
            )
            new_tables_dict["data"][DB_id] = new_table

    return new_tables_dict


def generate_table(header, DB_id, ref_url, table_rows, table_note):
    """ """
    parts = [
        header.replace("DB_link", DB_id),
        "&nbsp;\n" + f"# {ref_url}" + "\n\n",
    ]
    parts.append(table_note + "\n\n")

    if "cmmts" in header:
        parts.append("| Name | Comment |\n| --- | :-: |\n")
        parts.append(
            "\n".join(
                f"| {row.ID_url} | {row.cmmt} |"
                for row in table_rows.itertuples(index=False)
            )
        )
    else:
        cols = [
            "ID_url",
            "RA_ICRS",
            "DE_ICRS",
            "GLON",
            "GLAT",
            "Plx_m_round",
            "N_50",
            "C3_abcd",
            "P_dup",
            "UTI",
        ]
        parts.append(
            "| Name | RA | DEC | LON | LAT | Plx | N<sub>50</sub> | C3 |"
            " P<sub>dup</sub> | UTI |\n"
            "| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\n"
        )
        arr = table_rows[cols].astype(str).to_numpy()
        parts.append("\n".join("| " + " | ".join(row) + " |" for row in arr) + "\n")

    parts.append(table_sort_js)
    return "".join(parts)


def general_table_update(
    logging,
    root_path_db: Path,
    root_path_cmmts: Path,
    temp_path_db: Path,
    temp_path_cmmts: Path,
    new_tables_dict: dict,
) -> None:
    """
    Updates a markdown table file if the content has changed.
    """
    sections = {
        "data": (root_path_db, temp_path_db),
        "comments": (root_path_cmmts, temp_path_cmmts),
    }
    N_tot = 0
    for section, (root_path, temp_path) in sections.items():
        for DB_id, new_table in new_tables_dict[section].items():
            fname = f"{DB_id}_table.md"
            src = root_path / fname
            dst = temp_path / fname

            old_table = src.read_text() if src.exists() else ""

            if old_table != new_table:
                dst.write_text(new_table)
                logging.info(f"Table {DB_id} {'updated' if old_table else 'generated'}")
                N_tot += 1
    logging.info(f"{N_tot} tables updated/generated")


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
