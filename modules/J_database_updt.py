import gzip
import json
from itertools import islice

import numpy as np
import pandas as pd
from HARDCODED import (
    clusters_json,
    pages_folder,
    root_UCC_path,
)

from modules.ucc_entry import UCC_color


def count_OCs_classes(df_UCC, class_order):
    """Count the number of OCs per C3 class"""
    C3_classif, C3_count = np.unique(df_UCC["C3"], return_counts=True)
    C3_classif = list(C3_classif)
    OCs_per_class = []
    for c in class_order:
        i = C3_classif.index(c)
        OCs_per_class.append(C3_count[i])
    return OCs_per_class


def count_dups(df_UCC: pd.DataFrame) -> list:
    """
    Categorizes duplicate file names in the input DataFrame into five groups
    based on the number of duplicates. Each group is represented by a mask array.

    Args:
        df_UCC (pandas.DataFrame): A DataFrame containing a column named
        "dups_fnames_m", where each entry is a string of file names separated by ";"
        or NaN.

    Returns:
        list: A list of five numpy boolean arrays. Each array corresponds to a mask
              identifying rows with a specific range of duplicates:
              - Index 0: Rows with exactly 1 duplicate.
              - Index 1: Rows with exactly 2 duplicates.
              - Index 2: Rows with exactly 3 duplicates.
              - Index 3: Rows with exactly 4 duplicates.
              - Index 4: Rows with 5 or more duplicates.
    """
    dups_msk = [np.full(len(df_UCC), False) for _ in range(5)]
    for i, dups_fnames_m in enumerate(df_UCC["dups_fnames_m"]):
        if str(dups_fnames_m) == "nan":
            continue
        N_dup = len(dups_fnames_m.split(";"))
        if N_dup == 1:
            dups_msk[0][i] = True
        elif N_dup == 2:
            dups_msk[1][i] = True
        elif N_dup == 3:
            dups_msk[2][i] = True
        elif N_dup == 4:
            dups_msk[3][i] = True
        elif N_dup >= 5:
            dups_msk[4][i] = True

    return dups_msk


def count_N50membs(df_UCC: pd.DataFrame) -> list:
    """ """

    membs_msk = [df_UCC["N_50"] == 0]

    N_limi = 0
    for i, N_limf in enumerate((25, 50, 75, 100, 250, 500, 1000, 2000)):
        N50_r = (df_UCC["N_50"] > N_limi) & (df_UCC["N_50"] <= N_limf)
        membs_msk.append(N50_r)
        N_limi = N_limf

    Ninf = df_UCC["N_50"] > 2000
    membs_msk.append(Ninf)

    return membs_msk


def make_plots(df_UCC, dbs_used, OCs_per_class):
    """ """
    plot_path = "../../ucc/images/catalogued_ocs.webp"
    txt0 = files_handler.update_image(
        DRY_RUN, logging, plot_path, (ucc_plots.make_N_vs_year_plot, df_UCC)
    )
    if txt0 != "":
        logging.info(f"Plot {txt0}: number of OCs vs years")

    plot_path = "../../ucc/images/classif_bar.webp"
    txt0 = files_handler.update_image(
        DRY_RUN,
        logging,
        plot_path,
        (ucc_plots.make_classif_plot, OCs_per_class, class_order),
    )
    if txt0 != "":
        logging.info(f"Plot {txt0}: classification histogram")


def pc_radius(
    angular_radius_arcmin: np.ndarray, parallax_mas: np.ndarray
) -> np.ndarray:
    """
    NOT USED YET (24/12/04)

    Calculate the radius of an object in parsecs given its angular radius (arcmin)
    and parallax (mas).

    Parameters:
    angular_radius_arcmin (np.ndarray): Angular radius in arcminutes.
    parallax_mas (np.ndarray): Parallax in milliarcseconds.

    Returns:
    np.ndarray: Radius in parsecs.
    """
    msk = parallax_mas <= 0
    angular_radius_arcmin[msk] = np.nan
    parallax_mas[msk] = np.nan

    # Convert parallax from mas to arcsec and calculate distance
    distance_pc = 1 / (parallax_mas / 1000)
    # Convert arcmin to radians
    angular_radius_rad = (angular_radius_arcmin / 60) * (np.pi / 180)
    radius_pc = distance_pc * angular_radius_rad  # Radius in parsecs

    return radius_pc


def chunks(data, SIZE=2):
    """Split dictionary into chunks"""
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


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


def ucc_n_total_updt(N_cl_UCC, N_db_UCC, database_md):
    """Update the total number of entries and databases in the UCC"""
    delimiter_a = "<!-- NT1 -->"
    delimiter_b = "<!-- NT2 -->"
    replacement_text = str(N_cl_UCC)
    database_md_updt = replace_text_between(
        database_md, replacement_text, delimiter_a, delimiter_b
    )

    delimiter_a = "<!-- ND1 -->"
    delimiter_b = "<!-- ND2 -->"
    replacement_text = str(N_db_UCC)
    database_md_updt = replace_text_between(
        database_md_updt, replacement_text, delimiter_a, delimiter_b
    )

    if database_md_updt != database_md:
        logging.info("\nNumber of OCs and databases in the UCC updated")

    return database_md_updt


def updt_cats_used(df_UCC, dbs_used, database_md_in):
    """ """
    # Count DB occurrences in UCC
    all_DBs = list(dbs_used.keys())
    N_in_DB = {_: 0 for _ in all_DBs}
    for _ in df_UCC["DB"].values:
        for DB in _.split(";"):
            N_in_DB[DB] += 1

    md_table = "\n| Name | N | Name | N |\n"
    md_table += "| ---- | :-: | ---- | :-: |\n"
    for dict_chunk in chunks(dbs_used):
        row = ""
        for DB, DB_data in dict_chunk.items():
            row += f"| {DB_data['ref']} | [{N_in_DB[DB]}](/{DB}_table) "
        md_table += row + "|\n"
    md_table += "\n"

    delimeterA = "<!-- Begin table 1 -->\n"
    delimeterB = "<!-- End table 1 -->\n"
    database_md_updt = replace_text_between(
        database_md_in, md_table, delimeterA, delimeterB
    )

    if database_md_updt != database_md_in:
        logging.info("Table: catalogues used in the UCC updated")

    return database_md_updt


def updt_C3_classification(OCs_per_class, database_md_in):
    """ """
    C3_table = "\n| C3 |  N  | C3 |  N  | C3 |  N  | C3 |  N  |\n"
    C3_table += "|----| :-: |----| :-: |----| :-: |----| :-: |\n"
    classes_colors = []
    for C3 in class_order:
        col_row = UCC_color(C3)
        classes_colors.append(col_row)

    idx = -1
    for r in range(4):
        row = ""
        for c in range(4):
            idx += 1
            row += "| {} | [{}](/{}_table) ".format(
                classes_colors[idx], OCs_per_class[idx], class_order[idx]
            )
        C3_table += row + "|\n"
    C3_table += "\n"

    delimeterA = "<!-- Begin table 2 -->\n"
    delimeterB = "<!-- End table 2 -->\n"
    database_md_updt = replace_text_between(
        database_md_in, C3_table, delimeterA, delimeterB
    )

    if database_md_updt != database_md_in:
        logging.info("Table: C3 classification updated")

    return database_md_updt


def updt_OCs_per_quad(df_UCC, database_md_in):
    """Update table of OCs per quadrants"""
    quad_table = "\n| Region  | lon range  | lat range  |   N |\n"
    quad_table += "|---------|------------|------------| :-: |\n"
    quad_lines = (
        "| Q1P: 1st quadrant, positive latitude | [0, 90)    | [0, 90]    |",
        "| Q1N: 1st quadrant, negative latitude | [0, 90)    | (0, -90]   |",
        "| Q2P: 2nd quadrant, positive latitude | [90, 180)  | [0, 90]    |",
        "| Q2N: 2nd quadrant, negative latitude | [90, 180)  | (0, -90]   |",
        "| Q3P: 3rd quadrant, positive latitude | [180, 270) | [0, 90]    |",
        "| Q3N: 3rd quadrant, negative latitude | [180, 270) | (0, -90]   |",
        "| Q4P: 4th quadrant, positive latitude | [270, 360) | [0, 90]    |",
        "| Q4N: 4th quadrant, negative latitude | [270, 360) | (0, -90]   |",
    )

    df = df_UCC["quad"].values
    i = 0
    for quad_N in range(1, 5):
        for quad_s in ("P", "N"):
            i += 1
            quad = "Q" + str(quad_N) + quad_s
            msk = df == quad
            quad_table += quad_lines[i - 1] + f" [{msk.sum()}](/{quad}_table) |\n"
    quad_table += "\n"

    delimeterA = "<!-- Begin table 3 -->\n"
    delimeterB = "<!-- End table 3 -->\n"
    database_md_updt = replace_text_between(
        database_md_in, quad_table, delimeterA, delimeterB
    )

    if database_md_updt != database_md_in:
        logging.info("Table: OCs per quadrant updated")

    return database_md_updt


def updt_dups_table(dups_msk: list, database_md_in: str) -> str:
    """
    Updates a Markdown string with a summary table of duplicate counts, categorized
    by the number of duplicates.

    Args:
        database_md_in (str): The Markdown string to be updated with the duplicates
        table.
        dups_msk (list): A list of numpy boolean arrays, where each array masks rows
                         in `df_UCC` based on the number of duplicates.

    Returns:
        str: The updated Markdown string with the duplicates table inserted.
    """
    dups_table = "\n| Probable duplicates |   N  |\n"
    dups_table += "|---------------------| :--: |\n"

    for i, msk in enumerate(dups_msk):
        Nde = " N_dup ="
        if i == 4:
            Nde = "N_dup >="
        dups_table += f"|     {Nde} {i + 1}      | [{msk.sum()}](/Nd{i+1}_table) |\n"
    dups_table += "\n"

    delimeterA = "<!-- Begin table 4 -->\n"
    delimeterB = "<!-- End table 4 -->\n"
    database_md_updt = replace_text_between(
        database_md_in, dups_table, delimeterA, delimeterB
    )

    if database_md_updt != database_md_in:
        logging.info("Table: duplicated OCs updated")

    return database_md_updt


def memb_number_table(membs_msk, database_md_in):
    """
    Updates a Markdown string with a summary table categorized by the number of
    N_50 members.

    Args:
        membs_msk (list): List of boolean masks, where each mask identifies a range
        of N50 members.
        database_md_in (str): The Markdown string to be updated with the duplicates
        table.

    Returns:
        str: The updated Markdown string with the table inserted.
    """

    dups_table = "\n| N_50 |   N  | N_50 |   N  |\n"
    dups_table += "| :--: | :--: | :--: | :--: |\n"

    dups_table += f"| == 0 | [{membs_msk[0].sum()}](/N50_0_table) "

    N_limi = 0
    for i, N_limf in enumerate((25, 50, 75, 100, 250, 500, 1000, 2000)):
        dups_table += (
            f"| ({N_limi}, {N_limf}] | [{membs_msk[i + 1].sum()}](/N50_{N_limf}_table)"
        )
        if i % 2 == 0:
            dups_table += " |\n"
        else:
            dups_table += " "
        N_limi = N_limf

    dups_table += f"| > 2000 | [{membs_msk[-1].sum()}](/N50_inf_table) |\n"
    dups_table += "\n"

    delimeterA = "<!-- Begin table 5 -->\n"
    delimeterB = "<!-- End table 5 -->\n"
    database_md_updt = replace_text_between(
        database_md_in, dups_table, delimeterA, delimeterB
    )

    if database_md_updt != database_md_in:
        logging.info("Table: number of members updated")

    return database_md_updt


def updt_UCC(df_UCC: pd.DataFrame) -> pd.DataFrame:
    """
    Updates a DataFrame of astronomical cluster data by processing identifiers,
    coordinates, URLs, and derived quantities such as distances.

    Args:
        df_UCC (pd.DataFrame): The input DataFrame containing cluster data with columns:
                               - "ID": A semicolon-separated string of identifiers.
                               - "fnames": A semicolon-separated string of file names.
                               - "RA_ICRS", "DE_ICRS", "GLON", "GLAT": Coordinates.
                               - "Plx_m": Parallax measurements in milliarcseconds.

    Returns:
        pd.DataFrame: The updated DataFrame with the following changes:
                      - "ID": Extracts the first identifier from the "ID" column.
                      - "ID_url": Adds URLs linking to cluster details.
                      - "RA_ICRS", "DE_ICRS", "GLON", "GLAT": Rounded coordinates.
                      - "dist_pc": Adds parallax-based distances in parsecs, clipped
                                   to the range [10, 50000].
    """
    df = pd.DataFrame(df_UCC)

    # Extract the first identifier from the "ID" column
    df["ID"] = [_.split(";")[0] for _ in df_UCC["ID"]]

    # Generate URLs for names
    names_url = []
    for i, cl in df.iterrows():
        name = str(cl["ID"]).split(";")[0]
        fname = str(cl["fnames"]).split(";")[0]
        url = "/_clusters/" + fname + "/"
        names_url.append(f"[{name}]({url})")
    df["ID_url"] = names_url

    # Round coordinate columns
    df["RA_ICRS"] = np.round(df_UCC["RA_ICRS"].values, 2)
    df["DE_ICRS"] = np.round(df_UCC["DE_ICRS"].values, 2)
    df["GLON"] = np.round(df_UCC["GLON"].values, 2)
    df["GLAT"] = np.round(df_UCC["GLAT"].values, 2)

    # Compute parallax-based distances in parsecs
    dist_pc = 1000 / np.clip(np.array(df["Plx_m"]), a_min=0.0000001, a_max=np.inf)
    dist_pc = np.clip(dist_pc, a_min=10, a_max=50000)
    df["dist_pc"] = np.round(dist_pc, 0)

    return df


def general_table_update(new_table: str, table_name: str) -> None:
    """
    Updates a markdown table file if the content has changed.

    Args:
        logging: A logging object used to log messages.
        new_table: The updated table content as a string.
        table_name: The name of the table, used to construct the file path.

    Returns:
        None
    """
    # Read old entry, if any
    try:
        with open(
            root_UCC_path + pages_folder + "/tables/" + table_name + "_table.md", "r"
        ) as f:
            old_table = f.read()
    except FileNotFoundError:
        # This is a new table with no md entry yet
        old_table = ""

    # Write to file if any changes are detected
    if old_table != new_table:
        if DRY_RUN is False:
            with open(
                root_UCC_path + pages_folder + "/tables/" + table_name + "_table.md",
                "w",
            ) as file:
                file.write(new_table)
        logging.info(f"Table {table_name} updated")


def updt_DBs_tables(dbs_used, df_updt):
    """Update the DBs classification table files"""
    header = (
        """---\nlayout: page\ntitle: \n""" + """permalink: /DB_link_table/\n---\n\n"""
    )

    # Count DB occurrences in UCC
    all_DBs = list(dbs_used.keys())

    for DB_id in all_DBs:
        md_table = header.replace("DB_link", DB_id)
        md_table += "&nbsp;\n" + f"# {dbs_used[DB_id]['ref']}" + "\n\n"

        msk = []
        for _ in df_updt["DB"].values:
            if DB_id in _.split(";"):
                msk.append(True)
            else:
                msk.append(False)
        msk = np.array(msk)

        new_table = generate_table(df_updt, md_table, msk)
        general_table_update(new_table, DB_id)


def updt_n50members_tables(df_updt, membs_msk):
    """Update the duplicates table files"""
    header = (
        """---\nlayout: page\ntitle: nmembs_title\n"""
        + """permalink: /nmembs_link_table/\n---\n\n"""
    )

    # N==0 table
    md_table = header.replace("nmembs_title", "N50 members (==0)").replace(
        "nmembs_link", "N50_0"
    )
    md_table = generate_table(df_updt, md_table, membs_msk[0])
    general_table_update(md_table, "N50_0")

    Ni = 0
    for i, Nf in enumerate((25, 50, 75, 100, 250, 500, 1000, 2000)):
        title = f"N50 members ({Ni}, {Nf}]"
        Nmembs = f"N50_{Nf}"
        md_table = header.replace("nmembs_title", title).replace("nmembs_link", Nmembs)
        md_table = generate_table(df_updt, md_table, membs_msk[i + 1])
        Ni = Nf
        general_table_update(md_table, Nmembs)

    # N>2000 table
    md_table = header.replace("nmembs_title", "N50 members (>2000)").replace(
        "nmembs_link", "N50_inf"
    )
    md_table = generate_table(df_updt, md_table, membs_msk[-1])
    general_table_update(md_table, "N50_inf")


def updt_C3_tables(df_updt):
    """Update the C3 classification table files"""
    header = (
        """---\nlayout: page\ntitle: C3_title\n"""
        + """permalink: /C3_link_table/\n---\n\n"""
    )

    for C3_N in range(16):
        title = f"{class_order[C3_N]} classification"
        md_table = header.replace("C3_title", title).replace(
            "C3_link", class_order[C3_N]
        )
        msk = df_updt["C3"] == class_order[C3_N]
        md_table = generate_table(df_updt, md_table, msk)

        general_table_update(md_table, class_order[C3_N])


def updt_dups_tables(df_updt, dups_msk):
    """Update the duplicates table files"""
    header = (
        """---\nlayout: page\ntitle: dups_title\n"""
        + """permalink: /dups_link_table/\n---\n\n"""
    )

    for i, dups_N in enumerate(("Nd1", "Nd2", "Nd3", "Nd4", "Nd5")):
        title = f"{dups_N} duplicates"
        md_table = header.replace("dups_title", title).replace("dups_link", dups_N)
        msk = dups_msk[i]
        md_table = generate_table(df_updt, md_table, msk)

        general_table_update(md_table, dups_N)


def updt_quad_tables(df_updt):
    """Update the per-quadrant table files"""
    header = (
        """---\nlayout: page\ntitle: quad_title\n"""
        + """permalink: /quad_link_table/\n---\n\n"""
    )

    title_dict = {
        1: "1st",
        2: "2nd",
        3: "3rd",
        4: "4th",
        "P": "positive",
        "N": "negative",
    }

    for quad_N in range(1, 5):
        for quad_s in ("P", "N"):
            quad = "Q" + str(quad_N) + quad_s

            title = f"{title_dict[quad_N]} quadrant, {title_dict[quad_s]} latitude"
            md_table = header.replace("quad_title", title).replace("quad_link", quad)
            msk = df_updt["quad"] == quad
            md_table = generate_table(df_updt, md_table, msk)
            general_table_update(md_table, quad)


def generate_table(df_updt, md_table, msk):
    """ """
    md_table += "| Name | l | b | ra | dec | Plx | N50 | r50 | C3 |\n"
    md_table += "| ---- | - | - | -- | --- | --- | --  | --  |-- |\n"

    df_m = df_updt[msk]
    df_m = df_m.sort_values("ID_url")
    df_m["N_50"] = df_m["N_50"].astype(int)

    for i, row in df_m.iterrows():
        for col in (
            "ID_url",
            "GLON",
            "GLAT",
            "RA_ICRS",
            "DE_ICRS",
            "Plx_m",
            "N_50",
            "r_50",
        ):
            md_table += "| " + str(row[col]) + " "
        abcd = UCC_color(row["C3"])
        md_table += "| " + abcd + " |\n"

    return md_table


def updt_cls_JSON(df_updt):
    """
    Update cluster.json file used by 'ucc.ar' search
    """
    df = pd.DataFrame(
        df_updt[
            [
                "ID",
                "fnames",
                "RA_ICRS",
                "DE_ICRS",
                "GLON",
                "GLAT",
                "dist_pc",
                "N_50",
            ]
        ]
    )
    df = df.sort_values("GLON")

    df.rename(
        columns={
            "ID": "N",
            "fnames": "F",
            "RA_ICRS": "R",
            "DE_ICRS": "D",
            "GLON": "L",
            "GLAT": "B",
            "dist_pc": "P",
            "N_50": "M",
        },
        inplace=True,
    )
    json_new = df.to_dict(orient="records")

    # Load the old JSON data
    with gzip.open(root_UCC_path + clusters_json, "rt", encoding="utf-8") as file:
        json_old = json.load(file)

    # Check if new JSON is equal to the old one
    update_flag = False
    if len(json_old) != len(json_new):
        update_flag = True
    else:
        # True if JSONs are NOT equal
        update_flag = not all(a == b for a, b in zip(json_old, json_new))
        # Print differences to screen
        for i, (dict1, dict2) in enumerate(zip(json_old, json_new)):
            differing_keys = {
                key
                for key in dict1.keys() | dict2.keys()
                if dict1.get(key) != dict2.get(key)
            }
            if differing_keys:
                for key in differing_keys:
                    logging.info(
                        f"{i}, {key} --> OLD: {dict1.get(key)} | NEW: {dict2.get(key)}"
                    )

    # Update JSON if required
    if update_flag is True:
        if DRY_RUN is False:
            df.to_json(
                root_UCC_path + clusters_json,
                orient="records",
                indent=1,
                compression="gzip",
            )
        logging.info("File 'clusters.json.gz' updated")


if __name__ == "__main__":
    main()
