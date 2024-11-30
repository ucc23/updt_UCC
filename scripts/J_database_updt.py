import json
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from HARDCODED import (
    UCC_folder,
    all_DBs_json,
    clusters_json,
    dbs_folder,
    pages_folder,
    root_UCC_path,
)
from modules import UCC_new_match, logger, ucc_entry

# Order used for the C3 classes
class_order = [
    "AA",
    "AB",
    "BA",
    "AC",
    "CA",
    "BB",
    "AD",
    "DA",
    "BC",
    "CB",
    "BD",
    "DB",
    "CC",
    "CD",
    "DC",
    "DD",
]

# Use to process files without writing changes to files
DRY_RUN = True


def main():
    """ """
    logging = logger.main()
    logging.info("\nRunning 'J_database_updt' script\n")
    logging.info(f"DRY RUN IS {DRY_RUN}\n")

    # Read latest version of the UCC
    df_UCC, _ = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    # Load DATABASE.md file
    with open(root_UCC_path + pages_folder + "/" + "DATABASE.md") as file:
        database_md = file.read()

    # Load clusters data in JSON file
    with open(dbs_folder + all_DBs_json) as f:
        dbs_used = json.load(f)

    # Count number of OCs in each class
    C3_classif, C3_count = np.unique(df_UCC["C3"], return_counts=True)
    C3_classif = list(C3_classif)
    OCs_per_class = []
    for c in class_order:
        i = C3_classif.index(c)
        OCs_per_class.append(C3_count[i])

    # Update plots
    make_N_vs_year_plot(df_UCC)
    logging.info("\nPlot generated: number of OCs vs years")
    make_classif_plot(OCs_per_class)
    logging.info("Plot generated: classification histogram")

    # Update the total number of entries and databases in the UCC
    database_md_updt = ucc_n_total_updt(len(df_UCC), len(dbs_used), database_md)

    # Mask with duplicates
    dups_msk = count_dups(df_UCC)

    # Update DATABASE.md file
    database_md_updt = updt_cats_used(df_UCC, dbs_used, database_md_updt)
    logging.info("Table: catalogues used in the UCC updated")
    database_md_updt = updt_C3_classification(OCs_per_class, database_md_updt)
    logging.info("Table: C3 classification updated")
    database_md_updt = updt_OCs_per_quad(df_UCC, database_md_updt)
    logging.info("Table: OCs per quadrant updated")
    database_md_updt = updt_dups_table(df_UCC, database_md_updt, dups_msk)
    logging.info("Table: duplicated OCs updated")
    # Save updated page
    if DRY_RUN is False:
        with open(root_UCC_path + pages_folder + "/" + "DATABASE.md", "w") as file:
            file.write(database_md_updt)
    logging.info("DATABASE.md updated")

    # Prepare df_UCC to be used in the updating of the tables below
    df_updt = updt_UCC(df_UCC)

    # Update groups of tables
    updt_DBs_tables(dbs_used, df_updt, root_UCC_path, pages_folder)
    logging.info("\nDBs tables updated")
    updt_C3_tables(df_updt, root_UCC_path, pages_folder)
    logging.info("\nC3 tables updated")
    updt_quad_tables(df_updt, root_UCC_path, pages_folder)
    logging.info("\nQuadrant tables updated")
    updt_dups_tables(df_updt, root_UCC_path, pages_folder, dups_msk)
    logging.info("\nDuplicates tables updated")

    updt_cls_JSON(df_updt, root_UCC_path)
    logging.info("\nFile 'clusters.json.gz' updated")


def make_N_vs_year_plot(df_UCC):
    """ """
    plt.style.use("modules/science2.mplstyle")

    # Extract minimum year of publication for each catalogued OC
    years = []
    for i, row in df_UCC.iterrows():
        oc_years = []
        for cat0 in row["DB"].split(";"):
            cat = cat0.split("_")[0]
            oc_years.append(int("20" + cat[-2:]))
        years.append(min(oc_years))

    # Count number of OCs per year
    unique, counts = np.unique(years, return_counts=True)
    c_sum = np.cumsum(counts)

    # Combine with old years (previous to 1995)
    #
    # Source: http://www.messier.seds.org/open.html#Messier
    # Messier (1771): 33
    #
    # Source: https://spider.seds.org/ngc/ngc.html
    # "William Herschel first published his catalogue containing 1,000 entries in 1786
    # ...added 1,000 entries in 1789 and a final 500 in 1802 ... total number of entries
    # to 2,500. In 1864, Sir John Herschel the son of William then expanded the
    # catalogue into the General Catalogue of Nebulae and Clusters and Clusters of
    # Stars (GC), which contained 5,079 entries"
    # Herschel (1786): ???
    #
    # Source:
    # https://in-the-sky.org/data/catalogue.php?cat=NGC&const=1&type=OC&sort=0&view=1
    # Dreyer (1888): 640?
    #
    # Source: https://webda.physics.muni.cz/description.html#cluster_level
    # "The catalogue of cluster parameters prepared by Lyngå (1987, 5th edition, CDS
    # VII/92A) has been used to build the list of known open clusters."
    # Going to http://cdsweb.u-strasbg.fr/htbin/myqcat3?VII/92A leads
    # to a Vizier table with 1151 entries
    # Lyngå (1987): 1151
    #
    # Mermilliod 1988 (Bull. Inform. CDS 35, 77-91): 570
    # Mermilliod 1996ASPC...90..475M (BDA, 1996): ~500
    #
    years = [1771, 1888, 1987] + list(unique)
    values = [33, 640, 1151] + [int(1200 + c_sum[0])] + list(c_sum[1:])

    fig = plt.figure(figsize=(4, 3))
    plt.plot(years, values, alpha=0.5, lw=3, marker="o", ms=7, color="maroon", zorder=5)

    plt.annotate(
        "Messier",
        xy=(1775, 30),
        xytext=(1820, 30),
        verticalalignment="center",
        # Custom arrow
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )
    # plt.annotate(
    #     "Hipparcos + 2MASS",
    #     xy=(2015, 1000),
    #     xytext=(1850, 1000),  # fontsize=8,
    #     verticalalignment="center",
    #     # Custom arrow
    #     arrowprops=dict(arrowstyle="->", lw=0.7),
    # )
    plt.annotate(
        "Gaia data release",
        xy=(2015, 3600),
        xytext=(1850, 3600),  # fontsize=8,
        verticalalignment="center",
        # Custom arrow
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )
    plt.text(x=1860, y=120000, s="Approx total number of OCs in the Galaxy", fontsize=7)

    plt.axhline(100000, ls=":", lw=2, alpha=0.5, c="k")
    plt.xlim(1759, max(years) + 25)
    plt.title(r"Catalogued OCs in the literature", fontsize=7)
    # plt.xlabel("Year")  # )
    # plt.ylabel("N")  # , fontsize=15)
    # _, ymax = plt.gca().get_ylim()
    plt.ylim(20, 250000)
    plt.yscale("log")
    fig.tight_layout()
    if DRY_RUN is False:
        plt.savefig("../../ucc/images/catalogued_ocs.webp", dpi=300)


def make_classif_plot(height):
    """ """
    plt.style.use("modules/science2.mplstyle")

    def rescale(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    my_cmap = plt.get_cmap("RdYlGn_r")
    x1 = np.arange(0, len(class_order) + 1)

    fig, ax = plt.subplots(1, figsize=(6, 3))
    plt.bar(class_order, height, color=my_cmap(rescale(x1)))
    plt.xticks(rotation=30)
    plt.ylabel("N")
    ax.tick_params(axis="x", which="both", length=0)
    plt.minorticks_off()
    fig.tight_layout()
    if DRY_RUN is False:
        plt.savefig("../../ucc/images/classif_bar.webp", dpi=300)


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
    """ """
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
    return database_md_updt


def updt_cats_used(df_UCC, dbs_used, database_md_updt):
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
        database_md_updt, md_table, delimeterA, delimeterB
    )

    return database_md_updt


def updt_C3_classification(OCs_per_class, database_md_updt):
    """ """
    C3_table = "\n| C3 |  N  | C3 |  N  | C3 |  N  | C3 |  N  |\n"
    C3_table += "|----| :-: |----| :-: |----| :-: |----| :-: |\n"
    classes_colors = []
    for C3 in class_order:
        col_row = ucc_entry.UCC_color(C3)
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
        database_md_updt, C3_table, delimeterA, delimeterB
    )

    return database_md_updt


def updt_OCs_per_quad(df_UCC, database_md_updt):
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
        database_md_updt, quad_table, delimeterA, delimeterB
    )

    return database_md_updt


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


def updt_dups_table(df_UCC: pd.DataFrame, database_md_updt: str, dups_msk: list) -> str:
    """
    Updates a Markdown string with a summary table of duplicate counts, categorized
    by the number of duplicates.

    Args:
        df_UCC (pandas.DataFrame): The DataFrame containing data on duplicates.
        database_md_updt (str): The Markdown string to be updated with the duplicates
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
        database_md_updt, dups_table, delimeterA, delimeterB
    )

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
                               - "plx_m": Parallax measurements in milliarcseconds.

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
    dist_pc = 1000 / np.clip(np.array(df["plx_m"]), a_min=0.0000001, a_max=np.inf)
    dist_pc = np.clip(dist_pc, a_min=10, a_max=50000)
    df["dist_pc"] = np.round(dist_pc, 0)

    return df


def updt_DBs_tables(dbs_used, df_updt, root_UCC_path, pages_folder):
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

        md_table = generate_table(df_updt, md_table, msk)
        if DRY_RUN is False:
            with open(
                root_UCC_path + pages_folder + "/tables/" + DB_id + "_table.md", "w"
            ) as file:
                file.write(md_table)


def updt_C3_tables(df_updt, root_UCC_path, pages_folder):
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

        if DRY_RUN is False:
            with open(
                root_UCC_path
                + pages_folder
                + "/tables/"
                + class_order[C3_N]
                + "_table.md",
                "w",
            ) as file:
                file.write(md_table)


def updt_quad_tables(df_updt, root_UCC_path, pages_folder):
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

            if DRY_RUN is False:
                with open(
                    root_UCC_path + pages_folder + "/tables/" + quad + "_table.md", "w"
                ) as file:
                    file.write(md_table)


def updt_dups_tables(df_updt, root_UCC_path, pages_folder, dups_msk):
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

        if DRY_RUN is False:
            with open(
                root_UCC_path + pages_folder + "/tables/" + dups_N + "_table.md",
                "w",
            ) as file:
                file.write(md_table)


def generate_table(df_updt, md_table, msk):
    """ """
    md_table += "| Name | l | b | ra | dec | Plx | C1 | C2 | C3 |\n"
    md_table += "| ---- | - | - | -- | --- | --- | -- | -- | -- |\n"

    df_m = df_updt[msk]
    df_m = df_m.sort_values("ID_url")

    for i, row in df_m.iterrows():
        for col in (
            "ID_url",
            "GLON",
            "GLAT",
            "RA_ICRS",
            "DE_ICRS",
            "plx_m",
            "C1",
            "C2",
        ):
            md_table += "| " + str(row[col]) + " "
        abcd = UCC_color(row["C3"])
        md_table += "| " + abcd + " |\n"

    return md_table


def updt_cls_JSON(df_updt, root_UCC_path):
    """
    Update cluster.json file used by 'ucc.ar' search
    """
    df = pd.DataFrame(
        df_updt[
            [
                "ID",
                "fnames",
                "UCC_ID",
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
            "UCC_ID": "U",
            "RA_ICRS": "R",
            "DE_ICRS": "D",
            "GLON": "L",
            "GLAT": "B",
            "dist_pc": "P",
            "N_50": "M",
        },
        inplace=True,
    )

    if DRY_RUN is False:
        df.to_json(
            root_UCC_path + clusters_json,
            orient="records",
            indent=1,
            compression="gzip",
        )


if __name__ == "__main__":
    main()
