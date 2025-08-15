import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from modules.HARDCODED import (
    UCC_folder,
    UCC_members_file,
    articles_md_path,
    assets_folder,
    class_order,
    clusters_csv_path,
    databases_md_path,
    dbs_folder,
    dbs_tables_folder,
    images_folder,
    md_folder,
    name_DBs_json,
    pages_folder,
    parquet_dates,
    plots_folder,
    plots_sub_folders,
    tables_folder,
    tables_md_path,
    temp_fold,
)
from modules.update_site import ucc_entry, ucc_plots
from modules.update_site.main_files_updt import (
    count_N50membs,
    count_OCs_classes,
    count_shared_membs,
    ucc_n_total_updt,
    updt_articles_table,
    updt_C3_classif_main_table,
    updt_C3_classif_tables,
    updt_DBs_tables,
    updt_N50_main_table,
    updt_N50_tables,
    updt_OCs_per_quad_main_table,
    updt_OCs_per_quad_tables,
    updt_shared_membs_main_table,
    updt_shared_membs_tables,
)
from modules.utils import get_last_version_UCC, logger


def main():
    """ """
    logging = logger()

    # Get the latest version of the UCC catalogue
    UCC_last_version = get_last_version_UCC(UCC_folder)

    # Read paths
    (
        ucc_file_path,
        root_UCC_path,
        temp_gz_CSV_path,
        ucc_gz_CSV_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        ucc_tables_path,
        temp_tables_path,
        temp_entries_path,
        ucc_entries_path,
        temp_image_path,
        temp_data_date_path,
    ) = load_paths(UCC_last_version)

    # Load required files
    (
        df_UCC,
        df_UCC_edit,
        current_JSON,
        DBs_full_data,
        database_md,
        articles_md,
        tables_md,
        df_members,
        data_dates_json,
    ) = load_data(logging, ucc_file_path, root_UCC_path)

    # Update per cluster md and webp files. If no changes are expected, this step
    # can be skipped to save time
    if input("\nUpdate md files? (y/n): ").lower() == "y":
        updt_ucc_cluster_files(
            logging,
            root_UCC_path,
            ucc_entries_path,
            temp_entries_path,
            DBs_full_data,
            df_UCC,
            current_JSON,
        )
    if input("\nUpdate cluster plots? (y/n): ").lower() == "y":
        updt_ucc_cluster_plots(
            logging,
            root_UCC_path,
            df_UCC,
            df_members,
            data_dates_json,
            temp_data_date_path,
        )

    # Count number of OCs in each class
    OCs_per_class = count_OCs_classes(df_UCC["C3"], class_order)

    if input("\nUpdate main site plots? (y/n): ").lower() == "y":
        make_site_plots(logging, temp_image_path, df_UCC, OCs_per_class)

    # Mask with OCs with shared members
    shared_msk = count_shared_membs(df_UCC)

    # Mask with N50 members
    membs_msk = count_N50membs(df_UCC)

    N_members_UCC = len(df_members)

    # Update DATABASE, TABLES, ARTCILES .md files
    if input("\nUpdate 'DATABASE, TABLES, ARTICLES' files? (y/n): ").lower() == "y":
        update_main_pages(
            logging,
            N_members_UCC,
            current_JSON,
            df_UCC,
            database_md,
            articles_md,
            tables_md,
            OCs_per_class,
            membs_msk,
            shared_msk,
        )

    # Update tables files
    if input("\nUpdate individual tables files? (y/n): ").lower() == "y":
        updt_indiv_tables_files(
            logging,
            temp_dbs_tables_path,
            ucc_dbs_tables_path,
            ucc_tables_path,
            temp_tables_path,
            current_JSON,
            df_UCC_edit,
            membs_msk,
            shared_msk,
        )

    # Update CSV file
    if input("\nUpdate clusters CSV file? (y/n): ").lower() == "y":
        updt_cls_CSV(logging, ucc_gz_CSV_path, temp_gz_CSV_path, df_UCC_edit)

    if input("\nMove files to their final destination? (y/n): ").lower() == "y":
        move_files(logging, root_UCC_path, temp_data_date_path)
        logging.info("All files moved into place")

    # Check number of files
    N_UCC = len(df_UCC)
    file_checker(logging, N_UCC, root_UCC_path)

    logging.info("\nAll done!")


def load_paths(
    UCC_last_version: str,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]:
    """ """
    # Full path to the current UCC csv file
    ucc_file_path = UCC_folder + UCC_last_version

    # Root UCC path
    root_UCC_path = os.path.dirname(os.getcwd()) + "/"

    # UCC and temp path to compressed JSON file
    ucc_gz_CSV_path = root_UCC_path + clusters_csv_path
    temp_gz_CSV_path = temp_fold + clusters_csv_path

    # Temp path to the ucc assets and pages
    temp_assets_path = temp_fold + assets_folder
    if not os.path.exists(temp_assets_path):
        os.makedirs(temp_assets_path)
    temp_pages_path = temp_fold + pages_folder
    if not os.path.exists(temp_pages_path):
        os.makedirs(temp_pages_path)

    # Temp path to the ucc table files for each DB
    temp_dbs_tables_path = temp_fold + dbs_tables_folder
    if not os.path.exists(temp_dbs_tables_path):
        os.makedirs(temp_dbs_tables_path)
    # Root path to the ucc table files for each DB
    ucc_dbs_tables_path = root_UCC_path + dbs_tables_folder

    ucc_tables_path = root_UCC_path + tables_folder
    temp_tables_path = temp_fold + tables_folder

    # Temp path to the ucc cluster folder where each md entry is stored
    temp_entries_path = temp_fold + md_folder
    if not os.path.exists(temp_entries_path):
        os.makedirs(temp_entries_path)
    # Root path to the ucc cluster folder where each md entry is stored
    ucc_entries_path = root_UCC_path + md_folder

    # Temp path to ucc images folder
    temp_image_path = temp_fold + images_folder
    if not os.path.exists(temp_image_path):
        os.makedirs(temp_image_path)

    # Create temp quadrant folders for the plots
    for Nquad in range(1, 5):
        for lat in ("P", "N"):
            quad = "Q" + str(Nquad) + lat + "/"
            for fold in plots_sub_folders:
                out_path = temp_fold + quad + plots_folder + fold
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

    if not os.path.exists(temp_fold + UCC_folder):
        os.makedirs(temp_fold + UCC_folder)
    # Temporary path for updated data dates file
    temp_data_date_path = temp_fold + UCC_folder + parquet_dates

    return (
        ucc_file_path,
        root_UCC_path,
        temp_gz_CSV_path,
        ucc_gz_CSV_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        ucc_tables_path,
        temp_tables_path,
        temp_entries_path,
        ucc_entries_path,
        temp_image_path,
        temp_data_date_path,
    )


def load_data(
    logging, ucc_file_path: str, root_UCC_path: str
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, str, str, str, pd.DataFrame, dict
]:
    """ """

    # Current UCC catalogue
    df_UCC = pd.read_csv(ucc_file_path)
    logging.info(f"UCC {ucc_file_path} loaded (N={len(df_UCC)})")

    # Prepare df_UCC to be used in the updating of the table files
    df_UCC_edit = updt_UCC(df_UCC)

    # Load clusters data in JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    # Read the data for every DB in the UCC as a pandas DataFrame
    DBs_full_data = {}
    for k, v in current_JSON.items():
        DBs_full_data[k] = pd.read_csv(dbs_folder + k + ".csv")

    # Load DATABASE.md file
    with open(root_UCC_path + databases_md_path) as file:
        database_md = file.read()

    # Load ARTICLES.md file
    with open(root_UCC_path + articles_md_path) as file:
        articles_md = file.read()

    # Load TABLES.md file
    with open(root_UCC_path + tables_md_path) as file:
        tables_md = file.read()

    # Load current members file
    zenodo_members_file = UCC_folder + UCC_members_file
    df_members = pd.read_parquet(zenodo_members_file)

    # Load JSON file with last updated dates
    fname_json = UCC_folder + parquet_dates
    with open(fname_json, "r") as file:
        data_dates_json = json.load(file)

    return (
        df_UCC,
        df_UCC_edit,
        current_JSON,
        DBs_full_data,
        database_md,
        articles_md,
        tables_md,
        df_members,
        data_dates_json,
    )


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
    df["RA_ICRS"] = np.round(df_UCC["RA_ICRS"], 2)
    df["DE_ICRS"] = np.round(df_UCC["DE_ICRS"], 2)
    df["GLON"] = np.round(df_UCC["GLON"], 2)
    df["GLAT"] = np.round(df_UCC["GLAT"], 2)

    # Compute parallax-based distances in parsecs
    dist_pc = 1000 / np.clip(np.array(df["Plx_m"]), a_min=0.0000001, a_max=np.inf)
    dist_pc = np.clip(dist_pc, a_min=10, a_max=50000)
    df["dist_pc"] = np.round(dist_pc, 0)

    return df


def updt_ucc_cluster_files(
    logging,
    root_UCC_path,
    ucc_entries_path,
    temp_entries_path,
    DBs_full_data,
    df_UCC,
    current_JSON,
):
    """ """
    logging.info("\nGenerating md files")

    fnames_all = [_.split(";")[0] for _ in df_UCC["fnames"]]

    N_total = 0
    # Iterate trough each entry in the UCC database
    for i_ucc, UCC_cl in df_UCC.iterrows():
        fname0 = str(UCC_cl["fnames"]).split(";")[0]

        Qfold = UCC_cl["quad"] + "/"

        txt = f"{Qfold}{fname0}: "

        # Make catalogue entry
        txt_e = make_entry(
            root_UCC_path,
            ucc_entries_path,
            temp_entries_path,
            plots_folder,
            Qfold,
            df_UCC,
            fnames_all,
            UCC_cl,
            current_JSON,
            DBs_full_data,
            fname0,
        )

        if txt_e != "":
            N_total += 1
            logging.info(f"{N_total} -> " + txt + txt_e + f" ({i_ucc})")

    logging.info(f"\nN={N_total} OCs processed")


def make_entry(
    root_UCC_path,
    ucc_entries_path,
    temp_entries_path,
    plots_folder,
    Qfold,
    df_UCC,
    fnames_all,
    UCC_cl,
    current_JSON,
    DBs_full_data,
    fname0,
) -> str:
    """ """
    # Generate table with positional data: (ra, dec, plx, pmra, pmde, Rv)
    posit_table = ucc_entry.positions_in_lit(current_JSON, DBs_full_data, UCC_cl)

    # Generate CMD image carousel
    cl_name = UCC_cl["ID"].split(";")[0]
    img_cont = ucc_entry.carousel_div(
        root_UCC_path, plots_folder, cl_name, Qfold, fname0
    )

    # Generate fundamental parameters table
    fpars_table = ucc_entry.fpars_in_lit(
        current_JSON, DBs_full_data, UCC_cl["DB"], UCC_cl["DB_i"]
    )

    # Generate table with OCs that share members with this one
    shared_table = ucc_entry.table_shared_members(df_UCC, fnames_all, UCC_cl)

    # Get colors used by the 'CX' classification
    abcd_c = ucc_entry.UCC_color(UCC_cl["C3"])

    # Generate full entry
    new_md_entry = ucc_entry.make(
        UCC_cl, fname0, Qfold, posit_table, img_cont, abcd_c, fpars_table, shared_table
    )

    # Compare old md file (if it exists) with the new md file, for this cluster
    try:
        # Read old entry
        with open(ucc_entries_path + fname0 + ".md", "r") as f:
            old_md_entry = f.read()

        # Check if entry needs updating
        if new_md_entry != old_md_entry:
            txt = "md updated |"
            # with open("OLD.md", "w") as f:
            #     f.write(old_md_entry_no_date)
            # with open("NEW.md", "w") as f:
            #     f.write(new_md_entry_no_date)
            # breakpoint()
        else:
            # The existing md file has not changed
            txt = ""

    except FileNotFoundError:
        # This is a new OC with no md entry yet
        txt = "md generated |"

    if txt != "":
        # Generate/update entry
        with open(temp_entries_path + fname0 + ".md", "w") as f:
            f.write(new_md_entry)

    return txt


def updt_ucc_cluster_plots(
    logging, root_UCC_path, df_UCC, df_members, data_dates_json, temp_data_date_path
):
    """ """
    logging.info("\nGenerating plot files")
    N_total = 0
    # Iterate trough each entry in the UCC database
    for i_ucc, UCC_cl in df_UCC.iterrows():
        fname0 = str(UCC_cl["fnames"]).split(";")[0]

        Qfold = UCC_cl["quad"] + "/"

        txt = f"{Qfold}{fname0}: "

        # Make plots
        txt_p, data_dates_json = make_plots(
            root_UCC_path,
            temp_fold,
            Qfold,
            plots_folder,
            UCC_cl,
            fname0,
            df_members,
            data_dates_json,
        )

        if txt_p != "":
            N_total += 1
            logging.info(f"{N_total} -> " + txt + txt_p + f" ({i_ucc})")

    # Save temp file with updated dates to temp folder
    with open(temp_data_date_path, "w") as file:
        json.dump(data_dates_json, file, indent=2)

    logging.info(f"\nN={N_total} OCs processed")


def make_plots(
    root_UCC_path,
    temp_fold,
    Qfold,
    plots_folder,
    UCC_cl,
    fname0,
    df_members,
    data_dates_json,
) -> tuple[str, dict]:
    """
    Make CMD and Aladin plots.
    """
    txt = ""

    # Make Aladin plot if image files does not exist. These images are not updated
    # Path to original image (if it exists)
    orig_aladin_path = (
        root_UCC_path + Qfold + plots_folder + "aladin/" + fname0 + ".webp"
    )
    if Path(orig_aladin_path).is_file() is False:
        # Path were new image will be stored
        save_plot_file = temp_fold + Qfold + plots_folder + "aladin/" + fname0 + ".webp"
        ucc_plots.plot_aladin(
            UCC_cl["RA_ICRS_m"],
            UCC_cl["DE_ICRS_m"],
            UCC_cl["r_50"],
            save_plot_file,
        )
        txt += " Aladin plot generated |"

    # Make CMD plot
    # Check if the data for this OC new and the plot should be updated
    if "USED" not in data_dates_json[fname0]:
        # Read members
        df_membs = df_members[df_members["name"] == fname0]
        # Temp path were the image will be stored
        temp_plot_file = temp_fold + Qfold + plots_folder + "UCC/" + fname0 + ".webp"
        ucc_plots.plot_CMD(temp_plot_file, df_membs)
        txt += " CMD plot generated |"
        # Update file with current date of usage
        data_dates_json[fname0] = (
            f"USED ({datetime.datetime.now().strftime('%y%m%d%H')})"
        )

    return txt, data_dates_json


def make_site_plots(logging, temp_image_path, df_UCC, OCs_per_class):
    """ """
    ucc_plots.make_N_vs_year_plot(temp_image_path + "catalogued_ocs.webp", df_UCC)
    logging.info("Plot generated: number of OCs vs years")

    ucc_plots.make_classif_plot(
        temp_image_path + "classif_bar.webp", OCs_per_class, class_order
    )
    logging.info("Plot generated: classification histogram")


def update_main_pages(
    logging,
    N_members_UCC,
    current_JSON,
    df_UCC,
    database_md,
    articles_md,
    tables_md,
    OCs_per_class,
    membs_msk,
    shared_msk,
):
    """Update DATABASE, TABLES, ARTICLES .md files"""
    logging.info("\nUpdating DATABASE, TABLES, ARTICLES .md files")

    # Update DATABASE
    N_db_UCC, N_cl_UCC = len(current_JSON), len(df_UCC)
    # Update the total number of entries, databases, and members in the UCC
    database_md_updt = ucc_n_total_updt(
        logging, N_db_UCC, N_cl_UCC, N_members_UCC, database_md
    )
    with open(temp_fold + databases_md_path, "w") as file:
        file.write(database_md_updt)
    if database_md != database_md_updt:
        logging.info("DATABASE.md updated")
    else:
        logging.info("DATABASE.md not updated (no changes)")

    # Update TABLES
    tables_md_updt = updt_C3_classif_main_table(class_order, OCs_per_class, tables_md)
    tables_md_updt = updt_OCs_per_quad_main_table(df_UCC, tables_md_updt)
    tables_md_updt = updt_shared_membs_main_table(shared_msk, tables_md_updt)
    tables_md_updt = updt_N50_main_table(membs_msk, tables_md_updt)
    with open(temp_fold + tables_md_path, "w") as file:
        file.write(tables_md_updt)
    if tables_md != tables_md_updt:
        logging.info("TABLES.md updated")
    else:
        logging.info("TABLES.md not updated (no changes)")

    # Update ARTICLES
    articles_md_updt = updt_articles_table(df_UCC, current_JSON, articles_md)
    with open(temp_fold + articles_md_path, "w") as file:
        file.write(articles_md_updt)
    if articles_md != articles_md_updt:
        logging.info("ARTICLES.md updated")
    else:
        logging.info("ARTICLES.md not updated (no changes)")


def updt_indiv_tables_files(
    logging,
    temp_dbs_tables_path,
    ucc_dbs_tables_path,
    ucc_tables_path,
    temp_tables_path,
    current_JSON,
    df_UCC_edit,
    membs_msk,
    shared_msk,
):
    """ """
    logging.info("\nUpdating individual tables")

    # # TODO: radius in parsec, unused yet (24/12/04)
    # pc_rad = pc_radius(df_UCC["r_50"].values, df_UCC["Plx_m"].values)

    # Update pages for individual databases
    new_tables_dict = updt_DBs_tables(current_JSON, df_UCC_edit)
    general_table_update(
        logging, ucc_dbs_tables_path, temp_dbs_tables_path, new_tables_dict
    )

    # Update page with N members
    new_tables_dict = updt_N50_tables(df_UCC_edit, membs_msk)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    #
    new_tables_dict = updt_C3_classif_tables(df_UCC_edit, class_order)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    #
    new_tables_dict = updt_shared_membs_tables(df_UCC_edit, shared_msk)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    #
    new_tables_dict = updt_OCs_per_quad_tables(df_UCC_edit)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)


def general_table_update(
    logging, root_path: str, temp_path: str, new_tables_dict: dict
) -> None:
    """
    Updates a markdown table file if the content has changed.
    """
    for table_name, new_table in new_tables_dict.items():
        # Read old entry, if any
        try:
            with open(root_path + table_name + "_table.md", "r") as f:
                old_table = f.read()
        except FileNotFoundError:
            # This is a new table with no md entry yet
            old_table = ""

        # Write to file if any changes are detected
        if old_table != new_table:
            with open(temp_path + table_name + "_table.md", "w") as file:
                file.write(new_table)
            txt = "updated" if old_table != "" else "generated"
            logging.info(f"Table {table_name} {txt}")


def updt_cls_CSV(
    logging, ucc_gz_CSV_path: str, temp_gz_CSV_path: str, df_tables: pd.DataFrame
) -> None:
    """
    Update compressed cluster.csv file used by 'ucc.ar' search
    """
    df = pd.DataFrame(
        df_tables[
            [
                "ID",
                "fnames",
                "RA_ICRS",
                "DE_ICRS",
                "GLON",
                "GLAT",
                "dist_pc",
                "N_50",
                "r_50",
                "C3",
            ]
        ]
    )
    df_new = df.sort_values("ID")

    # Load the current compressed CSV file
    df_old = pd.read_csv(ucc_gz_CSV_path, compression="gzip")

    # Update CSV if required
    if not df_old.equals(df_new):
        df.to_csv(
            temp_gz_CSV_path,
            index=False,
            compression="gzip",
        )
        logging.info("File 'clusters.csv.gz' updated")
    else:
        logging.info("File 'cluster.csv.gz' not updated (no changes)")


def move_files(
    logging,
    root_UCC_path,
    temp_data_date_path,
) -> None:
    """ """
    logging.info("\nUpdate files:")

    # Move all files inside 'temp_fold/ucc/'
    temp_ucc_fold = temp_fold + "ucc/"
    for root, dirs, files in os.walk(temp_ucc_fold):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ucc = file_path.replace(temp_fold, root_UCC_path)
            if "_clusters" not in root:
                logging.info(file_path + " --> " + file_ucc)
            os.rename(file_path, file_ucc)
        if "_clusters" in root:
            file_path = root + "/*.md"
            file_ucc = file_path.replace(temp_fold, root_UCC_path)
            logging.info(file_path + " --> " + file_ucc)
    logging.info("")

    # Move the 'temp_fold/zenodo/data_dates.json' file
    if os.path.exists(temp_data_date_path):
        os.rename(temp_data_date_path, UCC_folder + parquet_dates)
        logging.info(temp_data_date_path + " --> " + UCC_folder + parquet_dates)
        logging.info("")

    # Move all plots
    all_plot_folds = [[], []]
    for qN in range(1, 5):
        for lat in ("P", "N"):
            qfold = "Q" + str(qN) + lat + "/"
            qplots_fold = qfold + plots_folder
            for fold in plots_sub_folders:
                temp_fpath = temp_fold + qplots_fold + fold + "/"
                fpath = root_UCC_path + qplots_fold + fold + "/"
                # Check if folder exists
                if os.path.exists(temp_fpath):
                    # For every file in this folder
                    for file in os.listdir(temp_fpath):
                        plot_temp = temp_fpath + file
                        plot_stored = fpath + file
                        all_plot_folds[0].append(temp_fpath)
                        all_plot_folds[1].append(fpath)
                        os.rename(plot_temp, plot_stored)
    all_plot_folds[0] = list(set(all_plot_folds[0]))
    all_plot_folds[1] = list(set(all_plot_folds[1]))

    for i, plot_temp in enumerate(all_plot_folds[0]):
        plot_stored = all_plot_folds[1][i]
        logging.info(plot_temp + "/*.webp" + " --> " + plot_stored + "/*.webp")


def file_checker(
    logging,
    N_UCC: int,
    root_UCC_fold: str,
) -> None:
    """Check the number and types of files in directories for consistency.

    Parameters:
    - logging: Logger instance for recording messages.

    Returns:
    - None
    """
    logging.info(f"\nChecking number of files against N_UCC={N_UCC}\n")
    flag_error = False

    N_all = {}
    for qN in range(1, 5):
        for lat in ("P", "N"):
            qfold = "Q" + str(qN) + lat + "/"
            N_all[qfold], N_extra = {}, 0
            for fold in plots_sub_folders:
                qplots_fold = root_UCC_fold + qfold + plots_folder + fold
                N_all[qfold][fold] = len(os.listdir(qplots_fold))
                N_extra += sum(not f.endswith(".webp") for f in os.listdir(qplots_fold))
            N_all[qfold]["extra"] = N_extra

    Ntot = {_: 0 for _ in plots_sub_folders}
    Ntot["extra"] = 0
    for qf, vals in N_all.items():
        mark = "V"
        txt = f"{qf} --> " + "; ".join(f"{k}: {v}" for k, v in vals.items())
        if vals["extra"] > 0 or len(set(vals[_] for _ in plots_sub_folders)) > 1:
            mark, flag_error = "X", True
        logging.info(f"{txt} <-- {mark}")
        Ntot["extra"] += vals["extra"]
        for k, v in vals.items():
            Ntot[k] += v
    logging.info("\nTotal  --> " + "; ".join(f"{k}: {v}" for k, v in Ntot.items()))

    # Check .md files
    clusters_md_fold = root_UCC_fold + md_folder
    NT_md = len(os.listdir(clusters_md_fold))
    NT_extra = NT_md - N_UCC
    mark = "V" if (NT_extra == 0) else "X"
    logging.info("\nN_UCC   md      extra")
    logging.info(f"{N_UCC}   {NT_md}   {NT_extra}     <-- {mark}")
    if mark == "X":
        flag_error = True

    if flag_error:
        raise ValueError("The file check was unsuccessful")


if __name__ == "__main__":
    main()
