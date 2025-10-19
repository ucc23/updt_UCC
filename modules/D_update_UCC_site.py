import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .D_funcs import ucc_entry, ucc_plots
from .D_funcs.main_files_updt import (
    UTI_ranges,
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
    updt_shared_membs_main_table,
    updt_shared_membs_tables,
    updt_UTI_main_table,
    updt_UTI_tables,
)
from .utils import logger
from .variables import (
    UCC_members_file,
    articles_md_path,
    assets_folder,
    class_order,
    clusters_csv_path,
    data_folder,
    databases_md_path,
    dbs_folder,
    dbs_tables_folder,
    images_folder,
    md_folder,
    merged_dbs_file,
    name_DBs_json,
    pages_folder,
    plots_folder,
    plots_sub_folders,
    root_ucc_path,
    tables_folder,
    tables_md_path,
    temp_folder,
    ucc_cat_file,
    ucc_path,
    zenodo_folder,
)


def main():
    """ """
    logging = logger()

    # Read paths
    (
        temp_gz_CSV_path,
        ucc_gz_CSV_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        ucc_tables_path,
        temp_tables_path,
        temp_entries_path,
        ucc_entries_path,
        temp_image_path,
    ) = load_paths()

    cols_from_B_to_C = ["Names", "DB", "DB_i"]

    # Load required files
    (
        df_UCC,
        current_JSON,
        DBs_full_data,
        database_md,
        articles_md,
        tables_md,
        df_members,
    ) = load_data(logging, cols_from_B_to_C)

    # Update per cluster md files. If no changes are expected, this step can be skipped
    if input("\nUpdate md files? (y/n): ").lower() == "y":
        updt_ucc_cluster_files(
            logging,
            ucc_entries_path,
            temp_entries_path,
            DBs_full_data,
            df_UCC,
            current_JSON,
        )

    # Update per cluster webp files. If no changes are expected, this step can be skipped
    if input("\nUpdate cluster plots? (y/n): ").lower() == "y":
        # Returns dataframe with 'plot_used' column updated
        df_UCC_updated = updt_ucc_cluster_plots(
            logging,
            df_UCC,
            df_members,
        )
        # Update and save updated UCC catalog file if any plots were generated
        if df_UCC_updated.empty is False:
            temp_C_path = temp_folder + data_folder + ucc_cat_file
            # Drop added columns from B
            df_UCC_C_new = df_UCC_updated.drop(columns=cols_from_B_to_C)
            df_UCC_C_new.to_csv(
                temp_C_path,
                na_rep="nan",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info(f"\nFile '{temp_C_path}' updated")

    # Count number of OCs in each class
    OCs_per_class = count_OCs_classes(df_UCC["C3"], class_order)

    if input("\nUpdate main site plots? (y/n): ").lower() == "y":
        make_site_plots(logging, temp_image_path, df_UCC, OCs_per_class)

    # Mask with OCs with shared members
    shared_msk = count_shared_membs(df_UCC)

    UTI_msk = UTI_ranges(df_UCC)
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
            UTI_msk,
            membs_msk,
            shared_msk,
        )

    df_UCC_edit = updt_UCC(df_UCC)

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
            UTI_msk,
            membs_msk,
            shared_msk,
        )

    # Update CSV file
    if input("\nUpdate clusters CSV file? (y/n): ").lower() == "y":
        updt_cls_CSV(logging, ucc_gz_CSV_path, temp_gz_CSV_path, df_UCC_edit)

    if input("\nMove files to their final destination? (y/n): ").lower() == "y":
        move_files(logging)

    # Check number of files
    file_checker(logging)
    logging.info("\nAll done!")


def load_paths() -> tuple[
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
    # UCC and temp path to compressed JSON file
    ucc_gz_CSV_path = root_ucc_path + clusters_csv_path
    temp_gz_CSV_path = temp_folder + clusters_csv_path

    # Temp path to the ucc assets and pages
    temp_assets_path = temp_folder + assets_folder
    if not os.path.exists(temp_assets_path):
        os.makedirs(temp_assets_path)
    temp_pages_path = temp_folder + pages_folder
    if not os.path.exists(temp_pages_path):
        os.makedirs(temp_pages_path)

    # Temp path to the ucc table files for each DB
    temp_dbs_tables_path = temp_folder + dbs_tables_folder
    if not os.path.exists(temp_dbs_tables_path):
        os.makedirs(temp_dbs_tables_path)
    # Root path to the ucc table files for each DB
    ucc_dbs_tables_path = root_ucc_path + dbs_tables_folder

    ucc_tables_path = root_ucc_path + tables_folder
    temp_tables_path = temp_folder + tables_folder

    # Temp path to the ucc cluster folder where each md entry is stored
    temp_entries_path = temp_folder + md_folder
    if not os.path.exists(temp_entries_path):
        os.makedirs(temp_entries_path)
    # Root path to the ucc cluster folder where each md entry is stored
    ucc_entries_path = root_ucc_path + md_folder

    # Temp path to ucc images folder
    temp_image_path = temp_folder + images_folder
    if not os.path.exists(temp_image_path):
        os.makedirs(temp_image_path)

    # Create temp folders using all letters in the alphabet
    for letter in "abcdefghijklmnopqrstuvwxyz":
        for fold in plots_sub_folders:
            out_path = temp_folder + plots_folder + f"plots_{letter}/" + fold
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    if not os.path.exists(temp_folder + data_folder):
        os.makedirs(temp_folder + data_folder)

    return (
        temp_gz_CSV_path,
        ucc_gz_CSV_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        ucc_tables_path,
        temp_tables_path,
        temp_entries_path,
        ucc_entries_path,
        temp_image_path,
    )


def load_data(
    logging,
    cols_from_B_to_C,
) -> tuple[pd.DataFrame, dict, dict, str, str, str, pd.DataFrame]:
    """ """

    # Load current CSV data files
    ucc_B_file = data_folder + merged_dbs_file
    ucc_C_file = data_folder + ucc_cat_file
    df_UCC_B = pd.read_csv(ucc_B_file)
    logging.info(f"\nFile {ucc_B_file} loaded ({len(df_UCC_B)} entries)")
    df_UCC_C = pd.read_csv(
        ucc_C_file,
        dtype={
            "frame_limit": "string",
            "shared_members": "string",
            "shared_members_p": "string",
        },
    )
    # Replace NaN values with "nan" string only in selected columns
    selected_columns = ["frame_limit", "shared_members", "shared_members_p"]
    df_UCC_C[selected_columns] = df_UCC_C[selected_columns].fillna("nan")

    logging.info(f"File {ucc_C_file} loaded ({len(df_UCC_C)} entries)")
    # Check the 'fnames' columns in df_UCC_B and df_UCC_C_final dataframes are equal
    if not df_UCC_B["fnames"].equals(df_UCC_C["fnames"]):
        raise ValueError("The 'fnames' columns in B and final C dataframes differ")
    # Add required columns to df_UCC_C
    df_UCC = df_UCC_C.copy()
    df_UCC[cols_from_B_to_C] = df_UCC_B[cols_from_B_to_C]

    # Load clusters data in JSON file
    with open(name_DBs_json) as f:
        current_JSON = json.load(f)

    # Read the data for every DB in the UCC as a pandas DataFrame
    DBs_full_data = {}
    for k, v in current_JSON.items():
        DBs_full_data[k] = pd.read_csv(dbs_folder + k + ".csv")

    # Load DATABASE.md file
    with open(root_ucc_path + databases_md_path) as file:
        database_md = file.read()

    # Load ARTICLES.md file
    with open(root_ucc_path + articles_md_path) as file:
        articles_md = file.read()

    # Load TABLES.md file
    with open(root_ucc_path + tables_md_path) as file:
        tables_md = file.read()

    # Load current members file
    zenodo_members_file = zenodo_folder + UCC_members_file
    df_members = pd.read_parquet(zenodo_members_file)

    return (
        df_UCC,
        current_JSON,
        DBs_full_data,
        database_md,
        articles_md,
        tables_md,
        df_members,
    )


def updt_ucc_cluster_files(
    logging,
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

        # Generate full entry
        new_md_entry = ucc_entry.make(
            current_JSON, DBs_full_data, df_UCC, UCC_cl, fnames_all, fname0
        )

        # Compare old md file (if it exists) with the new md file, for this cluster
        txt = ""
        try:
            # Read old entry
            with open(ucc_entries_path + fname0 + ".md", "r") as f:
                old_md_entry = f.read()
            # Check if entry needs updating
            if new_md_entry != old_md_entry:
                txt = "md updated |"
        except FileNotFoundError:
            # This is a new OC with no md entry yet
            txt = "md generated |"

        if txt != "":
            # Generate/update entry
            with open(temp_entries_path + fname0 + ".md", "w") as f:
                f.write(new_md_entry)
            N_total += 1
            logging.info(f"{N_total} -> {fname0}: " + txt + f" ({i_ucc})")

    logging.info(f"\nN={N_total} OCs processed")


def updt_ucc_cluster_plots(logging, df_UCC, df_members) -> pd.DataFrame:
    """ """
    logging.info("\nGenerating plot files")
    N_total = 0
    # Iterate trough each entry in the UCC database
    for i_ucc, UCC_cl in df_UCC.iterrows():
        fname0 = str(UCC_cl["fnames"]).split(";")[0]
        txt = ""

        # Make Aladin plot if image files does not exist. These images are not updated
        # Path to original image (if it exists)
        orig_aladin_path = (
            f"{root_ucc_path}{plots_folder}plots_{fname0[0]}/aladin/{fname0}.webp"
        )
        if Path(orig_aladin_path).is_file() is False:
            # Path were new image will be stored
            save_plot_file = (
                f"{temp_folder}{plots_folder}plots_{fname0[0]}/aladin/{fname0}.webp"
            )
            ucc_plots.plot_aladin(
                UCC_cl["RA_ICRS_m"],
                UCC_cl["DE_ICRS_m"],
                UCC_cl["r_50"],
                save_plot_file,
            )
            txt += " Aladin plot generated |"

        # Make CMD plot
        # Check if this OC's plot should be updated or generated
        if UCC_cl["plot_used"] == "n":
            # Read members
            df_membs = df_members[df_members["name"] == fname0]
            # Temp path were the image will be stored
            temp_plot_file = (
                f"{temp_folder}{plots_folder}plots_{fname0[0]}/UCC/{fname0}.webp"
            )
            ucc_plots.plot_CMD(temp_plot_file, df_membs)
            txt += " CMD plot generated |"
            # Update value indicating that the plot was generated
            df_UCC.loc[i_ucc, "plot_used"] = "y"

        if txt != "":
            N_total += 1
            logging.info(f"{N_total} -> {fname0}" + txt + f" ({i_ucc})")

    logging.info(f"\nN={N_total} OCs processed")

    if N_total > 0:
        return df_UCC.copy()
    else:
        return pd.DataFrame([])


def make_site_plots(logging, temp_image_path, df_UCC, OCs_per_class):
    """ """
    ucc_plots.make_N_vs_year_plot(temp_image_path + "catalogued_ocs.webp", df_UCC)
    logging.info("Plot generated: number of OCs vs years")

    ucc_plots.make_classif_plot(
        temp_image_path + "classif_bar.webp", OCs_per_class, class_order
    )
    logging.info("Plot generated: classification histogram")

    ucc_plots.make_UTI_plot(temp_image_path + "UTI_values.webp", df_UCC["UTI"])
    logging.info("Plot generated: UTI histogram")


def update_main_pages(
    logging,
    N_members_UCC,
    current_JSON,
    df_UCC,
    database_md,
    articles_md,
    tables_md,
    OCs_per_class,
    UTI_msk,
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
    with open(temp_folder + databases_md_path, "w") as file:
        file.write(database_md_updt)
    if database_md != database_md_updt:
        logging.info("DATABASE.md updated")
    else:
        logging.info("DATABASE.md not updated (no changes)")

    # Update TABLES
    tables_md_updt = updt_UTI_main_table(UTI_msk, tables_md)
    tables_md_updt = updt_C3_classif_main_table(
        class_order, OCs_per_class, tables_md_updt
    )
    # tables_md_updt = updt_OCs_per_quad_main_table(df_UCC, tables_md_updt)
    tables_md_updt = updt_shared_membs_main_table(shared_msk, tables_md_updt)
    tables_md_updt = updt_N50_main_table(membs_msk, tables_md_updt)
    with open(temp_folder + tables_md_path, "w") as file:
        file.write(tables_md_updt)
    if tables_md != tables_md_updt:
        logging.info("TABLES.md updated")
    else:
        logging.info("TABLES.md not updated (no changes)")

    # Update ARTICLES
    articles_md_updt = updt_articles_table(df_UCC, current_JSON, articles_md)
    with open(temp_folder + articles_md_path, "w") as file:
        file.write(articles_md_updt)
    if articles_md != articles_md_updt:
        logging.info("ARTICLES.md updated")
    else:
        logging.info("ARTICLES.md not updated (no changes)")


def updt_UCC(df_UCC: pd.DataFrame) -> pd.DataFrame:
    """
    Updates a DataFrame of astronomical cluster data by processing identifiers,
    coordinates, URLs, and derived quantities such as distances.
    """
    df = df_UCC.copy()

    # Extract the first identifier from the "ID" column
    df["Name"] = [_.split(";")[0] for _ in df_UCC["Names"]]

    # Generate URLs for names
    names_url = []
    for i, cl in df.iterrows():
        name = str(cl["Name"]).split(";")[0]
        fname = str(cl["fnames"]).split(";")[0]
        url = "/_clusters/" + fname + "/"
        names_url.append(f"[{name}]({url})")
    df["ID_url"] = names_url

    # Round coordinate columns
    df["RA_ICRS"] = np.round(df_UCC["RA_ICRS_m"], 2)
    df["DE_ICRS"] = np.round(df_UCC["DE_ICRS_m"], 2)
    df["GLON"] = np.round(df_UCC["GLON_m"], 2)
    df["GLAT"] = np.round(df_UCC["GLAT_m"], 2)

    # Compute parallax-based distances in parsecs
    dist_pc = 1000 / np.clip(np.array(df["Plx_m"]), a_min=0.0000001, a_max=np.inf)
    dist_pc = np.clip(dist_pc, a_min=10, a_max=50000)
    df["dist_pc"] = np.round(dist_pc, 0)

    df["Plx_m_round"] = np.round(df["Plx_m"], 2)
    df["C3_abcd"] = [ucc_entry.color_C3(_) for _ in df["C3"]]

    return df


def updt_indiv_tables_files(
    logging,
    temp_dbs_tables_path,
    ucc_dbs_tables_path,
    ucc_tables_path,
    temp_tables_path,
    current_JSON,
    df_UCC_edit,
    UTI_msk,
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
    new_tables_dict = updt_UTI_tables(df_UCC_edit, UTI_msk)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

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
    # new_tables_dict = updt_OCs_per_quad_tables(df_UCC_edit)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)


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
    logging, ucc_gz_CSV_path: str, temp_gz_CSV_path: str, df_UCC_edit: pd.DataFrame
) -> None:
    """
    Update compressed cluster.csv file used by 'ucc.ar' search
    """
    df = pd.DataFrame(
        df_UCC_edit[
            [
                "Name",
                "fnames",
                "RA_ICRS",
                "DE_ICRS",
                "GLON",
                "GLAT",
                "dist_pc",
                "N_50",
                "r_50",
                "C3",
                "UTI",
            ]
        ]
    )
    df_new = df.sort_values("Name")

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
) -> None:
    """ """
    logging.info("\nMoving files:")

    # Move updated C file
    final_C_path = data_folder + ucc_cat_file
    temp_C_path = temp_folder + final_C_path
    if os.path.exists(temp_C_path):
        os.rename(temp_C_path, final_C_path)
        logging.info(temp_C_path + " --> " + final_C_path)

    # Move all files inside temporary 'ucc/'
    temp_ucc_fold = temp_folder + ucc_path
    for root, dirs, files in os.walk(temp_ucc_fold):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ucc = file_path.replace(temp_folder, root_ucc_path)
            if "_clusters" not in root:
                logging.info(file_path + " --> " + file_ucc)
            os.rename(file_path, file_ucc)
        if "_clusters" in root:
            file_path = root + "/*.md"
            file_ucc = file_path.replace(temp_folder, root_ucc_path)
            logging.info(file_path + " --> " + file_ucc)
    logging.info("")

    # Move all plots
    all_plot_folds = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        letter_fold = plots_folder + f"plots_{letter}/"
        for fold in plots_sub_folders:
            temp_fpath = temp_folder + letter_fold + fold + "/"
            fpath = root_ucc_path + letter_fold + fold + "/"
            # Check if folder exists
            if os.path.exists(temp_fpath):
                # For every file in this folder
                for file in os.listdir(temp_fpath):
                    all_plot_folds.append(fpath)
                    os.rename(temp_fpath + file, fpath + file)
    unq_plot_folds = list(set(all_plot_folds))
    for plot_stored in unq_plot_folds:
        N = all_plot_folds.count(plot_stored)
        logging.info(
            plot_stored.replace(temp_folder, root_ucc_path)
            + "*.webp"
            + " --> "
            + plot_stored
            + f"*.webp (N={N})"
        )

    logging.info("\nAll files moved into place")


def file_checker(logging) -> None:
    """Check the number and types of files in directories for consistency.

    Parameters:
    - logging: Logger instance for recording messages.

    Returns:
    - None
    """
    logging.info("\nChecking files\n")
    # Read stored final version
    df_UCC_C = pd.read_csv(data_folder + ucc_cat_file, usecols=["fnames", "plot_used"])
    flag_error = False

    # Check that all entries in df_UCC_C have plot_used='y'
    if any(df_UCC_C["plot_used"] == "n"):
        logging.warning("Some entries in final C dataframe still have plot_used='n'\n")

    # Check the 'fnames' columns in df_UCC_B and df_UCC_C_final dataframes are equal
    df_UCC_B = pd.read_csv(data_folder + merged_dbs_file, usecols=["fnames"])
    df_UCC_fnames = df_UCC_C["fnames"].to_list()
    if not df_UCC_B["fnames"].to_list() == df_UCC_fnames:
        flag_error = True
        logging.warning("The 'fnames' columns in B and final C dataframes differ\n")

    ucc_fnames = sorted([_.split(";")[0] for _ in df_UCC_fnames])

    # Check that all md_files match the elements in df_UCC_fnames
    md_files = os.listdir(root_ucc_path + md_folder)
    md_fnames = sorted([_[:-3] for _ in md_files])
    # Print to screen which elements are different in both lists
    for f in md_fnames:
        if f not in ucc_fnames:
            logging.warning(f"{f} (.md) not in UCC catalog")
            flag_error = True
    for f in ucc_fnames:
        if f not in md_fnames:
            logging.warning(f"{f} (UCC) not in md files")
            flag_error = True
    logging.info("")

    for letter in "abcdefghijklmnopqrstuvwxyz":
        # Count how many elements in df_UCC_fnames start with this letter
        ucc_webp = [_ for _ in ucc_fnames if _.startswith(letter)]
        for fold in plots_sub_folders:
            letter_fold = root_ucc_path + plots_folder + f"plots_{letter}/" + fold
            if os.path.isdir(letter_fold):
                subf_webp = [_[:-5] for _ in os.listdir(letter_fold)]
                for f in subf_webp:
                    if f not in ucc_webp:
                        logging.warning(f"{fold}/{f}.webp not in UCC catalog")
                        flag_error = True
                for f in ucc_webp:
                    if f not in subf_webp:
                        logging.warning(f"{f}(.webp) not in {fold} folder")
                        flag_error = True

    if flag_error:
        raise ValueError("\nErrors were detected associated to the files")


if __name__ == "__main__":
    main()
