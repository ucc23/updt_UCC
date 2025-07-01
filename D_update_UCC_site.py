import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from modules.HARDCODED import (
    UCC_folder,
    assets_folder,
    clusters_csv_path,
    databases_md_path,
    dbs_folder,
    dbs_tables_folder,
    images_folder,
    md_folder,
    members_folder,
    name_DBs_json,
    pages_folder,
    parquet_dates,
    plots_folder,
    temp_fold,
)
from modules.update_site import ucc_entry, ucc_plots
from modules.update_site.main_files_updt import (
    count_N_members_UCC,
    ucc_n_total_updt,
    updt_cats_used,
    updt_DBs_tables,
)
from modules.utils import file_checker, get_last_version_UCC, logger

# # Order used for the C3 classes
# class_order = [
#     "AA",
#     "AB",
#     "BA",
#     "AC",
#     "CA",
#     "BB",
#     "AD",
#     "DA",
#     "BC",
#     "CB",
#     "BD",
#     "DB",
#     "CC",
#     "CD",
#     "DC",
#     "DD",
# ]


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
        temp_entries_path,
        ucc_entries_path,
        temp_image_path,
    ) = load_paths(UCC_last_version)

    # Load required files
    df_UCC, df_tables, current_JSON, DBs_full_data, database_md = load_data(
        logging, ucc_file_path, root_UCC_path
    )

    # Update per cluster md and webp files. If no changes are expected, this step
    # can be skipped to save time
    if input("\nUpdate OCs files? (y/n): ").lower() == "y":
        updt_ucc_cluster_files(
            logging,
            root_UCC_path,
            ucc_entries_path,
            temp_entries_path,
            DBs_full_data,
            df_UCC,
            current_JSON,
        )

    # Update the main ucc site files
    if input("\nUpdate UCC site files? (y/n): ").lower() == "y":
        updt_ucc_main_files(
            logging,
            temp_image_path,
            temp_dbs_tables_path,
            ucc_dbs_tables_path,
            df_UCC,
            current_JSON,
            database_md,
            df_tables,
        )

    # Update CSV file
    if input("\nUpdate clusters CSV file? (y/n): ").lower() == "y":
        updt_cls_CSV(logging, ucc_gz_CSV_path, temp_gz_CSV_path, df_tables)

    if input("\nMove files to their final destination? (y/n): ").lower() == "y":
        move_files(
            logging,
            root_UCC_path,
        )
        logging.info("All files moved into place")

    # Check number of files
    N_UCC = len(df_UCC)
    file_checker(
        logging, N_UCC, root_UCC_path, datafiles_only=False, md_folder=md_folder
    )

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

    # # Temp path to the ucc table files
    # temp_tables_path = temp_fold + tables_folder
    # if not os.path.exists(temp_tables_path):
    #     os.makedirs(temp_tables_path)
    # # Root path to the ucc table files
    # ucc_tables_path = root_UCC_path + tables_folder

    # Temp path to the ucc table files for each DB
    temp_dbs_tables_path = temp_fold + dbs_tables_folder
    if not os.path.exists(temp_dbs_tables_path):
        os.makedirs(temp_dbs_tables_path)
    # Root path to the ucc table files for each DB
    ucc_dbs_tables_path = root_UCC_path + dbs_tables_folder

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

    return (
        ucc_file_path,
        root_UCC_path,
        temp_gz_CSV_path,
        ucc_gz_CSV_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        temp_entries_path,
        ucc_entries_path,
        temp_image_path,
    )


def load_data(
    logging, ucc_file_path: str, root_UCC_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, str]:
    """ """

    df_UCC = pd.read_csv(ucc_file_path)
    logging.info(f"UCC {ucc_file_path} loaded (N={len(df_UCC)})")

    # Prepare df_UCC to be used in the updating of the table files
    df_tables = updt_UCC(df_UCC)

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

    # # Load TABLES.md file
    # with open(root_UCC_path + tables_md_path) as file:
    #     tables_md = file.read()

    return df_UCC, df_tables, current_JSON, DBs_full_data, database_md


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
    # Create temp quadrant folders
    for Nquad in range(1, 5):
        for lat in ("P", "N"):
            quad = "Q" + str(Nquad) + lat + "/"
            out_path = temp_fold + quad + plots_folder
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    logging.info("\nGenerating md and plot files")
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
            UCC_cl,
            current_JSON,
            DBs_full_data,
            fname0,
        )

        # Make plots
        txt_p = make_plots(
            root_UCC_path,
            temp_fold,
            Qfold,
            members_folder,
            plots_folder,
            UCC_cl,
            fname0,
        )

        if txt_e != "" or txt_p != "":
            N_total += 1
            logging.info(f"{N_total} -> " + txt + txt_e + txt_p + f" ({i_ucc})")

    logging.info(f"\nN={N_total} OCs processed")


def make_entry(
    root_UCC_path,
    ucc_entries_path,
    temp_entries_path,
    plots_folder,
    Qfold,
    df_UCC,
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

    # Generate table with close OCs
    close_table = ucc_entry.close_cat_cluster(df_UCC, UCC_cl)

    # # Get colors used by the 'CX' classification
    # abcd_c = ucc_entry.UCC_color(UCC_cl["C3"])

    # Generate full entry
    new_md_entry = ucc_entry.make(
        UCC_cl, fname0, Qfold, posit_table, img_cont, fpars_table, close_table
    )

    # Compare old md file (if it exists) with the new md file, for this cluster
    try:
        # Read old entry
        with open(ucc_entries_path + fname0 + ".md", "r") as f:
            old_md_entry = f.read()

        # If this is an OC that already has an entry
        # Remove dates before comparing
        new_md_entry_no_date = new_md_entry.split("Last modified")[0]
        old_md_entry_no_date = old_md_entry.split("Last modified")[0]
        # Check if entry needs updating
        if new_md_entry_no_date != old_md_entry_no_date:
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


def make_plots(
    root_UCC_path,
    temp_fold,
    Qfold,
    members_folder,
    plots_folder,
    UCC_cl,
    fname0,
) -> str:
    """
    Make CMD and Aladin plots.

    Only images that do not exist are generated, this script DOES NOT UPDATE
    images that already exist.
    """
    txt = ""

    # Make CMD plot
    make_CMD_plot = False
    # Path to original image (if it exists)
    orig_CMD_file = root_UCC_path + Qfold + plots_folder + fname0 + ".webp"
    if Path(orig_CMD_file).is_file() is False:
        # If image files does not exist --> generate
        make_CMD_plot = True
    else:
        # Load JSON file with last updated date for this Q
        fname_json = root_UCC_path + Qfold + parquet_dates
        with open(fname_json, "r") as file:
            json_data = json.load(file)
        # This indicates that this is a new parquet file and the plot should be re-done
        if "USED" not in json_data[fname0]:
            make_CMD_plot = True

    if make_CMD_plot:
        # Read members .parquet file
        parquet_path = root_UCC_path + Qfold + members_folder + fname0 + ".parquet"
        df_membs = pd.read_parquet(parquet_path)
        # Path were new image will be stored
        save_plot_file = temp_fold + Qfold + plots_folder + fname0 + ".webp"
        ucc_plots.plot_CMD(save_plot_file, df_membs)
        txt += " CMD plot generated |"

    # Make Aladin plot
    # Path to original image (if it exists)
    orig_aladin_path = root_UCC_path + Qfold + plots_folder + fname0 + "_aladin.webp"
    # If image files does not exist --> generate
    if Path(orig_aladin_path).is_file() is False:
        # Path were new image will be stored
        save_plot_file = temp_fold + Qfold + plots_folder + fname0 + "_aladin.webp"
        ucc_plots.plot_aladin(
            UCC_cl["RA_ICRS_m"],
            UCC_cl["DE_ICRS_m"],
            UCC_cl["r_50"],
            save_plot_file,
        )
        txt += " Aladin plot generated |"

    return txt


def updt_ucc_main_files(
    logging,
    temp_image_path,
    temp_dbs_tables_path,
    ucc_dbs_tables_path,
    df_UCC,
    current_JSON,
    database_md,
    df_tables,
):
    """ """
    logging.info("\nUpdating ucc.ar files")

    # # TODO: radius in parsec, unused yet (24/12/04)
    # pc_rad = pc_radius(df_UCC["r_50"].values, df_UCC["Plx_m"].values)

    # Count number of OCs in each class
    # OCs_per_class = count_OCs_classes(df_UCC["C3"], class_order)
    # # Mask with duplicates
    # dups_msk = count_dups(df_UCC)
    # Mask with N50 members
    # membs_msk = count_N50membs(df_UCC)

    # Update site plots
    make_site_plots(logging, temp_image_path, df_UCC)

    # Update DATABASE.md
    update_main_pages(logging, current_JSON, df_UCC, database_md)

    # Update pages for individual databases
    new_tables_dict = updt_DBs_tables(current_JSON, df_tables)
    general_table_update(
        logging, ucc_dbs_tables_path, temp_dbs_tables_path, new_tables_dict
    )

    # # Update page with N members
    # new_tables_dict = updt_n50members_tables(df_tables, membs_msk)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    # #
    # new_tables_dict = updt_C3_tables(df_tables, class_order)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    #
    # new_tables_dict = updt_dups_tables(df_tables, dups_msk)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    #
    # new_tables_dict = updt_quad_tables(df_tables)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)


def make_site_plots(logging, temp_image_path, df_UCC):
    """ """
    ucc_plots.make_N_vs_year_plot(temp_image_path + "catalogued_ocs.webp", df_UCC)
    logging.info("Plot generated: number of OCs vs years")

    # ucc_plots.make_classif_plot(
    #     temp_image_path + "classif_bar.webp", OCs_per_class, class_order
    # )
    # logging.info("Plot generated: classification histogram")


def update_main_pages(
    logging,
    current_JSON,
    df_UCC,
    database_md,
):
    """Update DATABASE.md"""

    logging.info("\nCounting total number of members")
    N_members_UCC = count_N_members_UCC(members_folder)
    logging.info(f"Total number of members extracted: {N_members_UCC}")

    # Update the total number of entries, databases, and members in the UCC
    N_db_UCC, N_cl_UCC = len(current_JSON), len(df_UCC)
    database_md_updt = ucc_n_total_updt(
        logging, N_db_UCC, N_cl_UCC, N_members_UCC, database_md
    )

    # Update the table with the catalogues used in the UCC
    database_md_updt = updt_cats_used(logging, df_UCC, current_JSON, database_md_updt)

    # Save updated page (temp)
    if database_md != database_md_updt:
        with open(temp_fold + databases_md_path, "w") as file:
            file.write(database_md_updt)
        logging.info("DATABASE.md updated")

    # tables_md_updt = updt_C3_classification(
    #     logging, class_order, OCs_per_class, tables_md_updt
    # )

    # tables_md_updt = updt_OCs_per_quad(logging, df_UCC, tables_md_updt)

    # tables_md_updt = updt_dups_table(logging, dups_msk, tables_md_updt)

    # tables_md_updt = memb_number_table(logging, membs_msk, tables_md_updt)

    # # Save updated page (temp)
    # if database_md != tables_md_updt:
    #     with open(temp_fold + tables_md_path, "w") as file:
    #         file.write(tables_md_updt)
    #     logging.info("TABLES.md updated")


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
    Update cluster.csv file used by 'ucc.ar' search
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
            ]
        ]
    )
    df_new = df.sort_values("ID")

    # df.rename(
    #     columns={
    #         "ID": "N",
    #         "fnames": "F",
    #         "RA_ICRS": "R",
    #         "DE_ICRS": "D",
    #         "GLON": "L",
    #         "GLAT": "B",
    #         "dist_pc": "P",
    #         "N_50": "M",
    #     },
    #     inplace=True,
    # )
    # json_new = df.to_dict(orient="records")

    # Load the old JSON data
    # with gzip.open(ucc_gz_JSON_path, "rt", encoding="utf-8") as file:
    #     json_old = json.load(file)
    df_old = pd.read_csv(ucc_gz_CSV_path, compression="gzip")

    # Check if the two DataFrames are equal
    update_flag = not df_old.equals(df_new)

    # # Check if new JSON is equal to the old one
    # update_flag = False
    # if len(json_old) != len(df_new):
    #     update_flag = True
    # else:
    #     # True if JSONs are NOT equal
    #     update_flag = not all(a == b for a, b in zip(json_old, json_new))
    #     # Print differences to screen
    #     for i, (dict1, dict2) in enumerate(zip(json_old, json_new)):
    #         differing_keys = {
    #             key
    #             for key in dict1.keys() | dict2.keys()
    #             if dict1.get(key) != dict2.get(key)
    #         }
    #         if differing_keys:
    #             for key in differing_keys:
    #                 logging.info(
    #                     f"{i}, {key} --> OLD: {dict1.get(key)} | NEW: {dict2.get(key)}"
    #                 )

    # Update CSV if required
    if update_flag is True:
        # df.to_json(
        #     temp_gz_JSON_path,
        #     orient="records",
        #     indent=1,
        #     compression="gzip",
        # )
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
) -> None:
    """ """
    logging.info("\nUpdate files:")

    # Move all files inside 'temp_fold/ucc/', this includes the cluster.json.gz file
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

    # Move all CMD and Aladin plots in Q folders
    for qN in range(1, 5):
        for lat in ("P", "N"):
            qfold = "Q" + str(qN) + lat + "/"
            # Check if folder exists
            qplots_fold = temp_fold + qfold + plots_folder
            if os.path.exists(qplots_fold):
                # Load JSON file with last updated date for this Q
                fname_json = root_UCC_path + qfold + parquet_dates
                with open(fname_json, "r") as file:
                    json_data = json.load(file)

                # For every file in this folder
                for file in os.listdir(qplots_fold):
                    plot_temp = qplots_fold + file
                    plot_stored = root_UCC_path + qfold + plots_folder + file
                    os.rename(plot_temp, plot_stored)

                    # Update entry for this file
                    fname0 = file.split("/")[-1].split(".")[0]
                    json_data[fname0] += (
                        f"; USED ({datetime.datetime.now().strftime('%y%m%d%H')})"
                    )

                # Update JSON file
                with open(fname_json, "w") as file:
                    json.dump(json_data, file, indent=2)

    plot_temp = temp_fold + "QXX/" + plots_folder + "*.webp"
    plot_stored = root_UCC_path + "QXX/" + plots_folder + "*.webp"
    logging.info(plot_temp + " --> " + plot_stored)


if __name__ == "__main__":
    main()
