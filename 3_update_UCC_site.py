import gzip
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from modules.HARDCODED import (
    UCC_folder,
    clusters_json,
    dbs_folder,
    images_folder,
    md_folder,
    members_folder,
    name_DBs_json,
    pages_folder,
    plots_folder,
    temp_fold,
)
from modules.update_site import ucc_entry, ucc_plots
from modules.update_site.main_files_updt import (
    count_dups,
    count_N50membs,
    count_OCs_classes,
    memb_number_table,
    ucc_n_total_updt,
    updt_C3_classification,
    updt_C3_tables,
    updt_cats_used,
    updt_DBs_tables,
    updt_dups_table,
    updt_dups_tables,
    updt_n50members_tables,
    updt_OCs_per_quad,
    updt_quad_tables,
)
from modules.update_site.zenodo_updt import (
    create_csv_UCC,
    create_membs_UCC,
    updt_readme,
)
from modules.utils import file_checker, get_last_version_UCC, logger

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


def main():
    """ """
    logging = logger()

    # Read paths and load required files
    (
        root_current_path,
        root_UCC_path,
        ucc_gz_JSON_path,
        temp_gz_JSON_path,
        temp_pages_path,
        ucc_pages_path,
        temp_entries_path,
        ucc_entries_path,
        temp_image_path,
        ucc_images_path,
        temp_tables_path,
        ucc_tables_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        last_version,
        df_UCC,
        DBs_full_data,
        current_JSON,
        database_md,
    ) = load_UCC_and_paths(logging)

    #
    # updat_zenodo_files(logging, root_UCC_path, last_version, df_UCC)

    #
    # updt_ucc_cluster_files(
    #     logging,
    #     root_UCC_path,
    #     root_current_path,
    #     ucc_entries_path,
    #     temp_entries_path,
    #     DBs_full_data,
    #     df_UCC,
    #     current_JSON,
    # )

    # Prepare df_UCC to be used in the updating of the table files below
    df_updt = updt_UCC(df_UCC)

    # #
    # updt_ucc_main_files(
    #     logging,
    #     temp_image_path,
    #     temp_pages_path,
    #     temp_tables_path,
    #     ucc_tables_path,
    #     temp_dbs_tables_path,
    #     ucc_dbs_tables_path,
    #     df_UCC,
    #     current_JSON,
    #     database_md,
    #     df_updt,
    # )

    # # Update JSON file
    # updt_cls_JSON(logging, ucc_gz_JSON_path, temp_gz_JSON_path, df_updt)

    if input("\nMove files to their final destination? (y/n): ").lower() != "y":
        sys.exit()
    move_files(
        logging,
        root_UCC_path,
        root_current_path,
    )
    logging.info("All files moved into place")

    # Check number of files
    N_UCC = len(df_updt)
    file_checker(logging, N_UCC, root_UCC_path, datafiles_only=False)

    logging.info("\nAll done!")


def load_UCC_and_paths(
    logging,
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
    str,
    str,
    str,
    pd.DataFrame,
    dict,
    dict,
    str,
]:
    """ """
    # Create quadrant folders
    for Nquad in range(1, 5):
        for lat in ("P", "N"):
            quad = "Q" + str(Nquad) + lat + "/"
            out_path = temp_fold + quad + plots_folder
            if not os.path.exists(out_path):
                os.makedirs(out_path)

    # Load the latest version of the combined catalogue
    last_version = get_last_version_UCC(UCC_folder)
    # Path to the current UCC csv file
    ucc_file = UCC_folder + last_version

    df_UCC = pd.read_csv(ucc_file)
    logging.info(f"UCC version {last_version} loaded (N={len(df_UCC)})")

    # Current root + main UCC folder
    root_current_path = os.getcwd()
    # Go up one level
    root_UCC_path = os.path.dirname(root_current_path) + "/"
    root_current_path = root_current_path + "/"

    # Load clusters data in JSON file
    with open(dbs_folder + name_DBs_json) as f:
        current_JSON = json.load(f)

    ucc_gz_JSON_path = root_UCC_path + clusters_json
    temp_gz_JSON_path = root_current_path + temp_fold + clusters_json

    # Read the data for every DB in the UCC as a pandas DataFrame
    DBs_full_data = {}
    for k, v in current_JSON.items():
        DBs_full_data[k] = pd.read_csv(dbs_folder + k + ".csv")

    # Folder name where the ucc pages are stored
    temp_pages_path = root_current_path + temp_fold + pages_folder
    if not os.path.exists(temp_pages_path):
        os.makedirs(temp_pages_path)
    ucc_pages_path = root_UCC_path + pages_folder

    # Load DATABASE.md file
    with open(ucc_pages_path + "DATABASE.md") as file:
        database_md = file.read()

    # Folder name where the datafile is stored
    temp_entries_path = root_current_path + temp_fold + md_folder
    if not os.path.exists(temp_entries_path):
        os.makedirs(temp_entries_path)
    ucc_entries_path = root_UCC_path + md_folder

    # Folder name where the ucc images are stored
    temp_image_path = root_current_path + temp_fold + images_folder
    if not os.path.exists(temp_image_path):
        os.makedirs(temp_image_path)
    ucc_images_path = root_UCC_path + images_folder

    temp_tables_path = temp_pages_path + "tables/"
    if not os.path.exists(temp_tables_path):
        os.makedirs(temp_tables_path)
    ucc_tables_path = root_UCC_path + pages_folder + "/tables/"

    temp_dbs_tables_path = temp_pages_path + "tables/dbs/"
    if not os.path.exists(temp_dbs_tables_path):
        os.makedirs(temp_dbs_tables_path)
    ucc_dbs_tables_path = root_UCC_path + pages_folder + "/tables/dbs/"

    return (
        root_current_path,
        root_UCC_path,
        ucc_gz_JSON_path,
        temp_gz_JSON_path,
        temp_pages_path,
        ucc_pages_path,
        temp_entries_path,
        ucc_entries_path,
        temp_image_path,
        ucc_images_path,
        temp_tables_path,
        ucc_tables_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        last_version,
        df_UCC,
        DBs_full_data,
        current_JSON,
        database_md,
    )


def updat_zenodo_files(logging, root_UCC_path, last_version, df_UCC):
    """ """
    #
    upld_zenodo_file = temp_fold + UCC_folder + "UCC_cat.csv"
    create_csv_UCC(upld_zenodo_file, df_UCC)
    logging.info("\nZenodo 'UCC_cat.csv' file generated")

    logging.info("Reading member files...")

    zenodo_members_file = temp_fold + UCC_folder + "UCC_members.parquet"
    create_membs_UCC(logging, root_UCC_path, members_folder, zenodo_members_file)
    logging.info("Zenodo 'UCC_members.parquet' file generated")

    zenodo_readme = temp_fold + UCC_folder + "README.txt"
    updt_readme(UCC_folder, last_version, zenodo_readme)
    logging.info("Zenodo README file updated")


def updt_ucc_cluster_files(
    logging,
    root_UCC_path,
    root_current_path,
    ucc_entries_path,
    temp_entries_path,
    DBs_full_data,
    df_UCC,
    current_JSON,
):
    """ """
    logging.info("\nGenerating md and plot files")
    N_total = 0
    # Iterate trough each entry in the UCC database
    for i_ucc, UCC_cl in df_UCC.iterrows():
        fname0 = str(UCC_cl["fnames"]).split(";")[0]
        Qfold = UCC_cl["quad"] + "/"

        # txt0 = f"{Qfold}{fname0}: "
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
            root_current_path,
            temp_fold,
            Qfold,
            members_folder,
            plots_folder,
            UCC_cl,
            fname0,
        )

        if txt_e != "" or txt_p != "":
            N_total += 1
            logging.info(f"{i_ucc}: " + txt + txt_e + txt_p + f" ({N_total})")

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

    # Generate image carousel
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

    # Color used by the 'CX' classification
    abcd_c = ucc_entry.UCC_color(UCC_cl["C3"])

    # Generate entry
    new_md_entry = ucc_entry.make(
        UCC_cl,
        fname0,
        Qfold,
        posit_table,
        img_cont,
        fpars_table,
        close_table,
        abcd_c,
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
    root_current_path,
    temp_fold,
    Qfold,
    members_folder,
    plots_folder,
    UCC_cl,
    fname0,
) -> str:
    """ """
    txt = ""

    # Make CMD plot
    # Path to original image (if it exists)
    orig_CMD_file = root_UCC_path + Qfold + plots_folder + fname0 + ".webp"
    # If image files does not exist --> generate
    if Path(orig_CMD_file).is_file() is False:
        parquet_path = root_UCC_path + Qfold + members_folder + fname0 + ".parquet"
        df_membs = pd.read_parquet(parquet_path)
        # Path were new image will be stored
        save_plot_file = (
            root_current_path + temp_fold + Qfold + plots_folder + fname0 + ".webp"
        )
        ucc_plots.plot_CMD(save_plot_file, df_membs)
        txt += " CMD plot generated |"

    # Make Aladin plot
    # Path to original image (if it exists)
    orig_aladin_path = root_UCC_path + Qfold + plots_folder + fname0 + "_aladin.webp"
    # If image files does not exist --> generate
    if Path(orig_aladin_path).is_file() is False:
        # Path were new image will be stored
        save_plot_file = (
            root_current_path
            + temp_fold
            + Qfold
            + plots_folder
            + fname0
            + "_aladin.webp"
        )
        ucc_plots.plot_aladin(
            UCC_cl["RA_ICRS_m"],
            UCC_cl["DE_ICRS_m"],
            UCC_cl["r_50"],
            save_plot_file,
        )
        txt += " Aladin plot generated |"

    return txt


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


def updt_ucc_main_files(
    logging,
    temp_image_path,
    temp_pages_path,
    temp_tables_path,
    ucc_tables_path,
    temp_dbs_tables_path,
    ucc_dbs_tables_path,
    df_UCC,
    current_JSON,
    database_md,
    df_updt,
):
    """ """
    logging.info("\nUpdating ucc.ar files")

    # # TODO: radius in parsec, unused yet (24/12/04)
    # pc_rad = pc_radius(df_UCC["r_50"].values, df_UCC["Plx_m"].values)

    # Count number of OCs in each class
    OCs_per_class = count_OCs_classes(df_UCC["C3"], class_order)
    # Mask with duplicates
    dups_msk = count_dups(df_UCC)
    # Mask with N50 members
    membs_msk = count_N50membs(df_UCC)

    # Update site plots
    make_site_plots(logging, temp_image_path, df_UCC, OCs_per_class)

    # Update main DATABASE.md file
    update_main_database_page(
        logging,
        temp_pages_path,
        current_JSON,
        df_UCC,
        database_md,
        OCs_per_class,
        dups_msk,
        membs_msk,
    )

    # Update pages for individual databases
    new_tables_dict = updt_DBs_tables(current_JSON, df_updt)
    general_table_update(
        logging, ucc_dbs_tables_path, temp_dbs_tables_path, new_tables_dict
    )

    # Update page with N members
    new_tables_dict = updt_n50members_tables(df_updt, membs_msk)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    #
    new_tables_dict = updt_C3_tables(df_updt, class_order)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    #
    new_tables_dict = updt_dups_tables(df_updt, dups_msk)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    #
    new_tables_dict = updt_quad_tables(df_updt)
    general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)


def make_site_plots(logging, temp_image_path, df_UCC, OCs_per_class):
    """ """
    ucc_plots.make_N_vs_year_plot(temp_image_path + "catalogued_ocs.webp", df_UCC)
    logging.info("Plot generated: number of OCs vs years")

    ucc_plots.make_classif_plot(
        temp_image_path + "classif_bar.webp", OCs_per_class, class_order
    )
    logging.info("Plot generated: classification histogram")


def update_main_database_page(
    logging,
    temp_pages_path,
    current_JSON,
    df_UCC,
    database_md,
    OCs_per_class,
    dups_msk,
    membs_msk,
):
    """Update DATABASE.md file"""
    #
    database_md_updt = ucc_n_total_updt(
        logging, len(df_UCC), len(current_JSON), database_md
    )

    database_md_updt = updt_cats_used(logging, df_UCC, current_JSON, database_md_updt)

    database_md_updt = updt_C3_classification(
        logging, class_order, OCs_per_class, database_md_updt
    )

    database_md_updt = updt_OCs_per_quad(logging, df_UCC, database_md_updt)

    database_md_updt = updt_dups_table(logging, dups_msk, database_md_updt)

    database_md_updt = memb_number_table(logging, membs_msk, database_md_updt)

    # Save updated page
    if database_md != database_md_updt:
        with open(temp_pages_path + "DATABASE.md", "w") as file:
            file.write(database_md_updt)
        logging.info("DATABASE.md updated")


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


def updt_cls_JSON(logging, ucc_gz_JSON_path: str, temp_gz_JSON_path: str, df_updt):
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
    df = df.sort_values("ID")

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
    with gzip.open(ucc_gz_JSON_path, "rt", encoding="utf-8") as file:
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
        df.to_json(
            temp_gz_JSON_path,
            orient="records",
            indent=1,
            compression="gzip",
        )
        logging.info("File 'clusters.json.gz' updated")


def move_files(
    logging,
    root_UCC_path,
    root_current_path,
) -> None:
    """ """
    logging.info("\nUpdate files:")

    temp_ucc_fold = root_current_path + temp_fold + "ucc/"
    for root, dirs, files in os.walk(temp_ucc_fold):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ucc = file_path.replace(root_current_path + temp_fold, root_UCC_path)
            # os.rename(file_path, file_ucc)
            logging.info(file_path + " --> " + file_ucc)

    logging.info("")
    # Move all CMD and Aladin plots in Q folders
    for qN in range(1, 5):
        for lat in ("P", "N"):
            qfold = "Q" + str(qN) + lat + "/"
            # Check if folder exists
            qplots_fold = temp_fold + qfold + plots_folder
            if os.path.exists(qplots_fold):
                # For every file in this folder
                for file in os.listdir(qplots_fold):
                    plot_temp = root_current_path + qplots_fold + file
                    plot_stored = root_UCC_path + qfold + plots_folder + file
                    # os.rename(plot_temp, plot_stored)
                    # logging.info(plot_temp + " --> " + plot_stored)


if __name__ == "__main__":
    main()
