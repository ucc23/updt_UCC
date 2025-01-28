import os

import pandas as pd

from modules.HARDCODED import (
    UCC_folder,
    members_folder,
    temp_fold,
)
from modules.update_site.zenodo_updt import (
    create_csv_UCC,
    create_membs_UCC,
    updt_readme,
)
from modules.utils import get_last_version_UCC, logger

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

    # Read latest version of the UCC
    last_version, df_UCC = load_UCC(logging)

    upld_zenodo_file = temp_fold + UCC_folder + "UCC_cat.csv"
    create_csv_UCC(upld_zenodo_file, df_UCC)
    logging.info("\nZenodo 'UCC_cat.csv' file generated")

    logging.info("Reading member files...")

    # Current root + main UCC folder
    root_current_folder = os.getcwd()
    # Go up one level
    root_UCC_path = os.path.dirname(root_current_folder)

    zenodo_members_file = temp_fold + UCC_folder + "UCC_members.parquet"
    create_membs_UCC(logging, root_UCC_path, members_folder, zenodo_members_file)
    logging.info("Zenodo 'UCC_members.parquet' file generated")

    zenodo_readme = temp_fold + UCC_folder + "README.txt"
    updt_readme(UCC_folder, last_version, zenodo_readme)
    logging.info("Zenodo README file updated")

    ######################################
    # J script
    logging.info("Updating ucc.ar files\n")

    df_UCC, dbs_used, database_md = load_data(logging)

    # # TODO: radius in parsec, unused yet (24/12/04)
    # pc_rad = pc_radius(df_UCC["r_50"].values, df_UCC["Plx_m"].values)

    # Count number of OCs in each class
    OCs_per_class = count_OCs_classes(df_UCC)
    # Mask with duplicates
    dups_msk = count_dups(df_UCC)
    # Mask with N50 members
    membs_msk = count_N50membs(df_UCC)

    # Update plots
    make_plots(df_UCC, dbs_used, OCs_per_class)

    # Update DATABASE.md file
    database_md_updt = ucc_n_total_updt(len(df_UCC), len(dbs_used), database_md)
    database_md_updt = updt_cats_used(df_UCC, dbs_used, database_md_updt)
    database_md_updt = updt_C3_classification(OCs_per_class, database_md_updt)
    database_md_updt = updt_OCs_per_quad(df_UCC, database_md_updt)
    database_md_updt = updt_dups_table(dups_msk, database_md_updt)
    database_md_updt = memb_number_table(membs_msk, database_md_updt)
    # Save updated page
    if database_md != database_md_updt:
        if DRY_RUN is False:
            with open(root_UCC_path + pages_folder + "/" + "DATABASE.md", "w") as file:
                file.write(database_md_updt)
        logging.info("DATABASE.md updated")

    # Prepare df_UCC to be used in the updating of the table files below
    df_updt = updt_UCC(df_UCC)

    # Update groups of tables
    updt_DBs_tables(dbs_used, df_updt)
    updt_n50members_tables(df_updt, membs_msk)
    updt_C3_tables(df_updt)
    updt_dups_tables(df_updt, dups_msk)
    updt_quad_tables(df_updt)

    # Update JSON file
    updt_cls_JSON(df_updt)


def load_UCC(logging) -> pd.DataFrame:
    """ """
    # Load the latest version of the combined catalogue
    last_version = get_last_version_UCC(UCC_folder)
    # Path to the current UCC csv file
    ucc_file = UCC_folder + last_version

    df_UCC = pd.read_csv(ucc_file)
    logging.info(f"UCC version {last_version} loaded (N={len(df_UCC)})")

    # # Load clusters data in JSON file
    # with open(temp_fold + dbs_folder + name_DBs_json) as f:
    #     dbs_used = json.load(f)
    # logging.info("JSON file loaded")

    # # Load DATABASE.md file
    # with open(root_UCC_path + pages_folder + "/" + "DATABASE.md") as file:
    #     database_md = file.read()

    return last_version, df_UCC


if __name__ == "__main__":
    main()
