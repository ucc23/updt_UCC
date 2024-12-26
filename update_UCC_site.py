import json

import pandas as pd

    aux,
    standardize_and_match,
)
from modules.HARDCODED import UCC_folder, dbs_folder, name_DBs_json, temp_fold

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
    logging = aux.logger()
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


def load_data(logging, last_version: str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """ """
    # Load the latest version of the combined catalogue
    df_UCC = pd.read_csv(UCC_folder + "UCC_cat_" + last_version + ".csv")
    logging.info(f"UCC version {last_version} loaded (N={len(df_UCC)})")

    # Load clusters data in JSON file
    with open(temp_fold + dbs_folder + name_DBs_json) as f:
        dbs_used = json.load(f)
    logging.info("JSON file loaded")

    # Load DATABASE.md file
    with open(root_UCC_path + pages_folder + "/" + "DATABASE.md") as file:
        database_md = file.read()

    return df_UCC, dbs_used, database_md


if __name__ == '__main__':
    main()
