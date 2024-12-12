import json
from pathlib import Path

import pandas as pd
from HARDCODED import (
    UCC_folder,
    all_DBs_json,
    dbs_folder,
    md_folder,
    members_folder,
    plots_folder,
    root_UCC_path,
)
from modules import UCC_new_match, files_handler, logger, ucc_entry, ucc_plots

logging = logger.main()

# Use to process files without writing changes to files
DRY_RUN = True


def main():
    """ """
    logging.info("Running 'make_entries' script\n")
    logging.info(f"DRY RUN IS {DRY_RUN}\n")

    # Folder name where the datafile is stored
    entries_path = root_UCC_path + f"{md_folder}/"

    logging.info("Reading databases...")
    with open(dbs_folder + all_DBs_json) as f:
        DBs_json = json.load(f)
    # Read the data for every DB in the UCC as a pandas DataFrame
    DBs_full_data = {}
    for k, v in DBs_json.items():
        DBs_full_data[k] = pd.read_csv(dbs_folder + k + ".csv")

    # Read latest version of the UCC
    df_UCC, _ = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    logging.info("\nProcessing UCC")
    N_total = 0
    # Iterate trough each entry in the UCC database
    for _, UCC_cl in df_UCC.iterrows():
        fname0 = str(UCC_cl["fnames"]).split(";")[0]
        Qfold = UCC_cl["quad"]

        txt0 = f"{Qfold}/{fname0}: "
        txt = f"{Qfold}/{fname0}: "

        # Make catalogue entry
        txt1 = make_entry(
            df_UCC, UCC_cl, DBs_json, DBs_full_data, entries_path, Qfold, fname0
        )
        if txt1 != "":
            txt += f" md {txt1}"

        # Make plots
        txt = make_plots(UCC_cl, Qfold, fname0, txt)

        if txt != txt0:
            logging.info(f"{N_total}, " + txt)
            N_total += 1

    logging.info(f"\nN={N_total} OCs processed")


def make_entry(df_UCC, UCC_cl, DBs_json, DBs_full_data, entries_path, Qfold, fname0):
    """ """
    # Generate entry
    new_md_entry = ucc_entry.main(
        df_UCC, UCC_cl, DBs_json, DBs_full_data, fname0, Qfold
    )

    # Read old entry, if any
    try:
        with open(entries_path + fname0 + ".md", "r") as f:
            old_md_entry = f.read()
    except FileNotFoundError:
        # This is a new OC with no md entry yet
        old_md_entry = ""

    file_flag = ""
    if old_md_entry == "":
        # If this is a new entry with no old md file
        file_flag = "generated"
    else:
        # If this is an OC that already has an entry
        # Remove dates before comparing
        new_md_entry_no_date = new_md_entry.split("Last modified")[0]
        old_md_entry_no_date = old_md_entry.split("Last modified")[0]
        # Check if entry needs updating
        if new_md_entry_no_date != old_md_entry_no_date:
            file_flag = "updated"
            # with open("OLD.md", "w") as f:
            #     f.write(old_md_entry_no_date)
            # with open("NEW.md", "w") as f:
            #     f.write(new_md_entry_no_date)
            # breakpoint()

    if file_flag != "":
        if DRY_RUN is False:
            # Generate/update entry
            with open(entries_path + fname0 + ".md", "w") as f:
                f.write(new_md_entry)

    return file_flag


def make_plots(UCC_cl, Qfold, fname0, txt):
    """ """
    # Path to image (if it exists)
    plot_path = root_UCC_path + Qfold + f"/{plots_folder}/" + fname0 + ".webp"
    parquet_path = root_UCC_path + Qfold + f"/{members_folder}/" + fname0 + ".parquet"
    # Update/generate
    txt0 = files_handler.update_image(
        DRY_RUN, logging, plot_path, (ucc_plots.members_plot, parquet_path)
    )
    if txt0 != "":
        txt += f" plot {txt0}"

    # Make Aladin plot. Never update these images, only generate if they do not
    # exist
    plot_aladin_path = (
        root_UCC_path + Qfold + f"/{plots_folder}/" + fname0 + "_aladin.webp"
    )
    if Path(plot_aladin_path).is_file() is False:
        # If 'old' image files does not exist --> generate
        if DRY_RUN is False:
            ucc_plots.make_aladin_plot(
                UCC_cl["RA_ICRS_m"],
                UCC_cl["DE_ICRS_m"],
                UCC_cl["r_50"],
                plot_aladin_path,
            )
        txt += " plot_aladin generated"

    return txt


if __name__ == "__main__":
    main()
