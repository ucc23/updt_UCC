from pathlib import Path

import pandas as pd

from . import ucc_entry, ucc_plots


def make_entry(
    root_UCC_path,
    ucc_entries_path,
    temp_entries_path,
    plots_folder,
    Qfold,
    df_UCC,
    UCC_cl,
    DBs_json,
    DBs_full_data,
    fname0,
):
    """ """

    # Generate table with positional data: (ra, dec, plx, pmra, pmde, Rv)
    posit_table = ucc_entry.positions_in_lit(DBs_json, DBs_full_data, UCC_cl)

    # Generate image carousel
    cl_name = UCC_cl["ID"].split(";")[0]
    img_cont = ucc_entry.carousel_div(
        root_UCC_path, plots_folder, cl_name, Qfold, fname0
    )

    # Generate fundamental parameters table
    fpars_table = ucc_entry.fpars_in_lit(
        DBs_json, DBs_full_data, UCC_cl["DB"], UCC_cl["DB_i"]
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
            file_flag = "updated"
            with open("OLD.md", "w") as f:
                f.write(old_md_entry_no_date)
            with open("NEW.md", "w") as f:
                f.write(new_md_entry_no_date)
            breakpoint()
        else:
            # The existing md file has not changed
            file_flag = ""

    except FileNotFoundError:
        # This is a new OC with no md entry yet
        file_flag = "generated"

    if file_flag != "":
        # Generate/update entry
        with open(temp_entries_path + fname0 + ".md", "w") as f:
            f.write(new_md_entry)

    return file_flag


def make_plots(
    root_UCC_path,
    root_current_path,
    temp_fold,
    Qfold,
    members_folder,
    plots_folder,
    UCC_cl,
    fname0,
    txt,
):
    """ """
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
        txt += " CMD plot generated"

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
        txt += " Aladin plot generated"

    return txt
