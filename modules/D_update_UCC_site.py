import datetime
import gzip
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .D_funcs import ucc_entry, ucc_plots, ucc_summ_cmmts, ucc_updt_tables
from .utils import comments_check, get_fnames, load_BC_cats, logger, members_hashes
from .variables import (
    UCC_cat_B_out,
    UCC_cat_C_out,
    UCC_cat_D_in,
    UCC_cmmts_folder,
    UCC_members_file,
    all_OC_names,
    archive_folder_path,
    articles_md_path,
    assets_folder,
    class_order,
    clusters_csv_path,
    clusters_manifest_path,
    cmmts_tables_folder,
    data_folder,
    databases_md_path,
    dbs_folder,
    dbs_tables_folder,
    eq_positions_path,
    fpars_order,
    images_folder,
    md_folder,
    members_files_folder,
    name_DBs_json,
    pages_folder,
    plots_folder,
    plots_sub_folders,
    root_ucc_path,
    temp_folder,
    ucc_path,
    zenodo_folder,
)


def main():
    """
    Main function to update the UCC site with new data and visualizations.
    """
    logging = logger()

    # Read paths
    (
        ucc_B_file_out,
        ucc_C_file_out,
        zenodo_members_file,
        plots_record_path,
        temp_plots_record_path,
        temp_entries_path,
        ucc_entries_path,
        temp_members_files_folder,
        temp_image_path,
        temp_dbs_tables_path,
        temp_cmmts_tables_path,
        ucc_dbs_tables_path,
        ucc_cmmts_tables_path,
        old_gz_CSV_path,
        new_clusters_csv_path,
    ) = load_paths(logging)

    # Load required files
    (
        df_members,
        df_BC,
        fnames_plots_updt,
        df_hash_old,
        df_hash_curr,
        DBs_JSON,
        DBs_full_data,
        cmmts_JSONS_dict,
        database_md,
        articles_md,
    ) = load_data(
        logging,
        ucc_B_file_out,
        ucc_C_file_out,
        zenodo_members_file,
        plots_record_path,
    )

    comments_check(DBs_JSON)

    ###########################################
    # Update clusters .webp files
    N_plots_updt = len(fnames_plots_updt)
    if (
        N_plots_updt > 0
        and input(f"\nUpdate {N_plots_updt} cluster plots? (y/n): ").strip().lower()
        == "y"
    ):
        fnames_processed = updt_ucc_cluster_plots(
            logging,
            df_BC,
            fnames_plots_updt,
            df_members,
        )
        N_total = len(fnames_processed)
        if N_total > 0:
            logging.info(f"\n{N_total} OCs processed")
        else:
            logging.info("No plots were generated/updated")
        N_missing = len(fnames_plots_updt - set(fnames_processed))
        if N_missing > 0:
            logging.info(
                f"\nWARNING: {N_missing} OCs "
                "were not processed in the plot update/generate step."
            )
    # Record the members used for the plots that will be moved to the site
    updt_plots_record(
        logging,
        fnames_plots_updt,
        df_hash_old,
        df_hash_curr,
        plots_record_path,
        temp_plots_record_path,
    )
    ###########################################

    ###########################################
    # Update per cluster md files
    if input("\nUpdate md files ? (y/n): ").lower() == "y":
        updt_ucc_cluster_files(
            logging,
            ucc_entries_path,
            temp_entries_path,
            DBs_full_data,
            df_BC,
            DBs_JSON,
            cmmts_JSONS_dict,
        )
    ###########################################

    ###########################################
    # This needs to happen before generating the ARTICLES.md file
    if input("\nUpdate per-article tables? (y/n): ").lower() == "y":
        # Update tables files
        updt_indiv_tables(
            logging,
            temp_dbs_tables_path,
            temp_cmmts_tables_path,
            ucc_dbs_tables_path,
            ucc_cmmts_tables_path,
            DBs_JSON,
            df_BC,
            cmmts_JSONS_dict,
        )

    if input("\nUpdate UCC plots, pages & tables? (y/n): ").lower() == "y":
        # Update site-wide plots
        make_site_plots(logging, temp_image_path, df_BC)
        # Update main .md files
        N_fpars = count_fpars(df_BC)
        N_members_UCC = len(df_members)
        update_main_pages(
            logging,
            N_fpars,
            N_members_UCC,
            DBs_JSON,
            df_BC,
            database_md,
            articles_md,
            temp_cmmts_tables_path,
        )
    ###########################################

    ###########################################
    # Update assets
    if input("\nUpdate equatorial positions JSON file? (y/n): ").lower() == "y":
        updt_eq_positions(logging, df_BC, DBs_JSON, DBs_full_data)

    if input("\nUpdate split member files? (y/n): ").lower() == "y":
        updt_members_files(df_BC, df_members, temp_members_files_folder)

    if input("\nUpdate clusters CSV file? (y/n): ").lower() == "y":
        updt_cls_CSV(logging, new_clusters_csv_path, df_BC.copy())
    ###########################################

    if input("\nMove files to their final destination? (y/n): ").lower() == "y":
        move_files(
            logging,
            plots_record_path,
            temp_plots_record_path,
            old_gz_CSV_path,
            new_clusters_csv_path,
        )

    # Check number of files
    file_checker(logging, df_BC)
    logging.info("\nAll done!")


def load_paths(
    logging,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    str,
    str,
]:
    """
    Load paths for input and output files
    """
    data_folder_p = Path(data_folder)
    temp_folder_p = Path(temp_folder)
    root_ucc_path_p = Path(root_ucc_path)

    # Path to main data files
    ucc_B_file_out = data_folder_p / UCC_cat_B_out
    ucc_C_file_out = data_folder_p / UCC_cat_C_out

    # Path to members file uploaded to Zenodo
    zenodo_members_file = Path(zenodo_folder) / UCC_members_file

    # Record of the members used to generate the plots in the site
    plots_record_path = data_folder_p / UCC_cat_D_in
    temp_plots_record_path = temp_folder_p / data_folder / UCC_cat_D_in

    # Create temp folders for storing plots
    plots_fold_exist = False
    for letter in "abcdefghijklmnopqrstuvwxyz":
        for fold in plots_sub_folders:
            out_path = temp_folder_p / (plots_folder + f"plots_{letter}/" + fold)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            else:
                plots_fold_exist = True
    if plots_fold_exist:
        logging.info(
            f"At least one folder exists: {temp_folder_p / (plots_folder + 'plots_*/')}"
        )

    def ensure_dir(path: Path):
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        if existed:
            logging.info(f"Folder exists: {path}")

    temp_data_folder = temp_folder_p / data_folder
    temp_entries_path = temp_folder_p / md_folder
    temp_assets_path = temp_folder_p / assets_folder
    temp_pages_path = temp_folder_p / pages_folder
    temp_members_files_folder = temp_folder_p / members_files_folder
    temp_image_path = temp_folder_p / images_folder
    temp_dbs_tables_path = temp_folder_p / dbs_tables_folder
    temp_cmmts_tables_path = temp_folder_p / cmmts_tables_folder

    for p in [
        temp_data_folder,
        temp_entries_path,
        temp_assets_path,
        temp_pages_path,
        temp_members_files_folder,
        temp_image_path,
        temp_dbs_tables_path,
        temp_cmmts_tables_path,
    ]:
        ensure_dir(p)

    ucc_entries_path = root_ucc_path_p / md_folder
    ucc_dbs_tables_path = root_ucc_path_p / dbs_tables_folder
    ucc_cmmts_tables_path = root_ucc_path_p / cmmts_tables_folder

    # UCC path to compressed CSV file
    folder = Path(root_ucc_path, assets_folder)
    matches = list(folder.glob(clusters_csv_path))
    if matches:
        old_gz_CSV_path = str(matches[0])
    else:
        raise ValueError(f"No file matching '{clusters_csv_path}' found")

    # Generate a timestamped filename for the new clusters CSV file
    date = pd.Timestamp.now().strftime("%y%m%d%H")
    new_clusters_csv_path = clusters_csv_path.replace("*", f"{date}")

    return (
        ucc_B_file_out,
        ucc_C_file_out,
        zenodo_members_file,
        plots_record_path,
        temp_plots_record_path,
        temp_entries_path,
        ucc_entries_path,
        temp_members_files_folder,
        temp_image_path,
        temp_dbs_tables_path,
        temp_cmmts_tables_path,
        ucc_dbs_tables_path,
        ucc_cmmts_tables_path,
        old_gz_CSV_path,
        new_clusters_csv_path,
    )


def load_data(
    logging,
    ucc_B_file_out,
    ucc_C_file_out,
    zenodo_members_file,
    plots_record_path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    set,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict,
    dict,
    str,
    str,
]:
    """
    Load required data files and return them as a tuple
    """

    # Load current members file
    df_members = pd.read_parquet(zenodo_members_file)
    # Entries whose plots need to be generated/updated
    fnames_plots_updt, df_hash_old, df_hash_curr = get_fnames_plots_updt(
        logging, df_members, plots_record_path
    )

    # Load current CSV data files
    all_names = pd.read_csv(data_folder + all_OC_names)
    df_UCC_B = load_BC_cats("B", ucc_B_file_out)
    # Add columns to B cat
    df_UCC_B[["fnames", "Names"]] = all_names[["fnames", "Names"]]

    logging.info(f"\nFile {ucc_B_file_out} loaded ({len(df_UCC_B)} entries)")
    df_UCC_C = load_BC_cats("C", ucc_C_file_out)
    logging.info(f"File {ucc_C_file_out} loaded ({len(df_UCC_C)} entries)")

    # Check B and C alignment
    if df_UCC_B["fname"].to_list() == df_UCC_C["fname"].to_list() is False:
        raise ValueError("The 'fname' columns in B and C dataframes differ")
    # Drop fname from B to avoid duplicate column
    df_UCC_B = df_UCC_B.drop(columns=["fname"])
    # Merge df_UCC_B and df_UCC_C dataframes
    df_BC = pd.concat([df_UCC_B, df_UCC_C], axis=1)

    # Load clusters data in JSON file
    with open(name_DBs_json) as f:
        DBs_JSON = json.load(f)

    # Read the data for every DB in the UCC as a pandas DataFrame
    DBs_full_data = {}
    for k, v in DBs_JSON.items():
        # Don't load DBs that only have comments files, as they dont have a CSV
        # file with the full data
        if v["data_cmmts"] != "comments":
            DBs_full_data[k] = pd.read_csv(dbs_folder + k + ".csv")

    # Build a dictionary mapping each fname to its canonical fname
    all_fnames_dict = {}
    for fnames_lst in df_BC["fnames"]:
        fnames = fnames_lst.split(";")
        fname0 = fnames[0]
        for fname in fnames:
            all_fnames_dict[fname] = fname0

    cmmts_JSONS_dict = {}
    # For each DB stored in the cmmts/ folder
    for fname_csv in os.listdir(UCC_cmmts_folder):
        DB_id = fname_csv.replace(".csv", "")
        multiple_names_check = []
        df = pd.read_csv(os.path.join(UCC_cmmts_folder, fname_csv))

        # Sanity check
        if "Cluster" not in df.columns or "Comment" not in df.columns:
            raise ValueError(
                f"File {fname_csv} must contain 'Cluster' and 'Comment' columns"
            )

        # Extract the fnames for all objects in this DB's comments file
        DB_names = [c.replace("_", " ").replace(",", ", ") for c in df["Cluster"]]
        DB_fnames = get_fnames(DB_names)

        fnames_cmmts = defaultdict(list)
        fnames_orig_names = defaultdict(list)
        # For each cluster in this DB
        for fnames, comment, orig_name in zip(DB_fnames, df["Comment"], DB_names):
            # Find any assigned name that exists in the UCC
            fnames0 = list(
                dict.fromkeys(
                    all_fnames_dict[_]
                    for _ in fnames
                    if all_fnames_dict.get(_) is not None
                )
            )

            # This should not happen
            if len(fnames0) > 1:
                multiple_names_check.append((fnames, fnames0))
                fnames0 = [fnames0[0]]

            if fnames0:
                fname0 = fnames0[0]
                fnames_cmmts[fname0].append(comment)
                fnames_orig_names[fname0].append(orig_name)

        if len(multiple_names_check) > 0:
            logging.warning(
                f"\nWARNING: Multiple UCC names found for the same cluster in {DB_id}"
            )
            logging.warning("[DB cmmt entry] --> [multiple names in all_names]")
            for names_db, names_all in multiple_names_check:
                logging.warning(f"{names_db} --> {names_all}")
            logging.warning("\nUsing the first  canonical fname")

        cmmts_JSONS_dict[DB_id] = {
            "art_name": DBs_JSON[DB_id]["authors"],
            "art_year": DBs_JSON[DB_id]["year"],
            "art_url": DBs_JSON[DB_id]["SCIX_url"],
            "clusters": dict(fnames_cmmts),
            "cl_orig_names": dict(fnames_orig_names),
        }

    # --- sort by year (descending) ---
    cmmts_JSONS_dict = dict(
        sorted(
            cmmts_JSONS_dict.items(),
            key=lambda x: x[1]["art_year"],
            reverse=True,
        )
    )

    # Assign GLON bins to clusters (used to generate split members files)
    edges = [int(_) for _ in np.linspace(0, 360, 91)]
    labels = [f"{edges[i]}_{edges[i + 1]}" for i in range(len(edges) - 1)]
    df_BC["bin"] = pd.cut(
        df_BC["GLON_m"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    # Load DATABASE.md file
    with open(root_ucc_path + databases_md_path) as file:
        database_md = file.read()

    # Load ARTICLES.md file
    with open(root_ucc_path + articles_md_path) as file:
        articles_md = file.read()

    return (
        df_members,
        df_BC,
        fnames_plots_updt,
        df_hash_old,
        df_hash_curr,
        DBs_JSON,
        DBs_full_data,
        cmmts_JSONS_dict,
        database_md,
        articles_md,
    )


def get_fnames_plots_updt(
    logging, df_members: pd.DataFrame, plots_record_file: Path
) -> tuple[set, pd.DataFrame, pd.DataFrame]:
    """
    Compare the hashes of the current members with those stored in
    'plots_record_file', i.e. the members used to generate the plots currently
    in the site, and return the entries whose plots need to be
    generated/updated: those not in the record, or with a different number of
    members or hash.

    If the record does not exist yet, the latest members file archived by the C
    script is used instead (to bootstrap the record).

    Returns the entries to update, and the old and current hashes.
    """
    if plots_record_file.is_file():
        df_hash_old = pd.read_csv(plots_record_file, dtype={"hash": str})
        ref_txt = str(plots_record_file)
    else:
        pattern = re.compile(
            re.escape(Path(UCC_members_file).stem) + r"_\d{8}\.parquet"
        )
        archived = sorted(
            f for f in os.listdir(archive_folder_path) if pattern.fullmatch(f)
        )
        if archived:
            ref_txt = archive_folder_path + archived[-1]
            df_hash_old = members_hashes(pd.read_parquet(ref_txt))
        else:
            # Assume all the plots in the site are up to date
            ref_txt = "current members file (no record or archived file found)"
            df_hash_old = members_hashes(df_members)
        logging.info(
            f"\nWARNING: file '{plots_record_file}' not found, using '{ref_txt}'"
        )

    df_hash_curr = members_hashes(df_members)
    df_m = df_hash_curr.merge(df_hash_old, on="name", how="left", suffixes=("", "_old"))
    msk = (df_m["N_membs"] != df_m["N_membs_old"]) | (df_m["hash"] != df_m["hash_old"])
    fnames_updt = set(df_m.loc[msk, "name"])

    logging.info(
        f"\n{len(fnames_updt)} entries with new/changed members vs '{ref_txt}'"
    )
    return fnames_updt, df_hash_old, df_hash_curr


def updt_plots_record(
    logging,
    fnames_plots_updt: set,
    df_hash_old: pd.DataFrame,
    df_hash_curr: pd.DataFrame,
    plots_record_file: Path,
    temp_plots_record_file: Path,
) -> None:
    """
    Generate the updated plots record, moved into place by 'move_files' along
    with the plots.

    Entries flagged for update take their current hash only if both their GC and
    CMD plots exist in the temp folder (i.e., they will be moved to the site);
    otherwise they keep their old hash (or are left out of the record if new),
    so they are flagged again in the next run. All other entries take their
    current hash, and entries no longer in the members file are dropped.
    """
    pending = {
        fname
        for fname in fnames_plots_updt
        if not all(
            Path(
                f"{temp_folder}{plots_folder}plots_{fname[0]}/{fold}/{fname}.webp"
            ).is_file()
            for fold in ("gcpos", "UCC")
        )
    }
    df_hash_new = pd.concat(
        [
            df_hash_curr[~df_hash_curr["name"].isin(pending)],
            df_hash_old[df_hash_old["name"].isin(pending)],
        ]
    ).sort_values("name", ignore_index=True)

    if plots_record_file.is_file():
        df_stored = pd.read_csv(plots_record_file, dtype={"hash": str})
        if df_stored.equals(df_hash_new):
            return

    df_hash_new.to_csv(temp_plots_record_file, index=False)
    logging.info(
        f"\nPlots record file '{temp_plots_record_file}' generated "
        f"({len(pending)} flagged entries still without plots)"
    )


def updt_ucc_cluster_plots(
    logging, df_BC, fnames_plots_updt, df_members, min_UTI=0.5
) -> list:
    """
    Generate plots for each cluster in the UCC database and update the 'plot_used'
    column in the dataframe.
    """
    # Ask if the Aladin plots should be generated if the file already exists
    overwrite_aladin = (
        input("\nRe-generate existing Aladin plots? (y/n): ").strip().lower() == "y"
    )
    overwrite_temp = (
        input("\nOverwrite stored temp plots? (y/n): ").strip().lower() == "y"
    )
    logging.info("\nGenerating plot files")

    # Velocities used for GC plot
    vx, vy, vz, vR = ucc_plots.velocity(
        df_BC["RA_ICRS_m"].values,
        df_BC["DE_ICRS_m"].values,
        df_BC["Plx_m"].values,
        df_BC["pmRA_m"].values,
        df_BC["pmDE_m"].values,
        df_BC["Rv_m"].values,
        df_BC["X_GC"].values,
        df_BC["Y_GC"].values,
        df_BC["Z_GC"].values,
        df_BC["R_GC"].values,
    )

    # Good OCs used for GC plot
    msk = df_BC["UTI"] > min_UTI
    Z_uti = df_BC["Z_GC"][msk]
    R_uti = df_BC["R_GC"][msk]

    # plots_generated = {"aladin": [], "GC": [], "CMD": []}
    fnames_processed = []
    # Iterate trough each entry in the UCC database
    for i_ucc, UCC_cl in df_BC.iterrows():
        fname0 = str(UCC_cl["fname"])
        txt = ""

        # Make Aladin plot if image file does not exist or if marked for update.
        # Path to original image
        orig_aladin_path = (
            f"{root_ucc_path}{plots_folder}plots_{fname0[0]}/aladin/{fname0}.webp"
        )
        # Path to temp file
        temp_aladin_path = (
            f"{temp_folder}{plots_folder}plots_{fname0[0]}/aladin/{fname0}.webp"
        )
        generate_aladin = False
        if Path(orig_aladin_path).is_file() is False:
            # Always generate the plot if the original image does not exist
            generate_aladin = True
        else:  # the original image exists
            # If file is flagged for update and overwriting is enabled
            if fname0 in fnames_plots_updt and overwrite_aladin is True:
                # If the temporary image does not exist
                if Path(temp_aladin_path).is_file() is False:
                    generate_aladin = True
                else:
                    if overwrite_temp is True:
                        generate_aladin = True
        if generate_aladin:
            ucc_plots.plot_aladin(
                logging,
                UCC_cl["RA_ICRS_m"],
                UCC_cl["DE_ICRS_m"],
                UCC_cl["r_50"],
                temp_aladin_path,
            )
            txt += " Aladin plot generated |"

        # Make GC and CMD plots
        # Check if this OC's plot should be generated/updated
        if fname0 in fnames_plots_updt:
            # Read members
            df_membs = df_members[df_members["name"] == fname0]

            # Temp path were the GC file will be stored
            temp_gc_path = (
                f"{temp_folder}{plots_folder}plots_{fname0[0]}/gcpos/{fname0}.webp"
            )
            # Generate the GC plot if the temporary image does not exist
            if Path(temp_gc_path).is_file() is False or overwrite_temp is True:
                ucc_plots.plot_gcpos(
                    temp_gc_path,
                    Z_uti,
                    R_uti,
                    UCC_cl["X_GC"],
                    UCC_cl["Y_GC"],
                    UCC_cl["Z_GC"],
                    UCC_cl["R_GC"],
                    vx[i_ucc],
                    vy[i_ucc],
                    vz[i_ucc],
                    vR[i_ucc],
                )
                txt += " GC plot generated |"

            # Temp path were the CMD file will be stored
            temp_cmd_path = (
                f"{temp_folder}{plots_folder}plots_{fname0[0]}/UCC/{fname0}.webp"
            )
            # Generate the CMD plot if the temporary image does not exist
            if Path(temp_cmd_path).is_file() is False or overwrite_temp is True:
                ucc_plots.plot_CMD(temp_cmd_path, df_membs)
                txt += " CMD plot generated |"

        if txt != "":
            logging.info(f"{fname0} -->" + txt + f" ({i_ucc})")
            fnames_processed.append(fname0)

    return fnames_processed


def UTI_to_hex(df_UCC):
    """Convert UTI value and C coefficients to hex colors"""

    def build_lut(cmap, n=256, soft=0.65):
        xs = np.linspace(0, 1, n)
        rgb = cmap(xs)[:, :3]
        rgb = soft + (1 - soft) * rgb
        return (rgb * 255).astype(np.uint8)

    cmap = plt.get_cmap("RdYlGn")
    lut = build_lut(cmap)

    def UTI_to_hex_array(x):
        idx = np.clip((x * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in lut[idx]]

    UTI_colors = {
        "UTI": UTI_to_hex_array(df_UCC["UTI"]),
        "C_N": UTI_to_hex_array(df_UCC["C_N"]),
        "C_dens": UTI_to_hex_array(df_UCC["C_dens"]),
        "C_C3": UTI_to_hex_array(df_UCC["C_C3"]),
        "C_lit": UTI_to_hex_array(df_UCC["C_lit"]),
        "C_dup": UTI_to_hex_array(df_UCC["C_dup"]),
    }
    return UTI_colors


def updt_ucc_cluster_files(
    logging,
    ucc_entries_path,
    temp_entries_path,
    DBs_full_data,
    df_BC,
    DBs_JSON,
    cmmts_JSONS_dict,
):
    """
    Generate/update markdown files for each cluster in the UCC database.
    """
    logging.info("\nGenerating md files")

    # Pre-process the data used by every entry, to avoid slow per-entry lookups
    DBs_pos = ucc_entry.pos_columns(DBs_JSON, DBs_full_data)
    shared_data = ucc_entry.shared_members_data(df_BC)
    plots_existing = ucc_entry.existing_plots()

    UTI_colors = UTI_to_hex(df_BC)

    members_files_mapping = {
        fname: bin_label for fname, bin_label in df_BC[["fname", "bin"]].values
    }

    current_year = datetime.datetime.now(tz=datetime.UTC).year

    # ran_i = np.random.randint(0, len(df_BC), size=50)

    N_total = 0
    # Iterate trough each entry in the UCC database
    cols = df_BC.columns
    for i_ucc, UCC_cl in enumerate(df_BC.itertuples(index=False, name=None)):
        UCC_cl = dict(zip(cols, UCC_cl))
        fname0 = str(UCC_cl["fname"])

        # if fname0 not in ("ngc2516",):
        #     continue
        # if "melotte" not in fname0:
        #     continue
        # if i_ucc not in ran_i or "cwnu" in fname0 or "cwwdl" in fname0:
        #     continue

        summary, descriptors, fpars_badges, badges_url, comments_lst = (
            ucc_summ_cmmts.run(current_year, UCC_cl, DBs_JSON, cmmts_JSONS_dict)
        )

        # Generate full entry
        new_md_entry = ucc_entry.make(
            fname0,
            i_ucc,
            DBs_JSON,
            members_files_mapping,
            DBs_pos,
            shared_data,
            plots_existing,
            UCC_cl,
            UTI_colors,
            summary,
            descriptors,
            fpars_badges,
            badges_url,
            comments_lst,
        )

        # Compare old md file (if it exists) with the new md file, for this cluster
        txt = ""
        try:
            # Read old entry
            with open(ucc_entries_path / (fname0 + ".md"), "r") as f:
                old_md_entry = f.read()
            # Check if entry needs updating
            if new_md_entry != old_md_entry:
                txt = "md updated |"
        except FileNotFoundError:
            # This is a new OC with no md entry yet
            txt = "md generated |"

        if txt != "":
            # Generate/update entry
            with open(temp_entries_path / (fname0 + ".md"), "w") as f:
                f.write(new_md_entry)
            N_total += 1
            if N_total < 1000:
                logging.info(f"{N_total} -> {fname0}: " + txt + f" ({i_ucc})")
            elif N_total == 1000:
                logging.info("updating more files...")

    logging.info(f"\nN={N_total} OCs processed")

    # # Delete all files in folder2 and move all files from folder1 to folder2
    # folder1 = "/home/gabriel/Github/UCC/updt_UCC/temp_updt/ucc/_clusters"
    # folder2 = "/home/gabriel/Github/UCC/ucc/_clusters2"
    # for file in os.listdir(folder2):
    #     file_path = os.path.join(folder2, file)
    #     os.remove(file_path)
    # for file in os.listdir(folder1):
    #     file_path = os.path.join(folder1, file)
    #     new_file_path = os.path.join(folder2, file)
    #     os.rename(file_path, new_file_path)
    # print("\nAll files moved")
    # breakpoint()


def updt_eq_positions(logging, df_BC, DBs_JSON, DBs_full_data):
    """
    Generate the gzipped JSON file with the (RA, DEC) positions given by each article
    for every cluster in the UCC. Same data used in the 'Astrometry' table.

    To reduce its size, each reference is stored once in the 'refs' list and the
    clusters point to it by its index. The UCC (RA, DEC) values come right after
    the cluster's name (null if missing):

    {"refs": ["Alfonso et al. 2024", ...],
     "clusters": {"ngc2516": ["NGC 2516", RA_UCC, DEC_UCC, [ref_idx, RA, DEC], ...]}}
    """

    def ucc_coord(val):
        if val == "" or pd.isna(val):
            return None
        return round(float(val), 3)

    DBs_pos = ucc_entry.pos_columns(DBs_JSON, DBs_full_data)

    cl_positions = {}
    cols = ["fname", "Names", "DB", "DB_i", "RA_ICRS_m", "DE_ICRS_m"]
    for fname, names, DB, DB_i, ra, dec in df_BC[cols].values:
        UCC_cl = {"DB": DB, "DB_i": DB_i}
        cl_positions[str(fname)] = (
            str(names).split(";")[0],
            ucc_coord(ra),
            ucc_coord(dec),
            ucc_entry.eq_positions_in_lit(DBs_JSON, DBs_pos, UCC_cl),
        )

    # Sorted so that the indexes are stable across runs
    refs = sorted({r[0] for *_, pos in cl_positions.values() for r in pos})
    refs_idx = {r: i for i, r in enumerate(refs)}
    eq_positions = {
        "refs": refs,
        "clusters": {
            fname: [name, ra, dec] + [[refs_idx[r[0]], r[1], r[2]] for r in pos]
            for fname, (name, ra, dec, pos) in cl_positions.items()
        },
    }

    # Only write the file if it changed
    ucc_eq_pos_path = root_ucc_path + assets_folder + eq_positions_path
    try:
        with gzip.open(ucc_eq_pos_path, "rt") as f:
            old_eq_positions = json.load(f)
    except FileNotFoundError:
        old_eq_positions = None

    if eq_positions == old_eq_positions:
        logging.info(f"File '{eq_positions_path}' not updated (no changes)")
        return

    data = json.dumps(eq_positions, separators=(",", ":")).encode()
    with open(temp_folder + assets_folder + eq_positions_path, "wb") as f:
        f.write(gzip.compress(data, mtime=0))
    logging.info(f"File '{eq_positions_path}' updated")


def write_bin(args):
    """Function to write a single bin to disk"""
    out_fname, df_bin = args
    df_bin.to_csv(out_fname, index=False, compression="gzip")
    return out_fname


def updt_members_files(df_ucc, df_membs, temp_members_files_folder):
    """
    Split the large members file into smaller files based on GLON bins for each cluster.
    """
    # Map members to bins
    cluster_to_bin = df_ucc.set_index("fname")["bin"].map(
        lambda _bin: temp_members_files_folder / f"membs_{_bin}.csv.gz"
    )

    membs = pd.DataFrame(df_membs)
    membs["bin"] = membs["name"].map(cluster_to_bin)
    membs = membs.dropna(subset=["bin"])

    # Group and write in parallel
    groups = list(membs.groupby("bin", sort=False, observed=True))

    print(f"Writing {len(groups)} .csv.gz files in parallel...")
    with ProcessPoolExecutor() as exe:
        for _ in exe.map(write_bin, groups):
            pass


def updt_cls_CSV(
    logging,
    new_clusters_csv_path: str,
    df_BC: pd.DataFrame,
) -> None:
    """
    Update compressed cluster.csv.gz file used by 'ucc.ar' search
    """
    # Extract the first identifier from the "ID" column
    df_BC["Name"] = [_.split(";")[0] for _ in df_BC["Names"]]
    df_BC["RA_ICRS"] = np.round(df_BC["RA_ICRS_m"], 2)
    df_BC["DE_ICRS"] = np.round(df_BC["DE_ICRS_m"], 2)
    df_BC["GLON"] = np.round(df_BC["GLON_m"], 2)
    df_BC["GLAT"] = np.round(df_BC["GLAT_m"], 2)
    df_BC["N_membs"] = df_BC["N_membs"].astype(int)

    # Replace "OC": "O", "EC": "E", "EC;OC": "EO"
    df_BC["Type"] = df_BC["Type"].replace({"OC": "O", "EC": "E", "EC;OC": "EO"})

    # Compute parallax-based distances in parsecs
    dist_pc = 1000 / np.clip(np.array(df_BC["Plx_m"]), a_min=0.0000001, a_max=np.inf)
    dist_pc = np.clip(dist_pc, a_min=10, a_max=50000)
    df_BC["dist_plx_pc"] = np.round(dist_pc, 0)

    df_new = pd.DataFrame(
        df_BC[
            [
                "Name",
                "Type",
                "fnames",
                "RA_ICRS",
                "DE_ICRS",
                "GLON",
                "GLAT",
                "dist_median",
                "av_median",
                "diff_ext_median",
                "age_median",
                "met_median",
                "mass_median",
                "bi_frac_median",
                "blue_str_median",
                "N_membs",
                "P_dup",
                "UTI",
                "bad_oc",
                "dist_plx_pc",  # clreg_plot, mapPlotter
                "r_50",  # clreg_plot
            ]
        ]
    )
    df_new = df_new.sort_values("Name").reset_index(drop=True)

    df_new.rename(
        columns={
            "dist_median": "dist",
            "av_median": "av",
            "diff_ext_median": "diff_ext",
            "age_median": "age",
            "met_median": "met",
            "mass_median": "mass",
            "bi_frac_median": "bi_frac",
            "blue_str_median": "blue_str",
        },
        inplace=True,
    )

    # Update CSV
    temp_gz_CSV_path = temp_folder + assets_folder + new_clusters_csv_path
    df_new.to_csv(
        temp_gz_CSV_path,
        index=False,
        compression="gzip",
    )
    # Update the 'latest' key in the 'clusters_manifest.json' JSON file
    csv_manifest_path = root_ucc_path + assets_folder + clusters_manifest_path
    with open(csv_manifest_path, "r") as f:
        manifest_data = json.load(f)
    manifest_data["latest"] = new_clusters_csv_path
    with open(temp_folder + assets_folder + clusters_manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logging.info(f"File '{new_clusters_csv_path}' updated")


def make_site_plots(logging, temp_image_path, df_BC):
    """
    Generate site-wide plots for the UCC database.
    """
    ucc_plots.make_N_vs_year_plot(temp_image_path / "catalogued_ocs.webp", df_BC)
    logging.info("Plot generated: number of OCs vs years")

    # Count number of OCs in each class
    OCs_per_class = ucc_updt_tables.count_OCs_classes(df_BC["C3"], class_order)
    ucc_plots.make_classif_plot(
        temp_image_path / "classif_bar.webp", OCs_per_class, class_order
    )
    logging.info("Plot generated: classification histogram")

    ucc_plots.make_UTI_plot(temp_image_path / "UTI_values.webp", df_BC["UTI"])
    logging.info("Plot generated: UTI histogram")


def count_fpars(df):
    """
    Count the number of non-empty fundamental parameters in the UCC database.
    """

    def is_number(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    N_pars = {_: 0 for _ in fpars_order}
    for col in fpars_order:
        count = 0
        for row in df[col].values:
            if str(row) != "nan":
                if ";" in str(row):
                    count += sum(
                        is_number(x.replace("*", "")) for x in str(row).split(";")
                    )
                else:
                    count += 1
        N_pars[col] = count

    return sum([v for k, v in N_pars.items()])


def updt_indiv_tables(
    logging,
    temp_dbs_tables_path,
    temp_cmmts_tables_path,
    ucc_dbs_tables_path,
    ucc_cmmts_tables_path,
    current_JSON,
    df_BC,
    cmmts_JSONS_dict: dict,
):
    """
    Update tables for individual databases and comments, and save them to temporary
    paths.
    """
    # New columns used to display in tables
    df_BC["Name"] = [_.split(";")[0] for _ in df_BC["Names"]]
    names_url = []
    for _, cl in df_BC.iterrows():
        name = str(cl["Name"]).split(";")[0]
        color = "red" if cl["bad_oc"] == "y" else "$blue"
        fname = str(cl["fname"])
        url = r"{{ site.baseurl }}/_clusters/" + fname + "/"
        clname = rf'<a href="{url}" target="_blank" style="color: {color};">{name}</a>'
        # names_url.append(f"[{name}]({url})")
        names_url.append(clname)
    df_BC["ID_url"] = names_url
    df_BC["RA_ICRS"] = np.round(df_BC["RA_ICRS_m"], 2)
    df_BC["DE_ICRS"] = np.round(df_BC["DE_ICRS_m"], 2)
    df_BC["GLON"] = np.round(df_BC["GLON_m"], 2)
    df_BC["GLAT"] = np.round(df_BC["GLAT_m"], 2)
    df_BC["Plx_m_round"] = np.round(df_BC["Plx_m"], 2)
    df_BC["N_membs"] = df_BC["N_membs"].astype(int)
    df_BC["C3_abcd"] = [ucc_entry.color_C3(_) for _ in df_BC["C3"]]
    df_BC = df_BC.sort_values("Name").reset_index()

    DBs_dups_badOCs = ucc_updt_tables.count_dups_bad_OCs(current_JSON, df_BC)

    # Update pages for individual databases
    new_tables_dict = ucc_updt_tables.updt_DBs_tables(
        current_JSON, df_BC, cmmts_JSONS_dict, DBs_dups_badOCs
    )

    # Update/generate files with tables for individual databases
    ucc_updt_tables.general_table_update(
        logging,
        ucc_dbs_tables_path,
        ucc_cmmts_tables_path,
        temp_dbs_tables_path,
        temp_cmmts_tables_path,
        new_tables_dict,
    )


def update_main_pages(
    logging,
    N_fpars,
    N_members_UCC,
    current_JSON,
    df_UCC,
    database_md,
    articles_md,
    temp_cmmts_tables_path,
):
    """Update main .md files"""
    logging.info("\nUpdating main .md files")
    N_updt = 0

    # Update DATABASE
    N_db_UCC, N_cl_UCC = len(current_JSON), len(df_UCC)
    # Update the total number of entries, databases, and members in the UCC
    database_md_updt = ucc_updt_tables.ucc_n_total_updt(
        logging, N_db_UCC, N_cl_UCC, N_fpars, N_members_UCC, database_md
    )
    if database_md != database_md_updt:
        with open(temp_folder + databases_md_path, "w") as file:
            file.write(database_md_updt)
        logging.info("DATABASE.md updated")
        N_updt += 1

    N_in_DB, N_cmmts_dict = ucc_updt_tables.count_OCs_in_tables(
        df_UCC, current_JSON, temp_cmmts_tables_path
    )

    # Update ARTICLES
    articles_md_updt = ucc_updt_tables.updt_articles_table(
        current_JSON, articles_md, N_in_DB, N_cmmts_dict
    )
    if articles_md != articles_md_updt:
        with open(temp_folder + articles_md_path, "w") as file:
            file.write(articles_md_updt)
        logging.info("ARTICLES.md updated")
        N_updt += 1

    if N_updt == 0:
        logging.info("No tables updated (DATABASE, ARTICLES)")


def move_files(
    logging,
    plots_record_path: Path,
    temp_plots_record_path: Path,
    old_gz_CSV_path: str,
    new_clusters_csv_path: str,
) -> None:
    """Move files with user confirmation."""

    planned_actions = []

    # --- Collect plot moves ---
    all_plot_folds = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        letter_fold = plots_folder + f"plots_{letter}/"
        for fold in plots_sub_folders:
            temp_fpath = temp_folder + letter_fold + fold + "/"
            fpath = root_ucc_path + letter_fold + fold + "/"
            if os.path.exists(temp_fpath):
                for file in os.listdir(temp_fpath):
                    planned_actions.append(("move", temp_fpath + file, fpath + file))
                    all_plot_folds.append(fpath)

    # --- Updated plots record file ---
    if os.path.exists(temp_plots_record_path):
        planned_actions.append(("move", temp_plots_record_path, plots_record_path))

    # --- Delete old clusters CSV file ---
    if os.path.exists(new_clusters_csv_path):
        planned_actions.append(("delete", old_gz_CSV_path, ""))

    # --- Move files inside temporary ucc/ ---
    temp_ucc_fold = temp_folder + ucc_path
    for root, dirs, files in os.walk(temp_ucc_fold):
        for filename in files:
            src = os.path.join(root, filename)
            dst = src.replace(temp_folder, root_ucc_path)
            planned_actions.append(("move", src, dst))

    if not planned_actions:
        logging.info("\nNo operations to apply.")
        return

    # --- Show planned actions ---
    logging.info("\nPlanned actions:")

    # Extract actions related to '_clusters', 'plots', and 'members' separately
    cluster_actions = [a for a in planned_actions if "_clusters" in str(a[2])]
    plot_actions = [a for a in planned_actions if "plots" in str(a[2])]
    members_actions = [a for a in planned_actions if "members" in str(a[2])]
    other_actions = [
        a
        for a in planned_actions
        if "_clusters" not in str(a[2])
        and "plots" not in str(a[2])
        and "members" not in str(a[2])
    ]
    for action, src, dst in other_actions:
        if action == "move":
            logging.info(f"  MOVE     {src} -> {dst}")
        elif action == "replace":
            logging.info(f"  REPLACE  {src} -> {dst}")
        elif action == "delete":
            logging.info(f"  DELETE   {src}")

    if len(cluster_actions) > 100:
        logging.info(
            f"  MOVE     {len(cluster_actions)} '_clusters/*.md' files will be updated"
        )
    else:
        for action, src, dst in cluster_actions:
            logging.info(f"  MOVE     {src} -> {dst}")
    if len(plot_actions) > 100:
        logging.info(f"  MOVE     {len(plot_actions)} 'plots/*' files will be updated")
    else:
        for action, src, dst in plot_actions:
            logging.info(f"  MOVE     {src} -> {dst}")
    if len(members_actions) > 0:
        logging.info(
            f"  MOVE     {len(members_actions)} 'members/*' files will be updated"
        )

    # --- Ask for confirmation ---
    resp = input("\nProceed with these actions? [y/N]: ").strip().lower()
    if resp != "y":
        logging.info("Operation cancelled by user. No changes applied.")
        return

    # --- Apply actions ---
    logging.info("\nApplying actions:")

    if cluster_actions:
        logging.info(
            f"_temp_updt/ucc/clusters/*.md -> ucc/_clusters/*.md ({len(cluster_actions)} files)"
        )

    clusters_plots_actions = []
    members_actions = 0
    for action, src, dst in planned_actions:
        if action == "move":
            os.rename(src, dst)
            if "_clusters" in str(dst) or "plots" in str(dst):
                clusters_plots_actions.append(f"{src} -> {dst}")
            elif "members" in str(dst):
                members_actions += 1
            else:
                logging.info(f"{src} -> {dst}")
        elif action == "replace":
            os.replace(src, dst)
            logging.info(f"{src} -> {dst}")
        elif action == "delete":
            os.remove(src)
            logging.info(f"Deleted: {src}")

    if members_actions > 0:
        logging.info(f"{members_actions} 'members/*' files updated")

    if len(clusters_plots_actions) > 100:
        logging.info(
            f"{len(clusters_plots_actions)} files in '_clusters/*.md' and 'plots/*' updated"
        )
    else:
        for action in clusters_plots_actions:
            logging.info(action)

    logging.info("\nAll files moved into place")


def file_checker(logging, df_BC: pd.DataFrame) -> None:
    """Check the number and types of files in directories for consistency.

    Parameters:
    - logging: Logger instance for recording messages.

    Returns:
    - None
    """
    dbs_plots_folders = {"HUNT2023": [], "CANTAT2020": []}
    for fname, DB in zip(df_BC["fname"], df_BC["DB"]):
        for db_name, db_lst in dbs_plots_folders.items():
            if db_name in DB:
                db_lst.append(fname)
    # Change key names to match folders
    dbs_plots_folders["HUNT23"] = dbs_plots_folders.pop("HUNT2023")
    dbs_plots_folders["CANTAT20"] = dbs_plots_folders.pop("CANTAT2020")

    logging.info("\nChecking files")
    # Read stored final version
    df_UCC_C = pd.read_csv(data_folder + UCC_cat_C_out, usecols=["fname"])
    flag_error = False

    df_UCC_fname = df_UCC_C["fname"].to_list()

    # Check that all md_files match the elements in df_UCC_fname
    md_files = os.listdir(root_ucc_path + md_folder)
    md_fnames = sorted([_[:-3] for _ in md_files])
    # Print to screen which elements are different in both lists
    for f in md_fnames:
        if f not in df_UCC_fname:
            logging.warning(f"{f} (.md) not in UCC catalog")
            flag_error = True
    for f in df_UCC_fname:
        if f not in md_fnames:
            logging.warning(f"{f} (UCC) not in md files")
            flag_error = True
    logging.info("")

    # Check that all plots match the elements in df_UCC_fname
    for letter in "abcdefghijklmnopqrstuvwxyz":
        # Extract elements in df_UCC_fname that start with this letter
        ucc_webp = [_ for _ in df_UCC_fname if _.startswith(letter)]
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

    missing_plots_h23_c20 = []
    for db_folder, fnames in dbs_plots_folders.items():
        for fname in fnames:
            end_path = db_folder + f"/{fname}.webp"
            fname_path = root_ucc_path + plots_folder + f"plots_{fname[0]}/" + end_path
            if not os.path.exists(fname_path):
                missing_plots_h23_c20.append(f"{fname_path} not found")
    if missing_plots_h23_c20:
        flag_error = True
        logging.warning(
            "WARNING: some plots from the original databases were not found\n"
        )
        for t in missing_plots_h23_c20:
            logging.warning(t)

    if flag_error:
        print("\n\n")
        raise ValueError("\nERRORS WERE DETECTED associated to the files")

    logging.warning("All checks passed\n")


if __name__ == "__main__":
    main()
