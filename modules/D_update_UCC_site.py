import csv
import datetime
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .D_funcs import ucc_entry, ucc_plots, ucc_summ_cmmts, ucc_updt_tables
from .utils import get_fnames, logger
from .variables import (
    UCC_cmmts_file,
    UCC_cmmts_folder,
    UCC_members_file,
    articles_md_path,
    assets_folder,
    class_order,
    clusters_csv_path,
    clusters_manifest_path,
    data_folder,
    databases_md_path,
    dbs_folder,
    dbs_tables_folder,
    fpars_order,
    images_folder,
    md_folder,
    members_files_folder,
    merged_dbs_file,
    name_DBs_json,
    pages_folder,
    plots_folder,
    plots_sub_folders,
    root_ucc_path,
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
        ucc_B_file,
        ucc_C_file,
        zenodo_members_file,
        ucc_C_file,
        temp_C_path,
        UCC_cmmts_path,
        temp_UCC_cmmts_path,
        temp_entries_path,
        ucc_entries_path,
        temp_members_files_folder,
        temp_image_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        old_gz_CSV_path,
    ) = load_paths(logging)

    # Load required files
    (
        df_members,
        df_C,
        df_BC,
        DBs_JSON,
        DBs_full_data,
        cmmts_JSONS_lst,
        database_md,
        articles_md,
        df_clusters_CSV_current,
    ) = load_data(logging, ucc_B_file, ucc_C_file, zenodo_members_file, old_gz_CSV_path)

    ###########################################
    # Update clusters .webp files
    N_plots_updt = (df_C["plot_used"] == "n").sum()
    if N_plots_updt > 0:
        if input(f"\nUpdate {N_plots_updt} cluster plots? (y/n): ").lower() == "y":
            # Returns df_C dataframe with 'plot_used' column updated
            df_C_updated = updt_ucc_cluster_plots(
                logging,
                df_C,
                df_members,
            )
            if df_C_updated.empty:
                raise ValueError("No plots were generated/updated")
            # Check that all entries in df_UCC_C have plot_used='y'
            if any(df_C_updated["plot_used"] == "n"):
                raise ValueError("Some entries in C dataframe still have plot_used='n'")
            # Drop added columns from B
            df_C_updated.to_csv(
                temp_C_path,
                na_rep="nan",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            logging.info(f"\nFile '{temp_C_path}' updated")
    ###########################################

    ###########################################
    # summ_cmmts_file = "STORED"
    # if input("\nUpdate summaries and comments? (y/n): ").lower() == "y":
    #     UCC_summ_cmmts_new = ucc_summ_cmmts.run(
    #         logging,
    #         df_BC,
    #         DBs_JSON,
    #         cmmts_JSONS,
    #         B_lookup,
    #         fnames_B,
    #     )

    #     if UCC_summ_cmmts_new != UCC_summ_cmmts:
    #         UCC_summ_cmmts = UCC_summ_cmmts_new
    #         # Store new JSON file
    #         with gzip.open(temp_UCC_cmmts_path, "wt", encoding="utf-8") as f:
    #             json.dump(UCC_summ_cmmts, f)
    #         logging.info(f"Updated {temp_UCC_cmmts_path} file.")
    #         summ_cmmts_file = "GENERATED"
    #     else:
    #         logging.info("No changes in summaries and comments.")

    ###########################################

    ###########################################
    # Update per cluster md files. If no changes are expected, this step can be skipped
    if input("\nUpdate md files ? (y/n): ").lower() == "y":
        updt_ucc_cluster_files(
            logging,
            ucc_entries_path,
            temp_entries_path,
            DBs_full_data,
            df_BC,
            DBs_JSON,
            cmmts_JSONS_lst,
        )
    ###########################################

    ###########################################
    if input("\nUpdate split member files? (y/n): ").lower() == "y":
        updt_members_files(df_BC, df_members, temp_members_files_folder)
    ###########################################

    ###########################################
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
        )
        # Update tables files
        updt_indiv_tables(
            logging,
            temp_dbs_tables_path,
            ucc_dbs_tables_path,
            DBs_JSON,
            df_BC,
        )
    ###########################################

    ###########################################
    # This function modifies the df_BC dataframe, so it should be run at the end
    new_clusters_csv_path = ""
    if input("\nUpdate clusters CSV file? (y/n): ").lower() == "y":
        new_clusters_csv_path = updt_cls_CSV(logging, df_BC, df_clusters_CSV_current)
    ###########################################

    if input("\nMove files to their final destination? (y/n): ").lower() == "y":
        move_files(
            logging,
            ucc_C_file,
            temp_C_path,
            UCC_cmmts_path,
            temp_UCC_cmmts_path,
            old_gz_CSV_path,
            new_clusters_csv_path,
        )

    # Check number of files
    file_checker(logging)
    logging.info("\nAll done!")


def load_paths(
    logging,
) -> tuple[
    Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, Path, str
]:
    """ """
    data_folder_p = Path(data_folder)
    temp_folder_p = Path(temp_folder)
    root_ucc_path_p = Path(root_ucc_path)

    # Path to main data files
    ucc_B_file = data_folder_p / merged_dbs_file
    ucc_C_file = data_folder_p / ucc_cat_file
    # Temp df_C path
    temp_C_path = temp_folder_p / ucc_C_file

    # Path to large members file uploaded to Zenodo
    zenodo_members_file = Path(zenodo_folder) / UCC_members_file

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

    # UCC comments file path
    UCC_cmmts_path = data_folder_p / UCC_cmmts_file
    temp_UCC_cmmts_path = temp_folder_p / UCC_cmmts_path

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

    for p in [
        temp_data_folder,
        temp_entries_path,
        temp_assets_path,
        temp_pages_path,
        temp_members_files_folder,
        temp_image_path,
        temp_dbs_tables_path,
    ]:
        ensure_dir(p)

    ucc_entries_path = root_ucc_path_p / md_folder
    ucc_dbs_tables_path = root_ucc_path_p / dbs_tables_folder

    # UCC path to compressed CSV file
    folder = Path(root_ucc_path, assets_folder)
    matches = list(folder.glob(clusters_csv_path))
    if matches:
        old_gz_CSV_path = str(matches[0])
    else:
        raise ValueError(f"No file matching '{clusters_csv_path}' found")

    return (
        ucc_B_file,
        ucc_C_file,
        zenodo_members_file,
        ucc_C_file,
        temp_C_path,
        UCC_cmmts_path,
        temp_UCC_cmmts_path,
        temp_entries_path,
        ucc_entries_path,
        temp_members_files_folder,
        temp_image_path,
        temp_dbs_tables_path,
        ucc_dbs_tables_path,
        old_gz_CSV_path,
    )


def load_data(
    logging,
    ucc_B_file,
    ucc_C_file,
    zenodo_members_file,
    old_gz_CSV_path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict,
    list,
    str,
    str,
    pd.DataFrame,
]:
    """ """

    # Load current members file
    df_members = pd.read_parquet(zenodo_members_file)

    # Load current CSV data files
    df_UCC_B = pd.read_csv(
        ucc_B_file,
        dtype={
            "blue_str_values": "string",
        },
    )
    logging.info(f"\nFile {ucc_B_file} loaded ({len(df_UCC_B)} entries)")
    # Replace NaN values with "nan" string only in selected columns
    selected = ["frame_limit", "shared_members", "shared_members_p"]
    df_UCC_C = pd.read_csv(
        ucc_C_file,
        converters={c: lambda x: "nan" if pd.isna(x) else str(x) for c in selected},
    )

    # Check B and C alignment
    fname_B = [_.split(";")[0] for _ in df_UCC_B["fnames"]]
    if fname_B == df_UCC_C["fname"].to_list() is False:
        raise ValueError("The 'fname' columns in B and C dataframes differ")

    logging.info(f"File {ucc_C_file} loaded ({len(df_UCC_C)} entries)")
    # Merge df_UCC_B and df_UCC_C dataframes
    df_BC = pd.concat([df_UCC_B, df_UCC_C], axis=1)

    # Process blue straggler values keeping the last one (most recent) as "median"
    df_BC.rename(columns={"blue_str_values": "blue_str_median"}, inplace=True)
    df_BC["blue_str_median"] = pd.to_numeric(
        df_BC["blue_str_median"].astype(str).str.split(";").str[-1], errors="coerce"
    )

    # Load clusters data in JSON file
    with open(name_DBs_json) as f:
        DBs_JSON = json.load(f)

    # Read the data for every DB in the UCC as a pandas DataFrame
    DBs_full_data = {}
    for k, v in DBs_JSON.items():
        DBs_full_data[k] = pd.read_csv(dbs_folder + k + ".csv")

    # Load (and check) all comments JSON files
    cmmts_JSONS_lst = []
    json_cmmts_path = f"{data_folder}{UCC_cmmts_folder}"
    for fpath_json in os.listdir(json_cmmts_path):
        # Check for duplicate clusters keys in JSON and raise error if any.
        # 'json.load()' silently drops duplicates, hence the check here
        with open(f"{json_cmmts_path}{fpath_json}", "r") as f:
            raw_json = json.load(f, object_pairs_hook=list)

        clusters = dict(raw_json)["clusters"]
        if isinstance(clusters, list):
            seen = set()
            dups = {k for k, _ in clusters if k in seen or seen.add(k)}
            if dups:
                raise ValueError(
                    f'Duplicate keys in "clusters": {", ".join(sorted(dups))}'
                )

        # Generate dictionary with fnames as keys and comments as values
        cluster_names, cmmts = zip(*clusters)
        cluster_fnames = get_fnames(cluster_names)
        fnames_cmmts = {}
        for i, fname in enumerate(cluster_fnames):
            fnames_cmmts[fname[0]] = cmmts[i]
        # Store contents of comments file
        raw_json = dict(raw_json)
        raw_json["clusters"] = fnames_cmmts
        cmmts_JSONS_lst.append(raw_json)

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

    # UCC path to compressed CSV file
    df_clusters_CSV_current = pd.read_csv(old_gz_CSV_path, compression="gzip")

    return (
        df_members,
        df_UCC_C,
        df_BC,
        DBs_JSON,
        DBs_full_data,
        cmmts_JSONS_lst,
        database_md,
        articles_md,
        df_clusters_CSV_current,
    )


def updt_ucc_cluster_plots(logging, df_UCC, df_members, min_UTI=0.5) -> pd.DataFrame:
    """ """
    logging.info("\nGenerating plot files")

    # Velocities used for GC plot
    vx, vy, vz, vR = ucc_plots.velocity(
        df_UCC["RA_ICRS_m"].values,
        df_UCC["DE_ICRS_m"].values,
        df_UCC["Plx_m"].values,
        df_UCC["pmRA_m"].values,
        df_UCC["pmDE_m"].values,
        df_UCC["Rv_m"].values,
        df_UCC["X_GC"].values,
        df_UCC["Y_GC"].values,
        df_UCC["Z_GC"].values,
        df_UCC["R_GC"].values,
    )

    # Good OCs used for GC plot
    msk = df_UCC["UTI"] > min_UTI
    Z_uti = df_UCC["Z_GC"][msk]
    R_uti = df_UCC["R_GC"][msk]

    N_total = 0
    # Iterate trough each entry in the UCC database
    for i_ucc, UCC_cl in df_UCC.iterrows():
        fname0 = str(UCC_cl["fname"])
        txt = ""

        # Make Aladin plot if image file does not exist. These images are not updated
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
                logging,
                UCC_cl["RA_ICRS_m"],
                UCC_cl["DE_ICRS_m"],
                UCC_cl["r_50"],
                save_plot_file,
            )
            txt += " Aladin plot generated |"

        # Make GC and CMD plots
        # Check if this OC's plot should be updated or generated
        if UCC_cl["plot_used"] == "n":
            # Read members
            df_membs = df_members[df_members["name"] == fname0]

            # Temp path were the GC file will be stored
            temp_plot_file = (
                f"{temp_folder}{plots_folder}plots_{fname0[0]}/gcpos/{fname0}.webp"
            )
            ucc_plots.plot_gcpos(
                temp_plot_file,
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
            temp_plot_file = (
                f"{temp_folder}{plots_folder}plots_{fname0[0]}/UCC/{fname0}.webp"
            )
            ucc_plots.plot_CMD(temp_plot_file, df_membs)
            txt += " CMD plot generated |"

            # Update value indicating that the plots were generated
            df_UCC.loc[i_ucc, "plot_used"] = "y"

        if txt != "":
            N_total += 1
            logging.info(f"{N_total} -> {fname0}" + txt + f" ({i_ucc})")

    logging.info(f"\nN={N_total} OCs processed")

    if N_total > 0:
        return df_UCC.copy()
    else:
        return pd.DataFrame([])


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
    cmmts_JSONS_lst,
):
    """ """
    logging.info("\nGenerating md files")

    fname_all = df_BC["fname"].to_list()

    UTI_colors = UTI_to_hex(df_BC)

    # Add stellar density
    r_50 = df_BC["r_50"].to_numpy()
    dist_pc = 1000 / np.clip(df_BC["Plx_m"], 0.01, 50)
    r_pc = dist_pc * np.tan(np.deg2rad(r_50 / 60))
    df_BC["dens_pc2"] = np.round(df_BC["N_50"] / r_pc**2, 1)

    members_files_mapping = {
        fname: bin_label for fname, bin_label in df_BC[["fname", "bin"]].values
    }

    current_year = datetime.datetime.now().year

    # ran_i = np.random.randint(0, len(df_BC), size=50)

    N_total = 0
    # Iterate trough each entry in the UCC database
    cols = df_BC.columns
    for i_ucc, UCC_cl in enumerate(df_BC.itertuples(index=False, name=None)):
        UCC_cl = dict(zip(cols, UCC_cl))
        fname0 = str(UCC_cl["fname"])

        # if fname0 not in ("alessi19",):
        #     continue
        # if "melotte" not in fname0:
        #     continue
        # if i_ucc not in ran_i or "cwnu" in fname0 or "cwwdl" in fname0:
        #     continue

        summary, descriptors, fpars_badges, badges_url, comments_lst = (
            ucc_summ_cmmts.run(current_year, UCC_cl, DBs_JSON, cmmts_JSONS_lst)
        )

        # Generate full entry
        new_md_entry = ucc_entry.make(
            fname0,
            i_ucc,
            DBs_JSON,
            members_files_mapping,
            DBs_full_data,
            df_BC,
            UCC_cl,
            fname_all,
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

    # profiler.stop()
    # profiler.open_in_browser()

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


def write_bin(args):
    """Function to write a single bin to disk"""
    out_fname, df_bin = args
    df_bin.to_csv(out_fname, index=False, compression="gzip")
    return out_fname


def updt_members_files(df_ucc, df_membs, temp_members_files_folder):
    """ """
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
    df_BC: pd.DataFrame,
    df_clusters_CSV_current: pd.DataFrame,
) -> str:
    """
    Update compressed cluster.csv.gz file used by 'ucc.ar' search
    """
    # Extract the first identifier from the "ID" column
    df_BC["Name"] = [_.split(";")[0] for _ in df_BC["Names"]]
    df_BC["RA_ICRS"] = np.round(df_BC["RA_ICRS_m"], 2)
    df_BC["DE_ICRS"] = np.round(df_BC["DE_ICRS_m"], 2)
    df_BC["GLON"] = np.round(df_BC["GLON_m"], 2)
    df_BC["GLAT"] = np.round(df_BC["GLAT_m"], 2)
    df_BC["N_50"] = df_BC["N_50"].astype(int)

    # Compute parallax-based distances in parsecs
    dist_pc = 1000 / np.clip(np.array(df_BC["Plx_m"]), a_min=0.0000001, a_max=np.inf)
    dist_pc = np.clip(dist_pc, a_min=10, a_max=50000)
    df_BC["dist_plx_pc"] = np.round(dist_pc, 0)

    df_new = pd.DataFrame(
        df_BC[
            [
                "Name",
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
                "N_50",
                "P_dup",
                "UTI",
                "bad_oc",
                "dist_plx_pc",  # radec_scatter, mapPlotter
                "r_50",  # radec_scatter
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

    # Update CSV if required
    new_clusters_csv_path = ""
    if not df_clusters_CSV_current.equals(df_new):
        date = pd.Timestamp.now().strftime("%y%m%d%H")
        new_clusters_csv_path = clusters_csv_path.replace("*", f"{date}")
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
        logging.info(f"File 'clusters_{date}.csv.gz' updated")

    else:
        logging.info("File 'clusters_XXXX.csv.gz' not updated (no changes)")

    return new_clusters_csv_path


def make_site_plots(logging, temp_image_path, df_BC):
    """ """
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
    """ """

    def is_number(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    N_pars = {_: 0 for _ in fpars_order}
    for col in fpars_order:
        count = 0
        rows = 0
        for row in df[col].values:
            rows += 1
            if str(row) != "nan":
                if ";" in str(row):
                    count += sum(
                        is_number(x.replace("*", "")) for x in str(row).split(";")
                    )
                else:
                    count += 1
        N_pars[col] = count

    return sum([v for k, v in N_pars.items()])


def update_main_pages(
    logging,
    N_fpars,
    N_members_UCC,
    current_JSON,
    df_UCC,
    database_md,
    articles_md,
):
    """Update main .md files"""
    logging.info("\nUpdating main .md files")

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
    else:
        logging.info("DATABASE.md not updated (no changes)")

    # # Update TABLES
    # tables_md_updt = updt_UTI_main_table(UTI_msk, tables_md)
    # tables_md_updt = updt_C3_classif_main_table(
    #     class_order, OCs_per_class, tables_md_updt
    # )
    # # tables_md_updt = updt_OCs_per_quad_main_table(df_UCC, tables_md_updt)
    # # tables_md_updt = updt_shared_membs_main_table(shared_msk, tables_md_updt)
    # tables_md_updt = updt_fund_params_main_table(Cfp_msk, tables_md_updt)
    # tables_md_updt = updt_Pdup_main_table(Pdup_msk, tables_md_updt)
    # tables_md_updt = updt_N50_main_table(membs_msk, tables_md_updt)
    # with open(temp_folder + tables_md_path, "w") as file:
    #     file.write(tables_md_updt)
    # if tables_md != tables_md_updt:
    #     logging.info("TABLES.md updated")
    # else:
    #     logging.info("TABLES.md not updated (no changes)")

    # Update CLUSTERS

    # Update ARTICLES
    articles_md_updt = ucc_updt_tables.updt_articles_table(
        df_UCC, current_JSON, articles_md
    )
    if articles_md != articles_md_updt:
        with open(temp_folder + articles_md_path, "w") as file:
            file.write(articles_md_updt)
        logging.info("ARTICLES.md updated")
    else:
        logging.info("ARTICLES.md not updated (no changes)")


def updt_indiv_tables(
    logging,
    temp_dbs_tables_path,
    ucc_dbs_tables_path,
    current_JSON,
    df,
):
    """ """
    logging.info("\nUpdating individual tables")

    # New columns used to display in tables
    df["Name"] = [_.split(";")[0] for _ in df["Names"]]
    names_url = []
    for _, cl in df.iterrows():
        name = str(cl["Name"]).split(";")[0]
        color = "red" if cl["bad_oc"] == "y" else "$blue"
        fname = str(cl["fname"])
        url = r"{{ site.baseurl }}/_clusters/" + fname + "/"
        clname = rf'<a href="{url}" target="_blank" style="color: {color};">{name}</a>'
        # names_url.append(f"[{name}]({url})")
        names_url.append(clname)
    #
    df["ID_url"] = names_url
    df["RA_ICRS"] = np.round(df["RA_ICRS_m"], 2)
    df["DE_ICRS"] = np.round(df["DE_ICRS_m"], 2)
    # df["GLON"] = np.round(df["GLON_m"], 2)
    # df["GLAT"] = np.round(df["GLAT_m"], 2)
    df["Plx_m_round"] = np.round(df["Plx_m"], 2)
    df["N_50"] = df["N_50"].astype(int)
    df["C3_abcd"] = [ucc_entry.color_C3(_) for _ in df["C3"]]
    df = df.sort_values("Name").reset_index()

    DBs_dups_badOCs = ucc_updt_tables.count_dups_bad_OCs(current_JSON, df)

    # Update pages for individual databases
    new_tables_dict = ucc_updt_tables.updt_DBs_tables(current_JSON, df, DBs_dups_badOCs)
    general_table_update(
        logging, ucc_dbs_tables_path, temp_dbs_tables_path, new_tables_dict
    )

    # # Update page with UTI values
    # new_tables_dict = updt_UTI_tables(df_UCC_edit, UTI_msk)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    # new_tables_dict = updt_N50_tables(df_UCC_edit, membs_msk)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    # new_tables_dict = updt_C3_classif_tables(df_UCC_edit, class_order)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    # new_tables_dict = updt_fund_params_table(df_UCC_edit, Cfp_msk)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)

    # new_tables_dict = updt_Pdup_tables(df_UCC_edit, Pdup_msk)
    # general_table_update(logging, ucc_tables_path, temp_tables_path, new_tables_dict)


def general_table_update(
    logging, root_path: Path, temp_path: Path, new_tables_dict: dict
) -> None:
    """
    Updates a markdown table file if the content has changed.
    """
    for table_name, new_table in new_tables_dict.items():
        # Read old entry, if any
        try:
            with open(root_path / (table_name + "_table.md"), "r") as f:
                old_table = f.read()
        except FileNotFoundError:
            # This is a new table with no md entry yet
            old_table = ""

        # Write to file if any changes are detected
        if old_table != new_table:
            with open(temp_path / (table_name + "_table.md"), "w") as file:
                file.write(new_table)
            txt = "updated" if old_table != "" else "generated"
            logging.info(f"Table {table_name} {txt}")


def move_files(
    logging,
    ucc_C_file: Path,
    temp_C_path: Path,
    UCC_cmmts_path: Path,
    temp_UCC_cmmts_path: Path,
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

    # --- Updated C file ---
    if os.path.exists(temp_C_path):
        planned_actions.append(("move", temp_C_path, ucc_C_file))

    # --- Delete old clusters file ---
    if new_clusters_csv_path != "":
        planned_actions.append(("delete", old_gz_CSV_path, None))

    # --- Updated comments file ---
    if os.path.exists(temp_UCC_cmmts_path):
        planned_actions.append(("replace", temp_UCC_cmmts_path, UCC_cmmts_path))

    # --- Move files inside temporary ucc/ ---
    temp_ucc_fold = temp_folder + ucc_path
    for root, dirs, files in os.walk(temp_ucc_fold):
        for filename in files:
            src = os.path.join(root, filename)
            dst = src.replace(temp_folder, root_ucc_path)
            planned_actions.append(("move", src, dst))

    # --- Show planned actions ---
    logging.info("\nPlanned actions:")

    # Extract actions related to _clusters folder
    cluster_actions = [a for a in planned_actions if "_clusters" in a[2]]
    other_actions = [a for a in planned_actions if "_clusters" not in a[2]]

    for action, src, dst in other_actions:
        if action == "move":
            logging.info(f"  MOVE     {src} -> {dst}")
        elif action == "replace":
            logging.info(f"  REPLACE  {src} -> {dst}")
        elif action == "delete":
            logging.info(f"  DELETE   {src}")
    if cluster_actions:
        logging.info(
            f"  MOVE     {len(cluster_actions)} '_clusters/*.md' files will be updated"
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

    for action, src, dst in planned_actions:
        if action == "move":
            os.rename(src, dst)
            if "_clusters" not in dst:
                logging.info(f"{src} -> {dst}")
        elif action == "replace":
            os.replace(src, dst)
            logging.info(f"{src} -> {dst}")
        elif action == "delete":
            os.remove(src)
            logging.info(f"Deleted: {src}")

    logging.info("\nAll files moved into place")


# def move_files(
#     logging,
#     ucc_C_file: Path,
#     temp_C_path: Path,
#     UCC_cmmts_path: Path,
#     temp_UCC_cmmts_path: Path,
#     old_gz_CSV_path: str,
#     new_clusters_csv_path: str,
# ) -> None:
#     """ """
#     logging.info("\nMoving files:")

#     # Move all plots
#     all_plot_folds = []
#     for letter in "abcdefghijklmnopqrstuvwxyz":
#         letter_fold = plots_folder + f"plots_{letter}/"
#         for fold in plots_sub_folders:
#             temp_fpath = temp_folder + letter_fold + fold + "/"
#             fpath = root_ucc_path + letter_fold + fold + "/"
#             # Check if folder exists
#             if os.path.exists(temp_fpath):
#                 # For every file in this folder
#                 for file in os.listdir(temp_fpath):
#                     all_plot_folds.append(fpath)
#                     os.rename(temp_fpath + file, fpath + file)
#     unq_plot_folds = list(set(all_plot_folds))
#     for plot_stored in unq_plot_folds:
#         N = all_plot_folds.count(plot_stored)
#         logging.info(
#             plot_stored.replace(temp_folder, root_ucc_path)
#             + "*.webp"
#             + " --> "
#             + plot_stored
#             + f"*.webp (N={N})"
#         )

#     # Move updated C file
#     if os.path.exists(temp_C_path):
#         os.rename(temp_C_path, ucc_C_file)
#         logging.info(str(temp_C_path) + " --> " + str(ucc_C_file))

#     # Move updated UCC comments file
#     if os.path.exists(temp_UCC_cmmts_path):
#         os.replace(temp_UCC_cmmts_path, UCC_cmmts_path)
#         logging.info(f"{temp_UCC_cmmts_path} --> UCC_cmmts_path")

#     # Move all files inside temporary 'ucc/'
#     temp_ucc_fold = temp_folder + ucc_path
#     for root, dirs, files in os.walk(temp_ucc_fold):
#         for filename in files:
#             file_path = os.path.join(root, filename)
#             file_ucc = file_path.replace(temp_folder, root_ucc_path)
#             if "_clusters" not in root:
#                 logging.info(file_path + " --> " + file_ucc)
#             os.rename(file_path, file_ucc)
#         if "_clusters" in root:
#             file_path = root + "/*.md"
#             file_ucc = file_path.replace(temp_folder, root_ucc_path)
#             logging.info(file_path + " --> " + file_ucc)

#     # Delete old cluster_XXXX.csv.gz file
#     if new_clusters_csv_path != "":
#         os.remove(old_gz_CSV_path)
#         logging.info(f"Old file removed: {old_gz_CSV_path}")

#     logging.info("\nAll files moved into place")


def file_checker(logging) -> None:
    """Check the number and types of files in directories for consistency.

    Parameters:
    - logging: Logger instance for recording messages.

    Returns:
    - None
    """
    logging.info("\nChecking files")
    # Read stored final version
    df_UCC_C = pd.read_csv(data_folder + ucc_cat_file, usecols=["fname", "plot_used"])
    flag_error = False

    # Check that all entries in df_UCC_C have plot_used='y'
    if any(df_UCC_C["plot_used"] == "n"):
        flag_error = True
        logging.warning("Some entries in final C dataframe still have plot_used='n'\n")

    # Check the 'fname' columns in df_UCC_B and df_UCC_C_final dataframes are equal
    df_UCC_B = pd.read_csv(data_folder + merged_dbs_file, usecols=["fnames"])
    df_UCC_B_fname = sorted([_.split(";")[0] for _ in df_UCC_B["fnames"]])
    df_UCC_fname = df_UCC_C["fname"].to_list()
    if not df_UCC_B_fname == df_UCC_fname:
        flag_error = True
        logging.warning("The 'fname' columns in B and final C dataframes differ\n")

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

    if flag_error:
        raise ValueError("\nErrors were detected associated to the files")

    logging.warning("All checks passed\n")


if __name__ == "__main__":
    main()
