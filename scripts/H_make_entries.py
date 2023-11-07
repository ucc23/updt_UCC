import warnings
from pathlib import Path
import numpy as np
import json
import pandas as pd
from modules import logger
from modules import read_ini_file
from modules import UCC_new_match
from modules import ucc_plots, ucc_entry


def main():
    """ """
    logging = logger.main()
    logging.info("Running 'make_entries' script\n")

    pars_dict = read_ini_file.main()
    (
        UCC_folder,
        dbs_folder,
        all_DBs_json,
        root_UCC_path,
        md_folder,
        members_folder,
        ntbks_folder,
        plots_folder,
    ) = (
        pars_dict["UCC_folder"],
        pars_dict["dbs_folder"],
        pars_dict["all_DBs_json"],
        pars_dict["root_UCC_path"],
        pars_dict["md_folder"],
        pars_dict["members_folder"],
        pars_dict["ntbks_folder"],
        pars_dict["plots_folder"],
    )

    # Folder name where the datafile is stored
    entries_path = root_UCC_path + f"{md_folder}/"

    print("Reading databases...")
    with open(dbs_folder + all_DBs_json) as f:
        DBs_used = json.load(f)
    DBs_data = {}
    for k, v in DBs_used.items():
        DBs_data[k] = pd.read_csv(dbs_folder + k + ".csv")

    # Read latest version of the UCC
    df_UCC, UCC_cat = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    # Load notebook template
    with open("notebook.txt", "r") as f:
        ntbk_str = f.readlines()

    logging.info("\nProcessing UCC")
    for index, UCC_cl in df_UCC.iterrows():
        fname0 = UCC_cl["fnames"].split(";")[0]
        Qfold = UCC_cl["quad"]

        txt0 = f"{Qfold}/{fname0}: "
        txt = f"{Qfold}/{fname0}: "

        # Make catalogue entry
        txt1 = make_entry(
            df_UCC, UCC_cl, DBs_used, DBs_data, entries_path, Qfold, fname0
        )
        if txt1 != "":
            txt += f" md ({txt1})"

        # Make notebook
        ntbk_fpath = Path(
            root_UCC_path + Qfold + f"/{ntbks_folder}/" + fname0 + ".ipynb"
        )
        if ntbk_fpath.is_file() is False:
            make_notebook(fname0, Qfold, ntbk_fpath, ntbk_str)
            txt += " ntbk"

        # Make data plot
        plot_fpath = Path(
            root_UCC_path + Qfold + f"/{plots_folder}/" + fname0 + ".webp"
        )
        if plot_fpath.is_file() is False:
            # Load datafile with members for this cluster
            files_path = root_UCC_path + Qfold + f"/{members_folder}/"
            df_cl = pd.read_parquet(files_path + fname0 + ".parquet")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ucc_plots.make_plot(plot_fpath, df_cl)
            txt += " plot"

        # Make Aladin plot
        plot_aladin_fpath = Path(
            root_UCC_path + Qfold + f"/{plots_folder}/" + fname0 + "_aladin.webp"
        )
        if plot_aladin_fpath.is_file() is False:
            ucc_plots.make_aladin_plot(
                UCC_cl["RA_ICRS_m"],
                UCC_cl["DE_ICRS_m"],
                UCC_cl["r_50"],
                plot_aladin_fpath,
            )
            if plot_aladin_fpath.is_file() is True:
                txt += " plot_aladin"
            else:
                txt += " plot_aladin could not be generated"

        if txt != txt0:
            logging.info(txt)


def make_entry(df_UCC, UCC_cl, DBs_used, DBs_data, entries_path, Qfold, fname0):
    """ """
    DBs, DBs_i = UCC_cl["DB"].split(";"), UCC_cl["DB_i"].split(";")
    fpars_table = fpars_in_lit(DBs_used, DBs_data, DBs, DBs_i)
    posit_table = positions_in_lit(DBs_used, DBs_data, DBs, DBs_i, UCC_cl)
    close_table = close_cat_cluster(df_UCC, UCC_cl)
    # Color used by the 'C1' classification
    abcd_c = UCC_color(UCC_cl["C3"])

    cl_names = UCC_cl["ID"].split(";")
    new_md_entry = ucc_entry.main(
        entries_path,
        cl_names,
        Qfold,
        fname0,
        UCC_cl["UCC_ID"],
        UCC_cl["C1"],
        UCC_cl["C2"],
        abcd_c,
        UCC_cl["r_50"],
        UCC_cl["N_50"],
        UCC_cl["RA_ICRS_m"],
        UCC_cl["DE_ICRS_m"],
        UCC_cl["plx_m"],
        UCC_cl["pmRA_m"],
        UCC_cl["pmDE_m"],
        fpars_table,
        posit_table,
        close_table,
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

    if file_flag != "":
        # Generate/update entry
        with open(entries_path + fname0 + ".md", "w") as f:
            f.write(new_md_entry)

    return file_flag


def UCC_color(abcd):
    """ """
    abcd_c = ""
    line = r"""<span style="color: {}; font-weight: bold;">{}</span>"""
    cc = {"A": "green", "B": "#FFC300", "C": "red", "D": "purple"}
    for letter in abcd:
        abcd_c += line.format(cc[letter], letter)  # + '\n'
    return abcd_c


def fpars_in_lit(DBs_used, DBs_data, DBs, DBs_i):
    """ """
    # Select DBs with parameters and re-arrange them by year
    DBs_w_pars, DBs_i_w_pars = [], []
    for i, db in enumerate(DBs):
        # If this database contains any estimated fundamental parameters
        pars_f = (np.array(DBs_used[db]["pars"].split(",")) != "None").any()
        if pars_f:
            DBs_w_pars.append(db)
            DBs_i_w_pars.append(DBs_i[i])
    # Extract years
    DBs_years = [int(_.split("_")[0][-2:]) for _ in DBs_w_pars]
    # Sort
    sort_idxs = np.argsort(DBs_years)
    # Re-arrange
    DBs_w_pars = np.array(DBs_w_pars)[sort_idxs]
    DBs_i_w_pars = np.array(DBs_i_w_pars)[sort_idxs]

    if len(DBs_w_pars) == 0:
        table = ""
        return table

    txt = ""
    for i, db in enumerate(DBs_w_pars):
        # Full 'db' database
        df = DBs_data[db]
        # Add reference
        txt_db = "| " + DBs_used[db]["ref"] + " | "

        txt_pars = ""
        # Add non-nan parameters
        for par in DBs_used[db]["pars"].split(","):
            # Read parameter value from DB as string
            if par != "None":
                par_v = str(df[par][int(DBs_i_w_pars[i])])
                # Remove empty spaces from param value (if any)
                par_v = par_v.replace(" ", "")
                try:
                    if par_v != "" and par_v != "nan":
                        par_v = float(df[par][int(DBs_i_w_pars[i])])
                    else:
                        par_v = np.nan
                    if not np.isnan(par_v):
                        txt_pars += par + "=" + str(round(par_v, 2)) + ", "
                except:
                    # DBs like SANTOS21 list more than 1 value per parameter
                    txt_pars += par + "=" + par_v + ", "

        if txt_pars != "":
            # Remove final ', '
            txt_pars = txt_pars[:-2]
            # Add quotes
            txt_pars = "`" + txt_pars + "`"
            # Combine and close row
            txt += txt_db + txt_pars + " |\n"

    if txt != "":
        # Remove final new line
        table = txt[:-1]
    else:
        table = ""

    return table


def positions_in_lit(DBs_used, DBs_data, DBs, DBs_i, row_UCC):
    """ """
    # Re-arrange DBs by year
    DBs_years = [int(_.split("_")[0][-2:]) for _ in DBs]
    # Sort
    sort_idxs = np.argsort(DBs_years)
    # Re-arrange
    DBs_sort = np.array(DBs)[sort_idxs]
    DBs_i_sort = np.array(DBs_i)[sort_idxs]

    table = ""
    for i, db in enumerate(DBs_sort):
        # Full 'db' database
        df = DBs_data[db]

        # Add positions
        row_in = ""
        for c in DBs_used[db]["pos"].split(","):
            if c != "None":
                # Read position as string
                pos_v = str(df[c][int(DBs_i_sort[i])])
                # Remove empty spaces if any
                pos_v = pos_v.replace(" ", "")
                if pos_v != "" and pos_v != "nan":
                    row_in += str(round(float(pos_v), 3)) + " | "
                else:
                    row_in += "--" + " | "
            else:
                row_in += "--" + " | "

        # See if the row contains any values
        if row_in.replace("--", "").replace("|", "").strip() != "":
            # Add reference
            table += "|" + DBs_used[db]["ref"] + " | "
            # Close and add row
            table += row_in[:-1] + "\n"

    # Add UCC positions
    table += "| **UCC** |"
    for col in ("RA_ICRS_m", "DE_ICRS_m", "plx_m", "pmRA_m", "pmDE_m", "Rv_m"):
        val = "--"
        if row_UCC[col] != "" and not np.isnan(row_UCC[col]):
            val = str(round(float(row_UCC[col]), 3))
        table += val + " | "
    # Close row
    table = table[:-1] + "\n"

    return table


def close_cat_cluster(df_UCC, row):
    """ """
    close_table = ""

    if str(row["dups_fnames_m"]) == "nan":
        return close_table

    fnames0 = [_.split(";")[0] for _ in df_UCC["fnames"]]

    dups_fnames = row["dups_fnames_m"].split(";")
    dups_probs = row["dups_probs_m"].split(";")

    for i, fname in enumerate(dups_fnames):
        j = fnames0.index(fname)
        name = df_UCC["ID"][j].split(";")[0]

        vals = []
        for col in ("RA_ICRS_m", "DE_ICRS_m", "plx_m", "pmRA_m", "pmDE_m", "Rv_m"):
            val = round(float(df_UCC[col][j]), 3)
            if np.isnan(val):
                vals.append("--")
            else:
                vals.append(val)
        val = round(float(dups_probs[i]), 3)
        if np.isnan(val):
            vals.append("--")
        else:
            vals.append(val)

        ra, dec, plx, pmRA, pmDE, Rv, prob = vals

        close_table += f"|[{name}](https://ucc.ar/_clusters/{fname}/)| "
        close_table += f"{int(100 * prob)} | "
        close_table += f"{ra} | "
        close_table += f"{dec} | "
        close_table += f"{plx} | "
        close_table += f"{pmRA} | "
        close_table += f"{pmDE} | "
        close_table += f"{Rv} |\n"
    # Remove final new line
    close_table = close_table[:-1]

    return close_table


def make_notebook(fname0, Qfold, ntbk_fpath, ntbk_str):
    """ """
    cl_str = r"""  "cluster = \"{}\"" """.format(fname0)
    ntbk_str[42] = "      " + cl_str + "\n"
    ntbk_str[90] = (
        r"""        "path = \"https://github.com/ucc23/{}/raw/main/datafiles/\"\n",""".format(
            Qfold
        )
        + "\n"
    )
    with open(ntbk_fpath, "w") as f:
        contents = "".join(ntbk_str)
        f.write(contents)


if __name__ == "__main__":
    main()
