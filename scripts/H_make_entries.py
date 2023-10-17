
import warnings
import numpy as np
import json
import pandas as pd
from modules import logger
from modules import read_ini_file
from modules import UCC_new_match
from modules import ucc_plots, ucc_entry


def main():
    """
    """
    logging = logger.main()
    logging.info("\nRunning 'make_entries' script\n")

    pars_dict = read_ini_file.main()
    UCC_folder, dbs_folder, all_DBs_json, new_OCs_fpath, root_UCC_path, \
        md_folder, members_folder, ntbks_folder, plots_folder = \
        pars_dict['UCC_folder'], pars_dict['dbs_folder'], \
        pars_dict['all_DBs_json'], pars_dict['new_OCs_fpath'], \
        pars_dict['root_UCC_path'], pars_dict['md_folder'], \
        pars_dict['members_folder'], pars_dict['ntbks_folder'], \
        pars_dict['plots_folder']

    # Folder name where the datafile is stored
    entries_path = root_UCC_path + f"{md_folder}/"

    # Read file produced by the `check_new_DB` script
    new_OCs_info = pd.read_csv(new_OCs_fpath)
    logging.info(f"Generating/updating {len(new_OCs_info)} entries")

    print("Reading databases...")
    with open(dbs_folder + all_DBs_json) as f:
        DBs_used = json.load(f)
    DBs_data = {}
    for k, v in DBs_used.items():
        DBs_data[k] = pd.read_csv(dbs_folder + k + '.csv')

    # Read latest version of the UCC
    df_UCC, UCC_cat = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    # Load notebook template
    with open("notebook.txt", "r") as f:
        ntbk_str = f.readlines()

    for index, new_cl in new_OCs_info.iterrows():
        # Only generate new entries for flagged OCs
        if new_cl['process_f'] is False:
            continue

        # Identify position in the UCC
        fname0 = new_cl['fnames'].split(',')[0]
        UCC_index = None
        for _, UCC_fnames in enumerate(df_UCC['fnames']):
            # for ucc_fname in UCC_fnames.split(';'):
            if fname0 == UCC_fnames:
                UCC_index = _
                break
        if UCC_index is None:
            logging.info(f"ERROR: could not find {fname0} in UCC DB")
            return
        UCC_cl = df_UCC.iloc[UCC_index]
        logging.info(f"{index} Processing {UCC_cl['fnames']}")

        fname0 = UCC_cl['fnames'].split(';')[0]

        Qfold = UCC_cl['quad']

        # Make catalogue entry
        DBs, DBs_i = UCC_cl['DB'].split(';'), UCC_cl['DB_i'].split(';')
        fpars_table = fpars_in_lit(DBs_used, DBs_data, DBs, DBs_i)
        posit_table = positions_in_lit(DBs_used, DBs_data, DBs, DBs_i, UCC_cl)
        close_table = close_cat_cluster(df_UCC, df_UCC['fnames'], UCC_cl)

        # Color used by the 'C1' classification
        abcd_c = UCC_color(UCC_cl['C3'])

        # All names for this cluster
        cl_names = UCC_cl['ID'].split(';')
        ucc_entry.make_entry(
            entries_path, cl_names, Qfold, fname0, UCC_cl['UCC_ID'],
            UCC_cl['C1'], UCC_cl['C2'], abcd_c, UCC_cl['r_50'],
            UCC_cl['N_50'], UCC_cl['RA_ICRS_m'], UCC_cl['DE_ICRS_m'],
            UCC_cl['plx_m'], UCC_cl['pmRA_m'], UCC_cl['pmDE_m'],
            UCC_cl['Rv_m'], fpars_table, posit_table, close_table)

        # Folder names where the members, plot and notebook files are stored
        files_path = root_UCC_path + Qfold + f"/{members_folder}/"
        notb_path = root_UCC_path + Qfold + f"/{ntbks_folder}/"
        plots_path = root_UCC_path + Qfold + f"/{plots_folder}/"

        # Make notebook
        make_notebook(Qfold, notb_path, ntbk_str, fname0)

        # Load datafile with members for this cluster
        df_cl = pd.read_parquet(files_path + fname0 + '.parquet')

        # Make plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ucc_plots.make_plot(plots_path, fname0, df_cl)

        fname0 = UCC_cl['fnames'].split(';')[0]
        logging.info(f"File+notebook+plot generated for {Qfold}/{fname0}\n")


def UCC_color(abcd):
    """
    """
    abcd_c = ""
    line = r"""<span style="color: {}; font-weight: bold;">{}</span>"""
    cc = {'A': 'green', 'B': '#FFC300', 'C': 'red', 'D': 'purple'}
    for letter in abcd:
        abcd_c += line.format(cc[letter], letter)# + '\n'
    return abcd_c


def fpars_in_lit(DBs_used, DBs_data, DBs, DBs_i):
    """
    """

    # Select DBs with parameters and re-arrange them by year
    DBs_w_pars, DBs_i_w_pars = [], []
    for i, db in enumerate(DBs):
        # If this database contains any estimated fundamental parameters
        pars_f = (np.array(DBs_used[db]['pars'].split(',')) != 'None').any()
        if pars_f:
            DBs_w_pars.append(db)
            DBs_i_w_pars.append(DBs_i[i])
    # Extract years
    DBs_years = [int(_.split('_')[0][-2:]) for _ in DBs_w_pars]
    # Sort
    sort_idxs = np.argsort(DBs_years)
    # Re-arrange
    DBs_w_pars = np.array(DBs_w_pars)[sort_idxs]
    DBs_i_w_pars = np.array(DBs_i_w_pars)[sort_idxs]

    txt = ''
    for i, db in enumerate(DBs_w_pars):

        # # If this database contains any estimated fundamental parameters
        # pars_f = (np.array(DBs_used[db]['pars'].split(',')) != 'None').any()
        # if not pars_f:
        #     continue

        txt += '| '
        # Full 'db' database
        df = DBs_data[db]
        # Add reference
        txt += DBs_used[db]['ref'] + ' | `'
        # Add non-nan parameters
        for par in DBs_used[db]['pars'].split(','):
            # Read parameter value from DB as string
            if par != 'None':
                par_v = str(df[par][int(DBs_i_w_pars[i])])
                # Remove empty spaces if any
                par_v = par_v.replace(' ', '')
                try:
                    if par_v != '' and par_v != 'nan':
                        par_v = float(df[par][int(DBs_i_w_pars[i])])
                    else:
                        par_v = np.nan
                    if not np.isnan(par_v):
                        txt += par + '=' + str(round(par_v, 2)) + ', '
                except:
                    # DBs like SANTOS21 list more than 1 value for
                    # parameter
                    txt += par + '=' + par_v + ', '
            # else:
            #     txt += '--=--, '
        # Close row
        txt = txt[:-2] + '` |\n'

    # Remove final new line
    if txt != '':
        table = txt[:-1]
    else:
        table = ''

    return table


def positions_in_lit(DBs_used, DBs_data, DBs, DBs_i, row):
    """
    """
    # Re-arrange DBs by year
    DBs_years = [int(_.split('_')[0][-2:]) for _ in DBs]
    # Sort
    sort_idxs = np.argsort(DBs_years)
    # Re-arrange
    DBs_sort = np.array(DBs)[sort_idxs]
    DBs_i_sort = np.array(DBs_i)[sort_idxs]

    table = ''
    for i, db in enumerate(DBs_sort):
        table += '|'
        # Full 'db' database
        df = DBs_data[db]
        # Add reference
        table += DBs_used[db]['ref'] + ' | '

        # Add positions
        for c in DBs_used[db]['pos'].split(','):
            if c != 'None':
                # Read position as string
                pos_v = str(df[c][int(DBs_i_sort[i])])
                # Remove empty spaces if any
                pos_v = pos_v.replace(' ', '')
                if pos_v != '' and pos_v != 'nan':
                    table += str(round(float(pos_v), 3)) + ' | '
                else:
                    table += '--' + ' | '
            else:
                table += '--' + ' | '
        # Close row
        table = table[:-1] + '\n'

    # Add UCC positions
    table += '| **UCC** |'
    for col in ('RA_ICRS_m', 'DE_ICRS_m', "plx_m", "pmRA_m", "pmDE_m", "Rv_m"):
        val = '--'
        if row[col] != '' and not np.isnan(row[col]):
            val = str(round(float(row[col]), 3))
        table += val + ' | '
    # Close row
    table = table[:-1] + '\n'

    return table


def close_cat_cluster(fMP_data, fMP_fnames, row):
    """
    """
    close_table = ''

    if str(row['dups_fnames_m']) == 'nan':
        return close_table

    fnames0 = [_.split(';')[0] for _ in fMP_fnames]

    dups_fnames = row['dups_fnames_m'].split(';')
    dups_probs = row['dups_probs_m'].split(';')

    for i, fname in enumerate(dups_fnames):

        j = fnames0.index(fname)
        name = fMP_data['ID'][j].split(';')[0]

        vals = []
        for col in (
                'RA_ICRS_m', 'DE_ICRS_m', 'plx_m', 'pmRA_m', 'pmDE_m', 'Rv_m'):
            val = round(float(fMP_data[col][j]), 3)
            if np.isnan(val):
                vals.append('--')
            else:
                vals.append(val)
        val = round(float(dups_probs[i]), 3)
        if np.isnan(val):
            vals.append('--')
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


def make_notebook(Qfold, notb_path, ntbk_str, fname0):
    """
    """
    cl_str = r"""  "cluster = \"{}\"" """.format(fname0)
    ntbk_str[42] = '      ' + cl_str + '\n'
    ntbk_str[90] = r"""        "path = \"https://github.com/ucc23/{}/raw/main/datafiles/\"\n",""".format(Qfold)  + '\n'
    with open(notb_path + fname0 + ".ipynb", "w") as f:
        contents = "".join(ntbk_str)
        f.write(contents)


if __name__ == '__main__':
    main()
