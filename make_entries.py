
import warnings
import numpy as np
import json
import pandas as pd
from modules import ucc_plots, ucc_entry

# from add_new_DB import new_DB
# This will process the entire UCC_cat_date_new catalogue
new_DB = ''

# Date of the latest version of the catalogue
UCC_cat_date_new = "230702"


def main(entries_path="../ucc/_clusters/"):
    """
    """
    print("Reading databases...")
    with open('databases/all_dbs.json') as f:
        DBs_used = json.load(f)
    DBs_data = {}
    for k, v in DBs_used.items():
        DBs_data[k] = pd.read_csv("databases/" + k + '.csv')

    # Read latest UCC catalogue
    UCC_data = pd.read_csv('UCC_cat_' + UCC_cat_date_new + '.csv')

    # Load notebook template
    with open("notebook.txt", "r") as f:
        ntbk_str = f.readlines()

    for i, row in UCC_data.iterrows():

        # Only generate new entries for those clusters in the recently added
        # database
        if new_DB not in row['DB']:
            continue

        fname0 = row['fnames'].split(';')[0]

        # if i < 12209:
        #     continue
        # if not ('ngc6791' in row['fnames'] or 'ngc6743' in row['fnames']):
        #     continue
        # if 'ngc2516' not in row['fnames']:
        #     continue
        if 'CHI23_3' not in row['DB']:
            continue

        Qfold = row['quad']
        # print(i, row['DB'], Qfold, fname0)
        # Folder name where the datafile is stored
        files_path = "../" + Qfold + "/datafiles/"
        # Folder names where the plot and notebook files will be stored
        notb_path = "../" + Qfold + "/notebooks/"
        plots_path = "../" + Qfold + "/plots/"

        # Load datafile with members for this cluster
        # df_cl = pd.read_parquet(files_path + fname0 + '.parquet')

        # Make catalogue entry
        DBs, DBs_i = row['DB'].split(';'), row['DB_i'].split(';')
        fpars_table = fpars_in_lit(DBs_used, DBs_data, DBs, DBs_i)
        posit_table = positions_in_lit(DBs_used, DBs_data, DBs, DBs_i, row)
        close_table = close_cat_cluster(UCC_data, UCC_data['fnames'], row)

        # Color used by the 'C1' classification
        abcd_c = UCC_color(row['C3'])

        # All names for this cluster
        cl_names = row['ID'].split(';')
        ucc_entry.make_entry(
            entries_path, cl_names, Qfold, fname0, row['UCC_ID'],
            row['C1'], row['C2'], abcd_c, row['r_50'],
            row['N_50'], row['RA_ICRS_m'], row['DE_ICRS_m'],
            row['plx_m'], row['pmRA_m'], row['pmDE_m'], row['Rv_m'],
            fpars_table, posit_table, close_table)

        # # Make notebook
        # make_notebook(Qfold, notb_path, ntbk_str, fname0)

        # # Make plot
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     ucc_plots.make_plot(plots_path, fname0, df_cl)

        # breakpoint()


# def split_membs_field(df_cl, N_membs_min, prob_min=0.5):
#     """
#     """
#     probs_final = df_cl['probs']
#     msk = probs_final > prob_min
#     if msk.sum() < N_membs_min:
#         idx = np.argsort(probs_final)[::-1][:N_membs_min]
#         msk = np.full(len(probs_final), False)
#         msk[idx] = True
#     df_membs, df_field = df_cl[msk], df_cl[~msk]

#     return df_membs, df_field


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
            else:
                txt += '--=--, '
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
