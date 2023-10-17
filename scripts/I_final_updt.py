
import os
import pandas as pd
import numpy as np
import csv
from modules import logger
from modules import read_ini_file
from modules import UCC_new_match
from H_make_entries import UCC_color


def main():
    """
    """
    logging = logger.main()
    logging.info("\nRunning 'final_updt' script\n")

    pars_dict = read_ini_file.main()
    UCC_folder, root_UCC_path, clusters_json, pages_folder, members_folder = \
        pars_dict['UCC_folder'], pars_dict['root_UCC_path'], \
        pars_dict['clusters_json'], pars_dict['pages_folder'], \
        pars_dict['members_folder']

    # Read latest version of the UCC
    df_UCC, _ = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    _ra = np.round(df_UCC['RA_ICRS'].values, 2)
    _dec = np.round(df_UCC['DE_ICRS'].values, 2)
    _lon = np.round(df_UCC['GLON'].values, 2)
    _lat = np.round(df_UCC['GLAT'].values, 2)

    updt_cls_JSON(df_UCC, root_UCC_path, clusters_json, _ra, _dec, _lon, _lat)
    logging.info("\nFile 'clusters.json' updated")

    updt_tables(df_UCC, root_UCC_path, pages_folder, _ra, _dec, _lon, _lat)
    logging.info("\nTables updated")

    zenodo_UCC(df_UCC, UCC_folder)
    logging.info("\nCompressed 'UCC_cat.csv.gz' file generated\n")

    zenodo_membs(logging, UCC_folder, root_UCC_path, members_folder)
    logging.info("\nCompressed 'UCC_members.parquet.gz' file generated")


def updt_cls_JSON(df_UCC, root_UCC_path, clusters_json, _ra, _dec, _lon, _lat):
    """
    Update cluster.json file used by 'ucc.ar' search 
    """
    df = pd.DataFrame(df_UCC[[
        'ID', 'fnames', 'UCC_ID', 'RA_ICRS', 'DE_ICRS', 'GLON', 'GLAT']])
    df['ID'] = [_.split(';')[0] for _ in df['ID']]

    df['RA_ICRS'] = _ra
    df['DE_ICRS'] = _dec
    df['GLON'] = _lon
    df['GLAT'] = _lat

    df = df.sort_values('GLON')

    df.to_json(root_UCC_path + clusters_json, orient="records", indent=1)


def updt_tables(df_UCC, root_UCC_path, pages_folder, _ra, _dec, _lon, _lat):
    """
    """
    df = pd.DataFrame(df_UCC)

    df['ID'] = [_.split(';')[0] for _ in df_UCC['ID']]
    names = []
    for i, cl in df.iterrows():
        name = cl['ID'].split(';')[0]
        url = "https://ucc.ar/_clusters/" + cl['fnames'].split(';')[0] + '/'
        names.append(f"[{name}]({url})")
    df['ID'] = names
    df['RA_ICRS'] = _ra
    df['DE_ICRS'] = _dec
    df['GLON'] = _lon
    df['GLAT'] = _lat

    # df_UCC['UCC_ID'] = [_.split(' ')[1] for _ in df['UCC_ID']]

    header = """---\nlayout: page\ntitle: quad_title\n""" \
             + """permalink: /quad_link_table/\n---\n\n"""

    title_dict = {
        1: '1st', 2: '2nd', 3: '3rd', 4: '4th', 'P': 'positive',
        'N': 'negative'}

    for quad_N in range(1, 5):
        for quad_s in ('P', 'N'):
            quad = 'Q' + str(quad_N) + quad_s

            title = f"{title_dict[quad_N]} quadrant, {title_dict[quad_s]} latitude"
            md_table = header.replace('quad_title', title).replace(
                'quad_link', quad)
            md_table += "| Name | l | b | ra | dec | Plx | C1 | C2 | C3 |\n"
            md_table += "| ---- | - | - | -- | --- | --- | -- | -- | -- |\n"

            msk = df['quad'] == quad
            df_m = df[msk]
            df_m = df_m.sort_values('ID')

            for i, row in df_m.iterrows():
                for col in ('ID', 'GLON', 'GLAT', 'RA_ICRS', 'DE_ICRS', 'plx_m', 'C1', 'C2'):
                    md_table += "| " + str(row[col]) + " "
                abcd = UCC_color(row['C3'])
                md_table += '| ' + abcd + ' |\n'

            with open(
                    root_UCC_path + pages_folder + '/' + quad + '_table.md',
                    'w') as file:
                file.write(md_table)


def zenodo_UCC(df_UCC, UCC_folder):
    """
    Generate the compressed file with the reduced UCC DB that is stored in
    the Zenodo repository
    """
    # Only keep certain columns
    drop_cols = [
        'DB', 'DB_i', 'GLON', 'GLAT', 'fnames', 'quad', 'dups_fnames',
        'dups_probs', 'N_fixed', 'N_membs', 'fixed_cent', 'cent_flags',
        'GLON_m', 'GLAT_m', 'dups_fnames_m', 'dups_probs_m']
    df = df_UCC.drop(columns=drop_cols)

    # Re-order columns
    df = df[[
        'ID', 'RA_ICRS', 'DE_ICRS', 'plx', 'pmRA', 'pmDE', 'UCC_ID', 'N_50',
        'r_50', 'RA_ICRS_m', 'DE_ICRS_m', 'plx_m', 'pmRA_m', 'pmDE_m', 'Rv_m',
        'N_Rv', 'C1', 'C2', 'C3']]

    # Store to compressed file
    df.to_csv(
        UCC_folder + 'UCC_cat.csv.gz', na_rep='nan', index=False,
        quoting=csv.QUOTE_NONNUMERIC, compression='gzip')


def zenodo_membs(logging, UCC_folder, root_UCC_path, members_folder):
    """
    Generate the compressed file with the estimated members that is stored in
    the Zenodo repository
    """

    # Now update the members file
    tmp = []
    for quad in ('1', '2', '3', '4'):
        for lat in ('P', 'N'):
            logging.info(f"Processing Q{quad}{lat}")
            path = root_UCC_path + 'Q' + quad + lat + f'/{members_folder}/'
            for file in os.listdir(path):
                df = pd.read_parquet(path + file)
                fname = file.replace('.parquet', '')
                name_col = [fname for _ in range(len(df))]
                df.insert(loc=0, column='name', value=name_col)
                tmp.append(df)

    df_comb = pd.concat(tmp, ignore_index=True)
    df_comb.to_parquet(
        UCC_folder + 'UCC_members.parquet.gz', index=False, compression='gzip')


if __name__ == '__main__':
    main()
