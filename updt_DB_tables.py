
import pandas as pd
import numpy as np
from make_entries import UCC_color


date = "230702"
out_path = '../ucc/_pages/'

df = pd.read_csv('UCC_cat_' + date + '.csv')
# df = pd.DataFrame(df_UCC)

df['ID'] = [_.split(';')[0] for _ in df['ID']]
names = []
for i, cl in df.iterrows():
    name = cl['ID'].split(';')[0]
    url = "https://ucc.ar/_clusters/" + cl['fnames'].split(';')[0] + '/'
    names.append(f"[{name}]({url})")
df['ID'] = names

# df['UCC_ID'] = [_.split(' ')[1] for _ in df['UCC_ID']]

df['RA_ICRS'] = np.round(df['RA_ICRS'].values, 2)
df['DE_ICRS'] = np.round(df['DE_ICRS'].values, 2)
df['GLON'] = np.round(df['GLON'].values, 2)
df['GLAT'] = np.round(df['GLAT'].values, 2)

header = """---
layout: page
title: quad_title
permalink: /quad_link_table/
---\n\n"""

title_dict = {
    1: '1st', 2: '2nd', 3: '3rd', 4: '4th', 'P': 'positive', 'N': 'negative'}

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

        with open(out_path + quad + '_table.md', 'w') as file:
            file.write(md_table)
        print("Finished", quad)
