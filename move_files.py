
from pathlib import Path
import pandas as pd

# path = "/home/gabriel/Github/web_sites/UCC/datafiles/datafiles_temp/"
path = "/home/gperren/fastMP/datafiles/"
df = pd.read_csv("UCC_cat_20230504.csv")

for i, fname in enumerate(df['fnames']):

    fname = fname.split(';')[0]
    UCC_ID = df['UCC_ID'][i]
    lonlat = UCC_ID.split('G')[1]
    lon = float(lonlat[:5])
    try:
        lat = float(lonlat[5:])
    except ValueError:
        lat = float(lonlat[5:-1])

    fold = "Q"
    if lon >= 0 and lon < 90:
        fold += "1"
    elif lon >= 90 and lon < 180:
        fold += "2"
    elif lon >= 180 and lon < 270:
        fold += "3"
    elif lon >= 270 and lon < 3600:
        fold += "4"
    if lat >= 0:
        fold += 'P'
    else:
        fold += "N"

    # if fold == 'Q1P':
    # print(fname, UCC_ID, fold)
    #     breakpoint()

    old_path = path + fname + '.csv.gz'
    new_path = path + fold + '/' + fname + '.csv.gz'  
    try:  
        Path(old_path).rename(new_path)
    except:
        print("err", fname, UCC_ID, fold)
    # breakpoint()
