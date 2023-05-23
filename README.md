# UCC database

This repository contains:

1. `databases/`: databases used to generate the UCC catalogue along with
   the JSON file (`all_dbs.json`) with their info
2. `modules/`:  modules used to update the catalogue and generate the landing
   pages and plots for each  cluster
3. `notebook.txt`: template used to generate the notebooks
4. `UCC_cat_XXXYYZZ.csv`: latest version of the catalogue
5. `add_new_DB.py`: script used to update the UCC catalogue with a new DB
6. `make_entries.py`: script used to generate new entries for the clusters
   added through the new DB. This includes: `.md` files for the site, plots
   files, notebook files, and datafiles


# Initial version of the catalogue

The initial version is generated using the `initial_DBs_comb` script in the
article's repo. This initial version is then processed with the `fastMP_process`
script in the same repo to generate the datafiles for each catalogued cluster,
through the use of the `fastMP` package. The `fastMP_process` script calls the
`call_fastMP` module that handles the datafile generation for each cluster and
the updating of the `UCC_cat_XXXYYZZ.csv` catalogue file with the obtained
parameters. Finally, the `add_duplicates` script is applied on the
`UCC_cat_XXXYYZZ.csv` catalogue file to add the probable duplicates for each
cluster

The datafiles generated in this initial run need to be processed with the
`make_entries` script once to generate the files that will populate the site
`ucc.ar`.

Summary:

1. `initial_DBs_comb` generates the initial version of the catalogue
2. `fastMP_process` (`call_fastMP` module) generates the the datafiles and
   updates the `UCC_cat_XXXYYZZ.csv` catalogue
3. `add_duplicates` adds the probable duplicates for each cluster


# Updating the catalogue

To add a new database to the UCC catalogue and generate the required files,
these steps must be followed.


## Adding a new DB

1. Save the new DB in proper `csv` format to the `databases`/ folder

2. Edit to change instances of 'vdBergh-Hagen' to 'VDBH' and 'vdBergh' to
'VDB', per CDS recommendation

3. Make sure no globular clusters are included

4. Replace possible empty entries in columns using:

```
import csv
import pandas as pd
df = pd.read_csv("newDB.csv")
df = df.replace(r'^\s*$', np.nan, regex=True)
df.to_csv("newDB.csv", na_rep='nan', index=False, quoting=csv.QUOTE_NONNUMERIC)
```

5. Edit the parameters column names (if any) following:
  - `Av`/`Ebv`: absorption / extinction
  - `dm`/`d_pc /d_kpc`: distance modulus / distance
  - `logt`: logarithmic age
  - `FeH`/`Z`: metallicity

6. Add the new DB to the `databases/all_dbs.json` file (adjusting its position
according to the publication year)


## Generating a new catalogue and datafiles

Run the `add_new_DB.py` script **making sure** to first edit it with the proper
ID of the new DB and the date of the latest UCC catalogue file

This script will combine the old `UCC_cat_XXXYYZZ.csv` catalogue with the new
database and generate a new `UCC_cat_XXXYYZZ.csv` catalogue with the
current date. It will also run the `fastMP` code for those clusters in the new
DB that ??????.

The clusters' datafiles are stored in the  `QXY` repositories in the
`datafiles/` subfolders.

The script will also update the `../ucc/_clusters/clusters.json` file used
by the `ucc.ar` site for searching.

Summary:

1. Updates the `UCC_cat_XXXYYZZ.csv` file with data from the new DB
2. Updates the `../ucc/_clusters/clusters.json` file with data from the new DB
3. Generates the new clusters' datafiles and stores them in the `QXY` repos


## Generate new cluster entries

After running the above script, run the `make_entries.py` script **making sure**
to first edit it with the proper date of the recently updated UCC catalogue
(should be the current date if both the above and this scripts are used on the
same day).

This script will process all the clusters with the new DB identifier in the
loaded `UCC_cat_XXXYYZZ.csv` file and generate a plot, `.ipynb` notebook, and
proper `.md` entry in the `../ucc/_clusters/` folder for each cluster.

Summary (for each processed cluster from the new DB):

1. Generate an `.md` entry in the `../ucc/_clusters/` folder
2. Generate a `.ipynb` notebook
3. Generate a plot