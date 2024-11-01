
# Update UCC with a new DB

To add a new database to the UCC catalogue and generate the required files,
follow these steps:

[1. Prepare DB](#1-prepare-db)
[2. Check format](#2-check-format)
[3. Check issues](#3-check-issues)
[4. Generate a new UCC version](#4-generate-a-new-ucc-version) (stored at `zenodo/UCC_cat_XXYYZZ.csv`)
[5. Generate members' datafiles](#5-generate-members-datafiles) (files stored at `../QXY/datafiles/`)
[6. Update the new UCC](#6-update-the-new-ucc)
[7. General check of new UCC](#7-general-check-of-new-ucc)
[8. Generate new Zenodo files](#8-generate-new-zenodo-files) (stored at parent folder)

The above steps prepare the files for the updated version of the UCC. The following
steps **apply** the required changes to the site's files.

[9. Generate new clusters entries](#9-generate-new-clusters-entries)
[10. Update site's files](#10-update-sites-files)

**Important**

Once all scripts have been applied, follow the final steps in the
[UCC public site build](https://github.com/ucc23/updt_UCC#ucc-public-site-build) section to update the public site.




---------------------------------

## 1. Prepare DB

**DB name format**
 
The format of the name for the DB is `SMITH24` or, if required, `SMITH24_1`.
The name of the DB **must not** contain any non letter characters except for the
`_` required to differentiate DBs with the same names published in the same year.

**Required columns**

The new database must contain a column with all the names assigned to a given
OC, a column with `RA`, and a column wit `DEC` values (no galactic coordinates
allowed).

**Naming convention**

OCs with multiple names must **not** use `;` as a separating character,
only `,` is allowed (surrounding the names with "")

**Preparing the files**

1. Save the new DB in proper `csv` format to the `/databases` folder
2. Add the name of the new DB and the column names for the `ID,RA,DEC` parameters
   to the `[General]` section of the `params.ini` file.

**Adding the DB to the JSON file**

Add the new DB to the `databases/all_dbs.json` file following the convention:

```
  "NAME_OF_NEW_DB": {
    "ref": "[Author et al. (2012)](url_to_DB))",
    "names": "name",
    "pos": "ra,dec,plx,pm_ra,pm_dec,Rv",
    "pars": "E_bv,dist,logt,FeH,Mass,binar_frac,blue_stragglers",
    "e_pars": "None,None,e_logt,None,None,None,None"
 },
```

The `pos` line must contain all six parameters. If either is not present, use `None`.

The `pars` line should only contain the available parameters in the suggested order. No
`None` are required here.

The `e_pars` line should follow the same order and length as the `pars` line, using
`None` wherever there are missing values.


## 2. Check format

Run the `B_new_DB_check.py`  script

This script checks the new DB for proper formatting:

- Check for "bad characters" in cluster's names: ';' or '_'
- Make sure that no GCs are very close to listed OCs
- Check for possible duplicates in the new DB
- Possible instances of 'vdBergh-Hagen'/'vdBergh' that must be changed to
  'VDBH' & 'VDB', per CDS recommendation
- Possible empty entries in columns are replaced by 'nan's

Once finished, correct any issues that are shown.

**Summary**

- Files edited: `databases/all_dbs.json` (manually)
- Files generated: properly formatted database for the new DB (manually)



## 3. Check issues

Run the `C_check_new_DB.py` script

This script will find the name-based matches and display any issues between OCs
that are present in the UCC and the new DB.

```
Is the OC already present in the UCC?
    |         |
    v         |--> No --> do nothing
   Yes
    |
    v
Is the difference between the old vs new centers values large?
    |         |
    v         |--> No --> do nothing
   Yes
    |
    v
Request attention
```

Fix any possible issues that were flagged for attention **before** running the
`D_add_new_DB.py` script.

**Summary**

- Files edited: None
- Files generated: None



## 4. Generate a new UCC version

Run the `D_add_new_DB.py` script

This script will update the current `UCC_cat_XXYYZZ.csv` catalogue with the
OCs in the new DB. The 5D coordinates are **not** updated here if the
OC(s) are already present in the UCC.

The new UCC version will have the same name format with an updated date stored
at `zenodo/UCC_cat_XXYYZZ.csv`.

**Summary**

- Files edited: None
- Files generated: `zenodo/UCC_cat_XXYYZZ.csv`



## 5. Generate members' datafiles

Run the `E_run_fastMP.py` script

This script generates new datafiles with members using `fastMP`, **only** for the
new OCs in the  new DB. These are identified as those with a
'nan' value in the 'C3' column of the new `zenodo/UCC_cat_XXYYZZ.csv` file.

The `.parquet` member files are automatically stored in the proper
`../QXY/datafiles/` repositories.

This script also generates a `new_OCs_data.csv` file with parameters for the new
OCs, obtained from their members and surrounding field stars.

**Summary**

- Files edited: None
- Files generated: `new_OCs_data.csv` + `../QXY/datafiles/*.parquet`



## 6. Update the new UCC

Run the `F_updt_UCC.py` script

**If no new OCs were processed by the previous script, this one can be skipped.**

This script uses the data in the `new_OCs_data.csv` file to update the new UCC
version `zenodo/UCC_cat_XXYYZZ.csv`.

It also estimates probable duplicates using data on positions obtained from the members
stars.

Delete `new_OCs_data.csv` once this script is finished.

**Summary**

- Files edited: `zenodo/UCC_cat_XXYYZZ.csv`, `new_OCs_data.csv` deleted
- Files generated: None



## 7. General check of new UCC

Run the `G_UCC_versions_check.py` script

**Before running this script**: write the name of the previous UCC version in the
section `[Check versions]` of the `params.ini` file, under the `old_UCC_name` variable.

This script checks the old UCC DB versus the new one to flag possible issues
for attention.

**Summary**

- Files edited: None
- Files generated: None



## 8. Generate new Zenodo files

Run the script `H_zenodo_updt.py`

This script will generate the two files that are to be uploaded to Zenodo. These files
contain all the UCC information.

**Summary**

- Files edited: None
- Files generated: `UCC_cat.csv.gz, UCC_members.parquet.gz`



## 9. Generate new clusters entries

Run the `I_make_entries.py` script

This script will process the **entire** UCC and generate an `md` file, `ipynb`
notebook, and plot, for every OC for which either of those files do not exist.

In the case of the `md` entries, it will also check if the new entry changed
compared to the old one, and it will update it **only** if it did.

For each processed OC that is missing either of those files:

1. Generate a `.md` entry, stored in `../ucc/_clusters/`
2. Generate a `.ipynb` notebook, stored in `../QXY/notebooks/`
3. Generate a plot, stored in `../QXY/plots/`

**Summary**

- Files edited: `../ucc/_clusters/*.md` entries (if there are changes in the new UCC)
- Files generated: `../ucc/_clusters/*.md` + `../QXY/notebooks/*.ipynb` +
  `../QXY/plots/*.webp` (if files are missing)




## 10. Update site's files

Run the script `J_database_updt.py`. This script will:

- update the `DATABASE.md` file used by the `ucc.ar` site
- update the tables files linked to the above file
- update the `../ucc/_clusters/clusters.json` file used for searching in `ucc.ar`

**Summary**

- Files edited: `ucc/_pages/DATABASE.md, ucc/_pages/QXY_table.md, ucc/clusters.json`
- Files generated: None
