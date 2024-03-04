
<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [UCC scripts](#ucc-scripts)
- [Fixing bad entries](#fixing-bad-entries)
- [Update UCC with a new DB](#update-ucc-with-a-new-db)
   * [Adding a new DB](#adding-a-new-db)
   * [Checking the new DB](#checking-the-new-db)
   * [Generating a new version of the UCC](#generating-a-new-version-of-the-ucc)
   * [Generate datafiles with members and update the new UCC](#generate-datafiles-with-members-and-update-the-new-ucc)
   * [Generate new cluster entries](#generate-new-cluster-entries)
   * [Update the UCC](#update-the-ucc)

<!-- TOC end -->


<!-- TOC --><a name="ucc-scripts"></a>
# UCC scripts

This repository contains all the necessary scripts to update/edit the UCC
database and its associated web site.



<!-- TOC --><a name="fixing-bad-entries"></a>
# Fixing bad entries

To fix bad entries, follow these steps.

1. Edit the latest UCC version

Create a copy of the latest UCC database, with the current date. From this
new version, manually remove the entries that need to be re-processed.

2. Fix DBs

Make sure that all the DBs are corrected for whatever issues are being fixed.

3. Run the `A_initial_DBs_comb.py` script

This script requires as input the names of the OCs that will be re-processed.
They should be the same OCs that were removed in the first step. It does not
matter if more than one name associated to a given OC is used in this list

The script will produce a `new_OCs_info.csv` file (used by `run_fastMP.py`) and
a `UCC_corrected.csv` file with the new data on the processed OCs.

4. Combine output with new UCC

Manually combine the output stored in the  `UCC_corrected.csv` file with the
new UCC version generated in the first step.

The re-processing by the previous script will likely have changed the UCC IDs of
some or all of the re-processed OCs. Check here if some of the UCC IDs need to
be manually corrected.

Delete the `UCC_corrected.csv` file when this step is over.

5. Continue the steps in the following sections

- **Generate datafiles with members and update the new UCC**
- **Generate new cluster entries**
- **Update the UCC**



<!-- TOC --><a name="update-ucc-with-a-new-db"></a>
# Update UCC with a new DB

To add a new database to the UCC catalogue and generate the required files,
these following steps must be followed.

1. Add the new DB with the proper format
2. Check the new DB for possible issues
3. Generate a new UCC database version
4. Process new clusters with `fastMP`
5. Generate the `ucc.ar` required files
6. Update all `git` repos to publish the changes in `ucc.ar`


<!-- TOC --><a name="adding-a-new-db"></a>
## Adding a new DB

0. The new database must contain an `ID` column with all the names assigned to a given
   OC, a column with `RA`, and a column wit `DEC` values (no galactic coordinates
   allowed). OCs with multiple names must **not** use ';' as a separating character,
   only ',' is allowed (surrounding the names with "")

1. Add the name of the new DB to the `[General]` section of the `params.ini` file,
   and the column names for the `ID,RA,DEC` parameters in the `[New DB check]` section.

   The format of the name for the DB is `SMITH24` or, if required, `SMITH24_1`.
   The name of the DB **must not** contain any non letter characters except for the
   `_` required to differentiate DBs with the same names published in the same year.

2. Save the new DB in proper `csv` format to the `databases`/ folder

3. Add the new DB to the `databases/all_dbs.json` file following the convention:

```
  "NAME_OF_NEW_DB": {
    "ref": "[Author et al. (2012)](url_to_DB))",
    "names": "name",
    "pars": "E_bv,dist,logt,FeH,Mass,binar_frac,blue_stragglers",
    "e_pars": "None,None,e_logt,None,None,None,None",
    "pos": "ra,dec,plx,pm_ra,pm_dec,Rv"
 },
```

4. Run the `B_new_DB_check.py`  script

This script checks the new DB for proper formatting:

- Make sure that no GCs are very close to listed OCs
- Check for possible duplicates in the new DB
- Possible instances of 'vdBergh-Hagen'/'vdBergh' that must be changed to
  'VDBH' & 'VDB', per CDS recommendation
- Possible empty entries in columns are replaced by 'nan's

Once finished, correct any issues that are shown.

### Summary
- Scripts used: `B_new_DB_check.py`
- Files edited: `databases/all_dbs.json` (manually)
- Files generated: properly formatted database for the new DB (manually)



<!-- TOC --><a name="checking-the-new-db"></a>
## Checking the new DB

1. Run the `C_check_new_DB.py` script

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

### Summary
- Scripts used: `C_check_new_DB.py`
- Files edited: None
- Files generated: None



<!-- TOC --><a name="generating-a-new-version-of-the-ucc"></a>
## Generating a new version of the UCC

1. Run the `D_add_new_DB.py` script

This script will update the current `UCC_cat_XXYYZZ.csv` catalogue with the
OCs in the new DB. The 5D coordinates are **not** updated here if the
OC(s) are already present in the UCC.

The new UCC version will have the same name format, with an updated date.

### Summary
- Scripts used: `D_add_new_DB.py`
- Files edited: None
- Files generated: `zenodo/UCC_cat_XXYYZZ.csv`



<!-- TOC --><a name="generate-datafiles-with-members-and-update-the-new-ucc"></a>
## Generate datafiles with members and update the new UCC

1. Run the `E_run_fastMP.py` script

This script generates new datafiles with members using `fastMP`, **only** for the
new OCs in the  new DB. These are identified as those with a
'nan' value in the 'C3' column of the new generated `zenodo/UCC_cat_XXYYZZ.csv`.
The `.parquet` member files are automatically stored in the proper
`../UCC/QXY/datafiles/` repositories.

This script also generates a `new_OCs_data.csv` file with parameters for the new
OCs, obtained from their members and surrounding field stars.

2. Run the `F_updt_UCC.py` script

**If no new OCs were processed by the previous script, this one can be skipped.**

This script uses the data in the `new_OCs_data.csv` file to update the new UCC
version `zenodo/UCC_cat_XXYYZZ.csv`. It also estimates probable duplicates
using data on positions obtained from the members stars.

Delete `new_OCs_data.csv` once this script is finished.

3. Run the `G_UCC_versions_check.py` script

**Before running this script**: write the name of the previous UCC version in the
section `[Check versions]` of the `params.ini` file, under the `old_UCC_name` variable.

This script checks the old UCC DB versus the new one to flag possible issues
for attention. It needs to be manually updated for whatever issues we are
trying to find.

### Summary
- Scripts used: `E_run_fastMP.py, F_updt_UCC.py, G_UCC_versions_check.py`
- Files edited: `zenodo/UCC_cat_XXYYZZ.csv`
- Files generated: `new_OCs_data.csv` (deleted after 2.) + members datafiles



<!-- TOC --><a name="generate-new-cluster-entries"></a>
## Generate new cluster entries

1.  Run the `H_make_entries.py` script

This script will process the **entire** UCC and generate an `md` file, `ipynb`
notebook, and plot, for every OC for which either of those files do not exist.
In the case of the `md` entries, it will also check if the entry changed
compared to the old one, and it will update it if it did.

For each processed OC that is missing either of those files:

1. Generate or update an `.md` entry, stored in `../ucc/_clusters/`
2. Generate a `.ipynb` notebook, stored in `../UCC/QXY/notebooks/`
3. Generate a plot, stored in `../UCC/QXY/plots/`

### Summary
- Scripts used: `H_make_entries.py`
- Files edited: `md` entries (if there are changes in the new UCC)
- Files generated: `md` + notebooks + plots (if files are missing)



<!-- TOC --><a name="update-the-ucc"></a>
## Update the UCC

1. Run the script `I_final_updt.py`

This script will:

- update the `../ucc/_clusters/clusters.json` file used for searching in `ucc.ar`
- update the tables files used by the `ucc.ar` site
- generate the files that contain all the UCC information, used in Zenodo

Once finished:

0. Run `$ vulture .` + `$ black .` on the repository
1. Push changes in `updt_UCC` repository
2. Push changes in `QXY` repositories
3. Update the Zenodo repository with the new files, creating a new release
4. Update the `CHANGELOG.md` file with the Zenodo URL for the release
5. Push changes in `ucc` repository


To check folders recursively for changes use:
`$ clear & find . -name '.git' | while read -r repo ; do repo=${repo%".git"}; (git -C "$repo" status -s | grep -q -v "^\$" && echo -e "\n\033[1m${repo}\033[m" && git -C "$repo" status -s) || true; done`

### Summary
- Scripts used: `I_final_updt`
- Files edited: `ucc/clusters.json, ucc/_pages/QXY_table.md`
- Files generated: `zenodo/UCC_cat.csv.gz, zenodo/UCC_members.parquet.gz`
