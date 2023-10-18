
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

To fix bad entries the following steps need to be followed.

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

1. Save the new DB in proper `csv` format to the `databases`/ folder

2. Run the `B_new_DB_check.py`  script

This script checks the new DB for proper formatting:

- Make sure that no GCs are very close to listed OCs
- Check for possible duplicates in the new DB
- Possible instances of 'vdBergh-Hagen'/'vdBergh' that must be changed to
  'VDBH' & 'VDB', per CDS recommendation
- OCs with multiple names must not use ';' as a separating character
- Possible empty entries in columns are replaced by 'nan's

Once finished, correct any issues that are shown.

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

JAEHNIG21
SANTOS21
HE21
CASTRO22
TARRICQ22
LI22
HE22
HE22_1
HAO22
PERREN22
HE23
HUNT23
QIN23
LI23
CHI23_2
CHI23
CHI23_3

### Summary

- Scripts used: `B_new_DB_check.py`
- Files edited: `databases/all_dbs.json`
- Files generated: properly formatted database for the new DB



<!-- TOC --><a name="checking-the-new-db"></a>
## Checking the new DB

1. Run the `C_check_new_DB.py` script

This script will find the name-based matches and display any issues with OCs
that are present in the UCC and the new DB. It will also generate the file
`new_OCs_info.csv` which contains information about what to process for each OC
in the new DB, based on an algorithm.

The algorithm to decide whether to run `fastMP` to estimate members for a
specific OC and produce its plot/notebook files (the `md` files will always be
updated/generated), is as follows:

```
Is the OC already present in the UCC?
    |         |
    v         |--> No --> flag=True
   Yes
    |
    v
Is the difference between the old vs new centers values large?
    |         |
    v         |--> No --> flag=False
   Yes
    |
    v
Request attention --> flag=False


flag:
True  --> add centers + fastMP + md entry + plot + notebook
False --> only update md entry
```

Go through the `new_OCs_info.csv` file and fix any possible issues flagged
before running the `add_new_DB.py` script.

### Summary

- Scripts used: `C_check_new_DB.py`
- Files edited: None
- Files generated: `new_OCs_info.csv`



<!-- TOC --><a name="generating-a-new-version-of-the-ucc"></a>
## Generating a new version of the UCC

1. Run the `D_add_new_DB.py` script

This script will use the information in the `new_OCs_info.csv` file to update
the current `UCC_cat_XXYYZZ.csv` catalogue. The new version will have the same
name format, with an updated date.

### Summary

- Scripts used: `D_add_new_DB.py`
- Files edited: None
- Files generated: `zenodo/UCC_cat_XXYYZZ.csv`



<!-- TOC --><a name="generate-datafiles-with-members-and-update-the-new-ucc"></a>
## Generate datafiles with members and update the new UCC

1. Run the `E_run_fastMP.py` script

This will process the clusters present in the `new_OCs_info.csv` file as
indicated by the OCs flags, and generate new datafiles with members using
`fastMP`.  The member files are automatically stored in the proper
`../UCC/QXY/datafiles/` repositories. This script generates a `new_OCs_data.csv`
file with parameters for the new OCs obtained from their members and surrounding
field stars.

2. Run the `F_updt_UCC.py` script

This script uses the data in the `new_OCs_data.csv` file to update the new UCC
version `zenodo/UCC_cat_XXYYZZ.csv`. It also estimates probable duplicates
using data on positions obtained from the members stars.

Delete `new_OCs_data.csv` once this script is finished.

3. Run the `G_UCC_versions_check.py` script

This script checks the old UCC DB versus the new one to flag possible issues
for attention. It needs to be manually updated for whatever issues are trying
to be found.

### Summary

- Scripts used: `E_run_fastMP.py, F_updt_UCC.py, G_UCC_versions_check.py`
- Files edited: `zenodo/UCC_cat_XXYYZZ.csv`
- Files generated: `new_OCs_data.csv` + members datafiles



<!-- TOC --><a name="generate-new-cluster-entries"></a>
## Generate new cluster entries

1.  Run the `H_make_entries.py` script

This script will process all OCs flagged to be processed in the
`new_OCs_info.csv` file and generate a plot, `.ipynb` notebook, and
proper `.md` entry in the `../ucc/_clusters/` folder for each one.

For each processed OC:

1. Generate an `.md` entry, stored in `../ucc/_clusters/`
2. Generate a `.ipynb` notebook, stored in `../UCC/QXY/notebooks/`
3. Generate a plot, stored in `../UCC/QXY/plots/`

Delete `new_OCs_info.csv` once this script is finished.

### Summary

- Scripts used: `H_make_entries.py`
- Files edited: None
- Files generated: md + notebooks + plots



<!-- TOC --><a name="update-the-ucc"></a>
## Update the UCC

1. Run the script `I_final_updt.py`

This script will

- update the `../ucc/_clusters/clusters.json` file used by the `ucc.ar` site
  for searching
- update the tables files used by the `ucc.ar` site
- generate the files that contain all the UCC information, used in Zenodo

Once finished:

1. Update the `CHANGELOG.md` file
2. Push changes in `ucc` & `QXY` repositories to Github
3. Update the Zenodo repository with the new files creating a new release

### Summary

- Scripts used: `I_final_updt`
- Files edited: `ucc/clusters.json, ucc/_pages/QXY_table.md`
- Files generated: `zenodo/UCC_cat.csv.gz, zenodo/UCC_members.parquet.gz`
