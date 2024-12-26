
# Update UCC

To add a new database to the UCC catalogue and generate the required files, the
following steps are required. Most of the process is automated.


## First script

The first script is called `update_UCC_DB.py`. **Before running the script**

1. Save the new DB in `csv` format to the `databases/` folder

The format of the name for the DB is `SMITH2024` or, if required, `SMITH2024_1`.
The name of the DB **must not** contain any non letter characters except for the
`_` required to differentiate DBs with the same names published in the same year.

2. Add the article name + url of the new DB and the column names to the `[New DB data]`
   section of the `params.ini` file.

The new database **must** contain a column with all the names assigned to a given
OC, a column with `RA`, and a column wit `DEC` values. Columns for Plx, PMs, and/or
Rv are not a requirement. Columns for any fundamental parameter and their uncertainties
are not a requirement either.

Once these new DB file and its parameters are in place, run the `update_UCC.py` script.
It handles the following tasks:

- [Prepare the DB](#prepare-the-db)
- [Check format and issues](#check-format-and-issues)
- [Generate a new UCC version](#generate-a-new-ucc-version)
- [Generate members' datafiles](#generate-members-datafiles)
- [General check of new UCC](#general-check-of-new-ucc)


### Prepare the DB

1. Checks if the new database is already in the JSON file.
2. Validates the new database's name.
3. Validates the new database's year.
4. Checks for required and non-required columns.
5. Checks for special characters in the name column (`;` or `_`).
6. Replaces empty positions with `NaN`s.
7. Adds the new database to the `databases/all_dbs.json` JSON file.

- Files edited: `databases/all_dbs.json`, new DB (empty spaces, naming, etc)
- Files generated: None


### Check format and issues

1. Checks for duplicate entries between the new database and the UCC.
2. Checks for nearby GCs.
3. Checks for OCs very close to each other within the new database.
4. Checks for OCs very close to each other between the new database and the UCC.
5. Checks for instances of 'vdBergh-Hagen' and 'vdBergh' (must be changed to
  'VDBH' & 'VDB', per CDS recommendation).
6. Checks positions and flags for attention if required.

The position flags are handled as follows:

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

- Files edited: None
- Files generated: None


### Generate a new UCC version

This process updates the current `UCC_cat_XXYYZZ.csv` catalogue with the OCs in the
new DB. The 5D coordinates are **not** updated if the OC(s) is already present in the
UCC.

1. Combines the UCC and the new database.
2. Assigns UCC IDs and quadrants for new clusters.
3. Performs a final duplicate check.

- Files edited: None
- Files generated: `zenodo/UCC_cat_XXYYZZ.csv` (updated date)


### Generate members' datafiles

If no **new** OCs were added by the previous script, this process is skipped. New OCs
are identified as those with a `nan` value in the `C3` column of the new
`zenodo/UCC_cat_XXYYZZ.csv` file.

1. Constructs a KD-tree for efficient spatial queries on the UCC.
2. Processes each new OC by:
    - Generating a frame for the OC.
    - Applying manual parameters if available.
    - Identifying close clusters.
    - Requesting data for the OC  using Gaia data.
    - Processing the OC with the `fastMP` method.
    - Splitting the data into members and field stars.
    - Extracting members data and updating the UCC.
    - Save members `.parquet` file in the proper Q folder

- Files edited: `zenodo/UCC_cat_XXYYZZ.csv`
- Files generated: `QXY/datafiles/*.parquet`


### General check of new UCC

Run checks on old and new UCC files to ensure consistency and identify possible issues
for attention.

- Files edited: None
- Files generated: None




## Second script

The above steps prepare the files for the updated version of the UCC. The following
steps **apply** the required changes to the site's files.

- [8. Generate new Zenodo files](#8-generate-new-zenodo-files) (stored at `zenodo_upload/` folder)
- [9. Generate new clusters entries](#9-generate-new-clusters-entries)
- [10. Update site's files](#10-update-sites-files)




## 8. Generate new Zenodo files

Run the script `H_zenodo_updt.py`

This script will generate the files that are to be uploaded to Zenodo. These files
contain all the UCC information.

**Summary**

- Files edited: None
- Files generated: `UCC_cat.csv, UCC_members.parquet,README.txt`



## 9. Generate new clusters entries

Run the `I_make_entries.py` script

This script will process the **entire** UCC and generate an `md` file and plot(s),
for every OC for which either of those files do not exist.

It will check if the new entry changed compared to the old one, and it will
update it **only** if it did.

For each processed OC that is missing either of those files:

1. Generate a `.md` entry, stored in `../ucc/_clusters/`
2. Generate a plot (two, if aladin plot is also generated), stored in `../QXY/plots/`

**Summary**

- Files edited: `../ucc/_clusters/*.md` entries (if there are changes in the new UCC)
- Files generated: `../ucc/_clusters/*.md` + `../QXY/plots/*.webp` (if files are missing)


## 10. Update site's files

Run the script `J_database_updt.py`. This script will:

- update the `DATABASE.md` file used by the `ucc.ar` site
- update the tables files linked to the above file
- update the `../ucc/_clusters/clusters.json` file used for searching in `ucc.ar`

**Summary**

- Files edited: `ucc/_pages/DATABASE.md, ucc/_pages/QXY_table.md, ucc/clusters.json`
- Files generated: `UCC_diff.csv`





## UCC public site build

Before updating the live site, generate a local site build and check the results
**carefully**. To build a local copy of the site we use Jekyll, see [Jekyll docs](https://jekyllrb.com/docs/).
Position a terminal in the `/ucc` folder (**not** the `/updt_ucc` folder) and run:

```
$ bundle exec jekyll serve --incremental
```

This will generate a full version of the site locally which can take a while. For a
faster build, avoid processing the files in the `_clusters` folder. To do this open
the `_config.yml` file and un-comment the last line in the `exclude:` section:

```
# Exclude these files from your production _site
exclude:
  - Gemfile
  - Gemfile.lock
  - LICENSE
  - README.md
  - CNAME
  # Un-comment to exclude for faster processing in a local build
  - _clusters
```

**IMPORTANT:** comment the `_clusters` folder in the `exclude:` section of the
`_config.yml` file before moving on.


Live build steps:

1. Push changes in each of the `QXY` repositories (if any)

2. Create a 'New version' in the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.8250523) 

2.1 Upload the three files stored in the `zenodo_upload/` folder by the `H` script
2.2 Get a DOI
2.3 Add a 'Publication date' with the format: YYYY-MM-DD
2.4 Add a 'Version' number with the format: YYMMDD

Publish new release and copy **its own url** (no the general repository url)

3. Update the `CHANGELOG.md` file, use the Zenodo URL for **this** release

Every change pushed to the [ucc](https://github.com/ucc23/ucc) repository triggers an
automatic build + deployment. **Check carefully before pushing.**

4. Push the changes to `ucc` repository
