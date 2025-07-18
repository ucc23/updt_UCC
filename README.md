
# UCC management

The UCC lives in several repositories within the [UCC23](https://github.com/ucc23) organization
and in its associated [Zenodo repository](https://doi.org/10.5281/zenodo.8250523).

- The main UCC database along with a single file with all the identified members are
  stored in the Zenodo repository

- For each entry in the database there is a corresponding entry in the [ucc](https://github.com/ucc23/ucc)
  repository in the form of an `.md` file. This repository also contains the
  files required to build the [public site](https://ucc.ar)

- The eight `QXX` repositories ([Q1P](https://github.com/ucc23/Q1P), etc.) contain the `.parquet` files with the
  identified members for each OC (in the `datafiles` folders), as well as a few plots
  per OC (in the `plots` folders): one with the four diagrams, one for Aladin Lite,
  and plots for the HUNT23 and/ot CANTAT20 members when available. The plots are loaded
  by the public site from this repository.

- Finally the [updt_ucc](https://github.com/ucc23/updt_UCC) repository, this one, contains the scripts and data files
  required to update the UCC (mostly) automatically


## 1. Adding a new DB

Given a new database (DB) to be added to the UCC, the  `A_get_new_DB.py` script
manages the updating of the JSON file that contains the information for each database
in the UCC, as well as downloading the Vizier database if available and/or requested:

1. Load the current JSON database file.
2. Check if the URL is already listed in the current database.
3. Fetch publication authors and year from NASA/ADS
4. Generate a new database name based on extracted metadata.
5. Handle temporary database files and check for existing data.
6. Fetch Vizier data or allow manual input for Vizier IDs.
7. Match new database columns with current JSON structure.
8. Update the JSON file and save the database as CSV.

Once the JSON file with the entry for the new DB and the generated CSV file with the
DB data are stored in the temporary folder, **check carefully both files before moving
on** as both might need manual intervention.

**Important**:
1. The column with the names of the new entries can contain several names; these must
   be separated using a ','.
2. The first name in the name(s) column (if there's more than one) will be used as
   the `fname` for that entry

### Input

- `ADS_bibcode`: NASA/ADS bibcode for the new DB (internal script parameter)
- `databases/all_dbs.json`: Current JSON file with UCC databases

### Output

- `temp_updt/databases/all_dbs.json`: Updated JSON file with new DB entry
- `temp_updt/databases/NEW_DB.csv`: New database in CSV format



## 2. Updating the UCC

This script handles two cases:

1. `run_mode="new_DB"`: Add a new DB to the UCC. The `A_get_new_DB.py` script must be
   run before this mode is used.

2. `run_mode="updt_DB"`: Update a DB that is **already present in the UCC**. The name
   of the DB must be given.

3. `run_mode="manual"`: Re-process entries already in the UCC. This requires a
   `manual_params.csv` file with the entries to re-process listed inside.

When using the `new_DB` or `updt_DB` mode this script will:

1. Check columns in the new DB
  a. Checks for required columns
  b. Checks for special characters in the name column
2. Standardize names in new DB and match with UCC
3. Check the new DB
  a. Checks for duplicate entries inside the new DB and between the new DB and the UCC
  b. Checks for nearby GCs
  c. Checks for OCs very close to each other within the new DB
  d. Checks for OCs very close to each other between the new DB and the UCC
  e. Checks for instances of 'vdBergh-Hagen' and 'vdBergh'
  f. Checks positions and flags for attention if required
4. Generate a new UCC version
  a. Update the UCC catalogue with the new DB
  b. Assigns UCC IDs for new clusters.
  c. Performs a final duplicate check.

If we use the `manual` mode the script will skip steps 1-4 described above. Instead
it will load the `manual_params.csv` file and re-process all entries listed there.

After either running steps 1-4 or loading the `manual_params.csv` file, the script
will:

5. Generate members' datafiles and update the UCC:
  a. Generate member files for processed OCs using fastMP (those with `N_50==nan`)

  The selection process for UCC members is as follows:
  
  ```
  if (P>0.0).sum()==0:
    raise ERROR
  
  if (P>0.5).sum()>=25:
    save members
  else:
    save stars with the largest probabilities as members (max 25)
  ```

  b. Save the member files (`temp_updt/datafiles/*.parquet`)
  c. Update the UCC with the processed OCs member's data 
6. Save updated UCC (`temp_updt/zenodo/UCC_cat_XXYYZZHH.csv`)
7. Move temporary files to their final destination

### Input

- `Gaia data files`: Gaia data files for a given release
- `manual_params.csv`: Configuration parameters
- `databases/globulars.csv`: Globular clusters data file
- `zenodo/UCC_cat_XXYYZZHH.csv`: Current UCC catalogue
- `zenodo/data_dates.json`: Current file with information about when the member files
  where created/updated
- `databases/all_dbs.json`: Current JSON database file
- `temp_updt/databases/all_dbs.json`: New JSON file (generated by script A)
- `temp_updt/databases/NEW_DB.json`: New database (generated by script A)

### Output

- `databases/all_dbs.json`: Updated UCC database JSON file
- `databases/NEW_DB.csv`: New database in CSV format
- `zenodo/UCC_cat_XXYYZZHH.csv`: Updated UCC database; previous version is archived
- `zenodo/UCC_members.parquet.temp`: Temporary file with all the members
- `zenodo/data_dates.json`: Updated file

These files are also generated, for exploratory use only:

- `temp_updt/df_UCC_updt.csv`: File with all the updated entries
- `temp_updt/UCC_diff_new.csv`: Non-matching entries in the new database
- `temp_updt/UCC_diff_old.csv`: Non-matching entries in the old database

The last two files are meant to be viewed with a diff app (e.g.: Meld) side by side



## 3. Updating Zenodo files

The `C_update_zenodo_files.py` script generates the three required files for uploading
to the Zenodo repository. The output files need to be manually uploaded to a new
Zenodo release.

### Input

- `zenodo/README.static.txt`: Static version of the README file
- `zenodo/UCC_cat_XXYYZZHH.csv`: Latest UCC database
- `zenodo/UCC_members.parquet.temp`: Member files for all the clusters. Deleted after
  it is used, may not exist if no new members were obtained

### Output

- `zenodo/README.txt`: New README file with the latest UCC information
- `zenodo/UCC_cat.csv`: New UCC database
- `zenodo/UCC_members.parquet`: New file with all the members
  **THIS FILE IS NOT TRACKED IN GITHUB BECAUSE IT IS LARGE**



## 4. Updating the site

The `D_update_UCC_site.py` script applies the required changes to update the ucc.ar
site. It processes the **entire** UCC catalogue and searches for modifications that
need to be applied to update the site.

1. Generate/update per cluster `.md` (stored in `ucc/_clusters/`) and `.webp` files
   (stored in the `QXY/plots/` folders)
2. Update the main ucc site files. This includes the`ucc/_pages/DATABASE.md` as well as
   all the tables and images used in the site
3. Update JSON file (`ucc/assets/clusters.json`)
4. Move all files to their final destination
5. Check that the number of files is correct

### Input

- `temp_updt/zenodo/README.txt`: Used to extract the total number of members
- `zenodo/UCC_cat_XXYYZZHH.csv`: Latest UCC database
- `databases/all_dbs.json`: Latest UCC database JSON file
- `ucc/_pages/DATABASE.md`: Current databases file for the site

### Output

- `ucc/assets/clusters.json.gz`: Updated JSON file with the latest UCC data
- `ucc/images/catalogued_ocs.webp`: Updated image with the latest UCC data
- `ucc/images/classif_bar.webp`: Updated image with the latest UCC data
- `ucc/_pages/DATABASE.md`: Update databases file for the site
- `ucc/_pages/TABLES.md`: Updated tables file for the site
- `ucc/_tables/*.md`: Updated tables for the site
- `ucc/_clusters/*.md`: Updated or generated files for each cluster
- `UCC/QXY/plots/*.webp`: CMD and Aladin plots, **only for new clusters**



## 5. Building the site

### 5.1 Local build

Before updating the live site, generate a local site build and check the results
**carefully**. To build a local copy of the site we use Jekyll, see [Jekyll docs](https://jekyllrb.com/docs/).

If this is a new installation, update the gems with:

```
$ bundle update --bundler
```

To build a local version of the site, position a terminal in the `/ucc` folder
(**not** the `/updt_ucc` folder) and run:

```
$ bundle exec jekyll serve --incremental
```

This will generate a full version of the site locally which can take a while. For a
faster build, avoid processing the files in the `_clusters, _tables` folder (for
example, using a different `include` with fewer/different selected folders)


### 5.2 Live build

1. Push changes (if any) to each of the `QXY` repositories. To do this, position
   the command line in the `UCC/` folder and run (change `YYMMDD` with version number):

```
for dir in Q*/; do (cd "$dir" && [ -d .git ] && git acp "version YYMMDD"); done
```

2. Create a 'New version' in the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.8250523) 

3.0 Make sure that the version number in the README matches than in the CHANGELOG
3.1 Upload the three files stored in the `temp_updt/zenodo/` folder
3.2 Get a DOI
3.3 Add a 'Publication date' with the format: YYYY-MM-DD
3.4 Use the version number in the README (format: YYMMDD) in the release

Publish new release and copy **its own url** (no the general repository url)

3. Update the `CHANGELOG.md` file, use the Zenodo URL for **this** release

4. Pull to update the new arXiv JSON file to the `ucc` repository <-- **IMPORTANT**

5. Push the changes to the `ucc` repository

6. Deploy the site using the Github workflow 'Build GitHub Page':

https://github.com/ucc23/ucc/actions/workflows/jekyll-gh-pages.yml
