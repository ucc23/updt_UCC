
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





# Update procedure for the UCC

To add a new database to the UCC catalogue and generate the required files, the
following steps are required. Most of the process is automated.


## First script

The  `1_get_new_DB.py` script manages the updating of the JSON file
that contains the information for each database in the UCC, as well as downloading
the Vizier database if available and/or requested.

Once the JSON file with the entry for the new DB and the generated CSV file are
stored in the temporary folder, **check carefully both files before moving on**. Both
might need manual intervention.

Files generated: `all_dbs.json`, `NEW_DB.csv` (both stored in temp `databases/` folder)


## Second script

The `2_update_UCC.py` script handles the following tasks:

1. Check columns in the new DB
  a. Checks for required columns
  b. Checks for special characters in the name column (`;` or `_`)
2. Standardize names in new DB and match with UCC
3. Check the new DB
  a. Checks for duplicate entries between the new database and the UCC
  b. Checks for nearby GCs
  c. Checks for OCs very close to each other within the new database
  d. Checks for OCs very close to each other between the new database and the UCC
  e. Checks for instances of 'vdBergh-Hagen' and 'vdBergh'
  f. Checks positions and flags for attention if required
4. Generate a new UCC version
  a. Update the UCC catalogue with the new DB
  b. Assigns UCC IDs and quadrants for new clusters.
  c. Performs a final duplicate check.
5. Generate members' datafiles and update UCC
  a. Generate member files for new OCs using fastMP
  b. Store the member files in temporary folders (`temp_updt/QXY/datafiles/*.parquet`)
  c. Update the UCC with the new OCs member's data 
6. Save updated UCC to a temporary folder (`temp_updt/zenodo/UCC_cat_XXYYZZHH.csv`)
7. Move temporary files to their final destination
8. Final check of new UCC

Once this script is finished:

- The `all_dbs.json` and `NEW_DB.csv` files created by the previous script are moved
  to the `databases/` folder
- The `parquet` files for each new OC are moved to the `QXY/` folders
- The new `UCC_cat_XXYYZZHH.csv` file is moved to the `zenodo/` folder, archiving
  the old one


## Third

The `3_update_zenodo_files.py` script generates the three required files for uploading
to the Zenodo repository: `UCC_cat.csv, UCC_members.parquet, README.txt`

The files are stored in the temporary folder `temp_updt/zenodo/` awaiting uploading.


## Fourth script

The `4_update_UCC_site.py` script applies the required changes to update the ucc.ar
site. It processes the **entire** UCC catalogue and and searches for modifications that
need to be applied to update the site.

1. Generate/update per cluster `.md` (stored in `ucc/_clusters/`) and `.webp` files
   (stored in the `QXY/plots/` folders)
2. Update the main ucc site files. This includes the`ucc/_pages/DATABASE.md` as well as
   all the tables and images used in the site
3. Update JSON file (`ucc/clusters.json`)
4. Move all files to their final destination
5. Check that the number of files is correct




## UCC public site build

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

1. Push changes (if any) to each of the `QXY` repositories. To do this, position
   the command line in the `UCC/` folder and run (change `YYMMDD` with version number):

```
for dir in Q*/; do (cd "$dir" && [ -d .git ] && git acp "version YYMMDD"); done
```

2. Create a 'New version' in the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.8250523) 

2.0 Make sure that the version number in the README matches than in the CHANGELOG
2.1 Upload the three files stored in the `temp_updt/zenodo/` folder
2.2 Get a DOI
2.3 Add a 'Publication date' with the format: YYYY-MM-DD
2.4 Use the version number in the README (format: YYMMDD) in the release

Publish new release and copy **its own url** (no the general repository url)

3. Update the `CHANGELOG.md` file, use the Zenodo URL for **this** release

4. Push the changes to the `ucc` repository

~NOT ANMORE~
Every change pushed to the [ucc](https://github.com/ucc23/ucc) repository triggers an automatic build and
deployment. **Check carefully before pushing.**
~NOT ANMORE~

5. Deploy the site using the Github workflow 'Build GitHub Page':

https://github.com/ucc23/ucc/actions/workflows/jekyll-gh-pages.yml

