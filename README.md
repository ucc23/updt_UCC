
# UCC management

The UCC files live in several repositories within the [UCC23](https://github.com/ucc23) organization.

- The [ucc](https://github.com/ucc23/ucc) repository contains the main files required to build the
  [public site](https://ucc.ar). Every pushed commit to this repo produces an
  **automatic build of the site**.
- The eight `QXX` repositories ([Q1P](https://github.com/ucc23/Q1P), etc.) contain the bulk of the files used 
  by the site,  within the folders `datafiles, notebooks, plots`.
- The [updt_ucc](https://github.com/ucc23/updt_UCC), this one, contains the scripts and data files required to
  update the UCC (mostly) automatically. This repository can be updated at any time
  since this **does not** trigger a build of the UCC site.


<!-- MarkdownTOC -->

- UCC public site build
- UCC local site build
- Fixing bad entries

<!-- /MarkdownTOC -->




```
Adds a new DB?
Yes --> Follow instructions in add_new_DB.md
No  --> Modify one or more existing entries?
        Yes  --> Follow `XXX.md` instructions
        No   --> Add one or more entries?
                 Yes --> Follow `YYY.md` instructions
                 No  --> Remove one or more entries?
                         Yes --> Follow `ZZZ.md` instructions
                         No  --> ???
```






## UCC public site build

Before committing to this repo, generate a full local build and check the site.

Every change made to this repository and pushed to Github will trigger and automatic
build + deployment. **Commit carefully.**

1. Push changes in `QXY` repositories
2. Update the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.8250523) with the new files, creating a new release.
   Remember to set the version number in Zenodo as YYMMDD
3. Update the `CHANGELOG.md` file, use the Zenodo URL for **this** release
4. Push changes to `/ucc` repository

**IMPORTANT:** Remember to comment the `_clusters` folder in the `exclude:` section of 
the `_config.yml` file.


## UCC local site build

To build a local copy of the site we use Jekyll, see [Jekyll docs](https://jekyllrb.com/docs/).

Position a terminal in the `/ucc` folder (**not** the `/updt_ucc` folder) and run:

```
$ bundle exec jekyll serve --incremental
```

This will generate a full version of the site locally which can take a while. For a
faster build **avoid processing** the files in the `_clusters` folder. To do this open
the `_config.yml` file and un-comment the last line in the `exclude:` section:

```
# Exclude these files from your production _site
exclude:
  - Gemfile
  - Gemfile.lock
  - LICENSE
  - README.md
  - CNAME
  - _clusters # Un-comment to exclude for faster processing in a local build
```











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
