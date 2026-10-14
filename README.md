
# UCC management

The UCC lives in several repositories within the [UCC23](https://github.com/ucc23)
organization and in its associated [Zenodo repository](https://doi.org/10.5281/zenodo.8250523).

- The [updt_ucc](https://github.com/ucc23/updt_UCC) repository, this one, contains the scripts and data files
  required to update the UCC (mostly) automatically. It also contains the
  main files that make up the UCC: `all_names.csv`, `UCC_cat_B_out.csv`,
  `UCC_cat_C_in.csv`, `UCC_cat_C_out.csv`, and the `UCC_members.parquet` file with
  all the identified members (stored in the `data/zenodo/` folder).

- The [ucc](https://github.com/ucc23/ucc) repository contains for each entry in the database a corresponding
  entry in the form of an `.md` file, in the `_clusters/` folder. This repository
  also contains all the files required to build the [public site](https://ucc.ar)

- The `plots_X` repositories contain several plots per OC in various folders:
  one with the four diagrams, one for Aladin, and plots for the HUNT23 and/or CANTAT20
  members when available. The plots are loaded by the public site from this repository.

The UCC update process is managed through five main scripts that perform the
required tasks in sequence, as described in the following sections.



## 1. Adding a new DB

Given a new database (DB) to be added to the UCC, the  `A_get_new_DB.py` script
manages the updating of the JSON file that contains the information for each database
in the UCC, as well as downloading the Vizier database if available and/or requested:

1. Load the current JSON database file and check that no two DBs share the same
   `received` date.
2. Offer to update the `citations_count` of the DBs whose counts are older than
   180 days (the script stops after updating them).
3. Check that every parameter format in the JSON file is valid (`pars_formats` in
   `variables.py`).
4. Ask for the NASA/ADS bibcode and check if it is already listed in the JSON file.
5. Fetch publication authors, year, title, and citations count from NASA/ADS.
6. Generate a new database name based on extracted metadata.
7. Fetch the Vizier data (or load the temporary CSV file if it already exists).
8. Auto-detect the columns of the new DB that match the `names`, `types`, `pos`,
   `pars`, and `e_pars` keys of the JSON file (see below).
9. Write the temporary JSON file with the new DB entry and check that every DB
   marked with `comments` in `data_cmmts` has a comments file (and vice versa).

The columns are matched against the column names already used for each key in the
JSON file (exact, normalized, and fuzzy matching). Names must be string columns, and
positions and parameters numeric. Each column and each key are used at most once.
Uncertainties (`e_pars`) are assigned from the matched parameters: columns such as
`e_X`, `s_X`, `delta_X`, `X_sigma`, or the percentiles `X_16`/`X_84` for a matched
parameter `X`. The parameter format (e.g. `loga`, `amyr`, `agyr` for the age) is
inferred from the column name or values. The `types` key is filled if a column
contains embedded cluster ids (see [JSON file format](#json-file-format)). The
patterns and ranges used are defined in the `A` section of `variables.py`.

Once the JSON file with the entry for the new DB and the generated CSV file with the
DB data are stored in the temporary folder, **check carefully both files before moving
on** as both might need manual intervention.

### Input

- `ADS_bibcode`: NASA/ADS bibcode for the new DB
- `data/databases_info.json`: Current JSON file with info about the databases included
  in the UCC
- `data/NASA_API_TOKEN`: NASA/ADS API token
- `data/databases/cmmts/*.csv`: Comments files (only checked)

### Output

- `temp_updt/data/databases_info.json`: Updated JSON file with new DB entry
- `temp_updt/data/databases/NEW_DB.csv`: New database in CSV format (if Vizier
  returns several tables they are stored as `NEW_DB_1.csv`, `NEW_DB_2.csv`, ...)

**Important**:
The column with the names of the new entries can contain several names; these must
be separated using a ','.

**Remember to carefully check the keys for the new DB in the JSON file. They are auto
detected by the script and they might need to be corrected and/or filled.**

The output files will be loaded and updated by the B script, they must **not be
manually moved**.


### JSON file format

The format of the `databases_info.json` file is mostly self-explanatory with the
exception of the `types` and parameters sections.

The `types` key identifies the embedded clusters (ECs) in the DB:

```
"types": {"col_name": "Type", "ec_id": ["EC", "ECC"]}
```

where `col_name` is the column that holds the type of each entry and `ec_id` lists
the values in that column that identify an EC. Use `"col_name": "ALL"` if every entry
in the DB is an EC, and `"col_name": "N/A"` if the DB does not identify ECs. The B
script uses this key to generate the `Type` column.

The parameters section This can have different formats depending on the
database structure. There are three possible formats:

1. **Default**

The default formatting corresponds to a single column with a given formatting/units
for each parameter:

```
"pars": {
  "par_general": {
    "par_format": "par_db_col"
  },
  ...
}
```

For some databases the "par_db_col" can contain multiple values separated by a
',' or ';'. Eg: KRONBERGER2006, ROSER2016, NIZOVKINA2025

2. **Duplicate values, different units**

The "par_general" parameter is present in more tan one column with different
units (eg, obtained by different methods). Eg: JAEHNIG2021, CARRASCO2025

```
"pars": {
  "par_general": {
    "par_format_0": "par_db_col_0",
    "par_format_1": "par_db_col_1",
  },
  ...
}
```

3. **Duplicate units**

There is more than one column with the same "par_format" units for the
same parameter. Eg: CHEN2003, PISKUNOV2008, SANTOS2021, HUNT2024, ZHANG2024, HU2025

```
"pars": {
  "par_general": {
    "par_format": ["par_db_col_0", "par_db_col_1"],
  },
  ...
}
```





## 2. Updating the UCC: first step

The `B_update_UCC` script handles two possible cases:

1. Rebuilding the entire UCC
2. Adding a new database to the UCC

The case is decided by the script according to whether it finds a temporary
`databases_info.json` file with new DBs to add. In both cases the B catalogue is
re-built from all the DBs (those with `data_cmmts="comments"` are skipped); the new
DBs are checked more thoroughly (the checks for the rest can be skipped).

The `all_names` file contains the following columns:

> "fnames","Names"

which are the main identifiers for each entry in the UCC. This file is re-built every
time the `B_update_UCC` script is run.

The final `UCC_cat_B_out.csv` file contains the following columns:

> "fname","DB","DB_i","F_score"

which identify the position of each entry in the original DBs. The `DB` column
contains the names of the DBs where the entry is present, and the `DB_i` column
contains the index of the entry in each original DB. Along with these columns,
the file also contains the following columns which are used only as centers to estimate
membership by the next script:

> "RA_ICRS","DE_ICRS","GLON","GLAT","Plx","pmRA","pmDE","DB_coords_used"

where `DB_coords_used` is the DB the center values were taken from (see
`DB_coords_hierarchy` in `variables.py`, and `UCC_cat_B_in.csv` for manual
centers). The fundamental parameters extracted from each DB where the entry is
present are stored in the columns:

> "dist","av","diff_ext","age","met","mass","bi_frac","blue_str"

followed by the `Type` column (`OC`, `EC`, or `EC;OC` if only some DBs catalogue
the entry as an EC; see the `types` key in the JSON file) and the `<par>_median`, `<par>_stddev` columns for each parameter.

### Input

- `data/all_names.csv`: File with the names of each entry in the UCC
- `data/UCC_cat_B_out.csv`: Current version of this file
- `data/UCC_cat_B_in.csv`: Manual center coordinates for selected entries
- `data/databases_info.json`: Current JSON database file
- `data/databases/*.csv`: All the DBs in the UCC
- `data/globulars.csv`: Globular clusters data file
- `temp_updt/data/databases_info.json`: New JSON file (generated by script A)
- `temp_updt/data/databases/NEW_DB.csv`: New database (generated by script A)

### Output

Files are generated in `temp_updt/` and, after confirmation, moved to their final
destination:

- `data/UCC_cat_B_out.csv`: Updated version of this file (old one is archived in
  `data/ucc_archived_nogit/`)
- `data/all_names.csv`: Updated version of this file

If new DBs were added:

- `data/databases_info.json`: Updated UCC database JSON file
- `data/databases/NEW_DB.csv`: New database in CSV format


### Handling embedded clusters

Embedded clusters are identified through the `types` key of each DB in the JSON
file (see [JSON file format](#json-file-format)), which is used to generate the
`Type` column of the B file. Check this key for every new DB.






## 3. Updating the UCC: second step

The `C_process_member_files.py` script generates the `UCC_cat_C.csv` file and updates
the parquet file with all the identified members. It also generates the required files
for uploading to the Zenodo repository (these files need to be manually uploaded to a
new Zenodo release).

The C catalogue is split into two files. The `UCC_cat_C_in.csv` file contains the
(manual) input columns used to process each entry:

```
- fname: name of each entry (must match those in the B file)
- process: (y/n) manual flag that indicates whether to (re)process this entry
- frame_limit: A string in the format "x_111.1" where "x" is one of the characters:

   - 'l' (left), 'r' (right) for GLON coordinates
   - 'b' (bottom), 't' (top) for GLAT coordinates
   - 'pml' (left), 'pmr' (right) for pmra
   - 'pmb' (bottom), 'pmt' (top), for pmde
   - 'plxl' (left), 'plxr' (right) for parallax

and the numbers are the limiting values for each. A single limit can be provided (eg,
"x_111.1") or several separated by a ';' (eg, "x_111.1;y_222.2;...").

- N_clust: Fixed number of cluster members (has precedence over N_clust_max)
- N_clust_max: Maximum number of cluster members
- use_mag_fastmp: (y/n) whether fastMP uses the G magnitudes
```

Manual values for new entries can be given in the (optional)
`data/new_ocs_manual_pars.csv` file.

The `UCC_cat_C_out.csv` file contains the columns generated using information taken
from the estimated members for each entry. This script performs the following main
operations:

### 3.1. Change detection
Compares two versions of the UCC catalog: a source catalog (`UCC_cat_B.csv`) and the
current catalog (`UCC_cat_C.csv`). It identifies four sets of open clusters:

- New clusters to be added (B entry not in C --> Add to C)
- Old clusters to be removed (C entry not in B and merged into another entry -->
  Remove from C)
- Renamed clusters (C entry not in B whose canonical name changed --> Rename in C)
- Existing clusters that have been manually marked for reprocessing
 (C entry with 'process=y' --> Reprocess in C)

### 3.2 Cluster processing
For each new or re-processed cluster, the script performs a detailed analysis:

- Uses the `fastMP` method to identify likely member stars
- Saves the list of member stars for each cluster into an individual .parquet file
- Calculates various parameters for the cluster based on its members

### 3.3 Data file updating
After processing all the necessary clusters, the script updates the main UCC data:

- Combines all the individual member .parquet files into a single, updated master
  members file
- Updates the `UCC_cat_C.csv` file with the newly derived data, adding new clusters
  and removing old ones

### 3.4 Post-processing
- Finds Shared Members: analyzes the entire member list to find stars that belong to
  more than one cluster and records this information (only if the members changed)
- Calculates the quality metrics (`C_N`, `C_dens`, `C_C3`, `C_lit`, `C_dup`), the
  duplicate probability `P_dup`, the UTI, and the `bad_oc` flag
- Renames/removes the `.md` and `.webp` files of renamed/removed entries in the
  `ucc/` and `plots/` repositories


### Input

- `data/UCC_cat_B_out.csv`: Current UCC B catalogue (produced by the B script)
- `data/all_names.csv`: File with the names of each entry in the UCC
- `data/UCC_cat_C_in.csv`, `data/UCC_cat_C_out.csv`: Current version of these files
- `data/new_ocs_manual_pars.csv`: (optional) Manual parameters for new entries
- `data/globulars.csv`: Globular clusters data file
- `data/databases_info.json`: Current UCC database JSON file
- `Gaia data files`: Gaia data files for a given release (`path_gaia_frames` in
  `variables.py`)
- `data/zenodo/README.txt`: Zenodo README file
- `data/zenodo/UCC_cat.csv`: Current Zenodo catalogue
- `data/zenodo/UCC_members.parquet`: File with estimated member for all the clusters
  (**FILE NOT TRACKED BECAUSE IT IS TOO LARGE**)

### Output

Files are generated in `temp_updt/` and, after confirmation, moved to their final
destination:

- `data/UCC_cat_C_in.csv`, `data/UCC_cat_C_out.csv`: Updated versions (previous
  versions are archived in `data/ucc_archived_nogit/`)
- `data/zenodo/README.txt`: Updated version
- `data/zenodo/UCC_cat.csv`: Updated (and simplified) version of the UCC catalog
- `data/zenodo/UCC_members.parquet`: Updated version (previous version is archived)
- `ucc/_clusters/*.md`, `plots/plots_*/*/*.webp`: Renamed/removed if required

The last three files are uploaded to Zenodo to generate a new release.





## 4. Updating the site

Updating the site requires running the `D_update_UCC_site.py` script.

This script applies the required changes to update the ucc.ar site. It processes the
UCC catalogue and searches for modifications that need to be applied to update the site.

- Generate/update per cluster `.webp` files (stored in the `plots/` folders)
- If plots were generated/updated, update the plots record file
  `data/UCC_cat_D_in.csv` (members hash and Aladin parameters used for each entry)
- Generate/update per cluster `.md` (stored in `ucc/_clusters/`) files
- Update the per-article tables (`ucc/_tables/dbs/`, `ucc/_tables/cmmts/`)
- Update the site-wide plots and the `DATABASE.md` and `ARTICLES.md` pages
- Update the equatorial positions file `ucc/assets/eq_positions.json.gz`
- Update the split members files (`ucc/assets/members/*.csv.gz`)
- Update the CSV clusters file `ucc/assets/clusters_*.csv.gz` (timestamped) and its
  associated JSON file `clusters_manifest.json`
- Move all files to their final destination
- Check that the number of files is correct

### Input

- `data/all_names.csv`: Used to access the Names columns
- `data/UCC_cat_B_out.csv`: Used to access certain columns like DB and parameters
- `data/UCC_cat_C_out.csv`: Primary source of data for most of the website content
- `data/UCC_cat_D_in.csv`: Record of the members/Aladin parameters used for the plots
- `data/zenodo/UCC_members.parquet`: Used for generating cluster plots
- `data/databases_info.json`: Latest UCC database JSON file
- `data/databases/*.csv`: All the DBs in the UCC
- `data/databases/cmmts/*.csv`: CSV files with comments

- `ucc/assets/clusters_*.csv.gz`: CSV file with the latest UCC data
- `ucc/assets/eq_positions.json.gz`: Equatorial positions of the clusters
- `ucc/_pages/DATABASE.md`, `ucc/_pages/ARTICLES.md`
- `ucc/_tables/dbs/{db_name}_table.md`, `ucc/_tables/cmmts/{db_name}_table.md`
  (individual table pages)
- `ucc/_clusters/{cluster_name}.md` (individual cluster pages)


### Output

Files are generated in `temp_updt/` and, after confirmation, moved to their final
destination:

- `plots/plots_X/{UCC,aladin,gcpos}/*.webp`: Updated or generated
- `data/UCC_cat_D_in.csv`: Updated if plots were generated

- `ucc/_clusters/*.md`: Updated or generated
- `ucc/assets/members/*.csv.gz`: Updated files
- `ucc/assets/clusters_*.csv.gz`: Updated (the old file is deleted)
- `ucc/assets/clusters_manifest.json`: Updated
- `ucc/assets/eq_positions.json.gz`: Updated
- `ucc/images/*.webp`: Updated
- `ucc/_pages/DATABASE.md`, `ucc/_pages/ARTICLES.md`: Updated
- `ucc/_tables/{dbs,cmmts}/*_table.md`: Updated






## 5. Checking the UCC

The `E_master_check.py` script runs a set of consistency checks on the UCC catalogues.
It is **optional** and does not modify any UCC file; it is meant to flag entries that
need to be manually inspected.

The script runs all the checks below with no user input, using the default
parameters stored in the `E` section of `variables.py` (`CHECK_CONFIG`, `MEMBS_COORDS_CONFIG`,
`PARAMS_CONFIG`, `B_VS_C_CONFIG`, `DUP_CONFIG`, `PROX_DUP_CONFIG`,
`H23_MEMBS_CONFIG`, `LIT_DIST_CONFIG`):

1. **Cross-DB consistency**: compares the position (angular separation), proper
   motion, and parallax values of each entry across the DBs listed in the B file. The
   `conflicts` mode reports the DBs most likely responsible for the discrepancies; the
   `groups` mode reports entries whose DB values split into two or more distinct
   groups. The parallax threshold grows with the parallax of the entry (5% of the
   median of its DB values, if larger than the absolute threshold), since the
   differences between DBs are larger for nearby clusters. Positions are only
   checked for entries whose B center is offset from the members' median center by
   at least `dist_norm` (the offset over the members' r50 radius).
2. **B center vs members**: compares the center coordinates in the B file with the
   median coordinates of the estimated members (absolute and `dist_norm`
   thresholds).
3. **Cross-DB parameters**: compares the fundamental parameters (dist, av, age, met,
   mass) of each entry across the DBs listed in the B file, against their median.
   Distances are compared as distance moduli and ages as log10(age/yr); the threshold
   is relative to the median for `mass` and absolute for the rest. These are *minor*
   checks (disagreements in the literature rather than UCC errors): they are scored,
   but an entry failing only these is not counted as flagged.
4. **B vs C**: compares the proper motions and parallaxes in the B file with those
   derived from the members in the C file (only for entries with UTI above
   `uti_min`).
5. **Duplicates**: uses the shared members in the C file to identify candidate
   duplicates (several runs, `dup_1` to `dup_5`, by number of candidates and
   publication date), and flags pairs of entries that share no members but are
   close in position, parallax, and proper motion (`dup_prox`).
6. **Members vs HUNT2023**: compares the number of members of each entry (rows in the
   members file) with the `N` column of the HUNT2023 DB. An entry is flagged if
   `max(N)/min(N) > 1 + thr * factor`, with `factor` decreasing with `max(N)`, so
   that differences in larger clusters are flagged and scored more strongly.
7. **Literature distance vs parallax**: compares the distance modulus of the median
   literature distance with that of the members' parallax (corrected for the Gaia
   zero point), for entries with a parallax above `plx_min`.

The integrity of the B and C catalogues and of the members file (repeated or
unmatched names, missing values, entries without members, negative parallaxes,
repeated stars) is also checked, with every problem issued as a warning.

Entries from the `hsc, theia, cwnu, ocsn` families are skipped in all checks except
the duplicates ones (`skip_pfx`; the proximity duplicates check only skips pairs
where both entries belong to these families).

Entire DBs can be excluded from the checks that compare per-DB values
(`DBs_<type>_conflicts`, `DBs_<type>_groups`, `params_<param>`) with the
`exclude_DBs` dictionary (`E` section of `variables.py`), eg:
`exclude_DBs = {"DBs_pos_conflicts": ("VDBH1975",)}`. The `params` key excludes its
DBs from all the `params_<param>` checks and from `lit_dist_plx`. The values of an
excluded DB are ignored only in the checks listed for it, and the exclusions are
listed in the output file. Conversely, the `include_DBs` dictionary (same keys)
restricts a check to the values of the listed DBs only; a check with an entry in
`include_DBs` ignores its entries in `exclude_DBs`. The script stops if a check does
not accept exclusions or a DB is not in `databases_info.json`.

Every failed check gives the entry a score of
`weight * min(1 + log2(ratio), sev_max) * (uti_floor + (1 - uti_floor) * UTI)`,
where `ratio` is the measured value over the check's threshold, so larger
discrepancies score higher, and entries with larger UTI values score higher (entries
without a UTI value use a factor of 1). The duplicates check has the largest weight
(the original entry of a duplicate pair is scored with `dup_orig_factor`).
An entry's total score is the sum over its failed checks; since the `conflicts` and
`groups` checks of the same quantity measure the same discrepancy, only the highest
of the two counts (the same applies to all the `dup_*` checks). The weights and other
factors are set in `SCORE_CONFIG`.

The output file lists, right below its `## Clusters` section, all the candidate
duplicates (entries that failed the duplicates check, as either the duplicate or the
original of a pair) in their own sub-section, followed by the `N_max_clusters` other
entries with the highest scores.

### Manual comments

The `UCC_cat_E_in.json` file stores the manual comments for entries that were
already inspected, as `{fname: {key: comment}}`:

```json
{
    "graham1": {
        "pos": "BUKOWIECKI2011 bad DEC"
    },
    "ic2944": {
        "pos": "VDBH1975 bad coords",
        "N_membs": "HUNT2023 recovers only the cluster core"
    }
}
```

An entry can have several keys. The checks matching each key are down-weighted (by
the `cmmt_factor` in `SCORE_CONFIG`), and all the comments are listed with the entry
in the output file.

| Key | Checks down-weighted |
|---|---|
| `pos` | `DBs_pos_conflicts`, `DBs_pos_groups`, `B_vs_C_coords` |
| `pm` | `DBs_pm_conflicts`, `DBs_pm_groups`, `B_vs_C_pm` |
| `plx` | `DBs_plx_conflicts`, `DBs_plx_groups`, `B_vs_C_plx` |
| `params` | `params_<param>` for every parameter (`dist`, `av`, `age`, `met`, `mass`), `lit_dist_plx` |
| `dup` | `dup_1` to `dup_5`, `dup_prox` (as either the duplicate or the original of a pair) |
| `N_membs` | `H23_membs` |

The file is validated before the checks run, and the script stops with a list of the
problems if it is not valid JSON (eg: a trailing comma), if an entry or key is
repeated, if a key is not in the table above, or if a comment is empty. Entries that
are not in the B file only produce a warning, since their comments never apply.
Comments whose checks no longer fail (orphan comments) are also reported with a
warning, since they can be removed.

#### Stale comments

A comment may no longer apply once its entry changes. The
`data/UCC_cat_E_hashes.csv` file (generated and updated by the script) stores, for
every comment (one row per entry and key), a hash of the comment and of the inputs
of the checks of its key when it was stored: the B columns, the values read from
the DB files, the C columns, and the members used by those checks
(`CMMTS_HASH_DEPS`):

- New or edited comments take the current hashes of their entry.
- If a comment did not change but the inputs of the checks of its key did, the
  comment is **stale**: the script prints a warning listing what changed, the
  comment no longer down-weights the entry's checks, and it is marked as `STALE`
  in the output file. Changes in parts of the entry that the checks of a key do not
  use never make its comments stale.
- A stale comment keeps being flagged until it is reviewed, and either edited, or
  confirmed by removing its entry's row from `UCC_cat_E_hashes.csv`.

### Input

- `data/UCC_cat_B_out.csv`: Current UCC B catalogue
- `data/UCC_cat_C_out.csv`: Current UCC C catalogue
- `data/UCC_cat_E_in.json`: Manual comments for already inspected entries
- `data/UCC_cat_E_hashes.csv`: Hashes of the commented entries when their comments
  were stored
- `data/databases_info.json`: Current UCC database JSON file
- `data/databases/*.csv`: Original DBs
- `data/zenodo/UCC_members.parquet`: File with estimated members for all the clusters

### Output

- `temp_updt/master_check.md`: Warnings issued during the run (also printed to
  screen), summary of the checks performed, and the highest scoring entries with
  their scores, failures, and manual comments
- `data/UCC_cat_E_hashes.csv`: Updated with the hashes of new or edited comments






## 6. Building the site

The Jekyll theme used by the site is a modified [Reverie](https://jekyllthemes.io/theme/reverie) theme.

### 6.1 Local build

Before updating the live site, generate a local site build and check the results
**carefully**. To build a local copy of the site we use Jekyll, see [Jekyll docs](https://jekyllrb.com/docs/).

If this is a new installation, update the gems with:

```
$ bundle update --all
```

To build a local version of the site, position a terminal in the `/ucc` folder
(**not** the `/updt_ucc` folder) and run:

```
$ bundle exec jekyll serve --incremental
```

This will generate a full version of the site locally which can take a while. For a
faster build, avoid processing the files in the `_clusters, _tables` folder (for
example, using a different `include` with fewer/different selected folders)

The script `test_build.sh` can also be used to check that the local build. It will
select `N` random clusters from the `_cluster/` folder and generate the site:

```
$ ./test_build.sh N
```

To generate `N` clusters while excluding some clusters, you can pass the prefixes
to exclude e.g. 'cwnu', 'cwwdl', 'ckcwdm', 'hsc' or 'theia':

```
$ ./test_build.sh N -cwnu -cwwdl -ckcwdm -hsc -theia
```

To select a specific cluster (eg, melotte55) instead of random clusters, run:

```
$ ./test_build.sh 0 melotte55
```

where `0` means "do not add random clusters", you can also select several clusters:

```
$ ./test_build.sh 0 melotte55 ngc2682
```

**Check the local version in both Chrome and Firefox**.


### 6.2 Live build

1. Create a 'New version' in the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.8250523) 

  -a Make sure that the version number in the `zenodo/README.txt` file matches that in
     the `_pages/CHANGELOG.md` file
  -b Upload the three files stored in the `data/zenodo/` folder:
     `README.txt, UCC_cat.csv, UCC_members.parquet`
  -c Get a new DOI from Zenodo
  -d Add a _Publication date_ with the format: YYYY-MM-DD
  -e Use the same version number from the README (format: YYMMDD) in the release

Publish this new release and copy **its own url** (not the general repository url)

2. Update the `_pages/CHANGELOG.md` file, use the Zenodo URL for **this** release

3. Push changes (if any) to each of the `plots/plots_*` repositories. To do this, run:

```
$ for dir in ../plots/plots_*/; do (cd "$dir" && [ -d .git ] && git acp "updt plots"); done
```

4. **IMPORTANT**: Make sure that the `_config.yml` file includes all the folders

5. Push any remaining changes to the `ucc` repository

6. Deploy the site using the Github workflow [Deploy Jekyll site to Pages](https://github.com/ucc23/ucc/actions/workflows/jekyll.yml)

7. Test live site both in Chrome and Firefox



Libraries and services used:

- pako (https://github.com/nodeca/pako): loadCSV.js
- d3 (https://github.com/d3/d3): map_search.js
- d3-geo-projection (https://github.com/d3/d3-geo-projection): map_search.js
- Plotly (https://github.com/plotly/plotly.py): posts.html
- flatgithub.com: ARTICLES.md