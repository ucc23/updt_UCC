
# UCC management

The UCC lives in several repositories within the [UCC23](https://github.com/ucc23) organization
and in its associated [Zenodo repository](https://doi.org/10.5281/zenodo.8250523).

- The main UCC database along, with a single file with all the identified members, is
  stored in the Zenodo repository

- For each entry in the database there is a corresponding entry in the [ucc](https://github.com/ucc23/ucc)
  repository in the form of an `.md` file. This repository also contains most of the
  files required to build the [public site](https://ucc.ar)

- The eight `QXX` repositories ([Q1P](https://github.com/ucc23/Q1P), etc.) contain the `.parquet` files with the
  identified members for each OC (in the `datafiles` folders), as well as two plots per
  OC: one with the four diagrams and one for the Aladin Lite plot (in the `plots`
  folders). The plots are loaded by the public site from this repository.

- Finally the [updt_ucc](https://github.com/ucc23/updt_UCC) repository, this one, contains the scripts and data files
  required to update the UCC (mostly) automatically


**What do you want to do?**

- Add a new DB? -> [add_new_DB.md](add_new_DB.md)
- Modify one or more existing entries? -> [modify_entries.md](modify_entries.md)
- Add one or more entries? -> [add_entries.md](add_entries.md)
- Remove one or more entries? -> 


# UCC public site build

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
