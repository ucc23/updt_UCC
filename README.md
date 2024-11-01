
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

- [UCC public site build](#ucc-public-site-build)
- [UCC local site build](#ucc-local-site-build)

<!-- /MarkdownTOC -->




Add a new DB? -> [add_new_DB.md](add_new_DB.md)

Modify one or more existing entries? -> [modify_entry.md](modify_entry.md)

Add one or more entries? -> 

Remove one or more entries?








## UCC public site build

Before committing to this repository, generate a full [UCC local site build](#ucc-local-site-build)
and check the resulting site **carefully**.

Every change made to this repository and pushed to Github will trigger and automatic
build + deployment. **Commit carefully.**

1. Push changes in `QXY` repositories
2. Update the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.8250523) with the new files, creating a new release.
   Remember to set the version number in Zenodo as YYMMDD
3. Update the `CHANGELOG.md` file, use the Zenodo URL for **this** release
4. **IMPORTANT:** comment the `_clusters` folder in the `exclude:` section
   of the `_config.yml` file, if left un-commented so far.
5. Push changes to `/ucc` repository




## UCC local site build

To build a local copy of the site we use Jekyll, see [Jekyll docs](https://jekyllrb.com/docs/).

Position a terminal in the `/ucc` folder (**not** the `/updt_ucc` folder) and run:

```
$ bundle exec jekyll serve --incremental
```

This will generate a full version of the site locally which can take a while. For a
faster build avoid processing the files in the `_clusters` folder. To do this open
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

