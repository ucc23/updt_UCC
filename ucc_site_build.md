
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
  - _clusters # Un-comment to exclude for faster processing in a local build
```

**IMPORTANT:** comment the `_clusters` folder in the `exclude:` section of the
`_config.yml` file before moving on.


1. Push changes in the `QXY` repositories
2. Update the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.8250523) with the new files, creating a new release.
   Remember to set the version number in Zenodo as YYMMDD
3. Update the `CHANGELOG.md` file, use the Zenodo URL for **this** release

Every change pushed to the [ucc](https://github.com/ucc23/ucc) repository triggers an
automatic build + deployment. **Check carefully before pushing.**

4. Push the changes to `ucc` repository
