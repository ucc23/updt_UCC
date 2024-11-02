
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


**What do you want to do?**

- Add a new DB? -> [add_new_DB.md](add_new_DB.md)
- Modify one or more existing entries? -> [modify_entries.md](modify_entries.md)
- Add one or more entries? -> [add_entries.md](add_entries.md)
- Remove one or more entries? -> 
