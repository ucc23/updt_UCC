
# UCC scripts

This repository contains all the necessary scripts to update/edit the UCC
database and its associated web site.


Adds a new DB?
Yes --> Follow [add_new_DB.md](add_new_DB.md) instructions
No  --> Modify one or more existing entries?
        Yes  --> Follow `XXX.md` instructions
        No   --> Add one or more entries?
                 Yes --> Follow `YYY.md` instructions
                 No  --> Remove one or more entries?
                         Yes --> Follow `ZZZ.md` instructions
                         No  --> ???





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






