import csv
import datetime
import os

import pandas as pd
from HARDCODED import UCC_folder, all_DBs_json, dbs_folder
from modules import (
    DBs_combine,
    UCC_new_match,
    combine_UCC_new_DB,
    duplicate_probs,
    logger,
    read_ini_file,
)

# Print entries to screen
show_entries = True


def main():
    """ """
    logging = logger.main()
    logging.info("\nRunning 'add_new_DB' script\n")

    date = datetime.datetime.now().strftime("%Y%m%d")[2:]
    new_ucc_path = UCC_folder + "UCC_cat_" + date + ".csv"
    # Check if file exists
    if os.path.exists(new_ucc_path):
        raise FileExistsError(f"File {new_ucc_path} already exists")

    pars_dict = read_ini_file.main()
    new_DB = pars_dict["new_DB"]
    logging.info(f"Adding new DB: {new_DB}")

    # Load the current UCC, the new DB, and its JSON values
    df_UCC, df_new, json_pars = UCC_new_match.load_data(
        logging, dbs_folder, all_DBs_json, UCC_folder
    )

    # Standardize and match the new DB with the UCC
    new_DB_fnames, db_matches = UCC_new_match.standardize_and_match(
        logging, df_UCC, df_new, json_pars, pars_dict, new_DB, show_entries
    )

    logging.info("")
    new_db_dict = combine_UCC_new_DB.main(
        logging, new_DB, df_UCC, df_new, json_pars, new_DB_fnames, db_matches
    )
    N_new = len(df_new) - sum(_ is not None for _ in db_matches)
    logging.info(f"\nN={N_new} new clusters in {new_DB}")
    logging.info("")

    # Add UCC_IDs and quadrants for new clusters
    ucc_ids_old = list(df_UCC["UCC_ID"].values)
    for i, UCC_ID in enumerate(new_db_dict["UCC_ID"]):
        # Only process new OCs
        if UCC_ID != "nan":
            continue
        # logging.info(f"New UCC_ID and quad for: {new_db_dict['fnames'][i]}")
        new_db_dict["UCC_ID"][i] = DBs_combine.assign_UCC_ids(
            new_db_dict["GLON"][i], new_db_dict["GLAT"][i], ucc_ids_old
        )
        new_db_dict["quad"][i] = DBs_combine.QXY_fold(new_db_dict["UCC_ID"][i])
        ucc_ids_old += [new_db_dict["UCC_ID"][i]]

    # Drop OCs from the UCC that are present in the new DB
    # Remove 'None' entries first from the indexes list
    idx_rm_comb_db = [_ for _ in db_matches if _ is not None]
    df_UCC_no_new = df_UCC.drop(list(df_UCC.index[idx_rm_comb_db]))
    df_UCC_no_new.reset_index(drop=True, inplace=True)
    df_all = pd.concat([df_UCC_no_new, pd.DataFrame(new_db_dict)], ignore_index=True)

    # Assign a 'duplicate probability' for each cluster in the UCC, based on the
    # literature data
    logging.info("\nFinding possible duplicates...")
    df_all["dups_fnames"], df_all["dups_probs"] = duplicate_probs.main(
        df_all["fnames"],
        df_all["GLON"],
        df_all["GLAT"],
        df_all["Plx"],
        df_all["pmRA"],
        df_all["pmDE"],
        prob_cut=0.5,
    )

    # Order by (lon, lat) first
    df_all = df_all.sort_values(["GLON", "GLAT"])
    df_all = df_all.reset_index(drop=True)

    # Final duplicate check
    dup_flag = duplicates_check(logging, df_all)
    if dup_flag:
        return

    # Save new version of the UCC catalogue to file before processing with
    # fastMP
    df_all.to_csv(
        new_ucc_path,
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    logging.info(f"UCC updated to version {date} (N={len(df_all)})")

    logging.info("\nThe UCC catalogue was updated\n")


def duplicates_check(logging, df_all):
    """ """

    def list_duplicates(seq):
        seen = set()
        seen_add = seen.add
        # adds all elements it doesn't know yet to seen and all other to seen_twice
        seen_twice = set(x for x in seq if x in seen or seen_add(x))
        # turn the set into a list (as requested)
        return list(seen_twice)

    def dup_check(df_all, col):
        dups = list_duplicates(list(df_all[col]))
        if len(dups) > 0:
            logging.info(f"\nWARNING! N={len(dups)} duplicates found in '{col}':")
            for dup in dups:
                print(dup)
            logging.info("UCC was not updated")
            return True
        else:
            return False

    for col in ("ID", "UCC_ID", "fnames"):
        dup_flag = dup_check(df_all, col)
        if dup_flag:
            return True

    return False


if __name__ == "__main__":
    main()
