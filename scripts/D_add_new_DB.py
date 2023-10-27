import datetime
import csv
import pandas as pd
from modules import logger
from modules import read_ini_file
from modules import combine_UCC_new_DB
from modules import DBs_combine
from modules import UCC_new_match


def main():
    """ """
    logging = logger.main()
    logging.info("\nRunning 'add_new_DB' script\n")

    pars_dict = read_ini_file.main()
    new_DB_ID, sep, UCC_folder = (
        pars_dict["new_DB"],
        pars_dict["sep"],
        pars_dict["UCC_folder"],
    )
    logging.info(f"Adding new DB: {new_DB_ID}")

    df_UCC, df_new, json_pars, new_DB_fnames, db_matches = UCC_new_match.main(logging)

    logging.info("")
    new_db_dict = combine_UCC_new_DB.main(
        logging, new_DB_ID, df_UCC, df_new, json_pars, new_DB_fnames, db_matches, sep
    )
    N_new = len(df_new) - sum(_ is not None for _ in db_matches)
    logging.info(f"\nN={N_new} new clusters in {new_DB_ID}")
    logging.info("")

    # Add UCC_IDs and quadrants for new clusters
    ucc_ids_old = list(df_UCC["UCC_ID"].values)
    for i, UCC_ID in enumerate(new_db_dict["UCC_ID"]):
        # Only process new OCs
        if UCC_ID != "nan":
            continue
        logging.info(f"New UCC_ID and quad for: {new_db_dict['fnames'][i]}")
        new_db_dict["UCC_ID"][i] = DBs_combine.assign_UCC_ids(
            new_db_dict["GLON"][i], new_db_dict["GLAT"][i], ucc_ids_old
        )
        new_db_dict["quad"][i] = DBs_combine.QXY_fold(new_db_dict["UCC_ID"][i])
        ucc_ids_old += [new_db_dict["UCC_ID"][i]]

    # Drop OCs from the UCC that are present in the new DB
    # Remove 'None' entries first from the indexes list
    idx_rm_comb_db = [_ for _ in db_matches if _ is not None]
    df_UCC_no_new = df_UCC.drop(df_UCC.index[idx_rm_comb_db])
    df_UCC_no_new.reset_index(drop=True, inplace=True)
    df_all = pd.concat([df_UCC_no_new, pd.DataFrame(new_db_dict)], ignore_index=True)

    # Used to remove close clusters from the field so that fastMP won't get
    # confused
    logging.info("\nFinding possible duplicates...")
    df_all["dups_fnames"], df_all["dups_probs"] = DBs_combine.dups_identify(df_all)

    # Order by (lon, lat) first
    df_all = df_all.sort_values(["GLON", "GLAT"])
    df_all = df_all.reset_index(drop=True)

    # Save new version of the UCC catalogue to file before processing with
    # fastMP
    date = datetime.datetime.now().strftime("%Y%m%d")[2:]
    df_all.to_csv(
        UCC_folder + "UCC_cat_" + date + ".csv",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    logging.info(f"UCC updated to version {date} (N={len(df_all)})")

    logging.info("\nThe UCC catalogue was updated\n")


if __name__ == "__main__":
    main()
