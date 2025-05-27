import pandas as pd

from B_update_UCC_DB import (
    diff_between_dfs,
    get_paths_check_paths,
    load_data,
    member_files_updt,
    save_final_UCC,
    update_UCC_membs_data,
)
from modules.utils import logger

"""
This code was used to address https://github.com/ucc23/ucc/issues/46 abd
update the members of 3531 entries.

1. Read the file with the entries to be re-processed
"""


def main():
    """ """
    logging = logger()

    df_entries = pd.read_csv("single_use_scripts/3500_entries.csv")
    # df_entries = df_entries[:10]

    # Generate paths and check for required folders and files
    (
        root_UCC_path,
        temp_database_folder,
        ucc_file,
        temp_zenodo_fold,
        new_ucc_file,
        temp_JSON_file,
        archived_UCC_file,
    ) = get_paths_check_paths(logging)

    (
        gaia_frames_data,
        df_GCs,
        manual_pars,
        df_UCC_old,
        current_JSON,
        df_new,
        newDB_json,
        new_DB_file,
        new_DB,
    ) = load_data(logging, ucc_file, temp_JSON_file, temp_database_folder)

    # Load clusters that requires re-processing, and mark them with C3=nan
    # Extract the first part of `fnames` by splitting at ';'
    df_UCC_old["first_fnames"] = df_UCC_old["fnames"].str.split(";").str[0]
    # Find matches between df1['name'] and df2['first_fnames']
    matches = df_UCC_old["first_fnames"].isin(df_entries["name"])
    # Update `C3` for matching rows
    df_UCC_old.loc[matches, "C3"] = "nan"
    # Drop the temporary column if not needed further
    df_UCC_old.drop(columns=["first_fnames"], inplace=True)
    #
    df_UCC_new2 = df_UCC_old

    # 5. Entries with no C3 value are identified as new and processed with fastMP
    N_new = (df_UCC_new2["C3"] == "nan").sum()
    logging.info(f"\nProcessing {N_new} new OCs in {new_DB} with fastMP...")

    # Generate member files for new OCs and obtain their data
    df_UCC_updt = member_files_updt(
        logging, df_UCC_new2, gaia_frames_data, df_GCs, manual_pars
    )

    # Update the UCC with the new OCs member's data
    df_UCC_new3 = update_UCC_membs_data(logging, df_UCC_new2, df_UCC_updt)
    df_UCC_new4 = diff_between_dfs(logging, df_UCC_new2, df_UCC_new3)

    # 6. Save updated UCC to CSV file
    save_final_UCC(logging, temp_zenodo_fold, new_ucc_file, df_UCC_new4)


if __name__ == "__main__":
    main()
