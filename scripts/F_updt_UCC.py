import csv

import pandas as pd
from HARDCODED import UCC_folder, new_OCs_fpath
from modules import DBs_combine, UCC_new_match, duplicate_probs, logger

logging = logger.main()


def main():
    """ """
    # Read latest version of the UCC
    df_UCC, UCC_cat = UCC_new_match.latest_cat_detect(logging, UCC_folder)
    UCC_old = df_UCC.copy()

    # Read file with parameters for the new OCs obtained from their members
    params_updt = pd.read_csv(new_OCs_fpath)

    df_UCC = updt_UCC(df_UCC, params_updt)

    df_UCC = members_duplicates(df_UCC)

    df_UCC.to_csv(UCC_cat, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC)

    DBs_combine.diff_between_dfs(logging, UCC_old, df_UCC, cols_exclude=None)
    logging.info("Files 'UCC_diff_xxx.csv' saved\n")


def updt_UCC(df_UCC, params_updt) -> pd.DataFrame:
    """
    Update these values for all the processed clusters
    """

    index_updt = list(params_updt["index_updt"])
    if len(index_updt) == 0:
        logging.info("\nNo update required")
        return df_UCC

    for i, idx in enumerate(index_updt):
        # Sanity check
        fname = params_updt["fname"][i]
        UCC_fnames = df_UCC["fnames"][idx]
        if fname not in UCC_fnames:
            raise ValueError("ERROR in 'updt_UCC': fnames do not match")

        df_UCC.at[idx, "N_fixed"] = params_updt["N_fixed"][i]
        # df_UCC.at[idx, "N_membs"] = int(params_updt["N_survived"][i])
        df_UCC.at[idx, "fixed_cent"] = params_updt["fixed_centers"][i]
        df_UCC.at[idx, "cent_flags"] = params_updt["cent_flags"][i]
        df_UCC.at[idx, "C1"] = params_updt["C1"][i]
        df_UCC.at[idx, "C2"] = params_updt["C2"][i]
        df_UCC.at[idx, "C3"] = params_updt["C3"][i]
        df_UCC.at[idx, "GLON_m"] = params_updt["GLON_m"][i]
        df_UCC.at[idx, "GLAT_m"] = params_updt["GLAT_m"][i]
        df_UCC.at[idx, "RA_ICRS_m"] = params_updt["RA_ICRS_m"][i]
        df_UCC.at[idx, "DE_ICRS_m"] = params_updt["DE_ICRS_m"][i]
        df_UCC.at[idx, "Plx_m"] = params_updt["Plx_m"][i]
        df_UCC.at[idx, "pmRA_m"] = params_updt["pmRA_m"][i]
        df_UCC.at[idx, "pmDE_m"] = params_updt["pmDE_m"][i]
        df_UCC.at[idx, "Rv_m"] = params_updt["Rv_m"][i]
        df_UCC.at[idx, "N_Rv"] = params_updt["N_Rv"][i]
        df_UCC.at[idx, "N_50"] = params_updt["N_50"][i]
        df_UCC.at[idx, "r_50"] = params_updt["r_50"][i]

    # Order by (lon, lat) first
    df_UCC = df_UCC.sort_values(["GLON", "GLAT"])
    df_UCC = df_UCC.reset_index(drop=True)

    logging.info("\nUCC updated")

    return df_UCC


def members_duplicates(df_UCC: pd.DataFrame, prob_cut: float = 0.25) -> pd.DataFrame:
    """
    Assign a 'duplicate probability' for each cluster in the UCC, based on its
    estimated members.

    Parameters:
        logging: Logger instance for logging messages.
        df_UCC: Dictionary containing cluster data, including keys 'fnames',
                'GLON_m', 'GLAT_m', 'plx_m', 'pmRA_m', and 'pmDE_m'.
        prob_cut: Float representing the probability cutoff for identifying duplicates.

    Returns:
        Updated df_UCC dictionary with added keys 'dups_fnames_m' and 'dups_probs_m',
        if duplicates are found.
    """
    logging.info("Finding final duplicates and their probabilities...")
    # Use members data
    dups_fnames_m, dups_probs_m = duplicate_probs.main(
        df_UCC["fnames"],
        df_UCC["GLON_m"],
        df_UCC["GLAT_m"],
        df_UCC["Plx_m"],
        df_UCC["pmRA_m"],
        df_UCC["pmDE_m"],
        prob_cut,
    )

    if dups_fnames_m:
        df_UCC["dups_fnames_m"] = dups_fnames_m
        df_UCC["dups_probs_m"] = dups_probs_m
        logging.info("Duplicates (using members data) added to UCC\n")
    else:
        logging.info("No duplicates added to UCC\n")

    return df_UCC


if __name__ == "__main__":
    main()
