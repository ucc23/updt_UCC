import pandas as pd
from modules import logger
from modules import read_ini_file
from modules import UCC_new_match


def main():
    """ """
    logging = logger.main()

    pars_dict = read_ini_file.main()
    UCC_folder, old_UCC_name = pars_dict["UCC_folder"], pars_dict["old_UCC_name"]

    # Read latest version of the UCC
    UCC_new, _ = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    UCC_old = pd.read_csv(UCC_folder + old_UCC_name)

    print("")
    fnames_old_all = list(UCC_old["fnames"])
    fnames_new_all = list(UCC_new["fnames"])

    for i_new, fnames_new in enumerate(fnames_new_all):
        i_old = None
        try:
            i_old = fnames_old_all.index(fnames_new)
        except ValueError:
            pass

        # If 'fnames_new' was found in the old UCC
        if i_old is not None:
            # Compare both entries
            check_rows(UCC_new, UCC_old, i_new, i_old)
            continue

        # If 'fnames_new' was NOT found in the old UCC
        # print(f"New entry (not in old UCC): {fnames_new}")

        # idxs_old_match = []
        # for i_old, fnames_old in enumerate(fnames_old_all):
        #     for fname_new in fnames_new.split(';'):
        #         for fname_old in fnames_old.split(';'):
        #             if fname_new == fname_old:
        #                 idxs_old_match.append(i_old)
        # idxs_old_match = list(set(idxs_old_match))
        # if len(idxs_old_match) > 1:
        #     print(f"ERROR: duplicate fname, new:{i_new}, old:{idxs_old_match}")


def check_rows(UCC_new, UCC_old, i_new, i_old):
    """ """
    row_new = UCC_new.iloc[i_new]
    row_old = UCC_old.iloc[i_old]

    df = row_new.compare(row_old)
    # If rows are not equal
    if df.empty is False:
        # If the only diffs are in these columns, don't show anything
        df_dict = df.to_dict()
        diff_cols = list(df_dict["self"].keys())
        if diff_cols == ["DB", "DB_i"]:
            return

        fname_old = UCC_old["fnames"][i_old]
        fname_new = UCC_new["fnames"][i_new]
        print(
            f"Diff old vs new {fname_old}: {UCC_old['dups_probs_m'][i_old]}, {UCC_new['dups_probs_m'][i_new]}"
        )
        # print(df, '\n')

        # print_f = False

        # if 'ID' in diff_cols:
        #     print_f = True
        # if abs(row_old['RA_ICRS']-row_new['RA_ICRS']) > 0.001:
        #     print_f = True
        # if abs(row_old['DE_ICRS']-row_new['DE_ICRS']) > 0.001:
        #     print_f = True

        # if print_f:
        #     print(f"Diff old vs new: {fname_old}, {fname_new} --> {diff_cols}")
        #     if 'ID' in diff_cols:
        #         print(row_old['ID'], row_new['ID'])
        #     if 'RA_ICRS' in diff_cols:
        #         print(row_old['RA_ICRS'], row_new['RA_ICRS'])
        #     if 'DE_ICRS' in diff_cols:
        #         print(row_old['DE_ICRS'], row_new['DE_ICRS'])


if __name__ == "__main__":
    main()
