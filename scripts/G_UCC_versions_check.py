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
            # print(f"OC not found in old UCC: {fnames_new}")
            continue

        # If 'fnames_new' was found in the old UCC
        # Compare both entries
        check_rows(UCC_new, UCC_old, i_new, i_old)

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
        df_dict = df.to_dict()
        diff_cols = list(df_dict["self"].keys())

        # If the only diffs are in these columns, don't show anything
        if diff_cols == ["DB", "DB_i"] or diff_cols == ["DB"] or diff_cols == ["DB_i"]:
            return

        for col in diff_cols:
            if col not in (
                "DB",
                "DB_i",
                "RA_ICRS",
                "DE_ICRS",
                "GLON",
                "GLAT",
                "dups_fnames",
                "dups_probs",
                "dups_fnames_m",
                "dups_probs_m",
            ):
                print(
                    i_old,
                    "col diff",
                    UCC_old["fnames"][i_old],
                    UCC_new["fnames"][i_new],
                    " --> ",
                    diff_cols,
                )
                return

        txt1 = ""
        if "RA_ICRS" in diff_cols:
            if abs(row_old["RA_ICRS"] - row_new["RA_ICRS"]) > 0.001:
                txt1 += "; RA_ICRS: " + str(
                    abs(row_old["RA_ICRS"] - row_new["RA_ICRS"])
                )
        if "DE_ICRS" in diff_cols:
            if abs(row_old["DE_ICRS"] - row_new["DE_ICRS"]) > 0.001:
                txt1 += "; DE_ICRS: " + str(
                    abs(row_old["DE_ICRS"] - row_new["DE_ICRS"])
                )
        if "GLON" in diff_cols:
            if abs(row_old["GLON"] - row_new["GLON"]) > 0.001:
                txt1 += "; GLON: " + str(abs(row_old["GLON"] - row_new["GLON"]))
        if "GLAT" in diff_cols:
            if abs(row_old["GLAT"] - row_new["GLAT"]) > 0.001:
                txt1 += "; GLAT: " + str(abs(row_old["GLAT"] - row_new["GLAT"]))

        if "dups_fnames" in diff_cols:
            aa = str(row_old["dups_fnames"]).split(";")
            bb = str(row_new["dups_fnames"]).split(";")
            if len(list(set(aa) - set(bb))) > 0:
                txt1 += (
                    "; dups_fnames: "
                    + str(row_old["dups_fnames"])
                    + " | "
                    + str(row_new["dups_fnames"])
                )
        if "dups_probs" in diff_cols:
            aa = str(row_old["dups_probs"]).split(";")
            bb = str(row_new["dups_probs"]).split(";")
            if len(list(set(aa) - set(bb))) > 0:
                txt1 += (
                    "; dups_probs: "
                    + str(row_old["dups_probs"])
                    + " | "
                    + str(row_new["dups_probs"])
                )

        if "dups_fnames_m" in diff_cols:
            aa = str(row_old["dups_fnames_m"]).split(";")
            bb = str(row_new["dups_fnames_m"]).split(";")
            if len(list(set(aa) - set(bb))) > 0:
                txt1 += (
                    "; dups_fnames_m: "
                    + str(row_old["dups_fnames_m"])
                    + " | "
                    + str(row_new["dups_fnames_m"])
                )
        if "dups_probs_m" in diff_cols:
            aa = str(row_old["dups_probs_m"]).split(";")
            bb = str(row_new["dups_probs_m"]).split(";")
            if len(list(set(aa) - set(bb))) > 0:
                txt1 += (
                    "; dups_probs_m: "
                    + str(row_old["dups_probs_m"])
                    + " | "
                    + str(row_new["dups_probs_m"])
                )

        if txt1 != "":
            print(
                i_old, UCC_old["fnames"][i_old], UCC_new["fnames"][i_new], " --> ", txt1
            )
        return

        # fname_old = UCC_old["fnames"][i_old]
        # fname_new = UCC_new["fnames"][i_new]
        # print(
        #     f"Diff old vs new {fname_old}: {UCC_old['dups_probs_m'][i_old]} | {UCC_new['dups_probs_m'][i_new]}"
        # )
        # print(df, '\n')

        # print_f = False

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
