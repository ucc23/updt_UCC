import csv
import json

import numpy as np
import pandas as pd
from HARDCODED import UCC_folder, all_DBs_json, dbs_folder, new_OCs_fpath
from modules import DBs_combine, UCC_new_match


def main():
    """
    Script used to re-match selected OCs using all the available databases.
    """
    # OCs_names: Names of the OCs that should be kept after the re-processing. It does
    # not matter if names are repeated
    OCs_names, sep = "XXXXX", ","

    OCs_to_keep = []
    # To 'fnames'
    for name in OCs_names.split(","):
        OCs_to_keep.append(DBs_combine.rm_chars_from_name(name))

    # Read latest version of the UCC
    df_UCC, _ = UCC_new_match.latest_cat_detect(None, UCC_folder)
    # Use the list of UCC IDs in the latest DB version
    ucc_ids_old = list(df_UCC["UCC_ID"])

    with open(dbs_folder + all_DBs_json) as f:
        dbs_used = json.load(f)

    print("Read DBs and extract names...")
    DB_data, DB_fnames, DB_names_orig = get_data_and_names(dbs_folder, dbs_used, sep)

    print("Extracting unique names...")
    unique_fnames, unique_names_orig = get_unique_names(DB_fnames, DB_names_orig)

    print("Matching databases...")
    cl_dict = get_matches(
        dbs_used, DB_data, DB_fnames, unique_fnames, unique_names_orig
    )

    print("Generating combined catalogue...")
    comb_dbs = combine_DBs(cl_dict)

    print("Assign fnames...")
    comb_dbs["ID"], comb_dbs["fnames"] = assign_fname(comb_dbs["ID"])
    dup_check(comb_dbs["fnames"])

    print("Finding duplicates...")
    dups_fnames, dups_probs = DBs_combine.dups_identify(comb_dbs)  ## TODO
    # This was removed, call duplicate_probs.main() with prob_cut=0.5 directly
    comb_dbs["dups_fnames"] = dups_fnames
    comb_dbs["dups_probs"] = dups_probs

    print("Removing all not re-processed OCs...")
    comb_dbs = keep_selected_OCs(OCs_to_keep, comb_dbs)

    print("Assign UCC IDs...")
    ucc_ids, quads = [], []
    for i, glon in enumerate(comb_dbs["GLON"]):
        glat = comb_dbs["GLAT"][i]
        ucc_id = DBs_combine.assign_UCC_ids(glon, glat, ucc_ids_old)
        ucc_ids_old.append(ucc_id)
        ucc_ids.append(ucc_id)
        quads.append(DBs_combine.QXY_fold(ucc_id))
    comb_dbs["UCC_ID"] = ucc_ids
    comb_dbs["quad"] = quads

    # Add empty columns used later by the 'fastMP_process' scripts
    N_tot = len(comb_dbs["ID"])
    comb_dbs["r_50"] = [np.nan for _ in range(N_tot)]
    comb_dbs["N_50"] = [np.nan for _ in range(N_tot)]
    comb_dbs["N_fixed"] = [np.nan for _ in range(N_tot)]
    # comb_dbs["N_membs"] = [np.nan for _ in range(N_tot)]
    comb_dbs["fixed_cent"] = [np.nan for _ in range(N_tot)]
    comb_dbs["cent_flags"] = [np.nan for _ in range(N_tot)]
    comb_dbs["C1"] = [np.nan for _ in range(N_tot)]
    comb_dbs["C2"] = [np.nan for _ in range(N_tot)]
    comb_dbs["C3"] = [np.nan for _ in range(N_tot)]
    comb_dbs["GLON_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["GLAT_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["RA_ICRS_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["DE_ICRS_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["plx_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["pmRA_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["pmDE_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["Rv_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["N_Rv"] = [np.nan for _ in range(N_tot)]
    comb_dbs["dups_fnames_m"] = [np.nan for _ in range(N_tot)]
    comb_dbs["dups_probs_m"] = [np.nan for _ in range(N_tot)]

    # Proper columns order
    cols_order = [
        "DB",
        "DB_i",
        "ID",
        "RA_ICRS",
        "DE_ICRS",
        "GLON",
        "GLAT",
        "plx",
        "pmRA",
        "pmDE",
        "fnames",
        "UCC_ID",
        "quad",
        "dups_fnames",
        "dups_probs",
        "r_50",
        "N_fixed",
        # "N_membs",
        "fixed_cent",
        "cent_flags",
        "C1",
        "C2",
        "C3",
        "N_50",
        "GLON_m",
        "GLAT_m",
        "RA_ICRS_m",
        "DE_ICRS_m",
        "plx_m",
        "pmRA_m",
        "pmDE_m",
        "Rv_m",
        "N_Rv",
        "dups_fnames_m",
        "dups_probs_m",
    ]
    comb_dbs = comb_dbs[cols_order]

    # Save to file
    pd.DataFrame(comb_dbs).to_csv(
        UCC_folder + "UCC_corrected" + ".csv",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    print("\n'UCC_corrected' database written to file")

    # Generate the 'new_OCs_info' file indicating that all these OCs should
    # have their members re-estimated and their md entries and plots
    # re-generated
    new_OCs_info = {"fnames": [], "process_f": []}
    for i, fnames in enumerate(comb_dbs["fnames"]):
        new_OCs_info["fnames"].append(fnames)
        new_OCs_info["process_f"].append(True)
    # Store info on the OCs for later processing
    df_info = pd.DataFrame(new_OCs_info)
    df_info.to_csv(new_OCs_fpath, index=False, quoting=csv.QUOTE_NONNUMERIC)


def get_data_and_names(dbs_folder, dbs_used, sep):
    """
    1. For each DB extract and store all its data --> DB_data
    2. For each cluster in each DB extract and standardize its name(s) --> DB_fnames
    3. For each cluster in each DB extract its original unedited name(s) -->
       DB_names_orig
    """
    DB_data, DB_fnames, DB_names_orig, N_all = {}, {}, {}, 0
    for DB, _ in dbs_used.items():
        # Load DB data
        data = pd.read_csv(dbs_folder + DB + ".csv", index_col=False)
        print(DB, len(data))
        N_all += len(data)
        # Store in dictionary
        DB_data[DB] = data

        # Extract and standardize all names
        names_all = data[dbs_used[DB]["names"]]
        names_final, names_orig = [], []
        for i, names in enumerate(names_all):
            names_l, names_l_orig = [], []
            names_s = names.split(sep)
            for name in names_s:
                name = name.strip()
                name = DBs_combine.rename_standard(name)
                names_l.append(DBs_combine.rm_chars_from_name(name))
                names_l_orig.append(name)

            names_final.append(names_l)
            # Store original names
            names_orig.append(names_l_orig)

        DB_fnames[DB] = names_final
        DB_names_orig[DB] = names_orig

    print(f"\nN={N_all} clusters in all DBs")

    return DB_data, DB_fnames, DB_names_orig


def get_unique_names(DB_fnames, DB_names_orig):
    """
    Identify unique names for all the DBs. An entry can store more than
    one unique name if the DB lists several names as belonging to the same
    cluster.

    If any cluster's name is identified with another name by any of the DBs
    (eg: Pismis 6 & NGC 2645),these two clusters *that could be shown as
    separate clusters by other DBs* will be merged into a single cluster
    in the final list.
    """
    all_fnames, all_names_orig = [], []
    for k, fnames in DB_fnames.items():
        all_fnames += fnames
        all_names_orig += DB_names_orig[k]

    match_dict = {}
    for i, fnames_l in enumerate(all_fnames):
        uid = "cl" + str(i)

        clid_m = []
        for fname in fnames_l:
            # Check if any of these names is already in dictionary
            for clid, d_names in match_dict.items():
                if fname in d_names[1]:
                    clid_m += [clid]
            # Remove possible duplicate id
            clid_m = list(set(clid_m))

        if len(clid_m) == 0:
            match_dict[uid] = [[], []]
            match_dict[uid][0] = list(dict.fromkeys(all_names_orig[i]))
            match_dict[uid][1] = list(dict.fromkeys(fnames_l))
        elif len(clid_m) == 1:
            clid_m = clid_m[0]
            match_dict[clid_m][0] += all_names_orig[i]
            match_dict[clid_m][1] += fnames_l
            # Remove duplicates
            match_dict[clid_m][0] = list(dict.fromkeys(match_dict[clid_m][0]))
            match_dict[clid_m][1] = list(dict.fromkeys(match_dict[clid_m][1]))
        else:
            # Create new entry
            match_dict[uid] = [[], []]
            match_dict[uid][0] += all_names_orig[i]
            match_dict[uid][1] += fnames_l
            # Merge old entries into new entry
            for clid in clid_m:
                match_dict[uid][0] += list(match_dict[clid][0])
                match_dict[uid][1] += list(match_dict[clid][1])
                # Remove old entries from dictionary
                del match_dict[clid]
            # Remove duplicates
            match_dict[uid][0] = list(dict.fromkeys(match_dict[uid][0]))
            match_dict[uid][1] = list(dict.fromkeys(match_dict[uid][1]))

    unique_fnames, unique_names_orig = [], []
    for k, v in match_dict.items():
        unique_names_orig.append(v[0])
        unique_fnames.append(v[1])

    print(f"N={len(unique_fnames)} unique names identified")

    return unique_fnames, unique_names_orig


def get_matches(dbs_used, DB_data, DB_fnames, unique_fnames, unique_names_orig):
    """
    If any name in the name lists stored in 'unique_fnames' matches any name
    in the lists of names in any DB, all those names are assumed to belong to
    the same unique cluster
    """
    cl_dict = {}
    # For each list of unique fnames
    for q, unique_fn in enumerate(unique_fnames):
        # Remove duplicates of the kind: Berkeley 102, Berkeley102,
        # Berkeley_102; keeping only the name with the space
        cl_str = DBs_combine.rm_name_dups(unique_names_orig[q])

        # For each name in list
        cl_dict[cl_str] = {
            "DB": [],
            "DB_i": [],
            "RA": [],
            "DE": [],
            "plx": [],
            "pmra": [],
            "pmde": [],
        }

        # For each DB
        for DB_ID, fnames_db in DB_fnames.items():
            df = DB_data[DB_ID]
            cols = []
            for v in dbs_used[DB_ID]["pos"].split(","):
                if str(v) == "None":
                    v = None
                cols.append(v)
            # Remove Rv column
            ra, de, plx, pmra, pmde = cols[:-1]

            # For each name in this list of unique names
            for fname in unique_fn:
                db_match = False

                # For each fname list in this DB
                for i, fname_l in enumerate(fnames_db):
                    # Unique fname is in this list of fnames for this DB
                    if fname in fname_l:
                        db_match = True
                        cl_dict[cl_str]["DB"].append(DB_ID)
                        cl_dict[cl_str]["DB_i"].append(i)
                        # Extract row from this DB
                        row = df.iloc[i]
                        cl_dict[cl_str]["RA"].append(row[ra])
                        cl_dict[cl_str]["DE"].append(row[de])

                        bad_DB = False
                        if DB_ID in ("KHARCHENKO12", "LOKTIN17"):
                            # These DBs have bad data for these parameters
                            bad_DB = True
                        if plx is None or bad_DB is True:
                            cl_dict[cl_str]["plx"].append(np.nan)
                        else:
                            cl_dict[cl_str]["plx"].append(row[plx])
                        if pmra is None or bad_DB is True:
                            cl_dict[cl_str]["pmra"].append(np.nan)
                        else:
                            cl_dict[cl_str]["pmra"].append(row[pmra])
                        if pmde is None or bad_DB is True:
                            cl_dict[cl_str]["pmde"].append(np.nan)
                        else:
                            cl_dict[cl_str]["pmde"].append(row[pmde])

                    if db_match:
                        break
                if db_match:
                    break

    return cl_dict


def combine_DBs(cl_dict):
    """
    Store unique values for each cluster
    """
    db_l, db_i_l, names_l, ra_l, dec_l, glon_l, glat_l, plx_l, pmRA_l, pmDE_l = [
        [] for _ in range(10)
    ]
    for names, v in cl_dict.items():
        DBs, DBs_i, ra, dec, plx, pmRA, pmDE = (
            v["DB"],
            v["DB_i"],
            v["RA"],
            v["DE"],
            v["plx"],
            v["pmra"],
            v["pmde"],
        )

        # Store DBs and the indexes in them where the cluster is located
        db_l.append(";".join(str(_) for _ in DBs))
        db_i_l.append(";".join(str(_) for _ in DBs_i))

        # USe 'CANTAT20' values exclusively if they are present as they are
        # of better quality than the rest
        try:
            i = DBs.index("CANTAT20")
            ra, dec = ra[i], dec[i]
            if len(plx) > 0:
                plx = plx[i]
            if len(pmRA) > 0:
                pmRA = pmRA[i]
            if len(pmDE) > 0:
                pmDE = pmDE[i]
        except ValueError:
            pass

        names_l.append(names)

        ra_m = np.nanmedian(ra)
        dec_m = np.nanmedian(dec)
        ra_l.append(round(ra_m, 5))
        dec_l.append(round(dec_m, 5))

        lon, lat = DBs_combine.radec2lonlat(ra_m, dec_m)
        glon_l.append(lon)
        glat_l.append(lat)

        if np.isnan(plx).all():
            plx_m = np.nan
        else:
            plx_m = np.nanmedian(plx)
        plx_l.append(round(plx_m, 3))

        if np.isnan(pmRA).all():
            pmRA_m = np.nan
        else:
            pmRA_m = np.nanmedian(pmRA)
        pmRA_l.append(round(pmRA_m, 3))

        if np.isnan(pmDE).all():
            pmDE_m = np.nan
        else:
            pmDE_m = np.nanmedian(pmDE)
        pmDE_l.append(round(pmDE_m, 3))

    # Store combined databases
    comb_dbs = {
        "DB": db_l,
        "DB_i": db_i_l,
        "ID": names_l,
        "RA_ICRS": ra_l,
        "DE_ICRS": dec_l,
        "GLON": glon_l,
        "GLAT": glat_l,
        "plx": plx_l,
        "pmRA": pmRA_l,
        "pmDE": pmDE_l,
    }

    return comb_dbs


def assign_fname(all_names):
    """
    The first entry in 'all_fnames_reorder' is used for files and urls
    """
    all_names_reorder, all_fnames_reorder = [], []
    for names in all_names:
        names = names.split(";")
        fnames_temp = []
        for i, name in enumerate(names):
            name = name.strip()
            fnames_temp.append(DBs_combine.rm_chars_from_name(name))

        names_reorder, fnames_reorder = preferred_names(names, fnames_temp)

        all_names_reorder.append(";".join(names_reorder))
        # Remove duplicated before storing
        all_fnames_reorder.append(";".join(list(dict.fromkeys(fnames_reorder))))

    return all_names_reorder, all_fnames_reorder


def preferred_names(names, fnames):
    """
    Use naming conventions according to this list of preferred names
    """
    names_lst = (
        "blanco",
        "westerlund",
        "ngc",
        "melotte",
        "trumpler",
        "ruprecht",
        "berkeley",
        "pismis",
        "vdbh",
        "loden",
        "kronberger",
        "collinder",
        "haffner",
        "tombaugh",
        "dolidze",
        "auner",
        "waterloo",
        "basel",
        "bochum",
        "hogg",
        "carraro",
        "lynga",
        "johansson",
        "mamajek",
        "platais",
        "harvard",
        "czernik",
        "koposov",
        "eso",
        "ascc",
        "teutsch",
        "alessi",
        "king",
        "saurer",
        "fsr",
        "juchert",
        "antalova",
        "stephenson",
    )

    # Replace with another name according to the preference list
    if len(names) == 1:
        return names, fnames

    # Always move 'MWSC to the last position'
    if "mwsc" in fnames[0]:
        names = names[1:] + [names[0]]
        fnames = fnames[1:] + [fnames[0]]

    # # Select the first name listed
    def find_preferred_name(fnames):
        """Replace with another name according to the preference list"""
        for name_prefer in names_lst:
            for i, name in enumerate(fnames):
                if name_prefer in name:
                    return i
        return None

    def reorder_names(nms_lst, i):
        nms_reorder = list(nms_lst)
        name0 = nms_reorder[i]
        del nms_reorder[i]
        nms_reorder = [name0] + nms_reorder
        return nms_reorder

    i = find_preferred_name(fnames)
    if i is not None:
        # Reorder
        names_reorder = reorder_names(names, i)
        fnames_reorder = reorder_names(fnames, i)
    else:
        names_reorder, fnames_reorder = names, fnames

    return names_reorder, fnames_reorder


def dup_check(fnames):
    """
    Check for duplicates in 'fnames' list
    """
    fnames0 = [_.split(";")[0] for _ in fnames]
    if len(fnames0) > len(list(set(fnames0))):
        print(len(fnames0) - len(list(set(fnames0))))
        breakpoint()

    for i, cl0 in enumerate(fnames):
        for j, cl1 in enumerate(fnames[i + 1 :]):
            for cl01 in cl0.split(";"):
                if cl01 == cl1.split(";")[0]:
                    print("dup fname0:", i, i + 1 + j, cl0, cl1)
                    breakpoint()
                    break


def keep_selected_OCs(OCs_to_keep, comb_dbs):
    """
    Keep only selected OCs
    """

    def check_OC_2_keep(row):
        for OC in OCs_to_keep:
            for fname in row["fnames"].split(";"):
                if OC == fname:
                    return True

    comb_dbs = pd.DataFrame(comb_dbs)
    idx_keep = []
    for index, row in comb_dbs.iterrows():
        keep_f = check_OC_2_keep(row)
        if keep_f:
            idx_keep.append(index)

    comb_dbs = comb_dbs.filter(items=idx_keep, axis=0)
    comb_dbs = comb_dbs.reset_index(drop=True)

    return comb_dbs


if __name__ == "__main__":
    # plt.style.use('science')
    main()
