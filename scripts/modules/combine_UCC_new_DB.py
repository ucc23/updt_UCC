import numpy as np

from .DBs_combine import radec2lonlat, rename_standard, rm_name_dups


def main(
    logging, new_DB_ID, df_UCC, df_new, json_pars, new_DB_fnames, db_matches, sep=","
):
    """ """
    # Extract names of (ra, dec, plx, pmRA, pmDE) columns
    cols = []
    for v in json_pars["pos"].split(","):
        if str(v) == "None":
            v = None
        cols.append(v)
    # Remove Rv column
    ra_c, dec_c, plx_c, pmra_c, pmde_c = cols[:-1]

    new_db_dict = {_: [] for _ in df_UCC.keys()}
    for i, new_cl in enumerate(new_DB_fnames):
        row_n = df_new.iloc[i]
        new_names = row_n[json_pars["names"]].split(sep)
        new_names_rename = []
        for _ in new_names:
            name = rename_standard(_.strip())
            new_names_rename.append(name)
        new_names = ";".join(new_names_rename)

        # Index of the match for this new cluster in the UCC (if any)
        if db_matches[i] is None:  # The cluster is not present in the UCC
            new_db_dict = new_OC_not_in_UCC(
                new_DB_ID,
                new_db_dict,
                i,
                new_cl,
                row_n,
                new_names,
                ra_c,
                dec_c,
                plx_c,
                pmra_c,
                pmde_c,
            )
            logging.info(f"{i} {','.join(new_cl)} is a new OC")
        else:  # The cluster is already present in the UCC
            # Identify row in UCC where this match is located
            row = df_UCC.iloc[db_matches[i]]
            new_db_dict = OC_in_UCC(new_DB_ID, new_db_dict, i, new_cl, new_names, row)
            logging.info(f"{i} {','.join(new_cl)} is in the UCC: update DB indexes")

    # Remove duplicates of the kind: Berkeley 102, Berkeley102,
    # Berkeley_102; keeping only the name with the space
    for q, names in enumerate(new_db_dict["ID"]):
        names_l = names.split(";")
        new_db_dict["ID"][q] = rm_name_dups(names_l)

    return new_db_dict


def OC_in_UCC(new_DB_ID, new_db_dict, i, new_cl, new_names, row):
    """Add an OC that is already present in the UCC"""

    # Check if this new DB is already in the UCC list
    if new_DB_ID in row["DB"]:
        # Remove new DB and its OC index from the UCC
        DB_ID_lst = row["DB"].split(";")
        DB_i_lst = row["DB_i"].split(";")
        # Index of new DB in the UCC
        i_rm = DB_ID_lst.index(new_DB_ID)
        del DB_ID_lst[i_rm]
        del DB_i_lst[i_rm]
        # Re-add new DB and its OC index
        if len(DB_ID_lst) > 0:
            DB_ID = ";".join(DB_ID_lst) + ";" + new_DB_ID
            DB_i = ";".join(DB_i_lst) + ";" + str(i)
        else:
            DB_ID = new_DB_ID
            DB_i = str(i)
    else:
        DB_ID = row["DB"] + ";" + new_DB_ID
        DB_i = row["DB_i"] + ";" + str(i)
    # Order by years before storing
    DB_ID, DB_i = date_order_DBs(DB_ID, DB_i)
    new_db_dict["DB"].append(DB_ID)
    new_db_dict["DB_i"].append(DB_i)

    # Add name in new DB
    ID = row["ID"] + ";" + new_names
    # Remove duplicates
    ID = ";".join(list(dict.fromkeys(ID.split(";"))))
    new_db_dict["ID"].append(ID)

    # Copy values from the UCC for these columns
    lon_n, lat_n = radec2lonlat(row["RA_ICRS"], row["DE_ICRS"])
    new_db_dict["RA_ICRS"].append(round(row["RA_ICRS"], 4))
    new_db_dict["DE_ICRS"].append(round(row["DE_ICRS"], 4))
    new_db_dict["GLON"].append(round(lon_n, 4))
    new_db_dict["GLAT"].append(round(lat_n, 4))
    new_db_dict["plx"].append(round(row["plx"], 4))
    new_db_dict["pmRA"].append(round(row["pmRA"], 4))
    new_db_dict["pmDE"].append(round(row["pmDE"], 4))

    # Add fnames in new DB
    fnames = row["fnames"] + ";" + ";".join(new_cl)
    # Remove duplicates
    fnames = ";".join(list(dict.fromkeys(fnames.split(";"))))
    new_db_dict["fnames"].append(fnames)

    # Copy values from the UCC for these columns
    new_db_dict["UCC_ID"].append(row["UCC_ID"])
    new_db_dict["quad"].append(row["quad"])
    new_db_dict["dups_fnames"].append(row["dups_fnames"])
    new_db_dict["dups_probs"].append(row["dups_probs"])
    new_db_dict["r_50"].append(row["r_50"])
    new_db_dict["N_50"].append(row["N_50"])
    new_db_dict["N_fixed"].append(row["N_fixed"])
    new_db_dict["N_membs"].append(row["N_membs"])
    new_db_dict["fixed_cent"].append(row["fixed_cent"])
    new_db_dict["cent_flags"].append(row["cent_flags"])
    new_db_dict["C1"].append(row["C1"])
    new_db_dict["C2"].append(row["C2"])
    new_db_dict["C3"].append(row["C3"])
    new_db_dict["GLON_m"].append(row["GLON_m"])
    new_db_dict["GLAT_m"].append(row["GLAT_m"])
    new_db_dict["RA_ICRS_m"].append(row["RA_ICRS_m"])
    new_db_dict["DE_ICRS_m"].append(row["DE_ICRS_m"])
    new_db_dict["plx_m"].append(row["plx_m"])
    new_db_dict["pmRA_m"].append(row["pmRA_m"])
    new_db_dict["pmDE_m"].append(row["pmDE_m"])
    new_db_dict["Rv_m"].append(row["Rv_m"])
    new_db_dict["N_Rv"].append(row["N_Rv"])
    new_db_dict["dups_fnames_m"].append(row["dups_fnames_m"])
    new_db_dict["dups_probs_m"].append(row["dups_probs_m"])

    return new_db_dict


def date_order_DBs(DB, DB_i):
    """Order DBs by year"""
    # Split lists
    all_dbs = DB.split(";")
    all_dbs_i = DB_i.split(";")

    # Extract years from DBs
    all_years = []
    for db in all_dbs:
        year = db.split("_")[0][-2:]
        all_years.append(year)
    # WILL BREAK IN 2090. If you are reading this, good luck to you :)
    idx = sort_year_digits(all_years)
    DB = ";".join(np.array(all_dbs)[idx].tolist())
    DB_i = ";".join(np.array(all_dbs_i)[idx].tolist())
    return DB, DB_i


def sort_year_digits(year_digits: list[str]) -> np.ndarray:
    """
    Sorts a list of two-digit year representations by converting them into
    full four-digit years based on a threshold. Years represented as 90-99 are
    treated as 1990-1999, while years represented as 00-89 are treated as 2000-2089.

    Args:
        year_digits: A list of two-digit strings representing years.

    Returns:
        A numpy array of indices representing the sorted order of the input list,
        based on the adjusted four-digit year values.

    Example:
        >>> sort_year_digits([1, 23, 1, 8, 0, 6, 99, 24])
        array([6, 4, 0, 2, 5, 3, 1, 7])
    """
    year_list = np.array(year_digits, dtype=int)
    msk = year_list >= 90
    year_list[msk] += 1900  # Treat 90-99 as 1990-1999
    year_list[~msk] += 2000  # Treat 00-89 as 2000-2089
    return np.argsort(year_list)


def new_OC_not_in_UCC(
    new_DB_ID,
    new_db_dict,
    i,
    new_cl,
    row_n,
    new_names,
    ra_c,
    dec_c,
    plx_c,
    pmra_c,
    pmde_c,
):
    """ """
    new_db_dict["DB"].append(new_DB_ID)
    new_db_dict["DB_i"].append(str(i))

    # Remove duplicates
    if ";" in new_names:
        ID = ";".join(list(dict.fromkeys(new_names.split(";"))))
    else:
        ID = new_names
    new_db_dict["ID"].append(ID)

    # Coordinates for this cluster in the new DB
    plx_n, pmra_n, pmde_n = np.nan, np.nan, np.nan
    ra_n, dec_n = row_n[ra_c], row_n[dec_c]
    if plx_c is not None:
        plx_n = row_n[plx_c]
    if pmra_c is not None:
        pmra_n = row_n[pmra_c]
    if pmde_c is not None:
        pmde_n = row_n[pmde_c]

    lon_n, lat_n = radec2lonlat(ra_n, dec_n)
    new_db_dict["RA_ICRS"].append(round(ra_n, 4))
    new_db_dict["DE_ICRS"].append(round(dec_n, 4))
    new_db_dict["GLON"].append(round(lon_n, 4))
    new_db_dict["GLAT"].append(round(lat_n, 4))
    new_db_dict["plx"].append(round(plx_n, 4))
    new_db_dict["pmRA"].append(round(pmra_n, 4))
    new_db_dict["pmDE"].append(round(pmde_n, 4))

    # Remove duplicates
    fnames = ";".join(new_cl)
    if ";" in fnames:
        fnames = ";".join(list(dict.fromkeys(fnames.split(";"))))
    new_db_dict["fnames"].append(fnames)

    # These values will be assigned later on for these new clusters
    new_db_dict["UCC_ID"].append("nan")
    new_db_dict["quad"].append("nan")
    new_db_dict["dups_fnames"].append("nan")
    new_db_dict["dups_probs"].append("nan")

    # These values will be assigned later when fastMP is run
    new_db_dict["r_50"].append(np.nan)
    new_db_dict["N_50"].append(0)
    new_db_dict["N_fixed"].append(0)
    new_db_dict["N_membs"].append(0)
    new_db_dict["fixed_cent"].append("nan")
    new_db_dict["cent_flags"].append("nan")
    new_db_dict["C1"].append(np.nan)
    new_db_dict["C2"].append(np.nan)
    new_db_dict["C3"].append("nan")
    new_db_dict["GLON_m"].append(np.nan)
    new_db_dict["GLAT_m"].append(np.nan)
    new_db_dict["RA_ICRS_m"].append(np.nan)
    new_db_dict["DE_ICRS_m"].append(np.nan)
    new_db_dict["plx_m"].append(np.nan)
    new_db_dict["pmRA_m"].append(np.nan)
    new_db_dict["pmDE_m"].append(np.nan)
    new_db_dict["Rv_m"].append(np.nan)
    new_db_dict["N_Rv"].append(0)
    new_db_dict["dups_fnames_m"].append("nan")
    new_db_dict["dups_probs_m"].append("nan")

    return new_db_dict
