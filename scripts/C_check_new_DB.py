import numpy as np
from modules import logger
from modules import read_ini_file
from modules import DBs_combine
from modules import UCC_new_match
from HARDCODED import dbs_folder, all_DBs_json, UCC_folder


def main():
    """ """
    logging = logger.main()
    logging.info("\nRunning 'check_new_DB' script\n")

    pars_dict = read_ini_file.main()
    new_DB = pars_dict["new_DB"]
    logging.info(f"Checking new DB: {new_DB}")

    df_UCC, df_new, json_pars, new_DB_fnames, db_matches = UCC_new_match.main(
        logging, dbs_folder, all_DBs_json, UCC_folder
    )

    new_db_info = prep_newDB(df_new, json_pars, new_DB_fnames, db_matches)

    # Store information on the OCs being added
    logging.info("\nOCs flagged for attention:\n")
    logging.info("{:<15} {:<5} {}".format(
        "name", "cent_flag", "[arcmin] [pmRA %] [pmDE %] [plx %]"))
    for i, fnames in enumerate(new_db_info["fnames"]):
        j = new_db_info["UCC_idx"][i]
        # If the OC is already present in the UCC
        if j is not None:
            flag_log(logging, df_UCC, new_db_info, fnames, i, j)


def prep_newDB(df_new, json_pars, new_DB_fnames, db_matches):
    """
    Prepare the info from the new Db matched with the UCC
    """
    # Extract names of (ra, dec, plx, pmRA, pmDE) columns
    cols = []
    for v in json_pars["pos"].split(","):
        if str(v) == "None":
            v = None
        cols.append(v)
    # Remove Rv column
    ra_c, dec_c, plx_c, pmra_c, pmde_c = cols[:-1]

    new_db_info = {
        "fnames": [],
        "RA_ICRS": [],
        "DE_ICRS": [],
        "pmRA": [],
        "pmDE": [],
        "plx": [],
        "UCC_idx": [],
    }

    for i, fnames_lst in enumerate(new_DB_fnames):
        # Use semi-colon here to math the UCC format
        fnames = ";".join(fnames_lst)
        row_n = df_new.iloc[i]

        # Coordinates for this cluster in the new DB
        plx_n, pmra_n, pmde_n = np.nan, np.nan, np.nan
        ra_n, dec_n = row_n[ra_c], row_n[dec_c]
        if plx_c is not None:
            plx_n = row_n[plx_c]
        if pmra_c is not None:
            pmra_n = row_n[pmra_c]
        if pmde_c is not None:
            pmde_n = row_n[pmde_c]

        # Index of the match for this new cluster in the old DB (if any)
        db_match_j = db_matches[i]

        # If the cluster is already present in the UCC
        if db_match_j is not None:
            new_db_info["fnames"].append(fnames)
            new_db_info["RA_ICRS"].append(ra_n)
            new_db_info["DE_ICRS"].append(dec_n)
            new_db_info["pmRA"].append(pmra_n)
            new_db_info["pmDE"].append(pmde_n)
            new_db_info["plx"].append(plx_n)
            new_db_info["UCC_idx"].append(db_match_j)
        else:
            # This is a new cluster
            new_db_info["fnames"].append(fnames)
            new_db_info["RA_ICRS"].append(np.nan)
            new_db_info["DE_ICRS"].append(np.nan)
            new_db_info["pmRA"].append(np.nan)
            new_db_info["pmDE"].append(np.nan)
            new_db_info["plx"].append(plx_n)
            new_db_info["UCC_idx"].append(None)

    lon_n, lat_n = DBs_combine.radec2lonlat(
        new_db_info["RA_ICRS"], new_db_info["DE_ICRS"]
    )
    new_db_info["GLON"] = lon_n
    new_db_info["GLAT"] = lat_n

    return new_db_info


def flag_log(logging, df_UCC, new_db_info, fnames, i, j):
    """ """
    bad_center = DBs_combine.check_cents_diff(
        (df_UCC["GLON_m"][j], df_UCC["GLAT_m"][j]),
        (df_UCC["pmRA_m"][j], df_UCC["pmDE_m"][j]),
        df_UCC["plx"][j],
        (new_db_info["GLON"][i], new_db_info["GLAT"][i]),
        (new_db_info["pmRA"][i], new_db_info["pmDE"][i]),
        new_db_info["plx"][i],
    )

    # Is the difference between the old vs new center values large?
    if bad_center == "000":
        return

    txt = ""
    if bad_center[0] == "1":
        d_arcmin = (
            np.sqrt(
                (df_UCC["GLON_m"][j] - new_db_info["GLON"][i]) ** 2
                + (df_UCC["GLAT_m"][j] - new_db_info["GLAT"][i]) ** 2
            )
            * 60
        )
        txt += "{:.1f} ".format(d_arcmin)
    else:
        txt += "nan "
    if bad_center[1] == "1":
        pmra_p = 100 * abs(
            (df_UCC["pmRA_m"][j] - new_db_info["pmRA"][i])
            / (df_UCC["pmRA_m"][j] + 0.001)
        )
        pmde_p = 100 * abs(
            (df_UCC["pmDE_m"][j] - new_db_info["pmDE"][i])
            / (df_UCC["pmDE_m"][j] + 0.001)
        )
        txt += "{:.1f} {:.1f} ".format(pmra_p, pmde_p)
    else:
        txt += "nan nan "
    if bad_center[2] == "1":
        plx_p = (
            100
            * abs(df_UCC["plx_m"][j] - new_db_info["plx"][i])
            / (df_UCC["plx_m"][j] + 0.001)
        )
        txt += "{:.1f}".format(plx_p)
    else:
        txt += "nan"
    txt = txt.split()
    logging.info("{:<15} {:<5} {:>12} {:>8} {:>8} {:>7}".format(
        fnames, bad_center, *txt))


if __name__ == "__main__":
    main()
