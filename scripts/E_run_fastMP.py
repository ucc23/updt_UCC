import numpy as np
import pandas as pd
from scipy import spatial
from modules import logger
from modules import read_ini_file
from modules import run_fastMP_funcs as fMPf
from modules import classif
from modules import main_process_GDR3_query as G3Q
from modules import UCC_new_match

# Load local version of fastMP
# insert at 1, 0 is the script path (or '' in REPL)
import sys

sys.path.insert(1, "/home/gabriel/Github/Gabriel_p/fastmp/")  # Path to fastMP
from fastmp import fastMP


def main():
    """ """
    logging = logger.main()

    pars_dict = read_ini_file.main()
    (
        new_DB,
        dbs_folder,
        GCs_cat,
        UCC_folder,
        frames_path,
        frames_ranges,
        max_mag,
        manual_pars_f,
        verbose,
        new_OCs_data,
        root_UCC_path,
    ) = (
        pars_dict["new_DB"],
        pars_dict["dbs_folder"],
        pars_dict["GCs_cat"],
        pars_dict["UCC_folder"],
        pars_dict["frames_path"],
        pars_dict["frames_ranges"],
        pars_dict["max_mag"],
        pars_dict["manual_pars_f"],
        pars_dict["verbose"],
        pars_dict["new_OCs_data"],
        pars_dict["root_UCC_path"],
    )

    # Read latest version of the UCC
    df_UCC, UCC_cat = UCC_new_match.latest_cat_detect(logging, UCC_folder)

    # Read extra input data
    frames_data = pd.read_csv(frames_ranges)
    df_gcs = pd.read_csv(dbs_folder + GCs_cat)

    # # Read info on what OCs to process
    # new_OCs_info = pd.read_csv(new_OCs_fpath)
    # Read OCs manual parameters
    manual_pars = pd.read_csv(manual_pars_f)

    # N_process = new_OCs_info['process_f'].sum()
    # logging.info(f"Processing {N_process} OCs with fastMP...\n")
    logging.info(f"Processing {new_DB} with fastMP...\n")

    # Parameters used to search for close-by clusters
    xys = np.array([df_UCC["GLON"].values, df_UCC["GLAT"].values]).T
    tree = spatial.cKDTree(xys)

    # Write header for ouput CSV file. Order matters!
    with open(new_OCs_data, "w") as myfile:
        myfile.write(
            "fname,index_updt,N_fixed,N_survived,fixed_centers,cent_flags,"
            + "C1,C2,C3,GLON_m,GLAT_m,RA_ICRS_m,DE_ICRS_m,plx_m,pmRA_m,"
            + "pmDE_m,Rv_m,N_Rv,N_50,r_50,N_ex_cls\n"
        )

    # For each new OC
    for index, new_cl in df_UCC.iterrows():
        params_updt = []

        # Check if this is a new OC that should be processed
        if str(new_cl["C3"]) != "nan":
            continue

        # Identify position in the UCC
        fname0 = new_cl["fnames"].split(";")[0]
        UCC_index = None
        for _, UCC_fnames in enumerate(df_UCC["fnames"]):
            if new_cl["fnames"] == UCC_fnames:
                UCC_index = _
                break
        if UCC_index is None:
            logging.info(f"ERROR: could not find {fname0} in UCC DB")
            return
        cl = df_UCC.iloc[UCC_index]
        logging.info(f"*** {index} Processing {cl['fnames']}")

        # Generate frame
        box_s, plx_min = fMPf.get_frame(cl)

        # fname0 = cl['fnames'].split(';')[0]
        # # These clusters are extended and require a larger frame
        # if fname0.startswith('ubc'):
        #     box_s *= 3
        #     box_s = min(box_s, 25)

        #
        fix_N_clust = None
        for _, row_manual_p in manual_pars.iterrows():
            if fname0 == row_manual_p["fname"]:
                if row_manual_p["Nmembs"] != "nan":
                    fix_N_clust = int(row_manual_p["Nmembs"])
                    logging.info(f"Manual N_membs applied: {fix_N_clust}")

                if row_manual_p["box_s"] != "nan":
                    box_s = float(row_manual_p["box_s"])
                    logging.info(f"Manual box size applied: {box_s}")

        # Get close clusters coords
        centers_ex = fMPf.get_close_cls(
            cl["GLON"],
            cl["GLAT"],
            tree,
            box_s,
            UCC_index,
            df_UCC,
            cl["dups_fnames"],
            df_gcs,
        )

        # Request data
        data = G3Q.run(
            frames_path,
            frames_data,
            cl["RA_ICRS"],
            cl["DE_ICRS"],
            box_s,
            plx_min,
            max_mag,
            verbose,
            logging,
        )

        # Extract center coordinates
        xy_c, vpd_c, plx_c = (cl["GLON"], cl["GLAT"]), None, None
        if not np.isnan(cl["pmRA"]):
            vpd_c = (cl["pmRA"], cl["pmDE"])
        if not np.isnan(cl["plx"]):
            plx_c = cl["plx"]

        fixed_centers = False
        if vpd_c is None and plx_c is None:
            fixed_centers = True

        # Generate input data array for fastMP
        X = np.array(
            [
                data["GLON"].values,
                data["GLAT"].values,
                data["pmRA"].values,
                data["pmDE"].values,
                data["Plx"].values,
                data["e_pmRA"].values,
                data["e_pmDE"].values,
                data["e_Plx"].values,
            ]
        )

        # Process with fastMP
        while True:
            logging.info(f"Fixed centers?: {fixed_centers}")
            probs_all, N_survived = fastMP(
                xy_c=xy_c,
                vpd_c=vpd_c,
                plx_c=plx_c,
                centers_ex=centers_ex,
                fixed_centers=fixed_centers,
                fix_N_clust=fix_N_clust,
            ).fit(X)

            cent_flags = fMPf.check_centers(*X[:5, :], xy_c, vpd_c, plx_c, probs_all)

            if cent_flags == "000" or fixed_centers is True:
                break
            else:
                # Re-run with fixed centers
                fixed_centers = True

        cent_flags = fMPf.check_centers(*X[:5, :], xy_c, vpd_c, plx_c, probs_all)

        logging.info(
            "Nsurv={}, (P>0.5)={}, cents={}".format(
                N_survived, (probs_all > 0.5).sum(), cent_flags
            )
        )

        df_membs, df_field = fMPf.split_membs_field(data, probs_all)
        # Write member stars for cluster and some field
        fMPf.save_cl_datafile(root_UCC_path, cl, df_membs, logging)

        C1, C2, C3 = classif.get_classif(df_membs, df_field)
        lon, lat, ra, dec, plx, pmRA, pmDE, Rv, N_Rv, N_50, r_50 = fMPf.extract_cl_data(
            df_membs
        )

        # Order matters!
        params_updt.append(fname0)
        params_updt.append(UCC_index)
        params_updt.append(fix_N_clust)
        params_updt.append(int(N_survived))
        params_updt.append(fixed_centers)
        params_updt.append(int(cent_flags))
        params_updt.append(C1)
        params_updt.append(C2)
        params_updt.append(C3)
        params_updt.append(lon)
        params_updt.append(lat)
        params_updt.append(ra)
        params_updt.append(dec)
        params_updt.append(plx)
        params_updt.append(pmRA)
        params_updt.append(pmDE)
        params_updt.append(Rv)
        params_updt.append(N_Rv)
        params_updt.append(N_50)
        params_updt.append(r_50)

        logging.info(f"*** Cluster {cl['ID']} processed with fastMP\n")

        # Update output CSV file
        pars_str = ""
        for p in params_updt:
            pars_str += str(p) + ","
        pars_str = pars_str[:-1] + "\n"
        with open(new_OCs_data, "a") as myfile:
            myfile.write(pars_str)


if __name__ == "__main__":
    main()
