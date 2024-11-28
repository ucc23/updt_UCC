import csv

import pandas as pd
from HARDCODED import new_OCs_fpath
from modules import logger, read_ini_file
from modules import main_process_GDR3_query as G3Q

#
cl_name = "HSC_906"
RA_ICRS, DE_ICRS = 195.183, 56.708
box_s, plx_min = 60, 15


def main():
    """ """
    logging = logger.main()

    pars_dict = read_ini_file.main()

    logging.info(f"\nMaximum magnitude: {pars_dict['max_mag']}")

    # Read extra input data
    frames_data = pd.read_csv(pars_dict["frames_ranges"])

    # Write header for ouput CSV file. Order matters!
    with open(new_OCs_fpath, "w") as myfile:
        myfile.write(
            "fname,index_updt,N_fixed,N_survived,fixed_centers,cent_flags,"
            + "C1,C2,C3,GLON_m,GLAT_m,RA_ICRS_m,DE_ICRS_m,plx_m,pmRA_m,"
            + "pmDE_m,Rv_m,N_Rv,N_50,r_50,N_ex_cls\n"
        )

    # Request data
    data = G3Q.run(
        pars_dict["frames_path"],
        frames_data,
        RA_ICRS,
        DE_ICRS,
        box_s,
        plx_min,
        pars_dict["max_mag"],
        pars_dict["verbose"],
        logging,
    )
    data.to_csv(
        cl_name + "_frame.csv",
        na_rep="nan",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


if __name__ == "__main__":
    main()
