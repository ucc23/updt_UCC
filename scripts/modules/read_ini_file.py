import configparser
import os


def main():
    """
    Read .ini config file
    """

    in_params = configparser.ConfigParser()

    # Read non-tracked 'default' file first if it exists
    for fpath in (".default", ""):
        full_path = f"../params.ini{fpath}"
        if os.path.exists(full_path):
            print(f"Loaded {full_path.split('/')[1]}")
            in_params.read(full_path)
            break

    pars_dict = {}

    pars = in_params["General"]
    pars_dict["new_DB"] = pars.get("new_DB")
    pars_dict["ID"] = pars.get("ID")
    pars_dict["RA"] = pars.get("RA")
    pars_dict["DEC"] = pars.get("DEC")

    pars = in_params["New DB check"]
    pars_dict["search_rad"] = pars.getfloat("search_rad")
    pars_dict["leven_rad"] = pars.getfloat("leven_rad")
    pars_dict["rad_dup"] = pars.getfloat("rad_dup")

    pars = in_params["Run fastMP / Updt UCC"]
    pars_dict["frames_path"] = pars.get("frames_path")
    pars_dict["frames_ranges"] = pars.get("frames_ranges")
    pars_dict["max_mag"] = pars.getfloat("max_mag")
    pars_dict["manual_pars_f"] = pars.get("manual_pars_f")
    pars_dict["verbose"] = pars.getint("verbose")

    pars = in_params["Check versions"]
    pars_dict["old_UCC_name"] = pars.get("old_UCC_name")

    return pars_dict
