import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from ..variables import plots_folder, root_ucc_path

header = """---
layout: post
title: {title}
style: style
title_flag: true
more_names: {more_names}
fname: {fname}
fov: {fov}
ra_icrs: {ra_icrs}
de_icrs: {de_icrs}
glon: {glon}
glat: {glat}
r50: {r50}
plx: {plx}
UTI: "{UTI}"
UTI_COLOR: "{UTI_COLOR}"
UTI_C_N_COL: "{UTI_C_N_COL}"
UTI_C_dens_COL: "{UTI_C_dens_COL}"
UTI_C_C3_COL: "{UTI_C_C3_COL}"
UTI_C_lit_COL: "{UTI_C_lit_COL}"
UTI_C_dup_COL: "{UTI_C_dup_COL}"
UTI_C_N: {UTI_C_N}
UTI_C_dens: {UTI_C_dens}
UTI_C_C3: {UTI_C_C3}
UTI_C_lit: {UTI_C_lit}
UTI_C_dup: {UTI_C_dup}
UTI_summary: |
    {UTI_summary}
class3: |
    {class3}
r_50_val: {r_50_val}
N_50_val: {N_50_val}
scix_url: {scix_url}
posit_table: |
    {posit_table}
cds_radec: {cds_radec}
carousel: {carousel}
fpars_table: |
    {fpars_table}
shared_table: |
    {shared_table}
---
"""


def make(current_JSON, DBs_full_data, df_UCC, UCC_cl, fnames_all, fname0):
    """ """
    cl_names = UCC_cl["Names"].split(";")

    more_names = ""
    if len(cl_names) > 1:
        more_names = "; ".join(cl_names[1:])

    UTI_summary = summarize_object(
        cl_names[0],
        UCC_cl["DB"],
        UCC_cl["N_50"],
        UCC_cl["Plx_m"],
        UCC_cl["shared_members_p"],
        UCC_cl["C_N"],
        UCC_cl["C_dens"],
        UCC_cl["C_C3"],
        UCC_cl["C_lit"],
        UCC_cl["C_dup"],
        UCC_cl["C_dup_same_db"],
    )

    # Get colors used by the 'CX' classification
    abcd_c = color_C3(UCC_cl["C3"])

    cds_dec = (
        "+" + str(UCC_cl["DE_ICRS_m"])
        if UCC_cl["DE_ICRS_m"] >= 0
        else str(UCC_cl["DE_ICRS_m"])
    )
    cds_radec = f"{UCC_cl['RA_ICRS_m']},{cds_dec}"

    # Generate table with positional data: (ra, dec, plx, pmra, pmde, Rv)
    posit_table = positions_in_lit(current_JSON, DBs_full_data, UCC_cl)

    # Check present plots. The order here determines the order in which plots will be
    # shown: UCC --> HUNT23 --> CANTAT20
    carousel = "UCC"
    for _db in ("HUNT23", "CANTAT20"):
        plot_fpath = Path(
            root_ucc_path
            + plots_folder
            + f"plots_{fname0[0]}/"
            + _db
            + "/"
            + fname0
            + ".webp"
        )
        if plot_fpath.is_file() is True:
            carousel += "_" + _db

    # Generate fundamental parameters table
    fpars_table = fpars_in_lit(
        current_JSON, DBs_full_data, UCC_cl["DB"], UCC_cl["DB_i"]
    )

    # Generate table with OCs that share members with this one
    shared_table = table_shared_members(df_UCC, fnames_all, UCC_cl)

    contents = header.format(
        title=cl_names[0],
        more_names=more_names,
        fname=fname0,
        fov=str(round(2 * (UCC_cl["r_50"] / 60.0), 3)),
        ra_icrs=str(UCC_cl["RA_ICRS_m"]),
        de_icrs=str(UCC_cl["DE_ICRS_m"]),
        glon=str(UCC_cl["GLON_m"]),
        glat=str(UCC_cl["GLAT_m"]),
        r50=str(UCC_cl["r_50"]),
        plx=str(UCC_cl["Plx_m"]),
        UTI="1.0" if UCC_cl["UTI"] == 1.0 else f"{UCC_cl['UTI']:.2f}",
        UTI_COLOR=UTI_to_hex(UCC_cl["UTI"]),
        UTI_C_N_COL=UTI_to_hex(UCC_cl["C_N"]),
        UTI_C_dens_COL=UTI_to_hex(UCC_cl["C_dens"]),
        UTI_C_C3_COL=UTI_to_hex(UCC_cl["C_C3"]),
        UTI_C_lit_COL=UTI_to_hex(UCC_cl["C_lit"]),
        UTI_C_dup_COL=UTI_to_hex(UCC_cl["C_dup"]),
        UTI_C_N=str(UCC_cl["C_N"]),
        UTI_C_dens=str(UCC_cl["C_dens"]),
        UTI_C_C3=str(UCC_cl["C_C3"]),
        UTI_C_lit=str(UCC_cl["C_lit"]),
        UTI_C_dup=str(UCC_cl["C_dup"]),
        UTI_summary=UTI_summary,
        class3=abcd_c,
        r_50_val=str(UCC_cl["r_50"]),
        N_50_val=str(int(UCC_cl["N_50"])),
        scix_url=cl_names[0].replace(" ", "%20"),
        posit_table=posit_table,
        cds_radec=cds_radec,
        carousel=carousel,
        fpars_table=fpars_table,
        shared_table=shared_table,
    )

    return contents


def UTI_to_hex(x: float, soft: float = 0.65) -> str:
    """
    Map a value in [0, 1] to a pastel hex color using the RdYlGn colormap.

    Parameters
    ----------
    x : float
        Value between 0 and 1.

    Returns
    -------
    str
        Hex color code.
    """
    # Clamp to [0, 1]
    x = max(0.0, min(1.0, x))

    # Get color from colormap (RGBA in [0,1])
    rgba = cm.RdYlGn(x)

    # Convert to pastel by blending toward white
    pastel = tuple(soft + (1 - soft) * c for c in rgba[:3])  # soften colors

    return mcolors.to_hex(pastel)


def summarize_object(
    name,
    cl_DB,
    N_50,
    plx,
    shared_members_p,
    C_N,
    C_dens,
    C_C3,
    C_lit,
    C_dup,
    C_dup_same_db,
) -> str:
    """
    Summarize an astronomical object based on five normalized parameters.

    Parameters
    ----------
    name : float
        Name of the object
    last_lit_year : int
        Latest year this object was mentioned in the literature
    N_50 : int
        Number of members with P>0.5
    C_N : float
        Number of estimated true members (1 = many).
    C_dens : float
        Stellar density (1 = dense).
    C_C3 : float
        Normalized C3 parameter (1 = best class AA, 0 = worst DD).
    C_lit : float
        Presence in literature (1 = frequently mentioned).
    C_dup : float
        Probability of duplication (1 = not a duplicate).

    Returns
    -------
    str
        Short textual summary.
    """

    def level(value, thresholds, labels):
        for t, lbl in zip(thresholds, labels):
            if value >= t:
                return lbl
        return labels[-1]

    distance = level(
        plx,
        [10, 5, 2, 1],
        ["very close", "close", "relatively close", "somewhat close", ""],
    )
    members = level(
        C_N,
        [0.9, 0.75, 0.5, 0.25],
        ["very rich", "rich", "moderately populated", "poorly populated", "sparse"],
    )
    density = level(
        C_dens,
        [0.9, 0.75, 0.5, 0.25],
        ["very dense", "dense", "moderately dense", "loose", "very loose"],
    )
    quality = level(
        C_C3,
        [0.99, 0.7, 0.5, 0.2],
        ["very high", "high", "intermediate", "low", "very low"],
    )
    literature = level(
        C_lit,
        [0.9, 0.75, 0.5, 0.25],
        [
            "very well-studied",
            "well-studied",
            "moderately studied",
            "poorly studied",
            "rarely studied",
        ],
    )

    txt_dist = ""
    if distance != "":
        txt_dist = f"{distance}, "
    summary = (
        f"{name} is a {txt_dist}{members}, {density} object of {quality} C3 quality."
    )

    first_lit_year = int(cl_DB.split(";")[0].split("_")[0][-4:])
    current_year = datetime.datetime.now().year
    years_gap_first = current_year - first_lit_year
    if years_gap_first <= 3:
        # The FIRST article associated to this entry is recent
        txt = ""
        if literature in ("well-studied", "moderately studied"):
            txt = f"but it is {literature} "
        summary += f" It was recently reported {txt}in the literature."
    else:
        # The FIRST article associated to this entry is NOT recent
        last_lit_year = int(cl_DB.split(";")[-1].split("_")[0][-4:])
        years_gap_last = current_year - last_lit_year

        if years_gap_last > 5:
            # This is an OC that has not been revisited in a while
            summary += f" It is {literature} in the literature, "
            summary += f"with no articles listed in the last {years_gap_last} years."
        else:
            summary += f" It is {literature} in the literature."

    html_warn = (
        """<br><br><span style="color: #99180f; font-weight: bold;">Warning: </span>"""
    )

    summary_dup, summary_unique = False, False
    if C_dup == 1.0:
        if C_dup_same_db == 1.0:
            # No duplicates found. Do nothing
            pass
        else:
            # Only same DB duplicates found
            summary_dup = True
            duplication = "likely a unique"
            dup_warn = "<br><br>"
            dup_amount = "significant"
            dup_amount = level(
                C_dup_same_db,
                [0.75, 0.5, 0.25],
                [
                    "very small",
                    "small",
                    "moderate",
                    "significant",
                ],
            )
            prev_or_same_db = "entry reported in the same catalogue."
    else:
        summary_dup = True
        duplication = level(
            C_dup,
            [0.999, 0.75, 0.5, 0.25, 0.1],
            [
                "a unique",
                "very likely a unique",
                "likely a unique",
                "possibly a duplicated",
                "likely a duplicate",
                "very likely a duplicate",
            ],
        )
        dup_warn = "<br><br>"
        if duplication not in ("very likely a unique", "likely a unique"):
            dup_warn = html_warn
        dup_amount = level(
            C_dup,
            [0.75, 0.5, 0.25],
            ["very small", "small", "moderate", "significant"],
        )
        if C_dup_same_db == 1.0:
            # Only different DB duplicates found
            if duplication == "a unique":
                if str(shared_members_p) != "nan:":
                    shared_members_p = shared_members_p.split(";")
                    N_shared = len(shared_members_p)
                    max_p = max(map(float, shared_members_p))
                    if max_p > 10:
                        entr, upto = "entries", "up to "
                        if N_shared == 1:
                            N_shared, entr, upto = "a", "entry", ""
                        dup_warn = ""
                        summary_unique = True
            else:
                prev_or_same_db = "previously reported entry."

        else:
            # Duplicates found in same and different DBs
            dup_amount_same = level(
                C_dup_same_db,
                [0.75, 0.5, 0.25],
                ["very small", "small", "moderate", "significant"],
            )
            prev_or_same_db = (
                f"previously reported entry, and a {dup_amount_same} "
                + "percentage with at least one entry reported in the same catalogue."
            )

    if summary_unique:
        summary += (
            f"{dup_warn}This is {duplication} object, with {N_shared} later "
            + f"reported {entr} sharing {upto}{max_p:.0f}% of its members."
        )
    elif summary_dup:
        summary += (
            f"{dup_warn}This is {duplication} object, which shares a {dup_amount} "
            + f"percentage of members with at least one {prev_or_same_db}"
        )

    if N_50 < 25:
        summary += (
            html_warn + "contains less than 25 stars with <i>P>0.5</i> estimated."
        )

    return summary


def positions_in_lit(DBs_json, DBs_full_data, row_UCC):
    """ """
    DBs_i_sort = row_UCC["DB_i"].split(";")

    table = (
        "| Reference    | RA    | DEC   | Plx  | pmRA  | pmDE   |  Rv  |\n"
        + "    | :---         | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )

    for i, db in enumerate(row_UCC["DB"].split(";")):
        # Full 'db' database
        df = DBs_full_data[db]

        # Add positions
        row_in = ""
        for c in ("RA", "DEC", "plx", "pmra", "pmde", "Rv"):
            # for c in DBs_json[db]["pos"].split(","):
            # if c != "None":
            if c in DBs_json[db]["pos"].keys():
                df_col_name = DBs_json[db]["pos"][c]
                # Read position as string
                pos_v = str(df[df_col_name][int(DBs_i_sort[i])])
                # Remove empty spaces if any
                pos_v = pos_v.replace(" ", "")
                if pos_v != "" and pos_v != "nan":
                    row_in += str(round(float(pos_v), 3)) + " | "
                else:
                    row_in += "--" + " | "
            else:
                row_in += "--" + " | "

        # See if the row contains any values
        if row_in.replace("--", "").replace("|", "").strip() != "":
            # Add reference
            ref_url = f"[{DBs_json[db]['authors']} ({DBs_json[db]['year']})]({DBs_json[db]['SCIX_url']})"
            table += "    |" + ref_url + " | "
            # Close and add row
            table += row_in[:-1] + "\n"

    # Add UCC positions
    table += "    | **UCC** |"
    for col in ("RA_ICRS_m", "DE_ICRS_m", "Plx_m", "pmRA_m", "pmDE_m", "Rv_m"):
        val = "--"
        if row_UCC[col] != "" and not np.isnan(row_UCC[col]):
            val = str(round(float(row_UCC[col]), 3))
        table += val + " | "

    return table


def table_shared_members(df_UCC, fnames_all, row):
    """ """
    shared_members_tab = ""

    if str(row["shared_members"]) == "nan":
        return shared_members_tab

    shared_members_tab = """| Cluster | <span title="Percentage of members that this OC shares with the ones listed">%</span>   | RA   | DEC   | Plx   | pmRA  | pmDE  | Rv    |\n"""
    shared_members_tab += (
        """    | :---:   | :-: |:---: | :---: | :---: | :---: | :---: | :---: |\n"""
    )

    shared_fnames = row["shared_members"].split(";")
    shared_percents = row["shared_members_p"].split(";")

    for i, fname in enumerate(shared_fnames):
        # Locate OC with shared members in the UCC
        j = fnames_all.index(fname)

        name = df_UCC["Names"][j].split(";")[0]
        vals = []
        for col in ("RA_ICRS_m", "DE_ICRS_m", "Plx_m", "pmRA_m", "pmDE_m", "Rv_m"):
            val = round(float(df_UCC[col][j]), 3)
            if np.isnan(val):
                vals.append("--")
            else:
                vals.append(val)
        val = shared_percents[i]
        if val == "nan":
            vals.append("--")
        else:
            vals.append(val)

        ra, dec, plx, pmRA, pmDE, Rv, perc = vals

        shared_members_tab += f"    |[{name}](/_clusters/{fname}/)| "
        shared_members_tab += f"{perc} | "
        shared_members_tab += f"{ra} | "
        shared_members_tab += f"{dec} | "
        shared_members_tab += f"{plx} | "
        shared_members_tab += f"{pmRA} | "
        shared_members_tab += f"{pmDE} | "
        shared_members_tab += f"{Rv} |\n"

    # Remove final new line
    shared_members_tab = shared_members_tab[:-1]

    return shared_members_tab


def fpars_in_lit(
    DBs_json: dict, DBs_full_data: dict, DBs: str, DBs_i: str, max_chars=7
):
    """
    DBs_json: JSON that contains the data for all DBs
    DBs_full_data: dictionary of pandas.DataFrame for all DBs
    DBs: DBs where this cluster is present
    DBs_i: Indexes where this cluster is present in each DB
    """
    DBs_lst, DBs_i_lst = DBs.split(";"), DBs_i.split(";")

    # Select DBs with parameters
    DBs_w_pars, DBs_i_w_pars = [], []
    for i, db in enumerate(DBs_lst):
        # If this database contains any estimated fundamental parameters
        if (not DBs_json[db]["pars"]) is False:
            DBs_w_pars.append(db)
            DBs_i_w_pars.append(DBs_i_lst[i])

    table = ""
    if len(DBs_w_pars) == 0:
        return table

    txt = ""
    for i, db in enumerate(DBs_w_pars):
        # Full 'db' database
        df = DBs_full_data[db]
        # Add reference
        ref_url = f"[{DBs_json[db]['authors']} ({DBs_json[db]['year']})]({DBs_json[db]['SCIX_url']})"
        txt_db = "    | " + ref_url + " | "

        txt_pars = ""
        # Add non-nan parameters
        for k, par in DBs_json[db]["pars"].items():
            # Some DBs have parameters as lists because they list more than one value
            # per parameter
            if isinstance(par, list):
                for par_i in par:
                    par_v = str(df[par_i][int(DBs_i_w_pars[i])])[:max_chars]
                    par_v = par_v.replace(" ", "")
                    if par_v != "" and par_v != "nan":
                        txt_pars += par_i + "=" + par_v + ", "

            else:
                par_v = str(df[par][int(DBs_i_w_pars[i])])[:max_chars]
                par_v = par_v.replace(" ", "")
                if par_v != "" and par_v != "nan":
                    txt_pars += par + "=" + par_v + ", "

        if txt_pars != "":
            # Remove final ', '
            txt_pars = txt_pars[:-2]
            # Add quotes
            txt_pars = "`" + txt_pars + "`"
            # Combine and close row
            txt += txt_db + txt_pars + " |\n"

    if txt != "":
        # If any parameter was added, add header to the table
        txt0 = """| Reference |  Values |\n    | :---  |  :---:  |\n"""
        # [:-1] -> Remove final new line
        table = txt0 + txt[:-1]

    return table


def color_C3(abcd):
    """ """
    abcd_c = ""
    line = r"""<span style="color: {}; font-weight: bold;">{}</span>"""
    cc = {"A": "green", "B": "#FFC300", "C": "red", "D": "purple"}
    for letter in abcd:
        abcd_c += line.format(cc[letter], letter)  # + '\n'
    return abcd_c
