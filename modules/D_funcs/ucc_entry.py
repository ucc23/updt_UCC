import re
from pathlib import Path

import numpy as np

from ..variables import fpars_headers, fpars_order, plots_folder, root_ucc_path

header = """---
layout: layout_cluster
style: style_cluster
title: {cl_name}
title_flag: true
more_names: {more_names}
fname: {fname}
members_file: "{members_file}"
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
UTI_C_N_desc: {UTI_C_N_desc}
UTI_C_dens_desc: {UTI_C_dens_desc}
UTI_C_C3_desc: {UTI_C_C3_desc}
UTI_C_lit_desc: {UTI_C_lit_desc}
UTI_C_dup_desc: {UTI_C_dup_desc}
summary: |
{summary}
badge_dist: "{badge_dist}"
badge_dist_url: "{badge_dist_url}"
badge_av: "{badge_av}"
badge_av_url: "{badge_av_url}"
badge_mass: "{badge_mass}"
badge_mass_url: "{badge_mass_url}"
badge_feh: "{badge_feh}"
badge_feh_url: "{badge_feh_url}"
badge_age: "{badge_age}"
badge_age_url: "{badge_age_url}"
badge_bss: "{badge_bss}"
badge_bss_url: "{badge_bss_url}"
badge_nofpars_url: "{badge_nofpars_url}"
comments: |
{comments}
class3: |
{class3}
dens_val: {dens_val}
N_50_val: {N_50_val}
scix_url: {scix_url}
posit_table: |
{posit_table}
cds_radec: {cds_radec}
carousel: {carousel}
fpars_table: |
{fpars_table}
note_asterisk: {note_asterisk}
shared_table: |
{shared_table}
---
"""


def make(
    fname0,
    i_ucc,
    current_JSON,
    members_files_mapping,
    DBs_full_data,
    df_BC,
    UCC_cl,
    fnames_all,
    UTI_colors,
    summary,
    descriptors,
    fpars_badges,
    badges_url,
    comments_lst,
    tsp="    ",
):
    """
    tsp: table spacing, added to the header to properly render tables and summary in
    .md files
    """

    cl_names = UCC_cl["Names"].split(";")

    more_names = ""
    if len(cl_names) > 1:
        more_names = "; ".join(cl_names[1:])

    members_file = members_files_mapping[fname0]

    # Generate fundamental parameters table
    fpars_table, mult_vals_note_flag = fpars_in_lit(current_JSON, UCC_cl, tsp)

    # summary = UCC_summ_cmmts[fname0]["summary"]
    UTI_C_N_desc, UTI_C_dens_desc, UTI_C_C3_desc, UTI_C_lit_desc, UTI_C_dup_desc = (
        descriptors
    )

    # Badges
    badges_names = ("dist", "av", "mass", "feh", "age", "bss")
    badge_dist, badge_av, badge_mass, badge_feh, badge_age, badge_bss = [
        fpars_badges.get(k, "") for k in badges_names
    ]

    def make_url(key):
        if not (val := badges_url.get(key)):
            return ""
        vmin, vmax = val
        return f"{key}_min={vmin}&{key}_max={vmax}"

    (
        badge_dist_url,
        badge_av_url,
        badge_mass_url,
        badge_feh_url,
        badge_age_url,
        badge_bss_url,
    ) = [make_url(k) for k in badges_names]

    # No fpars badge
    badge_nofpars_url = ""
    if all(
        b == ""
        for b in [badge_dist, badge_av, badge_mass, badge_feh, badge_age, badge_bss]
    ):
        vmin = "1e6"
        badge_nofpars_url = f"dav_min={vmin}&bf_min={vmin}&"
        badge_nofpars_url += "&".join([f"{k}_min={vmin}" for k in badges_names])
        badge_nofpars_url += "&nofpars=true"

    # Comments
    comments = ""
    if comments_lst:
        for cmmt in comments_lst:
            comments += (
                f"{tsp}<p><u><a href='{cmmt['url']}' target='_blank'>{cmmt['name']} ({cmmt['year']})</a></u>"
                + f"<br>{cmmt['comment']}</p>"
                + "\n"
            )

    # Get colors used by the 'CX' classification
    abcd_c = tsp + color_C3(UCC_cl["C3"])

    cds_dec = (
        "+" + str(UCC_cl["DE_ICRS_m"])
        if UCC_cl["DE_ICRS_m"] >= 0
        else str(UCC_cl["DE_ICRS_m"])
    )
    cds_radec = f"{UCC_cl['RA_ICRS_m']},{cds_dec}"

    # Generate table with positional data: (ra, dec, plx, pmra, pmde, Rv)
    posit_table = positions_in_lit(current_JSON, DBs_full_data, UCC_cl, tsp)

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

    # Generate table with OCs that share members with this one
    shared_table = table_shared_members(df_BC, fnames_all, UCC_cl, tsp)

    contents = header.format(
        cl_name=cl_names[0],
        more_names=more_names,
        fname=fname0,
        members_file=members_file,
        fov=str(round(2 * (UCC_cl["r_50"] / 60.0), 3)),
        ra_icrs=str(UCC_cl["RA_ICRS_m"]),
        de_icrs=str(UCC_cl["DE_ICRS_m"]),
        glon=str(UCC_cl["GLON_m"]),
        glat=str(UCC_cl["GLAT_m"]),
        r50=str(UCC_cl["r_50"]),
        plx=str(UCC_cl["Plx_m"]),
        UTI="1.0" if UCC_cl["UTI"] == 1.0 else f"{UCC_cl['UTI']:.2f}",
        UTI_COLOR=UTI_colors["UTI"][i_ucc],
        UTI_C_N_COL=UTI_colors["C_N"][i_ucc],
        UTI_C_dens_COL=UTI_colors["C_dens"][i_ucc],
        UTI_C_C3_COL=UTI_colors["C_C3"][i_ucc],
        UTI_C_lit_COL=UTI_colors["C_lit"][i_ucc],
        UTI_C_dup_COL=UTI_colors["C_dup"][i_ucc],
        UTI_C_N=str(UCC_cl["C_N"]),
        UTI_C_dens=str(UCC_cl["C_dens"]),
        UTI_C_C3=str(UCC_cl["C_C3"]),
        UTI_C_lit=str(UCC_cl["C_lit"]),
        UTI_C_dup=str(UCC_cl["C_dup"]),
        UTI_C_N_desc=UTI_C_N_desc,
        UTI_C_dens_desc=UTI_C_dens_desc,
        UTI_C_C3_desc=UTI_C_C3_desc,
        UTI_C_lit_desc=UTI_C_lit_desc,
        UTI_C_dup_desc=UTI_C_dup_desc,
        summary=summary,
        badge_dist=badge_dist,
        badge_dist_url=badge_dist_url,
        badge_av=badge_av,
        badge_av_url=badge_av_url,
        badge_mass=badge_mass,
        badge_mass_url=badge_mass_url,
        badge_feh=badge_feh,
        badge_feh_url=badge_feh_url,
        badge_age=badge_age,
        badge_age_url=badge_age_url,
        badge_bss=badge_bss,
        badge_bss_url=badge_bss_url,
        badge_nofpars_url=badge_nofpars_url,
        comments=comments,
        class3=abcd_c,
        dens_val=str(UCC_cl["dens_pc2"]),
        N_50_val=str(int(UCC_cl["N_50"])),
        scix_url=cl_names[0].replace(" ", "%20"),
        posit_table=posit_table,
        cds_radec=cds_radec,
        carousel=carousel,
        fpars_table=fpars_table,
        note_asterisk=mult_vals_note_flag,
        shared_table=shared_table,
    )

    return contents


# def UTI_to_hex(x: float, cmap, soft: float = 0.65) -> str:
#     """
#     Map a value in [0, 1] to a pastel hex color using the RdYlGn colormap.

#     Parameters
#     ----------
#     x : float
#         Value between 0 and 1.

#     Returns
#     -------
#     str
#         Hex color code.
#     """
#     # Get color from colormap (RGBA in [0,1])
#     rgba = cmap(max(0.0, min(1.0, x)))

#     # Convert to pastel by blending toward white
#     pastel = tuple(soft + (1 - soft) * c for c in rgba[:3])  # soften colors

#     return mcolors.to_hex(pastel)


def positions_in_lit(DBs_json, DBs_full_data, row_UCC, tsp):
    """ """
    DBs_sort = row_UCC["DB"].split(";")[::-1]
    DBs_i_sort = row_UCC["DB_i"].split(";")[::-1]

    table = (
        f"{tsp}| Reference | Year | RA [deg] | DEC [deg] | Plx [mas] | pmRA [mas/yr] | pmDE [mas/yr] | Rv [km/s] |\n"
        + "    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )

    # Add UCC positions
    table += f"{tsp}| **UCC** |" + '<span class="hidden-cell-val">99999</span>' + "-- |"
    for col in ("RA_ICRS_m", "DE_ICRS_m", "Plx_m", "pmRA_m", "pmDE_m", "Rv_m"):
        val = "--"
        if row_UCC[col] != "" and not np.isnan(row_UCC[col]):
            val = str(round(float(row_UCC[col]), 3))
        table += val + " | "
    table = table[:-1] + "\n"

    for i, db in enumerate(DBs_sort):
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
            ref_url = f"[{DBs_json[db]['authors']}]({DBs_json[db]['SCIX_url']})"
            year = DBs_json[db]["year"]
            # The spaces at the beginning is to add proper spacing to the final .md file
            table += tsp + "|" + ref_url + " | " + year + " | "
            # Close and add row
            table += row_in[:-1] + "\n"

    table = table[:-2]

    # #
    # table += (
    #     """    | <label for="toggle-pos-rows" class="toggle-btn"></label> | | | | | | |"""
    # ) + "\n"

    return table


def fpars_in_lit(
    DBs_json: dict,
    UCC_cl: dict,
    tsp: str,
):
    """
    DBs_json: JSON that contains the data for all DBs
    DBs_full_data: dictionary of pandas.DataFrame for all DBs
    DBs: DBs where this cluster is present
    DBs_i: Indexes where this cluster is present in each DB
    """

    mult_vals_note_flag = "false"

    # Reverse years order to show the most recent first
    DBs_lst = UCC_cl["DB"].split(";")[::-1]
    par_dict = {}
    for par in fpars_order:
        par_dict[par] = str(UCC_cl[par]).split(";")[::-1]

    rows = []
    for i, db in enumerate(DBs_lst):
        row = [f"[{DBs_json[db]['authors']}]({DBs_json[db]['SCIX_url']})"]
        row.append(DBs_json[db]["year"])

        for par in fpars_order:
            par_v = "--" if par_dict[par][i] == "nan" else par_dict[par][i]
            if "*" in par_v:
                mult_vals_note_flag = "true"
                par_v = re.sub(r"\*+", f"<sup>({par_v.count('*')})</sup>", par_v)
            row.append(par_v)
        # Filter rows where all columns after the year are '--'
        if not all(v == "--" for v in row[2:]):
            rows.append(row)

    if not rows:
        # Return empty values
        return "", mult_vals_note_flag

    def md_row(row):
        return "| " + " | ".join(row) + " |"

    # Properly format UCC median values
    medians = []
    for par in fpars_order:
        val = UCC_cl[par + "_median"]
        if str(val) == "nan":
            medians.append("--")
        else:
            if par in ("age", "mass"):
                medians.append(f"{val:.0f}")
            elif par == "met":
                medians.append(f"{val:.3f}")
            else:
                medians.append(f"{val}")

    UCC_medians = [
        "**UCC**",
        '<span class="hidden-cell-val">99999</span>' + "--",
    ] + medians
    rows = [UCC_medians] + rows

    align = [":---"] + [":---:"] * (len(fpars_headers) - 1)
    table = [
        md_row(fpars_headers),
        "| " + " | ".join(align) + " |",
        *(md_row(r) for r in rows),
    ]
    table = "\n".join(tsp + line for line in table)

    return table, mult_vals_note_flag


def color_C3(abcd):
    """ """
    abcd_c = ""
    line = r"""<span style="color: {}; font-weight: bold;">{}</span>"""
    cc = {"A": "green", "B": "#FFC300", "C": "red", "D": "purple"}
    for letter in abcd:
        abcd_c += line.format(cc[letter], letter)  # + '\n'
    return abcd_c


def table_shared_members(df_UCC, fnames_all, row, tsp):
    """ """
    shared_members_tab = ""

    if str(row["shared_members"]) == "nan":
        return shared_members_tab

    shared_members_tab = (
        tsp
        + """| Cluster | <span title="Percentage of members that this OC shares with the ones listed">%</span>   | RA   | DEC   | Plx   | pmRA  | pmDE  | Rv | UTI |\n"""
    )
    shared_members_tab += (
        tsp + "| :-: | :-: |:-: | :-: | :-: | :-: | :-: | :-: | :-: |\n"
    )

    shared_fnames = row["shared_members"].split(";")
    shared_percents = row["shared_members_p"].split(";")

    for i, fname in enumerate(shared_fnames):
        # Locate OC with shared members in the UCC
        j = fnames_all.index(fname)

        name = df_UCC["Names"][j].split(";")[0]
        vals = []
        for col in (
            "RA_ICRS_m",
            "DE_ICRS_m",
            "Plx_m",
            "pmRA_m",
            "pmDE_m",
            "Rv_m",
            "UTI",
        ):
            val = round(float(df_UCC[col][j]), 2)
            if np.isnan(val):
                vals.append("--")
            else:
                vals.append(val)
        val = shared_percents[i]
        if val == "nan":
            vals.append("--")
        else:
            vals.append(val)

        ra, dec, plx, pmRA, pmDE, Rv, UTI, perc = vals

        shared_members_tab += f"{tsp}|[{name}](/_clusters/{fname}/)| "
        shared_members_tab += f"{perc} | "
        shared_members_tab += f"{ra} | "
        shared_members_tab += f"{dec} | "
        shared_members_tab += f"{plx} | "
        shared_members_tab += f"{pmRA} | "
        shared_members_tab += f"{pmDE} | "
        shared_members_tab += f"{Rv} |"
        shared_members_tab += f"{UTI} |\n"

    # Remove final new line
    shared_members_tab = shared_members_tab[:-1]

    return shared_members_tab
