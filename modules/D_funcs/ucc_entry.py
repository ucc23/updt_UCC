import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from ..variables import plots_folder, root_ucc_path

header = """---
layout: post
title: {}
style: style
title_flag: true
---
"""

more_names = """<h3><span style="color: #808080;"><i>(cl_names_str)</i></span></h3>"""

aladin_snippet = r"""<div style="display: flex; justify-content: space-between; width:720px;height:250px">
<div style="text-align: center;">

<!-- Static image + data attributes for FOV and target -->
<img id="aladin_img"
     data-umami-event="aladin_load"
     src="https://raw.githubusercontent.com/ucc23/FOLD/main/aladin/FNAME.webp"
     alt="Click to load Aladin Lite" 
     style="width:355px;height:250px; cursor: pointer;"
     data-fov="FOV_VAL" 
     data-target="RA_ICRS DE_ICRS"/>
<!-- Div to contain Aladin Lite viewer -->
<div id="aladin-lite-div" style="width:355px;height:250px;display:none;"></div>
<!-- Aladin Lite script (will be loaded after the image is clicked) -->
<script src="{{ site.baseurl }}/scripts/aladin_load.js"></script>

</div>
<!-- Left block -->
"""

table_right_col = """
<table style="width:355px;height:250px;">
  <!-- Row 1 (title) -->
  <tr style="background-color: UTI_COLOR;">
      <td colspan="5">
      <div class="popup-container-focus" style="display: flex; justify-content: center; align-items: center; gap: 0.5em;">
          <button class="popup-button-focus"><a href="{{ site.baseurl }}/faq#what-is-the-uti-parameter" title="UTI parameter">UTI</a>: <span class="uti_value" data-umami-event="UTI">UCC_UTI</span></button>
          <div class="popup-message-focus">
              <strong>UCC Trust Index</strong><br><br>
                <table>
                  <tr style="background-color: #fafafa;">
                    <td style="text-align: center;">C_N</td>
                    <td style="text-align: center;">C_dens</td>
                    <td style="text-align: center;">C_C3</td>
                    <td style="text-align: center;">C_lit</td>
                    <td style="text-align: center;">C_dup</td>
                  </tr>
                  <tr>
                      <td style="text-align: center; background-color: UTI_C_N_COL; font-weight: bold;">UTI_C_N</td>
                      <td style="text-align: center; background-color: UTI_C_dens_COL; font-weight: bold;">UTI_C_dens</td>
                      <td style="text-align: center; background-color: UTI_C_C3_COL; font-weight: bold;">UTI_C_C3</td>
                      <td style="text-align: center; background-color: UTI_C_lit_COL; font-weight: bold;">UTI_C_lit</td>
                      <td style="text-align: center; background-color: UTI_C_dup_COL; font-weight: bold;">UTI_C_dup</td>
                  </tr>
                </table><br>
              <!-- Object summary follows below -->
              SUMMARY
          </div>
        </div>
    </td>
  </tr>
  <!-- Row 2 -->
  <tr>
    <th style="text-align: center;"><a href="https://ucc.ar/faq#what-is-the-c3-parameter" title="C3 class">C3</a></th>
    <th style="text-align: center;"><div title="Stars with membership probability >50%">N_50</div></th>
    <th style="text-align: center;"><div title="Radius that contains half the members [arcmin]">r_50</div></th>
  </tr>
  <!-- Row 3 -->
  <tr>
    <td style="text-align: center;">Class3</td>
    <td style="text-align: center;">N_50_val</td>
    <td style="text-align: center;">r_50_val</td>
  </tr>
</table>
</div>
"""

nasa_simbad_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Search object in <a data-umami-event="nasa_search" href="https://ui.adsabs.harvard.edu/search/q=%20collection%3Aastronomy%20body%3A%22XXNASAXX%22&sort=date%20desc%2C%20bibcode%20desc&p_=0" target="_blank">NASA/SAO ADS</a> | <a data-umami-event="simbad_search" href="https://simbad.cds.unistra.fr/simbad/sim-id-refs?Ident=XXSIMBADXX" target="_blank">Simbad</a></p>
"""

#  Mobile url
# https://simbad.cds.unistra.fr/mobile/bib_list.html?ident=

posit_table_top = """\n
### Positions

| Reference    | RA    | DEC   | Plx  | pmRA  | pmDE   |  Rv  |
| :---         | :---: | :---: | :---: | :---: | :---: | :---: |
"""

cds_simbad_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Search coordinates in <a data-umami-event="cds_coord_search" href="https://cdsportal.u-strasbg.fr/?target=RADEC_CDS" target="_blank">CDS</a> | <a data-umami-event="simbad_coord_search" href="https://simbad.cds.unistra.fr/mobile/object_list.html?coord=RADEC_SMB&output=json&radius=5&userEntry=XCLUSTX" target="_blank">Simbad</a></p>
"""

cl_plot = """
### Estimated members

{}

"""

notebook_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Explore data in <a data-umami-event="colab" href="https://colab.research.google.com/github/ucc23/ucc/blob/main/assets/notebook.ipynb" target="_blank">Colab</a></p>
"""

fpars_table_top = """\n
### Fundamental parameters

| Reference |  Values |
| :---      |  :---:  |
"""

bayestar_url = """\n
> <p style="text-align:center; font-weight: bold; font-size:20px">Search coordinates in <a data-umami-event="bayestar" href="http://argonaut.skymaps.info/query?lon=XXGLONXX%20&lat=XXGLATXX&coordsys=gal&mapname=bayestar2019" target="_blank">Bayestar19</a></p>
"""

cluster_region_plot = """\n
### Cluster region

<html lang="en">
  <body>
    <center>
    <div id="plot-params"
         data-oc-name="FOCNAME"
         data-ra-center="RAICRS"
         data-dec-center="DEICRS"
         data-rad-deg="R50"
         data-plx="PLX">
    </div>
    <div id="plot-container">
        <div id="plot"></div>
    </div>
    <script defer type="module" src="{{ site.baseurl }}/scripts/radec_scatter.js"></script>
    </center>
  </body>
</html>
<br>
"""

shared_OCs = """\n
#### Objects with shared members

| Cluster | <span title="Percentage of members that this OC shares with the ones listed">%</span>   | RA   | DEC   | Plx   | pmRA  | pmDE  | Rv    |
| :---:   | :-: |:---: | :---: | :---: | :---: | :---: | :---: |
"""


def make(UCC_cl, fname0, posit_table, img_cont, abcd_c, fpars_table, shared_table):
    """ """
    cl_names = UCC_cl["Names"].split(";")

    # Start generating the md file
    txt = ""
    txt += header.format(cl_names[0])
    if len(cl_names) > 1:
        txt += more_names.replace("cl_names_str", "; ".join(cl_names[1:]))

    fov = round(2 * (UCC_cl["r_50"] / 60.0), 3)
    txt += (
        aladin_snippet.replace("FOLD", f"plots_{fname0[0]}")
        .replace("FNAME", str(fname0))
        .replace("FOV_VAL", str(fov))
        .replace("RA_ICRS", str(UCC_cl["RA_ICRS_m"]))
        .replace("DE_ICRS", str(UCC_cl["DE_ICRS_m"]))
    )

    C_N, C_dens, C_C3, C_lit, C_dup = (
        UCC_cl["C_N"],
        UCC_cl["C_dens"],
        UCC_cl["C_C3"],
        UCC_cl["C_lit"],
        UCC_cl["C_dup"],
    )
    txt += (
        table_right_col.replace("UTI_COLOR", UTI_to_hex(UCC_cl["UTI"]))
        .replace("UCC_UTI", str(UCC_cl["UTI"]))
        .replace("UTI_C_N_COL", UTI_to_hex(C_N))
        .replace("UTI_C_dens_COL", UTI_to_hex(C_dens))
        .replace("UTI_C_C3_COL", UTI_to_hex(C_C3))
        .replace("UTI_C_lit_COL", UTI_to_hex(C_lit))
        .replace("UTI_C_dup_COL", UTI_to_hex(C_dup))
        .replace("UTI_C_N", str(C_N))
        .replace("UTI_C_dens", str(C_dens))
        .replace("UTI_C_C3", str(C_C3))
        .replace("UTI_C_lit", str(C_lit))
        .replace("UTI_C_dup", str(C_dup))
        .replace(
            "SUMMARY",
            summarize_object(
                cl_names[0],
                UCC_cl["DB"],
                UCC_cl["N_50"],
                UCC_cl["Plx_m"],
                UCC_cl["shared_members_p"],
                C_N,
                C_dens,
                C_C3,
                C_lit,
                C_dup,
            ),
        )
        .replace("Class3", abcd_c)
        .replace("r_50_val", str(UCC_cl["r_50"]))
        .replace("N_50_val", str(int(UCC_cl["N_50"])))
    )

    txt += nasa_simbad_url.replace("XXNASAXX", cl_names[0].replace(" ", "%20")).replace(
        "XXSIMBADXX", fname0
    )

    txt += posit_table_top
    txt += posit_table

    signed_dec = (
        "+" + str(UCC_cl["DE_ICRS_m"])
        if UCC_cl["DE_ICRS_m"] >= 0
        else str(UCC_cl["DE_ICRS_m"])
    )
    txt += (
        cds_simbad_url.replace(
            "RADEC_CDS", "{},{}".format(UCC_cl["RA_ICRS_m"], signed_dec)
        )
        .replace(
            "RADEC_SMB", "{}%20{}".format(UCC_cl["RA_ICRS_m"], UCC_cl["DE_ICRS_m"])
        )
        .replace("XCLUSTX", fname0)
    )

    txt += cl_plot.format(img_cont)

    txt += notebook_url

    if fpars_table != "":
        txt += fpars_table_top
        txt += fpars_table

    txt += bayestar_url.replace("XXGLONXX", str(UCC_cl["GLON_m"])).replace(
        "XXGLATXX", str(UCC_cl["GLAT_m"])
    )

    txt += (
        cluster_region_plot.replace("FOCNAME", fname0)
        .replace("RAICRS", str(round(UCC_cl["RA_ICRS_m"], 2)))
        .replace("DEICRS", str(round(UCC_cl["DE_ICRS_m"], 2)))
        .replace("R50", str(UCC_cl["r_50"]))
        .replace("PLX", str(UCC_cl["Plx_m"]))
        .replace("OCNAME", str(cl_names[0]))
    )

    if shared_table != "":
        txt += shared_OCs + shared_table

    contents = "".join(txt)

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
    name, cl_DB, N_50, plx, shared_members_p, C_N, C_dens, C_C3, C_lit, C_dup
):
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

    txt_warn = (
        """<br><br><span style="color: #99180f; font-weight: bold;">Warning: </span>"""
    )

    dup_warn, dup_amount = "<br><br>", ""
    if duplication == "a unique":
        if str(shared_members_p) != "nan:":
            shared_members_p = shared_members_p.split(";")
            N_shared = len(shared_members_p)
            max_p = max(map(float, shared_members_p))
            if max_p > 10:
                entr, upto = "entries", "up to "
                if N_shared == 1:
                    N_shared, entr, upto = "a", "entry", ""
                summary += f"This is {duplication} object, with {N_shared} later "
                summary += f"reported {entr} sharing {upto}{max_p:.0f}% of its members."
    elif duplication == "very likely a unique":
        dup_amount = "very small"
    elif duplication == "likely a unique":
        dup_amount = "small"
    elif duplication == "possibly a duplicated":
        dup_warn, dup_amount = txt_warn, "moderate"
    elif duplication in ("likely a duplicate", "very likely a duplicate"):
        dup_warn, dup_amount = txt_warn, "significant"

    if dup_amount != "":
        summary += dup_warn
        summary += f"This is {duplication} object, which shares a {dup_amount}"
        summary += " percentage of members with a previously reported entry."

    if N_50 < 25:
        summary += txt_warn + "contains less than 25 stars with <i>P>0.5</i> estimated."

    return summary


def positions_in_lit(DBs_json, DBs_full_data, row_UCC):
    """ """
    DBs_sort, DBs_i_sort = row_UCC["DB"].split(";"), row_UCC["DB_i"].split(";")

    table = ""
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
            ref_url = f"[{DBs_json[db]['authors']} ({DBs_json[db]['year']})]({DBs_json[db]['ADS_url']})"
            table += "|" + ref_url + " | "
            # Close and add row
            table += row_in[:-1] + "\n"

    # Add UCC positions
    table += "| **UCC** |"
    for col in ("RA_ICRS_m", "DE_ICRS_m", "Plx_m", "pmRA_m", "pmDE_m", "Rv_m"):
        val = "--"
        if row_UCC[col] != "" and not np.isnan(row_UCC[col]):
            val = str(round(float(row_UCC[col]), 3))
        table += val + " | "
    # Close row
    table = table[:-1] + "\n"

    return table


def carousel_div(cl_name, fname0):
    """Generate the plot or carousel of plots, depending on whether there is more than
    one plot for a cluster.
    """

    def slider_content(_db):
        slide_cont = (
            f"""<a href="https://raw.githubusercontent.com/ucc23/plots_{fname0[0]}/"""
            + f"""main/{_db}{fname0}.webp" target="_blank">\n"""
        )
        slide_cont += (
            f"""<img src="https://raw.githubusercontent.com/ucc23/plots_{fname0[0]}/"""
        )
        slide_cont += (
            f"""main/{_db}{fname0}.webp" alt="{cl_name} {_db[:-1]}">\n</a>\n"""
        )
        return slide_cont

    # Check present plots. The order here determines the order in which plots will be
    # shown: UCC --> HUNT23 --> CANTAT20
    cmd_plots = []
    for _db in ("UCC/", "HUNT23/", "CANTAT20/"):
        plot_fpath = Path(
            root_ucc_path
            + plots_folder
            + f"plots_{fname0[0]}/"
            + _db
            + fname0
            + ".webp"
        )
        if plot_fpath.is_file() is True:
            cmd_plots.append(_db)

    # Only UCC CMD is present
    if len(cmd_plots) == 1:
        img_cont = slider_content(cmd_plots[0])
        return img_cont

    # Initiate carousel
    img_cont = """<div class="carousel">\n"""
    img_cont += """<input type="radio" name="radio-btn" id="slide1" checked>\n"""

    # Buttons
    for _ in range(2, len(cmd_plots) + 1):
        img_cont += f"""<input type="radio" name="radio-btn" id="slide{_}">\n"""

    # Slider
    img_cont += """<div class="slides">\n"""
    for _db in cmd_plots:
        img_cont += """<div class="slide">\n"""
        img_cont += slider_content(_db)
        img_cont += """</div>\n"""

    # Button labels
    img_cont += """</div>\n<div class="indicators">\n"""
    for _ in range(1, len(cmd_plots) + 1):
        img_cont += f"""<label for="slide{_}">{_}</label>\n"""

    img_cont += """</div>\n</div>"""

    # with open("temp_imag_cont.txt", "w") as text_file:
    #     text_file.write(img_cont)
    # breakpoint()

    return img_cont


def table_shared_members(df_UCC, fnames_all, row):
    """ """
    shared_members_tab = ""

    if str(row["shared_members"]) == "nan":
        return shared_members_tab

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

        shared_members_tab += f"|[{name}](/_clusters/{fname}/)| "
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

    if len(DBs_w_pars) == 0:
        table = ""
        return table

    # Re-arrange DBs by year
    # DBs_w_pars, DBs_i_w_pars = date_order_DBs(
    #     ";".join(DBs_w_pars), ";".join(DBs_i_w_pars)
    # )
    # DBs_w_pars, DBs_i_w_pars = DBs_w_pars.split(";"), DBs_i_w_pars.split(";")

    txt = ""
    for i, db in enumerate(DBs_w_pars):
        # Full 'db' database
        df = DBs_full_data[db]
        # Add reference
        ref_url = f"[{DBs_json[db]['authors']} ({DBs_json[db]['year']})]({DBs_json[db]['ADS_url']})"
        txt_db = "| " + ref_url + " | "

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

    table = ""
    if txt != "":
        # Remove final new line
        table = txt[:-1]

    return table


def UCC_color(abcd):
    """ """
    abcd_c = ""
    line = r"""<span style="color: {}; font-weight: bold;">{}</span>"""
    cc = {"A": "green", "B": "#FFC300", "C": "red", "D": "purple"}
    for letter in abcd:
        abcd_c += line.format(cc[letter], letter)  # + '\n'
    return abcd_c
