from pathlib import Path

import numpy as np

# from HARDCODED import (
#     plots_folder,
#     root_UCC_path,
# )
# from modules import combine_UCC_new_DB
# from ..utils import date_order_DBs

header = """---
layout: post
title:  {}
---
"""

more_names = """<h3><span style="color: #808080;"><i>(cl_names_str)</i></span></h3>"""


aladin_snippet = r"""<div style="display: flex; justify-content: space-between; width:720px;height:250px">
<div style="text-align: center;">

<!-- Static image + data attributes for FOV and target -->
<img id="aladin_img"
     data-umami-event="aladin_load"
     src="https://raw.githubusercontent.com/ucc23/QFOLD/main/plots/aladin/FNAME.webp"
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
  <tr>
    <td colspan="5"><h3>UCC_ID</h3></td>
  </tr>
  <!-- Row 2 -->
  <tr>
    <th style="text-align: center;"><a href="https://ucc.ar/faq#what-is-the-c3-parameter" title="Combined class">C3</a></th>
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

no_membs_50_warning = """
<div style="text-align: center;">
   <span style="color: #99180f; font-weight: bold;">Warning: </span><span>less than 25 stars with <i>P>0.5</i> were found</span>
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

# data_foot = """\n
# <br>
# <font color="b3b1b1"><i>Last modified: {}</i></font>
# """


def make(
    UCC_cl, fname0, Qfold, posit_table, img_cont, abcd_c, fpars_table, shared_table
):
    """ """
    cl_names = UCC_cl["ID"].split(";")

    # Start generating the md file
    txt = ""
    txt += header.format(cl_names[0])
    if len(cl_names) > 1:
        txt += more_names.replace("cl_names_str", "; ".join(cl_names[1:]))

    fov = round(2 * (UCC_cl["r_50"] / 60.0), 3)
    txt += (
        aladin_snippet.replace("QFOLD", str(Qfold).replace("/", ""))
        .replace("FNAME", str(fname0))
        .replace("FOV_VAL", str(fov))
        .replace("RA_ICRS", str(UCC_cl["RA_ICRS_m"]))
        .replace("DE_ICRS", str(UCC_cl["DE_ICRS_m"]))
    )

    txt += (
        table_right_col.replace("UCC_ID", UCC_cl["UCC_ID"])
        # .replace("Class1", str(UCC_cl["C1"]))
        # .replace("Class2", str(UCC_cl["C2"]))
        .replace("Class3", abcd_c)
        .replace("r_50_val", str(UCC_cl["r_50"]))
        .replace("N_50_val", str(int(UCC_cl["N_50"])))
    )

    if UCC_cl["N_50"] < 25:
        txt += no_membs_50_warning

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

    txt += notebook_url.format(Qfold.replace("/", ""), fname0)

    if fpars_table != "":
        txt += fpars_table_top
        txt += fpars_table

    txt += bayestar_url.replace("XXGLONXX", str(UCC_cl["GLON_m"])).replace(
        "XXGLATXX", str(UCC_cl["GLAT_m"])
    )

    txt += (
        cluster_region_plot.replace("FOCNAME", fname0)
        .replace("RAICRS", str(round(UCC_cl["RA_ICRS"], 2)))
        .replace("DEICRS", str(round(UCC_cl["DE_ICRS"], 2)))
        .replace("R50", str(UCC_cl["r_50"]))
        .replace("PLX", str(UCC_cl["Plx_m"]))
        .replace("OCNAME", str(cl_names[0]))
    )

    if shared_table != "":
        txt += shared_OCs + shared_table

    contents = "".join(txt)

    return contents


def positions_in_lit(DBs_json, DBs_full_data, row_UCC):
    """ """
    # Re-arrange DBs by year
    # DBs_sort, DBs_i_sort = date_order_DBs(row_UCC["DB"], row_UCC["DB_i"])
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


def carousel_div(root_UCC_path, plots_folder, cl_name, Qfold, fname0):
    """Generate the plot or carousel of plots, depending on whether there is more than
    one plot for a cluster.
    """

    def slider_content(_db):
        slide_cont = (
            f"""<a href="https://raw.githubusercontent.com/ucc23/{Qfold}"""
            + f"""main/{plots_folder}{_db}{fname0}.webp" target="_blank">\n"""
        )
        slide_cont += f"""<img src="https://raw.githubusercontent.com/ucc23/{Qfold}"""
        slide_cont += f"""main/{plots_folder}{_db}{fname0}.webp" alt="{cl_name} {_db[:-1]}">\n</a>\n"""
        return slide_cont

    # Check present plots. The order here determines the order in which plots will be
    # shown: UCC --> HUNT23 --> CANTAT20
    cmd_plots = []
    for _db in ("UCC/", "HUNT23/", "CANTAT20/"):
        plot_fpath = Path(root_UCC_path + Qfold + plots_folder + _db + fname0 + ".webp")
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
    for _ in range(len(cmd_plots)):
        img_cont += f"""<input type="radio" name="radio-btn" id="slide{_ + 1}">\n"""

    # Slider
    img_cont += """<div class="slides">\n"""
    for _db in cmd_plots:
        img_cont += """<div class="slide">\n"""
        img_cont += slider_content(_db)
        img_cont += """</div>\n"""

    # Button labels
    img_cont += """</div>\n<div class="indicators">\n"""
    for _ in range(len(cmd_plots)):
        img_cont += f"""<label for="slide{_ + 1}">{_ + 1}</label>\n"""

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

        name = df_UCC["ID"][j].split(";")[0]
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
