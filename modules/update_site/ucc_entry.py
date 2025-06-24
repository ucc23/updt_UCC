from pathlib import Path

import numpy as np

# from HARDCODED import (
#     plots_folder,
#     root_UCC_path,
# )
# from modules import combine_UCC_new_DB
from ..utils import date_order_DBs

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
     src="https://raw.githubusercontent.com/ucc23/QFOLD/main/plots/FNAME_aladin.webp"
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
<table style="text-align: center; width:355px;height:250px;">
  <!-- Row 1 (title) -->
  <tr>
    <td colspan="5"><h3>UCC_ID</h3></td>
  </tr>
  <!-- Row 2 -->
  <tr>
    <th><a href="https://ucc.ar/faq#what-are-the-c1-c2-and-c3-parameters" title="Photometric class">C1</a></th>
    <th><a href="https://ucc.ar/faq#what-are-the-c1-c2-and-c3-parameters" title="Density class">C2</a></th>
    <th><a href="https://ucc.ar/faq#what-are-the-c1-c2-and-c3-parameters" title="Combined class">C3</a></th>
    <th><div title="Stars with membership probability >50%">N_50</div></th>
    <th><div title="Radius that contains half the members [arcmin]">r_50</div></th>
  </tr>
  <!-- Row 3 -->
  <tr>
    <td>Class1</td>
    <td>Class2</td>
    <td>Class3</td>
    <td>N_50_val</td>
    <td>r_50_val</td>
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

close_OCs = """\n
#### Objects with similar coordinates

| Cluster | RA    | DEC   | Plx   | pmRA  | pmDE  | Rv    |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: |
"""

# data_foot = """\n
# <br>
# <font color="b3b1b1"><i>Last modified: {}</i></font>
# """


def make(
    UCC_cl, fname0, Qfold, posit_table, img_cont, fpars_table, abcd_c, close_table
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
        .replace("Class1", str(UCC_cl["C1"]))
        .replace("Class2", str(UCC_cl["C2"]))
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

    txt += (
        cluster_region_plot.replace("FOCNAME", fname0)
        .replace("RAICRS", str(round(UCC_cl["RA_ICRS"], 2)))
        .replace("DEICRS", str(round(UCC_cl["DE_ICRS"], 2)))
        .replace("R50", str(UCC_cl["r_50"]))
        .replace("PLX", str(UCC_cl["Plx_m"]))
        .replace("OCNAME", str(cl_names[0]))
    )

    if close_table != "":
        txt += close_OCs + close_table

    contents = "".join(txt)

    return contents


def positions_in_lit(DBs_json, DBs_full_data, row_UCC):
    """ """
    # Re-arrange DBs by year
    DBs_sort, DBs_i_sort = date_order_DBs(row_UCC["DB"], row_UCC["DB_i"])
    DBs_sort, DBs_i_sort = DBs_sort.split(";"), DBs_i_sort.split(";")

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

    # Check present plots. The order here determines the order in which plots will be
    # shown: UCC --> HUNT23 --> CANTAT20
    dbs_plots = []
    for _db in ("HUNT23", "CANTAT20"):
        plot_fpath = Path(
            root_UCC_path + Qfold + plots_folder + fname0 + f"_{_db}.webp"
        )
        if plot_fpath.is_file() is True:
            dbs_plots.append(_db)

    if len(dbs_plots) == 0:
        img_cont = (
            f"![{cl_name}](https://raw.githubusercontent.com/ucc23/{Qfold}"
            + f"main/plots/{fname0}.webp)"
        )
    else:
        img_cont = """<div class="carousel">\n"""
        img_cont += """<input type="radio" name="radio-btn" id="slide1" checked>\n"""
        for _ in range(len(dbs_plots)):
            img_cont += f"""<input type="radio" name="radio-btn" id="slide{_ + 2}">\n"""
        img_cont += """<div class="slides">\n"""
        img_cont += """<div class="slide">\n"""
        img_cont += (
            f"""<a href="https://raw.githubusercontent.com/ucc23/{Qfold}"""
            + f"""main/plots/{fname0}.webp" target="_blank">\n"""
        )
        img_cont += f"""<img src="https://raw.githubusercontent.com/ucc23/{Qfold}"""
        img_cont += (
            f"""main/plots/{fname0}.webp" alt="{cl_name} UCC">\n</a>\n</div>\n"""
        )
        for _db in dbs_plots:
            img_cont += """<div class="slide">\n"""
            img_cont += (
                f"""<a href="https://raw.githubusercontent.com/ucc23/{Qfold}"""
                + f"""main/plots/{fname0}_{_db}.webp" target="_blank">\n"""
            )
            img_cont += f"""<img src="https://raw.githubusercontent.com/ucc23/{Qfold}"""
            img_cont += f"""main/plots/{fname0}_{_db}.webp" alt="{cl_name} {_db}">\n</a>\n</div>\n"""
        img_cont += """</div>\n<div class="indicators">\n"""
        img_cont += """<label for="slide1">1</label>\n"""
        for _ in range(len(dbs_plots)):
            img_cont += f"""<label for="slide{_ + 2}">{_ + 2}</label>\n"""
        img_cont += """</div>\n</div>"""

    with open("/home/gabriel/Descargas/imag_cont.txt", "w") as text_file:
        text_file.write(img_cont)

    return img_cont


def close_cat_cluster(df_UCC, row):
    """ """
    close_table = ""

    if str(row["close_entries"]) == "nan":
        return close_table

    fnames0 = [_.split(";")[0] for _ in df_UCC["fnames"]]

    dups_fnames = row["close_entries"].split(";")
    # dups_probs = row["dups_probs_m"].split(";")

    for i, fname in enumerate(dups_fnames):
        j = fnames0.index(fname)
        name = df_UCC["ID"][j].split(";")[0]

        vals = []
        for col in ("RA_ICRS_m", "DE_ICRS_m", "Plx_m", "pmRA_m", "pmDE_m", "Rv_m"):
            val = round(float(df_UCC[col][j]), 3)
            if np.isnan(val):
                vals.append("--")
            else:
                vals.append(val)
        # val = round(float(dups_probs[i]), 3)
        # if np.isnan(val):
        #     vals.append("--")
        # else:
        #     vals.append(val)

        ra, dec, plx, pmRA, pmDE, Rv = vals

        close_table += f"|[{name}](/_clusters/{fname}/)| "
        # close_table += f"{int(100 * prob)} | "
        close_table += f"{ra} | "
        close_table += f"{dec} | "
        close_table += f"{plx} | "
        close_table += f"{pmRA} | "
        close_table += f"{pmDE} | "
        close_table += f"{Rv} |\n"
    # Remove final new line
    close_table = close_table[:-1]

    return close_table


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
    DBs_w_pars, DBs_i_w_pars = date_order_DBs(
        ";".join(DBs_w_pars), ";".join(DBs_i_w_pars)
    )
    DBs_w_pars, DBs_i_w_pars = DBs_w_pars.split(";"), DBs_i_w_pars.split(";")

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
            # Read parameter value from DB as string
            par_v = str(df[par][int(DBs_i_w_pars[i])])

            # Trim parameter value
            if ";" in par_v:
                # DBs like SANTOS21 list more than 1 value per parameter
                par_v = par_v[: int(max_chars * 2)]
            else:
                par_v = par_v[:max_chars]

            # Remove empty spaces from param value (if any)
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
