from datetime import datetime
from pathlib import Path

import numpy as np
from HARDCODED import (
    plots_folder,
    root_UCC_path,
)

from modules import combine_UCC_new_DB

header = """---
layout: post
title:  {}
---
"""

more_names = """<h3><span style="color: #808080;"><i>(cl_names_str)</i></span></h3>"""


aladin_snippet = r"""<div style="display: flex; justify-content: space-between; width:720px;height:250px">
<div style="text-align: center;">
<!-- WEBP image -->
<img id="myImage" src="https://raw.githubusercontent.com/ucc23/QFOLD/main/plots/FNAME_aladin.webp" alt="Clickable Image" style="width:355px;height:250px; cursor: pointer;">

<!-- Div to contain Aladin Lite viewer -->
<div id="aladin-lite-div" style="width:355px;height:250px;display:none;"></div>

<!-- Aladin Lite script (will be loaded after the image is clicked) -->
<script type="text/javascript">
// Function to load Aladin Lite after image click and hide the image
function loadAladinLiteAndHideImage() {
    // Dynamically load the Aladin Lite script
    let aladinScript = document.createElement('script');
    aladinScript.src = "https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js";
    aladinScript.charset = "utf-8";
    aladinScript.onload = function () {
        A.init.then(() => {
            let aladin = A.aladin('#aladin-lite-div', {survey:"P/DSS2/color", fov:RAD_DEG, target: "RA_ICRS DE_ICRS"});
            // Remove the image
            document.getElementById('myImage').remove();
            // Hide the image
            //document.getElementById('myImage').style.visibility = "hidden";
            // Show the Aladin Lite viewer
            document.getElementById('aladin-lite-div').style.display = 'block';
        });
     };
    document.head.appendChild(aladinScript);
}
// Event listener for image click
document.getElementById('myImage').addEventListener('click', loadAladinLiteAndHideImage);
</script>
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
   <span style="color: #99180f; font-weight: bold;">Warning: </span><span>no stars with <i>P>0.5</i> were found</span>
</div>
"""

nasa_simbad_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Search object in <a data-umami-event="nasa_search" href="https://ui.adsabs.harvard.edu/search/q=%20collection%3Aastronomy%20body%3A%22XXNASAXX%22&sort=date%20desc%2C%20bibcode%20desc&p_=0" target="_blank">NASA/SAO ADS</a> | <a data-umami-event="simbad_search" href="https://simbad.cds.unistra.fr/simbad/sim-id-refs?Ident=XXSIMBADXX" target="_blank">Simbad</a></p>
"""

#  Mobile url
# https://simbad.cds.unistra.fr/mobile/bib_list.html?ident=

posit_table_top = """\n
### Position in UCC and published works (not exhaustive)

| Reference    | RA    | DEC   | Plx  | pmRA  | pmDE   |  Rv  |
| :---         | :---: | :---: | :---: | :---: | :---: | :---: |
"""

cds_simbad_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Search coordinates in <a data-umami-event="cds_coord_search" href="https://cdsportal.u-strasbg.fr/?target=RADEC_CDS" target="_blank">CDS</a> | <a data-umami-event="simbad_coord_search" href="https://simbad.cds.unistra.fr/mobile/object_list.html?coord=RADEC_SMB&output=json&radius=5&userEntry=XCLUSTX" target="_blank">Simbad</a></p>
"""

cl_plot = """
### Plots for selected probable members

{}

"""

notebook_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Explore data in <a data-umami-event="colab" href="https://colab.research.google.com/github/UCC23/{}/blob/master/notebooks/{}.ipynb" target="_blank">Colab</a></p>
"""

fpars_table_top = """\n
### Fundamental parameters in literature (not exhaustive)

| Reference |  Fundamental parameters |
| :---         |     :---:      |
"""

close_table_top = """\n
### Probable <a href="https://ucc.ar/faq#probable-duplicates" title="See FAQ for definition of proximity">duplicates</a> and clusters in proximity

| Cluster | P (%) | RA    | DEC   | Plx   | pmRA  | pmDE  | Rv    |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

data_foot = """\n
<br>
<font color="b3b1b1"><i>Last modified: {}</i></font>
"""


def main(df_UCC, UCC_cl, DBs_json, DBs_full_data, fname0, Qfold):
    """ """
    # DBs where this cluster is present
    DBs = UCC_cl["DB"].split(";")

    # Indexes where this cluster is present in each DB
    DBs_i = UCC_cl["DB_i"].split(";")

    cl_names = UCC_cl["ID"].split(";")

    posit_table = positions_in_lit(DBs_json, DBs_full_data, DBs, DBs_i, UCC_cl)
    img_cont = carousel_div(cl_names[0], Qfold, fname0)
    close_table = close_cat_cluster(df_UCC, UCC_cl)
    fpars_table = fpars_in_lit(DBs_json, DBs_full_data, DBs, DBs_i)

    # Color used by the 'C1' classification
    abcd_c = UCC_color(UCC_cl["C3"])

    # Start generating the md file
    txt = ""
    txt += header.format(cl_names[0])
    if len(cl_names) > 1:
        txt += more_names.replace("cl_names_str", "; ".join(cl_names[1:]))

    rad_deg = round(2 * (UCC_cl["r_50"] / 60.0), 3)
    txt += (
        aladin_snippet.replace("QFOLD", str(Qfold))
        .replace("FNAME", str(fname0))
        .replace("RAD_DEG", str(rad_deg))
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

    if UCC_cl["N_50"] == 0:
        txt += no_membs_50_warning

    txt += nasa_simbad_url.replace("XXNASAXX", cl_names[0].replace(" ", "%20")).replace(
        "XXSIMBADXX", fname0
    )

    txt += posit_table_top
    txt += posit_table

    if close_table != "":
        txt += close_table_top
        txt += close_table + "\n"

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

    txt += notebook_url.format(Qfold, fname0)

    if fpars_table != "":
        txt += fpars_table_top
        txt += fpars_table

    txt += data_foot.format(datetime.today().strftime("%Y-%m-%d"))
    contents = "".join(txt)

    return contents


def positions_in_lit(DBs_json, DBs_full_data, DBs, DBs_i, row_UCC):
    """ """
    # Re-arrange DBs by year
    DBs_years = [_.split("_")[0][-2:] for _ in DBs]
    # Sort
    sort_idxs = combine_UCC_new_DB.sort_year_digits(DBs_years)
    # Re-arrange
    DBs_sort = np.array(DBs)[sort_idxs]
    DBs_i_sort = np.array(DBs_i)[sort_idxs]

    table = ""
    for i, db in enumerate(DBs_sort):
        # Full 'db' database
        df = DBs_full_data[db]

        # Add positions
        row_in = ""
        for c in DBs_json[db]["pos"].split(","):
            if c != "None":
                # Read position as string
                pos_v = str(df[c][int(DBs_i_sort[i])])
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
            table += "|" + DBs_json[db]["ref"] + " | "
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


def carousel_div(cl_name, Qfold, fname0):
    """Generate the plot or carousel of plots, depending on whether there is more than
    one plot for a cluster.
    """

    # Check present plots. The order here determines the order in which plots will be
    # shown: UCC --> HUNT23 --> CANTAT20
    dbs_plots = []
    for _db in ("HUNT23", "CANTAT20"):
        plot_fpath = Path(
            root_UCC_path + Qfold + f"/{plots_folder}/" + fname0 + f"_{_db}.webp"
        )
        if plot_fpath.is_file() is True:
            dbs_plots.append(_db)

    if len(dbs_plots) == 0:
        img_cont = (
            f"![{cl_name}](https://raw.githubusercontent.com/ucc23/{Qfold}"
            + f"/main/plots/{fname0}.webp)"
        )
    else:
        img_cont = """<div class="carousel">\n"""
        img_cont += """<input type="radio" name="radio-btn" id="slide1" checked>\n"""
        for _ in range(len(dbs_plots)):
            img_cont += f"""<input type="radio" name="radio-btn" id="slide{_+2}">\n"""
        img_cont += """<div class="slides">\n"""
        img_cont += """<div class="slide">\n"""
        img_cont += (
            f"""<a href="https://raw.githubusercontent.com/ucc23/{Qfold}"""
            + f"""/main/plots/{fname0}.webp" target="_blank">\n"""
        )
        img_cont += f"""<img src="https://raw.githubusercontent.com/ucc23/{Qfold}"""
        img_cont += (
            f"""/main/plots/{fname0}.webp" alt="{cl_name} UCC">\n</a>\n</div>\n"""
        )
        for _db in dbs_plots:
            img_cont += """<div class="slide">\n"""
            img_cont += (
                f"""<a href="https://raw.githubusercontent.com/ucc23/{Qfold}"""
                + f"""/main/plots/{fname0}_{_db}.webp" target="_blank">\n"""
            )
            img_cont += f"""<img src="https://raw.githubusercontent.com/ucc23/{Qfold}"""
            img_cont += f"""/main/plots/{fname0}_{_db}.webp" alt="{cl_name} {_db}">\n</a>\n</div>\n"""
        img_cont += """</div>\n<div class="indicators">\n"""
        img_cont += """<label for="slide1">1</label>\n"""
        for _ in range(len(dbs_plots)):
            img_cont += f"""<label for="slide{_+2}">{_+2}</label>\n"""
        img_cont += """</div>\n</div>"""

    with open("/home/gabriel/Descargas/imag_cont.txt", "w") as text_file:
        text_file.write(img_cont)

    return img_cont


def close_cat_cluster(df_UCC, row):
    """ """
    close_table = ""

    if str(row["dups_fnames_m"]) == "nan":
        return close_table

    fnames0 = [_.split(";")[0] for _ in df_UCC["fnames"]]

    dups_fnames = row["dups_fnames_m"].split(";")
    dups_probs = row["dups_probs_m"].split(";")

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
        val = round(float(dups_probs[i]), 3)
        if np.isnan(val):
            vals.append("--")
        else:
            vals.append(val)

        ra, dec, plx, pmRA, pmDE, Rv, prob = vals

        close_table += f"|[{name}](/_clusters/{fname}/)| "
        close_table += f"{int(100 * prob)} | "
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
    DBs_json: dict, DBs_full_data: dict, DBs: list, DBs_i: list, max_chars=7
):
    """
    DBs_json: JSON that contains the data for all DBs
    DBs_full_data: dictionary of pandas.DataFrame for all DBs
    DBs: DBs where this cluster is present
    DBs_i: Indexes where this cluster is present in each DB
    """
    # Select DBs with parameters
    DBs_w_pars, DBs_i_w_pars = [], []
    for i, db in enumerate(DBs):
        # If this database contains any estimated fundamental parameters
        if DBs_json[db]["pars"] != "":
            DBs_w_pars.append(db)
            DBs_i_w_pars.append(DBs_i[i])

    if len(DBs_w_pars) == 0:
        table = ""
        return table
    else:
        # Re-arrange DBs by year
        # Extract DB year
        # This splits the DB name in a '_' and keeps the first part. It is meant to handle
        # DBs with names such as: 'SMITH23_3'.
        DBs_years = [_.split("_")[0][-2:] for _ in DBs_w_pars]
        # Sort
        sort_idxs = combine_UCC_new_DB.sort_year_digits(DBs_years)
        # Re-arrange
        DBs_w_pars = np.array(DBs_w_pars)[sort_idxs]
        DBs_i_w_pars = np.array(DBs_i_w_pars)[sort_idxs]

    txt = ""
    for i, db in enumerate(DBs_w_pars):
        # Full 'db' database
        df = DBs_full_data[db]
        # Add reference
        txt_db = "| " + DBs_json[db]["ref"] + " | "

        txt_pars = ""
        # Add non-nan parameters
        for par in DBs_json[db]["pars"].split(","):
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
