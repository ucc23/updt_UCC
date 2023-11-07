from datetime import datetime


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
> <p style="text-align:center; font-weight: bold; font-size:20px">Search object in <a href="https://ui.adsabs.harvard.edu/search/q=%20collection%3Aastronomy%20body%3A%22XXNASAXX%22&sort=date%20desc%2C%20bibcode%20desc&p_=0" target="_blank">NASA/SAO ADS</a> | <a href="https://simbad.cds.unistra.fr/simbad/sim-id-refs?Ident=XXSIMBADXX" target="_blank">Simbad</a></p>
"""

#  Mobile url
# https://simbad.cds.unistra.fr/mobile/bib_list.html?ident=

posit_table_top = """\n
### Position in UCC and published works (not exhaustive)

| Reference    | RA    | DEC   | plx  | pmRA  | pmDE   |  Rv  |
| :---         | :---: | :---: | :---: | :---: | :---: | :---: |
"""

cds_simbad_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Search coordinates in <a href="https://cdsportal.u-strasbg.fr/?target=RADEC_CDS" target="_blank">CDS</a> | <a href="https://simbad.cds.unistra.fr/mobile/object_list.html?coord=RADEC_SMB&output=json&radius=5&userEntry=XCLUSTX" target="_blank">Simbad</a></p>
"""

cl_plot = """
### Plots for selected probable members

![CLUSTER](https://raw.githubusercontent.com/ucc23/{}/main/plots/{}.webp)

"""

notebook_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Explore data in <a href="https://colab.research.google.com/github/UCC23/{}/blob/master/notebooks/{}.ipynb" target="_blank">Colab</a></p>
"""

fpars_table_top = """\n
### Fundamental parameters in literature (not exhaustive)

| Reference |  Fundamental parameters |
| :---         |     :---:      |
"""

close_table_top = """\n
### Probable <a href="https://ucc.ar/faq#probable-duplicates" title="See FAQ for definition of proximity">duplicates</a> and clusters in proximity

| Cluster | P (%) | RA    | DEC   | plx   | pmRA  | pmDE  | Rv    |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

data_foot = """\n
<br>
<font color="b3b1b1"><i>Last modified: {}</i></font>
"""


def main(
    entries_path,
    cl_names,
    Qfold,
    fname,
    ucc_id,
    C1,
    C2,
    abcd_c,
    r_50,
    N_50,
    ra,
    dec,
    plx,
    pmra,
    pmde,
    fpars_table,
    posit_table,
    close_table,
):
    """ """
    txt = ""
    txt += header.format(cl_names[0])
    if len(cl_names) > 1:
        txt += more_names.replace("cl_names_str", "; ".join(cl_names[1:]))

    rad_deg = round(2 * (r_50 / 60.0), 3)
    txt += (
        aladin_snippet.replace("QFOLD", str(Qfold))
        .replace("FNAME", str(fname))
        .replace("RAD_DEG", str(rad_deg))
        .replace("RA_ICRS", str(ra))
        .replace("DE_ICRS", str(dec))
    )

    txt += (
        table_right_col.replace("UCC_ID", ucc_id)
        .replace("Class1", str(C1))
        .replace("Class2", str(C2))
        .replace("Class3", abcd_c)
        .replace("r_50_val", str(r_50))
        .replace("N_50_val", str(int(N_50)))
    )

    if N_50 == 0:
        txt += no_membs_50_warning

    txt += nasa_simbad_url.replace("XXNASAXX", cl_names[0].replace(" ", "%20")).replace(
        "XXSIMBADXX", fname
    )

    txt += posit_table_top
    txt += posit_table

    if close_table != "":
        txt += close_table_top
        txt += close_table + "\n"

    signed_dec = "+" + str(dec) if dec >= 0 else str(dec)
    txt += (
        cds_simbad_url.replace("RADEC_CDS", "{},{}".format(ra, signed_dec))
        .replace("RADEC_SMB", "{}%20{}".format(ra, dec))
        .replace("XCLUSTX", fname)
    )

    txt += cl_plot.format(Qfold, fname)

    txt += notebook_url.format(Qfold, fname)

    if fpars_table != "":
        txt += fpars_table_top
        txt += fpars_table

    txt += data_foot.format(datetime.today().strftime("%Y-%m-%d"))
    contents = "".join(txt)

    return contents
