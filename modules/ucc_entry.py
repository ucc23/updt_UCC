
from datetime import datetime


header = """---
layout: post
title:  {}
---
"""

aladin_header = """
<!-- include Aladin Lite CSS file in the head section of your page -->
<link rel="stylesheet" href="https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css" />
<script type="text/javascript" src="https://code.jquery.com/jquery-1.12.1.min.js" charset="utf-8"></script>
<!-- Aladin Lite CS -->
"""

aladin_table1 = r"""
<div style="display: flex; justify-content: space-between;">
   <div style="text-align: center;">
      <!-- Left block -->
      <!-- Aladin Lite viewer -->
      <div id="aladin-lite-div" align="left" style="width:285px;height:250px;"></div>
      <script type="text/javascript" src="https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.js" charset="utf-8"></script>
      <script type="text/javascript">var aladin = A.aladin('#aladin-lite-div', {survey: "P/DSS2/color", fov:0.25, target: " """

aladin_table2 = r""""});</script>
   </div>
   <!-- Aladin Lite viewer -->
   <!-- Left block -->
"""

data_table1 = """   <!-- Right block -->
   <table style="text-align: center;">
      <!-- Row 0 (title) -->
      <tr>
         <td align="center" colspan="5"><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{} (<a href="#" title="Cluster class">{}</a>)</b></td>
      </tr>
      <!-- Row 1 -->
      <tr>
         <th>RA</th>
         <th>DEC</th>
         <th>GLON</th>
         <th>GLAT</th>
         <th>Class</th>
      </tr>
      <!-- Row 2 -->
      <tr>
         <td>{}</td>
         <td>{}</td>
         <td>{}</td>
         <td>{}</td>
         <td>
"""

data_table2 = """         </td>
      </tr>
      <!-- Row 3 -->
      <tr>
         <th>plx</th>
         <th>pmRA</th>
         <th>pmDE</th>
         <th>Rv</th>
         <th>N_20</th>
      </tr>
      <!-- Row 4 -->
      <tr>
         <td>{}</td>
         <td>{}</td>
         <td>{}</td>
         <td>{}</td>
         <td>{}</td>
      </tr>
   </table>
   <!-- Right block -->
</div>
"""


hidden_classif = """
<!-- Hidden for search purposes -->
<font color="#FFFFFF">{}</font>
"""

other_names = """
<div style="text-align: left;">
   <span style="color: #99180f; font-weight: bold;">Other denominations: </span><span>{}</span>
</div>
"""

cl_plot = """
![CLUSTER](https://raw.githubusercontent.com/ucc23/datafiles/main/plots/{}.png)

"""

notebook_url = """
> <p style="text-align:center; font-weight: bold; font-size:20px">Explore data in <a href="https://colab.research.google.com/github/UCC23/datafiles/blob/master/notebooks/{}.ipynb" target="_blank">Colab</a></p>
"""

fpars_table_top = """\n
### Fundamental parameters in literature (not exhaustive)

| Reference |  Fundamental parameters |
| :---         |     :---:      |
"""

nasa_url = """\n
> <p style="text-align:center; font-weight: bold; font-size:20px">Search name in <a href="https://ui.adsabs.harvard.edu/search/q=%20collection%3Aastronomy%20%3Dbody%3A%22{}%22&sort=date%20desc%2C%20bibcode%20desc&p_=0" target="_blank">NASA/SAO ADS</a></p>
"""

posit_table_top = """\n
### Position in published works (not exhaustive)

| Reference    | RA    | DEC   | plx  | pmRA  | pmDE   |  Rv  |
| :---         | :---: | :---: | :---: | :---: | :---: | :---: |
"""

cds_url = """\n
> <p style="text-align:center; font-weight: bold; font-size:20px">Search region in <a href="http://cdsportal.u-strasbg.fr/?target={}" target="_blank">CDS</a></p>
"""

close_table_top = """\n
### Catalogued clusters in <a href="faq.html" title="See FAQ for definition of proximity">proximity</a>

| Cluster | RA    | DEC   | plx   | pmRA  | pmDE  |
| :---    | :---: | :---: | :---: | :---: | :---: |
"""

data_foot = """\n
<br>
<font color="b3b1b1"><i>Last time modified {}</i></font>
"""


def make_entry(
    entries_path, cl_names, fname, ucc_id, abcd_v, abcd_c, Nmemb, lon, lat,
    ra, dec, plx, pmra, pmde, rv, fpars_table, posit_table, close_table
):
    """
    """
    txt = ""
    txt += header.format(cl_names[0])
    txt += aladin_header
    txt += aladin_table1[:-1] + "{}".format(ra) + " " + "{}".format(dec)
    txt += aladin_table2
    txt += data_table1.format(ucc_id, abcd_v, ra, dec, lon, lat, )
    txt += abcd_c
    txt += data_table2.format(plx, pmra, pmde, rv, Nmemb)
    if len(cl_names) > 1:
        txt += other_names.format(", ".join(cl_names[1:]))
    txt += cl_plot.format(fname)
    txt += notebook_url.format(fname)
    txt += fpars_table_top
    txt += fpars_table
    txt += nasa_url.format(cl_names[0].replace(' ', '%20'))
    txt += posit_table_top
    txt += posit_table
    ra_dec = "{}%20{}".format(ra, dec)
    txt += cds_url.format(ra_dec)
    if close_table != '':
        txt += close_table_top
        txt += close_table
    txt += data_foot.format(datetime.today().strftime('%Y-%m-%d'))

    with open(entries_path + fname + ".md", "w") as f:
        contents = "".join(txt)
        f.write(contents)
