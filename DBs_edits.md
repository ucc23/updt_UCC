
# Databases in UCC

List of all the databases used to generate the UCC and the corrections made
on each of them.


## Initial DBs

These are the list of 32 initial DBs used in the original UCC article.

### KHARCHENKO12
Global survey of star clusters in the Milky Way. I. The
Pipeline and ...; [Kharchenko et al. 2012](https://ui.adsabs.harvard.edu/abs/2012A%26A...543A.156K), [HEASARC](https://heasarc.gsfc.nasa.gov/W3Browse/all/mwsc.html)

We do not use these values in the DB.
I retrieved the data from the HEASARC service selecting all clusters with
'class' equal to "OPEN STAR CLUSTER", resulting in 2858 entries.
This DB lists the FSR clusters as "FSR XXXX" with leading zeros and the ESO
clusters as "ESO XXX-XX" with leading zeroes.

vdBergh-Hagen --> VDBH per CDS recommendation (added 'vdBergh-Hagen' so that the
naming isn't lost)
vdBergh       --> VDB per CDS recommendation (added 'vdBergh' so that the
naming isn't lost)
ESO 456-29 --> removed as its coordinates match GC 'Gran 1'
FSR 1716 --> removed as its a GC
FSR 1758 --> removed as its a GC
VDBH 140,vdBergh-Hagen 140 --> removed as its a GC

### LOKTIN17
Updated version of the `homogeneous catalog of open cluster parameters',
[Loktin & Popova 2017](https://ui.adsabs.harvard.edu/abs/2017AstBu..72..257L/abstract)

Many proper motion values in this Db are very wrong (see e.g: NGC 2516). We
thus do not use these values in the DB.

Contains 11 clusters with an extra name in parenthesis, these were fixed as
follows:

```
"Alpha_Persei, Melotte 20"
"Collinder258, Har5"
"Coma Star, Melotte 111"
"Herschel1, ASCC41"
"Hyades, Melotte 25"
"NGC2645, Pismis6"
"NGC2682, Melotte 67"
"NGC3247, Cr220"
"Pleiades, Melotte 45"
"Praesepe, NGC2632"
"Stephenson1, Del_L"
```

VDBergh_Hagen --> VDBH per CDS recommendation (added 'vdBergh-Hagen' to 43
clusters so that the naming isn't lost)
VDBergh       --> VDB per CDS recommendation

Sauer5 --> Saurer 5
Teusch61 --> Teutsch61
AlessiJ2327+55 --> Alessi J2327.0+55
Sigma_Ori --> Sigma_Orionis

berkeley42 --> removed as it is the GC NGC 6749
lynga7 --> removed as it is the GC BH 184

### CASTRO18
A new method for unveiling open clusters in Gaia. New nearby open
clusters confirmed by DR2, [Castro-Ginard et al. 2018](https://ui.adsabs.harvard.edu/abs/2018A%26A...618A..59C/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/618/A59)

The table contains 23 UBC clusters.

### BICA19
A Multi-band Catalog of 10978 Star Clusters ... in the Milky Way;
2913 clusters (OCs); [Bica et al. 2019](https://ui.adsabs.harvard.edu/abs/2019AJ....157...12B/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/AJ/157/12/table3)

The Vizier table contain 10978 entries. We keep only those with Class1 OC (open
cluster) or OCC (open cluster candidate); this reduces the list to 3564 entries.
Added 'RA_ICRS' and 'DE_ICRS' columns.

Fixes:

-MWSC 2776, name listed twice, we remove one of the repeated names.
-FSR 523, FSR 847, FSR 436; we remove the duplicates that showed as
single entries
-ESO 393-3, removed name from both entries (cluster not found in CDS)
-MWSC 1025, 1482, 948, 3123, 1997, 1840, 442, 1808, 2204: removed name from
both entries (clusters not found in KHARCHENKO12)
-ESO 97-2: removed from Loden 848 as it matches the position of Loden 894
according to CDS
-FSR 972, OCL 344, Collinder 384, FSR 179: removed from both entries, it does
not show in CDS or anywhere else
-MWSC 206: removed entry that also showed FSR 60 as the coordinates for FSR 60
are a better match in KHARCHENKO12 for the entry with the single FSR 60 name
-FSR 429.MWSC 3667 --> FSR 429,MWSC 3667
-Carraro 1.MWSC 1829 --> Carraro 1,MWSC 1829
-Cernik 39 --> "Czernik 39
-FSR343 --> FSR 343
-ESO456-13 --> ESO 456-13
-de Wit 1 --> Wit 1 (to match KHARCHENKO12)
-JS 1 --> Juchert-Saloran 1 (to match KHARCHENKO12)
-ESO 589-26,MW --> ESO 589-26
-Messineo 1,Cl 1813-18,SAI 126, --> Remove comma at the end
-BH       --> VDBH per CDS recommendation (added 'vdBergh-Hagen' so that the
naming isn't lost)
-vdBergh  --> VDB per CDS recommendation
-Alessi J2327.6+5535 --> Alessi J2327.0+55
-TRSG 1 --> RSG 1
-Added 'Dol-Dzim 9' to 'DoDz 9' to match KHARCHENKO12
-Added 'Dol-Dzim 11' to 'DoDz 11' to match KHARCHENKO12
-Removed 'Alessi J0715.6-0722' as it is an OCC and its position matches that of
'Alessi J0715.6-0727'
ESO 456-29,MWSC 2761 --> removed as its coordinates match GC 'Gran 1'
ESO 93-8,MWSC 1932 --> removed as it is a GC
FSR 1758,MWSC 2617 --> removed as its a GC
VDBH 140,vdBergh-Hagen 140,FSR 1632,MWSC 2071 --> removed as its a GC

This is the only DB that lists the Ryu & Lee (2018) clusters. The original
article claims to have found 721 new OCs (923 minus 202 embedded). BICA19 (page
11) says that the Ryu & Lee article lists 719 OCs (921 minus 202 embedded).
BICA19 lists in its Vizier table only 711 Ryu OCs, 4 of which are listed with
alternative names (Teutsch J1814.6-2814|Ryu 563, Quartet|Ryu 858,
GLIMPSE 70|Mercer 70|Ryu 273, LS 468|La Serena 468|Ryu 094). Hence there are
707 Ryu clusters in the final BICA19 Vizier table.

For the Ryu clusters:
*The average angular radius of the clusters is 1'.31±0 60.
More specifically, 902 (98%) clusters are smaller than 3′, and
823 (89%) clusters are even smaller than 2′.* [Ryu & Lee (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...856..152R/abstract)

### BOSSINI19

Added columns for the standard deviation of the fundamental parameters, combining the
16th and 84th percentiles.

The following OCs are duplicates and were combined using their average values:
ASCC124, Alessi37
ASCC22, Ferrero11
ASCC112, AlessiTeutsch11
Harvard5, Collinder258

The following OCs were renamed:
ngc0188, ngc0752, ngc0381, ngc0225, ngc0581
to:
ngc188, ngc752, ngc381, ngc225, ngc581

### SIM19
207 New Open Star Clusters within 1 kpc from Gaia Data Release,
[Sim et al. 2019](https://ui.adsabs.harvard.edu/abs/2019JKAS...52..145S/abstract), Data taken from Table 2 in online article

The table contains 207 UPK clusters. Added (ra, dec) columns and a plx column
estimated as 1000/dist_pc

### CASTRO19
Hunting for open clusters in Gaia DR2: the Galactic anticentre,
[Castro-Ginard et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..35C/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/627/A35)

The table contains 53 UBC clusters.

### LIUPANG19
A Catalog of Newly Identified Star Clusters in Gaia DR2,
[Liu & Pang 2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..245...32L/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/245/32)

Initial DB used:
The table contains 76 clusters with no acronym given. I added 'FoF_' to match
HUNT23. Added `logt` column.

Updated DB used:
The Vizier table contains 2443 entries, including the 76 new OCs.

### FERREIRA19
Three new Galactic star clusters discovered in the field of
the open cluster NGC 5999 ..., [Ferreira et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.5508F/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/483/5508)

The table contains 3 UFMG clusters. Added (ra, de) columns.

### CASTRO20
Hunting for open clusters in Gaia DR2: 582 new open clusters ..
Galactic disc, [Castro-Ginard et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...635A..45C/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/635/A45)

Fixed wrong ra coordinates for UBC595, UBC181

The table contains 570 UBC clusters.

### FERREIRA20
Discovery and astrophysical properties of Galactic open
clusters in dense..., [Ferreira et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.496.2021F/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/496/2021)

The table contains 25 UFMG clusters. Added (ra, de) columns.

### CANTAT20
Painting a portrait of the Galactic disc with its stellar clusters;
[Cantat-Gaudin et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...640A...1C), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/A%2bA/640/A1/table1)

The table contains 2017 clusters. This DB lists the FSR clusters as "FSR_XXXX"
with leading zeros and the ESO clusters as "ESO_XXX_XX" with leading zeroes.

BH        --> VDBH per CDS recommendation
vdBergh_  --> VDB per CDS recommendation
LP_ --> FoF_ to match the original work LIUPANG19
Sigma_Ori --> Sigma_Orionis

### HAO20
Sixteen Open Clusters Discovered with Sample-based Clustering Search of Gaia
DR2, [Hao et al. 2020](https://ui.adsabs.harvard.edu/abs/2020PASP..132c4502H/abstract), Data from Table 2

This table lists 16 clusters with no acronym. Used 'HXWHB_' to match HUNT23


### DONOR20

Did not use the ages listed as they are from the MWSC database.


### FERREIRA21
New star clusters discovered towards the Galactic bulge
direction using Gaia DR2, [Ferreira et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502L..90F/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/502/L90)

The table contains 34 UFMG clusters.

### HE21
A catalogue of 74 new open clusters found in Gaia Data-Release 2,
[He et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021RAA....21...93H/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/other/RAA/21.93)

The table contains 74 with no acronym, added 'HXHWL_' to match HUNT23.
 Added (ra, de) columns.

### DIAS21
Updated parameters of 1743 open clusters based on Gaia DR2,
[Dias et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504..356D), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/504/356)

The table contains 1743 clusters. This DB lists the FSR clusters as "FSR_XXXX"
with leading zeros and the ESO clusters as "ESO_XXX_XX" with leading zeroes.

The table lists 177 LIUPANG19 clusters because it includes clusters not
listed as new by the authors. Changed 'LP_' to 'FoF_' for consistency.
Removed the leading zero in 'FoF_XXXX'. Cluster 'LP_866' was duplicated
(LP_0866), removed the second entry. 

BH       --> VDBH per CDS recommendation
vdBergh  --> VDB per CDS recommendation
Sigma_Ori --> Sigma_Orionis

The `r50` column is listed with 'pc' units in Vizier, but it is degrees unit.
It also shows 4 clusters with very large `r50` values which could be listed
in pc units:

```
Cluster      DIAS21  CG20   BICA19
----------------------------------
Berkeley_58  32.969  0.06   0.058
Blanco_1     13.218  0.699  0.833
NGC_7789     9.324   0.211  0.133
Berkeley_59  3.097   0.137  nan
```

### HUNT21
Improving the open cluster census. I. Comparison of clustering algorithms
applied to Gaia DR2 data, [Hunt & Reffert (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A.104H/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/646/A104)

This table lists 41 'PHOC_' clusters.

### CASADO21
New open clusters found by manual mining of data based on Gaia DR2,
[Casado (2021)](https://ui.adsabs.harvard.edu/abs/2021RAA....21..117C/abstract), Data from Table 1 in article

This table lists 20 'Casado_' clusters.

### JAEHNIG21
Membership Lists for 431 Open Clusters in Gaia DR2 Using Extreme Deconvolution
Gaussian Mixture Models, [Jaehnig et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...923..129J/abstract), Data from Table 1 in article

This table lists 11 'XDOCC-' clusters. Changed 'XDOCC-0Y' to 'XDOCC-Y' to match
HUNT23.

### SANTOS21
Canis Major OB1 stellar group contents revealed by Gaia,
[Santos-Silva et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1033S/abstract), Data from Table 1 in article

This table lists 5 'CMa-' clusters. Added `logt` column.

### TARRICQ22
Structural parameters of 389 local open clusters,
[Tarricq et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...659A..59T/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/659/A59)

The table contains 467 clusters. This DB lists the FSR clusters as "FSR_XXX"
with leading zeros and the ESO clusters as "ESO_XXX_XX" with leading zeroes.

BH_99 --> VDBH_99
LP_ --> FoF_

### CASTRO22
Hunting for open clusters in Gaia EDR3: 628 new open clusters
found with OCfinder, [Castro-Ginard et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A.118C/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/661/A118)

The table contains 628 UBC clusters numbered from UBC 1001 in order to
differentiate them from the UBC clusters found using Gaia DR2 in the previous
articles.

### HE22
New Open-cluster Candidates Found in the Galactic Disk Using Gaia
DR2/EDR3 Data, [He et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..260....8H/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/260/8)

The table contains 541 with no acronym, added 'CWNU_'.
Replaced '---' with '' for RV values

### HE22_1
A Blind All-sky Search for Star Clusters in Gaia EDR3: 886
Clusters within 1.2 kpc of the Sun, [He et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJS..262....7H/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJS/262/7)

The table contains 270 new 'CWNU_' clusters. Removed Theia entries.
Changed 'LP_' to 'FoF_' to match HUNT23, LIUPANG19, TARRICQ22

ESO_489-01 --> ESO_489_01
vdBergh  --> VDB per CDS recommendation
Sigma_Ori --> Sigma_Orionis

### HAO22
Newly detected open clusters in the Galactic disk using Gaia EDR3,
[Hao et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...660A...4H/abstract), [Vizier](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/660/A4)

This table lists 704 'OC_' clusters.

OC 0586 --> removed, GC 'BH 140'

### LI22
LISC Catalog of Star Clusters. I. Galactic Disk Clusters in Gaia EDR3,
[Li et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJS..259...19L/abstract), [Zenodo](https://zenodo.org/record/5705371#.YZPASbFdsrs)

This table lists 61 'LISC_' clusters. The table contains a column called
`t/t_range` in Gyr and I'm not sure what it represents.

### HE23
Unveiling hidden stellar aggregates in the Milky Way: 1656 new star
clusters found in Gaia EDR3, [He et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..264....8H/abstract), [Vizier](https://cdsarc.cds.unistra.fr/ftp/vizier.submit/he22c/)

The table contains 1656 with no acronym, added 'CWNU_'. Added (ra, de) columns.

### HUNT23
Improving the open cluster census. II. An all-sky cluster catalogue
with Gaia DR3, [Hunt & Reffert (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv230313424H/abstract), [Vizier](https://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A%2BA/673/A114)

Removed all globular clusters marked as open clusters: Palomar 2, 7 (listed
as IC_1276) 8, 10, 11, 12, ESO_452-11 (listed as 1636-283), Pismis_26 (Ton 2),
Lynga_7 (BH 184). HSC_2890 and HSC_134 were removed as their position and
astrometry match those of the GCs Gran 3 and 4. HSC 2605 has very similar
coordinates and proper motions to globular NGC 5139 but its parallax is
different, so it was not removed.

In the initial UCC version all the Theia entries were removed along with the moving
groups. In the updated version the Theia entries labeled as OCs were re-incorporated.

Some ~160 HSC clusters have center values that do not align with the medians of
their members. Changed these centers to the member' medians.

Added standard deviation columns for 'AV50,MOD50,logAge50', obtained from the 84th
and 16th percentiles.

BH        --> VDBH per CDS recommendation (added 'vdBergh-Hagen' so that the
naming isn't lost)
vdBergh_  --> VDB_ per CDS recommendation (added 'vdBergh' so that the
naming isn't lost)

Fixed:
ESO_429-429 --> ESO_429-02 (according to CDS coords)
CMa_2 --> CMa_02
AH03_J0748+26.9 --> AH03_J0748-26.9
Juchert_J0644.8+0925 --> Juchert_J0644.8-0925
Teutsch_J0718.0+1642 --> Teutsch_J0718.0-1642
Teutsch_J0924.3+5313 --> Teutsch_J0924.3-5313
Teutsch_J1037.3+6034 --> Teutsch_J1037.3-6034
Teutsch_J1209.3+6120 --> Teutsch_J1209.3-6120

Removed Teutsch 182 because it is listed along with UBC 6 and these are the same OC
according to the UCC. Also Teutsch 182 has wrong center coordinates


### QIN23
Hunting for Neighboring Open Clusters with Gaia DR3: 101 New Open
Clusters within 500 pc, [Qin et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..265...12Q/abstract), Tables provided by Qin & Chen

This table lists 101 'OSCN_' clusters.

### LI23
LISC Catalog of Star Clusters. II. High Galactic Latitude Open Clusters in
Gaia EDR3; [Li & Mao (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..265....3L/abstract), [Zenodo](https://zenodo.org/record/7603419)

This table lists 56 'LISC' clusters but only 35 are kept as "real" objects.
The parallax distances are in very bad agreement with the the estimated
distance moduli. These appear to either be MC clusters or not real clusters
at all. HUNT23 recovers 0% of these clusters.

### CHI23_2
Identifying 46 New Open Cluster Candidates in Gaia EDR3 Using a Hybrid
pyUPMASK and Random Forest Method; [Chi et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..265...20C/abstract), data taken
from Table in the article (IOP)

This table lists 46 clusters with no acronym, 'CWWL_' was added to match
HUNT23. Added `logt` column.

### CHI23
LISC Catalog of Open Clusters.III. 83 Newly found Galactic disk open clusters
using Gaia EDR3; [Chi et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv230208926C/abstract)

The article mentions 83 clusters but only 82 are visible in the PDF. I sent an
email to zhongmuli@126.com but never got an answer. Added 'LISC-III' to the
names to match HUNT23. Added (ra, dec) columns.

### CHI23_3
**WARNING**: https://twitter.com/CantatGaudin/status/1638133660875456515
Blind Search of The Solar Neighborhood Galactic Disk within 5kpc: 1179 new
Star clusters found in Gaia DR3, [Chi et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv230310380C/abstract), Table requested to
authors

The table lists 1179 clusters. Added CWWDL following the convention by HUNT23
for DBs with no acronyms. Clusters CWWDL_3274 and CWWDL_3247 are very close


## Added DBs

### HE23_1
Survey for Distant Stellar Aggregates in the Galactic Disk: Detecting 2000 Star
Clusters and Candidates, along with the Dwarf Galaxy IC 10;
[He et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJS..267...34H/abstract), data from Table 1

Article lists 2085 clusters, 28 of which are flagged as 'GCC' (GC candidate)
and removed leaving 2057 entries.

'Ruprecht 123' is listed twice as: 'Ruprecht_123_v0' & 'Ruprecht_123'. Removed
the 'Ruprecht_123_v0' entry as the RA value of the other entry was more similar
to the values listed in the UCC. The final number of entries is 2056.

There are 20 'LP_XXX' OCs that were renamed 'FoF_XXX', removing leading zeros.
There are 102 OCs already present in the UCC.