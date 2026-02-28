# The Unified Cluster Catalogue (UCC)

These files correspond to the 260227 version of the UCC database (https://ucc.ar),
composed of 17780 clusters with a combined 1080169 members.
If you find this data useful please cite its associated article
([Perren et al. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.4107P))
using the following BibTex snippet:

```
@ARTICLE{2023MNRAS.526.4107P,
       author = {{Perren}, Gabriel I. and {Pera}, Mar{\'\i}a S. and {Navone}, Hugo D. and {V{\'a}zquez}, Rub{\'e}n A.},
        title = "{The Unified Cluster Catalogue: towards a comprehensive and homogeneous data base of stellar clusters}",
      journal = {\mnras},
     keywords = {methods: data analysis, catalogues, open clusters and associations: general, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = dec,
       volume = {526},
       number = {3},
        pages = {4107-4119},
          doi = {10.1093/mnras/stad2826},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.4107P}
}
```

You can alternatively mention the UCC in the 'Acknowledgements' section of your
paper:

"This research has made use of the Unified Cluster Catalogue
\citep[\url{https://ucc.ar};][]{Perren_2023}."


## Contents

--------------------------------------------------------------------------------
|        FileName      | Description                                           |
| ---------------------|--------------------------------------------------------
| README.txt           | This file                                             |
| UCC_cat.csv          | Full UCC catalogue                                    |
| UCC_members.parquet  | List of members for the clusters in the UCC catalogue |
--------------------------------------------------------------------------------

Columns listed in the `UCC_cat.csv` file (one row per cluster):

```
Name(s)          : Name(s) associated to the OC
name             : Formatted naming to match the members file 'main name'
N_50             : Number of estimated members with P>0.5
r_50             : Radius that contains half the members (in arcmin)
RA_ICRS          : Right ascension
DEC_ICRS         : Declination
GLON             : Galactic longitude
GLAT             : Galactic latitude
Plx              : Parallax [mas]
pmRA             : Right ascension proper motion [mas/yr]
pmDE             : Declination proper motion [mas/yr]
Rv               : Radial velocity [km/s]
N_Rv             : Number of members used to estimate the radial velocity
Dist_[kpc]       : Distance [kpc]
Dist_STDDEV      : Distance standard deviation
Av_[mag]         : Absorption in the V band [mag]
Av_STDDEV]       : Absorption standard deviation
Diff_ext_[mag]   : Differential extinction [mag]
Diff_ext_STDDEV  : Differential extinction standard deviation
Age_[Myr]        : Age [Myr]
Age_STDDEV       : Age standard deviation
FeH_[dex]        : Metallicity [dex]
FeH_STDDEV       : Metallicity standard deviation
Mass_[Msun]      : Mass [Msun]
Mass_STDDEV      : Mass standard deviation
Binary_fr        : Estimated binary fraction
Binary_fr_STDDEV : Estimated binary fraction standard deviation
Blue_str         : Blue stragglers value (fraction and/or total count)
C3               : Combined quality class
P_dup            : Probability that this entry is a duplicate of a previous one
UTI              : UCC Trust Index, a measure of the reliability of the cluster
bad_oc           : Flag that indicates that this is likely an asterism,
                   moving group or artifact (based on UTI value, duplication
                   probability and presence in literature)
```


Columns listed in the `UCC_members.parquet` file (one row per star):

```
name      : Cluster's main name
Source    : Gaia identification
RA_ICRS   : Right ascension
DE_ICRS   : Declination
Plx       : Parallax
e_Plx     : Parallax error
pmRA      : Proper motion in RA
e_pmRA    : Proper motion in RA error
pmDE      : Proper motion in DEC
e_pmDE    : Proper motion in DEC error
RV        : Radial velocity
e_RV      : Radial velocity error
GLON      : Galactic longitude
GLAT      : Galactic latitude
Gmag      : Gaia G magnitude
BP-RP     : Gaia BP-RP color
e_Gmag    : G magnitude error
e_BP-RP   : BP-RP color error
probs     : Probability of being a member
```

To access all of the members associated to a given cluster you can follow
these steps:

1. Install the required Python packages:

```
$ pip install pandas
$ pip install fastparquet
```

2. Open your code editor and write:

```
import pandas as pd

# Load the full members file
df = pd.read_parquet("UCC_members.parquet")

# Generate a mask using the cluster's main name (in this case Blanco 1)
msk = df['name'] == 'blanco1'

# Access the members of this cluster using the mask
print(df[msk])
```


