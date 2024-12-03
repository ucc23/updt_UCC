# The Unified Cluster Catalogue (UCC)

These files correspond to the XXXXXX version of the UCC database (https://ucc.ar).


## Referencing

If you find this data useful please cite its associated article
(Perren et al. 2023) using the following BibTex snippet:

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

"This research has made use of the Unified Cluster Catalogue (UCC)
(Perren et al. 2023)."


## Contents

--------------------------------------------------------------------------------
|        FileName      | Description                                           |
| ---------------------|--------------------------------------------------------
| README.txt           | This file                                             |
| UCC_cat.csv          | Full UCC catalogue                                    |
| UCC_members.parquet  | List of members for the clusters in the UCC catalogue |
--------------------------------------------------------------------------------

Columns listed in the `UCC_cat.csv` file:

```
ID        : Name(s) associated to the OC
RA_ICRS   : Median RA from the databases this OC is present
DE_ICRS   : Median DEC from the databases this OC is present
Plx       : Median parallax from the databases this OC is present
pmRA      : Median RA proper motion from the databases this OC is present
pmDE      : Median DEC proper motion from the databases this OC is present
UCC_ID    : Internal UCC naming
N_50      : Number of estimated members with P>0.5
r_50      : Radius that contains half the members (in arcmin)
RA_ICRS_m : Median RA estimated using the selected members
DE_ICRS_m : Median DEC estimated using the selected members
Plx_m     : Median parallax estimated using the selected members
pmRA_m    : Median RA proper motion estimated using the selected members
pmDE_m    : Median DEC proper motion estimated using the selected members
Rv_m      : Median radial velocity estimated using the selected members
N_Rv      : Number of members with Rv data
C1        : C_phot quality class
C2        : C_dens quality class
C3        : C_phot,C_dens, combined quality class
```

Columns listed in the `UCC_members.parquet` file:

```
name      : Cluster's main name
Source    : Gaia identification
RA_ICRS   : RA
DE_ICRS   : DEC
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


