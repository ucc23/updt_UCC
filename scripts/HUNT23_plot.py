import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

hunt23_membs_path = (
    "/home/gabriel/Github/articles/2023/UCC/0_data/hunt23_members.parquet"
)


def main():
    """Generate a plot using member of the HUNT23 article."""
    print("Reading HUNT23 members...")
    hunt23_membs = pd.read_parquet(hunt23_membs_path)

    hunt23_name = "Teutsch_182"
    make_plot(hunt23_membs, hunt23_name)
    breakpoint()


def make_plot(hunt23_membs, hunt23_name, Nmembs_manual=None):
    """ """
    if hunt23_name.startswith("VDBH_"):
        hunt23_name = "BH_" + hunt23_name.split("_")[1]
    if hunt23_name.startswith("VDB_"):
        hunt23_name = "vdBergh_" + hunt23_name.split("_")[1]

    # Read HUNT23 members data
    msk1 = hunt23_membs["Name"] == hunt23_name
    hunt23_cl = hunt23_membs[msk1]

    plot(hunt23_name, hunt23_cl)


def plot(name, clust1):
    """ """
    plt.suptitle(f"{name}, N_H23={len(clust1)}")
    plt.subplot(231)
    # plt.scatter(clust1["GLON"], clust1["GLAT"], alpha=0.5, label="HUNT23")
    plt.scatter(clust1["RA_ICRS"], clust1["DE_ICRS"], alpha=0.5, label="HUNT23")
    plt.legend()
    plt.xlabel("GLON")
    plt.ylabel("GLAT")

    plt.subplot(232)
    plt.scatter(clust1["pmRA"], clust1["pmDE"], alpha=0.5)
    plt.xlabel("pmRA")
    plt.ylabel("pmDE")

    plt.subplot(233)
    plt.hist(clust1["Plx"], alpha=0.5, density=True)
    plt.xlabel("plx")

    plt.subplot(234)
    plt.scatter(clust1["BP-RP"], clust1["Gmag"], alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("BP-RP")
    plt.ylabel("Gmag")

    plt.subplot(235)
    plt.hist(clust1["Prob"], alpha=0.5)  # , density=True)
    plt.xlabel("Probabilities")

    plt.subplot(236)
    d_pc = 1000 / clust1["Plx"].values
    d_pc = np.clip(d_pc, 1, 20000)

    gc_frame = coord.Galactocentric()
    # Galactic coordinates.
    eq = SkyCoord(
        ra=clust1["RA_ICRS"].values * u.degree,
        dec=clust1["DE_ICRS"].values * u.degree,
        frame="icrs",
    )
    lb = eq.transform_to("galactic")
    lon = lb.l.wrap_at(180 * u.deg).radian * u.radian
    lat = lb.b.radian * u.radian
    coords = SkyCoord(l=lon, b=lat, distance=d_pc * u.pc, frame="galactic")
    # Galactocentric coordinates.
    c_glct = coords.transform_to(gc_frame)
    x, y, z = c_glct.x, c_glct.y, c_glct.z
    # x_kpc, y_kpc, z_kpc = x_pc.value/1000, y_pc.value/1000, z_pc.value/1000
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()


if __name__ == "__main__":
    # plt.style.use('science')
    main()
