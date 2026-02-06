import warnings
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.io import fits
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy import ndimage

from ..utils import plx_to_pc
from ..variables import custom_style_path, spiral_arms


def velocity(ra, dec, plx, pmRA, pmDE, rad_v, x_gc, y_gc, z_gc, R_gc):
    """
    Draw the Galactocentric velocity vector projected in the XY plane.

    ra, dec: degrees
    distance_pc: pc
    pmRA, pmDE: mas/yr
    x_gc, y_gc, z_gc: Galactocentric X,Y,Z [kpc]
    """

    d_pc = plx_to_pc(plx)

    # Replace 'nan' values with '0' in rad_v
    rad_v = np.nan_to_num(rad_v, nan=0.0)

    stars = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=d_pc * u.pc,
        pm_ra_cosdec=pmRA * u.mas / u.yr,
        pm_dec=pmDE * u.mas / u.yr,
        radial_velocity=rad_v * u.km / u.s,
    )

    stars_galcen = stars.transform_to(Galactocentric())

    vx = stars_galcen.v_x.to(u.km / u.s).value
    vy = stars_galcen.v_y.to(u.km / u.s).value
    vz = stars_galcen.v_z.to(u.km / u.s).value
    # Velocity projection in R:
    vR = (x_gc * vx + y_gc * vy + z_gc * vz) / R_gc

    return vx, vy, vz, vR


def plot_gcpos(
    plot_fpath,
    Z_uti,
    R_uti,
    X,
    Y,
    Z,
    R,
    vx,
    vy,
    vz,
    vR,
    X_sun=-8.2,
    Y_sun=0,
    Z_sun=0,
    R_sun=8.2,
    R_MW=15.0,
    dpi=150,
):
    """ """

    plt.style.use(custom_style_path)

    fig = plt.figure(figsize=(6, 3))  # Taller than wide
    gs = GridSpec(1, 2, figure=fig, wspace=0.05)
    plt.suptitle(f"({X:.2f}, {Y:.2f}, {Z:.2f}) [kpc]")

    max_l = 7
    scale_xy = min(abs(max_l / vx), abs(max_l / vy))
    point_pos_X, point_pos_Y = abs(X), abs(Y)
    arrow_point_pos_X, arrow_point_pos_Y = (
        abs(X + vx * scale_xy),
        abs(Y + vy * scale_xy),
    )
    rmax = max(
        R_MW * 1.05,
        1.05 * max(point_pos_X, arrow_point_pos_X, point_pos_Y, arrow_point_pos_Y),
    )

    ax1 = fig.add_subplot(gs[0])
    # Plot spiral arms
    for arm, coords in spiral_arms.items():
        armx, army = np.array(coords).T
        ax1.plot(armx, army, linestyle="--", label=arm, lw=2, zorder=3)
    ax1.scatter(X_sun, Y_sun, marker="$\\odot$", c="y", ec="k", lw=0.5, s=150, zorder=5)
    ax1.scatter(0, 0, marker="x", c="k", s=100, zorder=5)
    ax1.scatter(X, Y, marker="o", c="orange", ec="k", s=80, zorder=10)
    ax1.quiver(
        X,
        Y,
        vx * scale_xy,
        vy * scale_xy,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.01,
        zorder=9,
    )
    circle = Circle((0.0, 0.0), R_MW, fill=False, edgecolor="k")
    ax1.add_patch(circle)
    ax1.hlines(y=0, xmin=-R_MW, xmax=R_MW, color="k", linestyle=":", alpha=0.5)
    ax1.vlines(x=0, ymin=-R_MW, ymax=R_MW, color="k", linestyle=":", alpha=0.5)
    ax1.set_xlabel(r"X$_{GC}$ [kpc]")
    ax1.set_ylabel(r"Y$_{GC}$ [kpc]", labelpad=1.5)
    ax1.set_xlim(-rmax, rmax)
    ax1.set_ylim(-rmax, rmax)

    ax2 = fig.add_subplot(gs[1])
    ax2.hlines(y=0, xmin=0, xmax=R_MW, color="k", linestyle="-", linewidth=3, alpha=0.5)
    ax2.scatter(0, 0, marker="x", c="k", s=100, zorder=5)
    ax2.scatter(R_sun, Z_sun, marker="$\\odot$", ec="k", lw=0.5, c="y", s=150, zorder=5)
    ax2.scatter(R, Z, marker="o", c="orange", ec="k", s=80, zorder=10)
    ax2.scatter(
        R_uti, Z_uti, marker=".", c="cyan", ec="k", lw=0.5, s=30, alpha=0.5, zorder=2
    )
    max_l_R, max_l_z = 2, 0.5
    scale_rz = min(abs(max_l_R / vR), abs(max_l_z / vz))
    ax2.quiver(
        R,
        Z,
        vR * scale_rz,
        vz * scale_rz,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.01,
        zorder=9,
    )
    point_pos_R = abs(R)
    arrow_point_pos_R = abs(R + vR * scale_rz)
    xlim = max(16.0, 1.05 * max(point_pos_R, arrow_point_pos_R))
    ax2.set_xlim(-0.5, xlim)
    point_pos_z = abs(Z)
    arrow_point_pos_z = abs(Z + vz * scale_rz)
    ylim = max(1.0, 1.1 * max(point_pos_z, arrow_point_pos_z))
    ax2.set_ylim(-ylim, ylim)
    ax2.set_xlabel(r"R$_{GC}$ [kpc]")
    ax2.set_ylabel(r"Z$_{GC}$ [kpc]", labelpad=1.5)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    plt.savefig(plot_fpath, dpi=dpi)
    fig.clear()
    plt.close(fig)


def plot_CMD(
    plot_fpath,
    df_membs,
    probs_col="probs",
    title="UCC",
    cmap="plasma",
    dpi=200,
):
    """ """
    plt.style.use(custom_style_path)

    # Sort by probabilities
    df_membs = df_membs.sort_values(probs_col, kind="stable")

    pr = df_membs[probs_col]
    vmin, vmax = min(pr), max(pr)
    if vmin > 0.01:
        if (vmax - vmin) < 0.001:
            vmin -= 0.01
    elif 0 < vmin <= 0.01:
        if (vmax - vmin) < 0.001:
            vmin = 0.0
    else:
        vmax += 0.01

    ec = "grey"
    fs = 7

    if len(df_membs) > 1000:
        size = 5
    elif len(df_membs) > 500:
        size = 10
    elif len(df_membs) > 100:
        size = 20
    else:
        size = 25

    # num=1, clear=True are there to release memory as per.
    # https://stackoverflow.com/a/65910539/1391441
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(5.5, 5.5), num=1, clear=True
    )
    fig.suptitle(f"{title} (N={len(df_membs)})", fontsize=fs + 1)

    ax1.scatter(
        df_membs["GLON"],
        df_membs["GLAT"],
        c=pr,
        alpha=0.8,
        ec=ec,
        lw=0.2,
        s=size,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax1.set_xlabel("GLON [deg]", fontsize=fs)
    ax1.set_ylabel("GLAT [deg]", fontsize=fs)

    xmin, xmax = np.nanmin(df_membs["GLON"]), np.nanmax(df_membs["GLON"])
    ymin, ymax = np.nanmin(df_membs["GLAT"]), np.nanmax(df_membs["GLAT"])
    xr = (xmax - xmin) * 0.15
    yr = (ymax - ymin) * 0.15
    xmin, xmax = xmin - xr, xmax + xr
    ymin, ymax = ymin - yr, ymax + yr
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    ax1.tick_params(axis="both", which="major", labelsize=fs)
    # Get and set font size of the offset text
    offset_text = ax1.xaxis.get_offset_text()
    offset_text.set_fontsize(fs)

    # Set a maximum of 5 xticks
    Nxt = len("".join([label.get_text() for label in ax1.get_xticklabels()]))
    if Nxt > 5:
        ax1.locator_params(axis="x", nbins=5)

    im2 = ax2.scatter(
        df_membs["pmRA"],
        df_membs["pmDE"],
        c=pr,
        alpha=0.8,
        ec=ec,
        lw=0.2,
        s=size,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    x_pos, y_pos, w, h = 0.985, 0.08, 0.02, 0.847
    cb_ax = fig.add_axes((x_pos, y_pos, w, h))
    cbar = fig.colorbar(im2, orientation="vertical", cax=cb_ax)
    # cbar.set_label('Probs')
    cbar.ax.tick_params(labelsize=fs)

    ax2.set_xlabel("pmRA [mas/yr]", fontsize=fs)
    ax2.set_ylabel("pmDE [mas/yr]", fontsize=fs)

    # Plot limits
    xmin, xmax = np.nanmin(df_membs["pmRA"]), np.nanmax(df_membs["pmRA"])
    xr = (xmax - xmin) * 0.1
    xmin, xmax = xmin - xr, xmax + xr
    ymin, ymax = np.nanmin(df_membs["pmDE"]), np.nanmax(df_membs["pmDE"])
    yr = (ymax - ymin) * 0.1
    ymin, ymax = ymin - yr, ymax + yr
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.tick_params(axis="both", which="major", labelsize=fs)

    x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd = diag_limits(
        df_membs["BP-RP"], df_membs["Gmag"]
    )

    ax3.scatter(
        df_membs["Plx"],
        df_membs["Gmag"],
        c=pr,
        alpha=0.8,
        s=size,
        ec=ec,
        lw=0.2,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax3.axvline(np.median(df_membs["Plx"]), ls=":", c="k", lw=2)
    ax3.invert_yaxis()
    ax3.set_xlabel("Plx [mas]", fontsize=fs)
    ax3.set_ylabel("G [mag]", fontsize=fs)
    # Plot limits
    xmin, xmax = np.nanmin(df_membs["Plx"]), np.nanmax(df_membs["Plx"])
    xr = (xmax - xmin) * 0.1
    xmin, xmax = xmin - xr, xmax + xr
    ax3.set_xlim(xmin, xmax)
    ax3.set_ylim(y_min_cmd, y_max_cmd)
    ax3.tick_params(axis="both", which="major", labelsize=fs)

    ax4.scatter(
        df_membs["BP-RP"],
        df_membs["Gmag"],
        c=pr,
        alpha=0.8,
        ec=ec,
        lw=0.2,
        s=size,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax4.invert_yaxis()
    ax4.set_xlabel("BP-RP [mag]", fontsize=fs)
    ax4.set_ylabel("G [mag]", fontsize=fs)
    # Plot limits
    ax4.set_xlim(x_min_cmd, x_max_cmd)
    ax4.set_ylim(y_min_cmd, y_max_cmd)
    ax4.tick_params(axis="both", which="major", labelsize=fs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()

    plt.savefig(plot_fpath, dpi=dpi)
    # https://stackoverflow.com/a/65910539/1391441
    fig.clear()
    plt.close(fig)


# def colorbar(mappable):
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     return fig.colorbar(mappable, cax=cax)


def diag_limits(phot_x, phot_y):
    """
    Define plot limits for *all* photometric diagrams.
    """
    x_delta = np.nanmax(phot_x) - np.nanmin(phot_x)
    x_min_cmd = np.nanmin(phot_x) - 0.05 * x_delta
    x_max_cmd = np.nanmax(phot_x) + 0.05 * x_delta

    # y_median, y_std = np.nanmedian(phot_y), np.nanstd(phot_y)
    # y limits.
    y_min_cmd = np.nanmax(phot_y) + 0.25
    # If photometric axis y is a magnitude, make sure the brightest star
    # is always plotted.
    y_max_cmd = np.nanmin(phot_y) - 0.5

    return x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd


def plot_aladin(
    logging,
    ra,
    dec,
    r_50,
    plot_aladin_fpath,
    dpi=100,
):
    """ """
    plt.style.use("default")

    rad_deg = round(2 * (r_50 / 60.0), 3)
    query_params = {
        "hips": "P/DSS2/color",
        "ra": ra,
        "dec": dec,
        "fov": rad_deg,
        "width": 350,
        "height": 245,
    }

    try:
        url = f"http://alasky.u-strasbg.fr/hips-image-services/hips2fits?{urlencode(query_params)}"
        hdul = fits.open(url)
        rotated_img = ndimage.rotate(hdul[0].data.T, 90)
    except Exception as e:
        logging.error(f"Error generating Aladin plot for {plot_aladin_fpath}: {e}")

    fig, ax = plt.subplots()
    plt.imshow(rotated_img)
    plt.axis("off")

    if rad_deg >= 1:
        fov = str(round(rad_deg, 1)) + "º"
    else:
        fov = str(round(rad_deg * 60, 1)) + "'"

    t = plt.text(
        0.015, 0.02, f"FoV: {fov}", fontsize=14, color="blue", transform=ax.transAxes
    )
    t.set_bbox(dict(facecolor="grey", alpha=0.75, linewidth=0))

    t = plt.text(
        0.56,
        0.02,
        "Click to load Aladin",
        fontsize=14,
        color="red",
        weight="bold",
        transform=ax.transAxes,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.75, linewidth=0))

    plt.scatter(0.5, 0.5, marker="+", s=400, color="#B232B2", transform=ax.transAxes)

    plt.savefig(plot_aladin_fpath, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    # https://stackoverflow.com/a/65910539/1391441
    fig.clear()
    plt.close(fig)


def make_N_vs_year_plot(file_out_name, df_UCC, fontsize=7, dpi=300):
    """ """
    plt.style.use(custom_style_path)

    # Extract minimum year of publication for each catalogued OC
    years = []
    # Go through each OC in the database
    for i, oc in enumerate(df_UCC["DB"]):
        oc_years = []
        # Extract the years of the DBs were this OC is catalogued
        for cat0 in oc.split(";"):
            cat = cat0.split("_")[0]
            oc_years.append(int(cat[-4:]))
        # Keep the smallest year where this OC was catalogued
        years.append(min(oc_years))

    # Count number of OCs per year
    unique, counts = np.unique(years, return_counts=True)
    c_sum = np.cumsum(counts)

    # Combine with old years (previous to 1995)
    #
    # Source: http://www.messier.seds.org/open.html#Messier
    # Messier (1771): 33
    #
    # Source: https://spider.seds.org/ngc/ngc.html
    # "William Herschel first published his catalogue containing 1,000 entries in 1786
    # ...added 1,000 entries in 1789 and a final 500 in 1802 ... total number of entries
    # to 2,500. In 1864, Sir John Herschel the son of William then expanded the
    # catalogue into the General Catalogue of Nebulae and Clusters and Clusters of
    # Stars (GC), which contained 5,079 entries"
    # Herschel (1786): ???
    #
    # Source:
    # https://in-the-sky.org/data/catalogue.php?cat=NGC&const=1&type=OC&sort=0&view=1
    # Dreyer (1888): 640?
    #
    # Source: https://webda.physics.muni.cz/description.html#cluster_level
    # "The catalogue of cluster parameters prepared by Lyngå (1987, 5th edition, CDS
    # VII/92A) has been used to build the list of known open clusters."
    # Going to http://cdsweb.u-strasbg.fr/htbin/myqcat3?VII/92A leads
    # to a Vizier table with 1151 entries
    # Lyngå (1987): 1151
    #
    # Mermilliod 1988 (Bull. Inform. CDS 35, 77-91): 570
    # Mermilliod 1996ASPC...90..475M (BDA, 1996): ~500
    #
    if min(unique) <= 1987:
        raise ValueError("DB year is smaller than 1987, check")
    years = [1771, 1888, 1987] + list(unique)
    min_N = 1151  # Lyngå (1987)
    values = [33, 640, min_N] + list(np.clip(c_sum, a_min=min_N, a_max=np.inf))

    fig = plt.figure(figsize=(4, 2.5))
    plt.plot(years, values, alpha=0.5, lw=3, marker="o", ms=4, color="maroon", zorder=5)

    plt.annotate(
        "Messier",
        xy=(1775, 30),
        xytext=(1820, 30),
        fontsize=fontsize,
        verticalalignment="center",
        # Custom arrow
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )
    # plt.annotate(
    #     "Hipparcos + 2MASS",
    #     xy=(2015, 1000),
    #     xytext=(1850, 1000),  # fontsize=8,
    #     verticalalignment="center",
    #     # Custom arrow
    #     arrowprops=dict(arrowstyle="->", lw=0.7),
    # )
    plt.annotate(
        "Gaia data release",
        xy=(2010, 3600),
        xytext=(1870, 3600),
        fontsize=fontsize,
        verticalalignment="center",
        # Custom arrow
        arrowprops=dict(arrowstyle="->", lw=0.7),
    )

    plt.text(
        x=1880,
        y=len(df_UCC) + 3000,
        s=r"N$_{UCC}$=" + f"{len(df_UCC)}",
        fontsize=fontsize,
    )
    plt.axhline(len(df_UCC), ls=":", lw=2, alpha=0.5, c="grey")

    plt.text(
        x=1820,
        y=120000,
        s="Approximate number of OCs in the Galaxy",
        fontsize=fontsize,
    )
    plt.axhline(100000, ls=":", lw=2, alpha=0.5, c="k")

    plt.xlim(1759, max(years) + 25)
    # plt.title(r"Catalogued OCs in the literature", fontsize=fontsize)
    plt.ylim(20, 250000)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Year of publication", fontsize=fontsize)
    plt.ylabel("Number of catalogued OCs", fontsize=fontsize)
    plt.yscale("log")
    fig.tight_layout()

    plt.savefig(file_out_name, dpi=dpi)
    # https://stackoverflow.com/a/65910539/1391441
    fig.clear()
    plt.close(fig)


def make_classif_plot(path, height, class_order, dpi=300):
    """ """
    plt.style.use(custom_style_path)

    def rescale(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    my_cmap = plt.get_cmap("RdYlGn_r")
    x1 = np.arange(0, len(class_order) + 1)

    fig, ax = plt.subplots(1, figsize=(6, 3))
    plt.bar(class_order, height, color=my_cmap(rescale(x1)))
    plt.xticks(rotation=30)
    plt.ylabel("N")
    ax.tick_params(axis="x", which="both", length=0)
    plt.minorticks_off()
    fig.tight_layout()

    plt.savefig(path, dpi=dpi)
    # https://stackoverflow.com/a/65910539/1391441
    fig.clear()
    plt.close(fig)


def make_UTI_plot(path, UTI_vals, dpi=300):
    """ """
    plt.style.use(custom_style_path)
    my_cmap = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(1, figsize=(6, 3))

    Y, X = np.histogram(UTI_vals, 25)
    x_span = X.max() - X.min()
    C = [my_cmap(((x - X.min()) / x_span)) for x in X]
    width = X[1] - X[0]
    plt.bar(0.5 * width + X[:-1], Y, color=C, width=width)

    plt.xlabel("UTI")
    plt.ylabel("N")

    fig.tight_layout()
    # plt.savefig(path, dpi=dpi)
    # https://stackoverflow.com/a/65910539/1391441
    fig.clear()
    plt.close(fig)


# def make_dbs_year_plot(path, df_UCC, all_dbs_json, dpi=300):
#     """ """
#     plt.style.use("modules/science2.mplstyle")

#     dbs = list(itertools.chain.from_iterable([_.split(';') for _ in df_UCC['DB']]))
#     N_dbs = Counter(dbs)

#     names, height = [], []
#     for i, name in enumerate(list(all_dbs_json.keys())):
#         names.append(all_dbs_json[name]['ref'][1:].split(']')[0])
#         height.append(N_dbs[name])

#     fig, ax = plt.subplots(1, figsize=(5, 10))
#     plt.barh(range(len(names)), height, color='grey', alpha=.5)
#     # plt.scatter(range(len(names)), height)

#     # Add labels on top of the bars
#     for i, value in enumerate(height):
#         # plt.text(i, 1.2, str(names[i]), ha='center', va='center', rotation=0)
#         plt.text(1.2, i, str(names[i]), va='center', ha='left', rotation=0)
#     #     if height[i] > 100:
#     #         plt.text(i, value + 0.1, str(height[i]), ha='center', va='bottom', rotation=30)

#     plt.gca().set_yticks([])
#     # plt.xlabel("Articles")
#     plt.xscale('log')
#     # plt.ylabel("N")
#     # ax.tick_params(axis="x", which="both", length=0)
#     # plt.minorticks_off()
#     # plt.xlim(-1, len(height) + .1)
#     # plt.ylim(1.15, max(height) + 10000)
#     fig.tight_layout()

#     plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
#     # https://stackoverflow.com/a/65910539/1391441
#     fig.clear()
#     plt.close(fig)


# if __name__ == "__main__":
#     import matplotlib

#     matplotlib.use("agg")
#     import pandas as pd
#     from files_handler import update_image

#     plt.style.use("science2.mplstyle")

#     fname0, Qfold = "dutrabica58", "Q1P"
#     # Load datafile with members for this cluster
#     membs_file = f"/home/gabriel/Github/UCC/{Qfold}/datafiles/{fname0}.parquet"
#     df_cl = pd.read_parquet(membs_file)
#     root = "/home/gabriel/Descargas/"
#     plots_path = root + fname0 + ".webp"
#     DRY_RUN, logging = False, None
#     txt = make_plot(DRY_RUN, logging, plots_path, df_cl)
#     print(txt)
