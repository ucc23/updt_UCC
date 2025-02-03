import warnings
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage


def plot_CMD(
    plot_fpath,
    df_membs,
    title="UCC",
    cmap="plasma",
    dpi=200,
):
    """ """
    # This is a modified style that removes the Latex dependence from the
    # 'scienceplots' package
    plt.style.use("modules/update_site/science2.mplstyle")

    # Sort by probabilities
    df_membs.sort_values("probs", inplace=True, kind="stable")

    pr = df_membs["probs"]
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
    xr = (xmax - xmin) * 0.1
    yr = (ymax - ymin) * 0.1
    xmin, xmax = xmin - xr, xmax + xr
    ymin, ymax = ymin - yr, ymax + yr
    if xmax - xmin < 0.1:
        xmin, xmax = xmin - 0.05, xmax + 0.05
    if ymax - ymin < 0.1:
        ymin, ymax = ymin - 0.05, ymax + 0.05
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    ax1.tick_params(axis="both", which="major", labelsize=fs)

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
    cb_ax = fig.add_axes([x_pos, y_pos, w, h])
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


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


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
    except Exception as _:
        return "ERROR"

    rotated_img = ndimage.rotate(hdul[0].data.T, 90)
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
    plt.style.use("modules/update_site/science2.mplstyle")

    # Extract minimum year of publication for each catalogued OC
    years = []
    for i, oc in enumerate(df_UCC["DB"]):
        oc_years = []
        for cat0 in oc.split(";"):
            cat = cat0.split("_")[0]
            oc_years.append(int(cat[-4:]))
        # Store the smallest year where this OC was catalogued
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
    min_N = 1151
    values = [33, 640, min_N] + list(np.clip(c_sum, a_min=min_N, a_max=np.inf))

    fig = plt.figure(figsize=(4, 2.5))
    plt.plot(years, values, alpha=0.5, lw=3, marker="o", ms=7, color="maroon", zorder=5)

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
    plt.yscale("log")
    fig.tight_layout()

    plt.savefig(file_out_name, dpi=dpi)
    # https://stackoverflow.com/a/65910539/1391441
    fig.clear()
    plt.close(fig)


def make_classif_plot(path, height, class_order, dpi=300):
    """ """
    plt.style.use("modules/update_site/science2.mplstyle")

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
