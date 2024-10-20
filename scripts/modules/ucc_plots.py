import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from urllib.parse import urlencode
from astropy.io import fits
from scipy import ndimage


def make_plot(plot_fpath, df_membs, cmap="plasma", dpi=200):
    """ """
    # Load local style file taken from https://github.com/garrettj403/SciencePlots
    # but with a line commented out to avoid an issue with LaTeX and the
    # logging module: https://github.com/garrettj403/SciencePlots/issues/103
    plt.style.use("./modules/science.mplstyle")

    pr = df_membs["probs"]
    vmin = min(pr)

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
        2, 2, figsize=(5.5, 5), num=1, clear=True
    )

    ax1.scatter(
        df_membs["GLON"],
        df_membs["GLAT"],
        c=pr,
        alpha=0.8,
        ec=ec,
        lw=0.2,
        s=size,
        cmap=cmap,
    )
    ax1.set_xlabel("GLON", fontsize=fs)
    ax1.set_ylabel("GLAT", fontsize=fs)

    # xmin, xmax = np.nanmin(df_membs['GLON']), np.nanmax(df_membs['GLON'])
    # ymin, ymax = np.nanmin(df_membs['GLAT']), np.nanmax(df_membs['GLAT'])
    # xr, yr = xmax - xmin, ymax - ymin
    # rad = max(xr, yr) * .6
    # xc, yc = np.nanmedian(df_membs['GLON']), np.nanmedian(df_membs['GLAT'])
    # ax1.set_xlim(xc - rad, xc + rad)
    # ax1.set_ylim(yc - rad, yc + rad)

    xmin, xmax = np.nanmin(df_membs["GLON"]), np.nanmax(df_membs["GLON"])
    ymin, ymax = np.nanmin(df_membs["GLAT"]), np.nanmax(df_membs["GLAT"])
    xr = (xmax - xmin) * 0.1
    yr = (ymax - ymin) * 0.1
    xmin, xmax = xmin - xr, xmax + xr
    ymin, ymax = ymin - yr, ymax + yr
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
        cmap=cmap,
    )

    # x_pos, y_pos, w, h = .98, 0.59, .02, .38
    x_pos, y_pos, w, h = 0.985, 0.103, 0.02, 0.866
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
        cmap=cmap,
    )
    ax3.axvline(np.median(df_membs["Plx"]), ls=":", c="k", lw=2)
    ax3.invert_yaxis()
    ax3.set_xlabel("Plx [mas]", fontsize=fs)
    ax3.set_ylabel("G", fontsize=fs)
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
        cmap=cmap,
    )
    ax4.invert_yaxis()
    ax4.set_xlabel("BP-RP", fontsize=fs)
    ax4.set_ylabel("G", fontsize=fs)
    # Plot limits
    ax4.set_xlim(x_min_cmd, x_max_cmd)
    ax4.set_ylim(y_min_cmd, y_max_cmd)
    ax4.tick_params(axis="both", which="major", labelsize=fs)

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


def make_aladin_plot(ra, dec, r_50, plot_aladin_fpath, dpi=100):
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
        return

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
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":
    import pandas as pd

    fname0 = "melotte111"
    # Load datafile with members for this cluster
    membs_file = f"/home/gabriel/Github/UCC/Q3P/datafiles/{fname0}.parquet"
    df_cl = pd.read_parquet(membs_file)
    plots_path = "/home/gabriel/Descargas/"
    make_plot(plots_path, fname0, df_cl)
