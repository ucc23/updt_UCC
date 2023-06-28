
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import ZScaleInterval

import scienceplots
plt.style.use('science')

# matplotlib.rc('font', family='sans-serif') 
# matplotlib.rc('font', serif='Helvetica Neue') 
# matplotlib.rc('text', usetex='false') 
# matplotlib.rcParams.update({'font.size': 22})


def make_plot(out_path, fname0, df, N_membs_min, cmap='viridis', dpi=200):
    """
    """
    # Select members and field stars
    msk_membs = df['probs'] > 0.5
    if msk_membs.sum() < N_membs_min:
        # Select the first '' stars assuming that the data is order by
        # probabilities
        idx = np.arange(N_membs_min)
        msk_membs = np.full(len(df), False)
        msk_membs[idx] = True
    df_membs = df[msk_membs]
    df_field = df[~msk_membs]

    pr = df_membs['probs']
    vmin = min(pr)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5.5, 5))

    ax1.scatter(
        df_field['GLON'], df_field['GLAT'], c='grey', alpha=.3, ec='w', lw=.35,
        s=10)
    ax1.scatter(
        df_membs['GLON'], df_membs['GLAT'], c=pr, alpha=.8, ec='k', lw=.35,
        s=mag_size(df_membs['Gmag'].values), cmap=cmap)
    ax1.set_xlabel("GLON")
    ax1.set_ylabel("GLAT")

    ax2.scatter(
        df_field['pmRA'], df_field['pmDE'], c='grey', alpha=.3, ec='w', lw=.35,
        s=10)
    im2 = ax2.scatter(
        df_membs['pmRA'], df_membs['pmDE'], c=pr, alpha=.8, ec='k', lw=.35,
        s=mag_size(df_membs['Gmag'].values), vmin=vmin, cmap=cmap)

    # x_pos, y_pos, w, h = .98, 0.59, .02, .38
    x_pos, y_pos, w, h = .985, 0.103, 0.02, 0.866
    cb_ax = fig.add_axes([x_pos, y_pos, w, h])
    cbar = fig.colorbar(im2, orientation='vertical', cax=cb_ax)
    cbar.set_label('Probs')

    ax2.set_xlabel("pmRA [mas/yr]")
    ax2.set_ylabel("pmDE [mas/yr]")
    # Plot limits
    pmra_c, pmde_c = np.median(df_membs['pmRA']), np.median(df_membs['pmDE'])
    xrad = np.percentile(abs(pmra_c-df_membs['pmRA']), 95) * 2
    yrad = np.percentile(abs(pmde_c-df_membs['pmDE']), 95) * 2
    ax2.set_xlim(pmra_c-xrad, pmra_c+xrad)
    ax2.set_ylim(pmde_c-yrad, pmde_c+yrad)

    ax3.scatter(df_field['Plx'], df_field['Gmag'], c='grey', alpha=.3, s=10,
                marker='x')
    ax3.scatter(df_membs['Plx'], df_membs['Gmag'], c=pr, alpha=.8, s=20,
                ec='k', lw=.35, cmap=cmap)
    ax3.axvline(np.median(df_membs['Plx']), ls=':', c='k', lw=2)
    ax3.invert_yaxis()
    ax3.set_xlabel("Plx [mas]")
    ax3.set_ylabel("Gmag")
    # Plot limits
    plx_c = np.median(df_membs['Plx'])
    xrad = np.percentile(abs(plx_c-df_membs['Plx']), 95) * 2
    ax3.set_xlim(plx_c-xrad, plx_c+xrad)
    ax3.set_ylim(max(df_membs['Gmag']) + .2, min(df_membs['Gmag']) - .5)

    ax4.scatter(
        df_field['BP-RP'], df_field['Gmag'], c='grey', alpha=.3, ec='w', lw=.35,
        s=10)
    ax4.scatter(
        df_membs['BP-RP'], df_membs['Gmag'], c=pr, alpha=.8, ec='k', lw=.35,
        s=20, cmap=cmap)
    ax4.invert_yaxis()
    ax4.set_xlabel("BP-RP")
    ax4.set_ylabel("Gmag")
    # Plot limits
    x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd = diag_limits(
        df_membs['BP-RP'], df_membs['Gmag'])
    ax4.set_xlim(x_min_cmd, x_max_cmd)
    ax4.set_ylim(y_min_cmd, y_max_cmd)

    fig.tight_layout()
    plt.savefig(out_path + fname0 + ".png", dpi=dpi)


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def mag_size(mag):
    """
    Convert magnitudes into intensities and define sizes of stars in
    finding chart.
    """
    N = len(mag)
    interval = ZScaleInterval()
    zmin, zmax = interval.get_limits(mag)

    mag = mag.clip(zmin, zmax)
    factor = 500. * (1 - 1 / (1 + 150 / N ** 0.85))
    flux = (10 ** ((mag - zmin) / -2.5))
    sizes = 10 + factor * flux
    return sizes


def plx_size(plx, max_s=100):
    plx = np.clip(plx, a_min=0.01, a_max=20)
    delta_p = plx.max() - plx.min()
    m = max_s / delta_p
    h = -m * plx.min()
    s = 10 + (h + m * plx)
    return s


def diag_limits(phot_x, phot_y):
    """
    Define plot limits for *all* photometric diagrams.
    """
    x_delta = np.nanmax(phot_x) - np.nanmin(phot_x)
    x_min_cmd = min(phot_x) - .2 * x_delta
    x_max_cmd = max(phot_x) + .1 * x_delta

    # y_median, y_std = np.nanmedian(phot_y), np.nanstd(phot_y)
    # y limits.
    y_min_cmd = np.nanmax(phot_y) + .5
    # If photometric axis y is a magnitude, make sure the brightest star
    # is always plotted.
    y_max_cmd = np.nanmin(phot_y) - 1.

    return x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd
