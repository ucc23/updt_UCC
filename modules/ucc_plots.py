
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


def make_plot(out_path, fnames, df, N_membs_min, cmap='cividis', dpi=200):
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

    ax3.scatter(df_field['Plx'], df_field['Gmag'], c='grey', alpha=.3, s=10,
                marker='x')
    ax3.scatter(df_membs['Plx'], df_membs['Gmag'], c=pr, alpha=.8, s=20,
                ec='k', lw=.35, cmap=cmap)
    ax3.axvline(np.median(df_membs['Plx']), ls=':', c='k', lw=2)
    ax3.invert_yaxis()
    ax3.set_xlabel("Plx [mas]")
    ax3.set_ylabel("Gmag")

    ax4.scatter(
        df_field['BP-RP'], df_field['Gmag'], c='grey', alpha=.3, ec='w', lw=.35,
        s=10)
    ax4.scatter(
        df_membs['BP-RP'], df_membs['Gmag'], c=pr, alpha=.8, ec='k', lw=.35,
        s=20, cmap=cmap)
    ax4.invert_yaxis()
    ax4.set_xlabel("BP-RP")
    ax4.set_ylabel("Gmag")

    fig.tight_layout()
    clname = fnames.split(';')[0]
    plt.savefig(out_path + clname + ".png", dpi=dpi)


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
