# Add ..modules.utils to the path to allow import
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

sys.path.append("/home/gabriel/Github/UCC/updt_UCC/")
from modules.utils import plx_to_pc

# New spiral arms
# https://www.aanda.org/articles/aa/full_html/2021/08/aa39751-20/aa39751-20.html

# Momany et al. (2006) "Outer structure of the..."
spiral_arms = {
    "Outer": (
        (7.559999999999999, 2.514124293785308),
        (7.237333333333332, 3.4180790960452008),
        (6.738666666666667, 4.46327683615819),
        (6.1519999999999975, 5.480225988700564),
        (5.535999999999998, 6.384180790960453),
        (4.861333333333334, 7.1186440677966125),
        (4.479999999999997, 7.514124293785311),
        (3.8053333333333335, 7.909604519774014),
        (2.8373333333333335, 8.474576271186443),
        (1.751999999999999, 9.011299435028253),
        (0.6373333333333306, 9.35028248587571),
        (-0.4480000000000022, 9.63276836158192),
        (-1.2693333333333356, 9.830508474576273),
        (-1.8560000000000016, 9.943502824858761),
        (-2.677333333333337, 10),
        (-3.4986666666666686, 9.943502824858761),
        (-4.085333333333336, 9.915254237288138),
        (-5.024000000000003, 9.830508474576273),
        (-5.845333333333336, 9.689265536723166),
        (-6.4906666666666695, 9.519774011299436),
        (-7.253333333333337, 9.322033898305087),
        (-7.986666666666669, 8.926553672316388),
        (-8.485333333333337, 8.559322033898308),
        (-9.160000000000004, 8.050847457627121),
        (-9.864000000000004, 7.570621468926557),
        (-10.333333333333336, 7.203389830508474),
        (-10.89066666666667, 6.525423728813561),
        (-11.389333333333337, 5.9604519774011315),
        (-11.77066666666667, 5.395480225988699),
        (-12.240000000000004, 4.661016949152543),
        (-12.65066666666667, 3.9830508474576263),
        (-13.061333333333337, 3.361581920903955),
        (-13.442666666666671, 2.7401129943502838),
    ),
    "Perseus": (
        (5.2719999999999985, 2.372881355932204),
        (4.9786666666666655, 3.27683615819209),
        (4.773333333333333, 3.6723163841807924),
        (4.3039999999999985, 4.46327683615819),
        (3.776, 5.141242937853107),
        (3.3359999999999985, 5.593220338983052),
        (2.690666666666665, 6.073446327683616),
        (2.074666666666662, 6.468926553672318),
        (1.4879999999999995, 6.751412429378529),
        (0.607999999999997, 7.005649717514125),
        (-0.15466666666666917, 7.146892655367235),
        (-0.8293333333333344, 7.259887005649716),
        (-1.621333333333336, 7.316384180790962),
        (-2.4426666666666694, 7.288135593220339),
        (-3.14666666666667, 7.1186440677966125),
        (-3.909333333333336, 7.005649717514125),
        (-4.6426666666666705, 6.666666666666668),
        (-5.3173333333333375, 6.3559322033898304),
        (-5.962666666666669, 5.903954802259886),
        (-6.57866666666667, 5.395480225988699),
        (-6.989333333333336, 5.028248587570623),
        (-7.63466666666667, 4.350282485875709),
        (-8.104000000000003, 3.7570621468926575),
        (-8.51466666666667, 3.10734463276836),
        (-8.896000000000004, 2.4011299435028235),
        (-9.218666666666671, 1.6384180790960414),
        (-9.424000000000003, 1.073446327683616),
        (-9.65866666666667, 0.1977401129943459),
        (-9.805333333333337, -0.5084745762711869),
        (-9.92266666666667, -0.8192090395480243),
        (-10.010666666666669, -1.2711864406779654),
        (-10.128000000000004, -2.005649717514128),
        (-10.186666666666671, -2.711864406779661),
        (-10.186666666666671, -2.909604519774014),
        (-10.128000000000004, -3.3615819209039586),
        (-9.952000000000004, -4.2372881355932215),
        (-9.83466666666667, -5),
        (-9.688000000000002, -5.310734463276839),
        (-9.48266666666667, -5.734463276836161),
        (-9.189333333333337, -6.29943502824859),
        (-8.86666666666667, -7.005649717514126),
        (-8.456000000000003, -7.598870056497178),
    ),
    "Orion-Cygnus": (
        (-7.341333333333337, 3.2485875706214706),
        (-7.63466666666667, 2.909604519774014),
        (-7.9280000000000035, 2.485875706214692),
        (-8.280000000000003, 1.9209039548022595),
        (-8.456000000000003, 1.4124293785310726),
        (-8.60266666666667, 1.1016949152542352),
        (-8.808000000000003, 0.5649717514124291),
        (-9.013333333333335, -0.197740112994353),
        (-9.13066666666667, -0.7627118644067821),
        (-9.160000000000004, -1.2146892655367267),
    ),
    "Carina-Sagittarius": (
        (2.8373333333333335, 3.6723163841807924),
        (2.5146666666666633, 4.152542372881356),
        (2.338666666666665, 4.350282485875709),
        (1.9280000000000008, 4.774011299435031),
        (1.2239999999999966, 5.16949152542373),
        (0.49066666666666237, 5.451977401129945),
        (-0.12533333333333552, 5.593220338983052),
        (-0.7706666666666688, 5.621468926553675),
        (-1.7680000000000025, 5.53672316384181),
        (-2.5600000000000023, 5.310734463276834),
        (-2.970666666666668, 5.112994350282488),
        (-3.645333333333337, 4.576271186440678),
        (-3.8800000000000026, 4.350282485875709),
        (-4.2613333333333365, 3.9830508474576263),
        (-4.613333333333337, 3.5593220338983045),
        (-4.789333333333337, 3.192090395480225),
        (-5.1413333333333355, 2.627118644067796),
        (-5.493333333333336, 2.033898305084744),
        (-5.845333333333336, 1.4124293785310726),
        (-6.22666666666667, 0.6497175141242906),
        (-6.608000000000002, -0.14124293785311082),
        (-6.842666666666668, -0.7627118644067821),
        (-7.048000000000004, -1.5536723163841835),
        (-7.19466666666667, -2.25988700564972),
        (-7.253333333333337, -3.1073446327683634),
        (-7.136000000000003, -3.6440677966101696),
        (-7.048000000000004, -3.98305084745763),
        (-6.813333333333338, -4.519774011299436),
        (-6.461333333333336, -5.254237288135597),
        (-6.05066666666667, -5.875706214689268),
        (-5.6106666666666705, -6.440677966101699),
        (-5.024000000000003, -6.977401129943505),
        (-4.466666666666669, -7.485875706214692),
        (-3.8800000000000026, -7.909604519774014),
        (-3.352000000000002, -8.163841807909607),
        (-3.0880000000000027, -8.361581920903957),
    ),
    "Crux-Scutum": (
        (1.663999999999998, 3.1355932203389827),
        (1.3119999999999976, 3.4180790960452008),
        (0.6666666666666643, 3.8135593220338997),
        (-0.09600000000000186, 3.9830508474576263),
        (-0.858666666666668, 4.0395480225988685),
        (-1.5626666666666686, 3.926553672316384),
        (-2.2666666666666693, 3.5875706214689274),
        (-2.9413333333333362, 3.192090395480225),
        (-3.5280000000000022, 2.6553672316384187),
        (-3.909333333333336, 2.033898305084744),
        (-4.29066666666667, 1.3276836158192076),
        (-4.584000000000003, 0.6214689265536713),
        (-4.760000000000003, -0.11299435028248794),
        (-4.906666666666668, -0.9322033898305087),
        (-4.965333333333335, -1.6666666666666714),
        (-4.994666666666669, -2.344632768361585),
        (-4.8480000000000025, -3.2203389830508478),
        (-4.672000000000002, -3.8983050847457648),
        (-4.320000000000004, -4.576271186440678),
        (-3.8800000000000026, -5.225988700564974),
        (-3.4106666666666694, -5.734463276836161),
        (-2.765333333333336, -6.158192090395483),
        (-2.032000000000002, -6.610169491525427),
        (-1.3866666666666685, -6.920903954802263),
        (-0.6826666666666696, -7.06214689265537),
        (0.05066666666666464, -7.25988700564972),
        (0.8719999999999999, -7.401129943502829),
        (1.575999999999997, -7.372881355932208),
        (2.2799999999999976, -7.316384180790964),
    ),
    "Norma": (
        (-3.14666666666667, 0.8474576271186436),
        (-3.3226666666666684, 0.1977401129943459),
        (-3.2933333333333366, -0.7627118644067821),
        (-3.2346666666666692, -1.4406779661016955),
        (-2.970666666666668, -2.1186440677966125),
        (-2.5600000000000023, -2.824858757062149),
        (-2.1200000000000028, -3.4463276836158236),
        (-1.6506666666666696, -3.9548022598870105),
        (-0.9760000000000026, -4.378531073446332),
        (-0.2720000000000038, -4.689265536723166),
        (0.4319999999999986, -4.858757062146896),
        (0.9600000000000009, -4.887005649717519),
        (1.370666666666665, -4.858757062146896),
        (2.0453333333333283, -4.717514124293789),
        (2.6613333333333316, -4.519774011299436),
        (3.042666666666662, -4.406779661016952),
        (3.4826666666666632, -4.152542372881356),
        (4.010666666666662, -3.7288135593220346),
        (4.538666666666664, -3.050847457627121),
    ),
}


def main():
    df = pd.read_csv("UCC_cat_C.csv")

    # kde_plot = get_KDE(df["X_GC"], df["Y_GC"], df["UTI"])
    msk = df["UTI"] > 0.5
    Z_uti = df["Z_GC"][msk]
    R_uti = df["R_GC"][msk]

    vx, vy, vz, vR = velocity(
        df["RA_ICRS_m"].values,
        df["DE_ICRS_m"].values,
        df["Plx_m"].values,
        df["pmRA_m"].values,
        df["pmDE_m"].values,
        df["Rv_m"].values,
        df["X_GC"].values,
        df["Y_GC"].values,
        df["Z_GC"].values,
        df["R_GC"].values,
    )

    # Plot random entries in (X_GC, Y_GC) and (R_GC, Z_GC)
    idx = np.random.choice(len(df), size=len(df), replace=False)
    for i in range(len(df)):
        fname = df["fname"][i]
        # if fname not in ("ubc605",):
        #     continue
        print(f"{i}, {fname}")
        mplot(
            Z_uti,
            R_uti,
            fname,
            df["X_GC"][i],
            df["Y_GC"][i],
            df["Z_GC"][i],
            df["R_GC"][i],
            vx[i],
            vy[i],
            vz[i],
            vR[i],
        )


def mplot(
    Z_uti,
    R_uti,
    fname,
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
):
    # fig = plt.figure(figsize=(6, 9))  # Taller than wide
    # gs = GridSpec(4, 3, figure=fig)
    fig = plt.figure(figsize=(6, 3))  # Taller than wide
    gs = GridSpec(1, 2, figure=fig, wspace=0.05)
    plt.style.use("/home/gabriel/Github/UCC/updt_UCC/modules/D_funcs/science2.mplstyle")

    plt.suptitle(f"({X:.2f}, {Y:.2f}, {Z:.2f}) [kpc]")

    # ax1 = fig.add_subplot(gs[0:3, :])
    ax1 = fig.add_subplot(gs[0])
    # ax1.set_aspect("equal")  # Square in data coordinates
    # ax1.set_title(f"({X:.2f}, {Y:.2f}, {Z:.2f}) [kpc]")

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

    # # Add warning text to top of the plot
    # if R == plx_min:
    #     ax1.text(
    #         0,
    #         0.85 * rmax,
    #         "Warning: parallax is too small, distance was capped",
    #         ha="center",
    #         fontsize=9,
    #     )  # color="red",

    # Plot spiral arms
    for arm, coords in spiral_arms.items():
        armx, army = np.array(coords).T
        ax1.plot(armx, army, linestyle="--", label=arm, lw=2, zorder=3)

    # xx, yy, zz, levels = kde_plot
    # ax1.contour(xx, yy, zz, levels=levels, colors="cyan", linewidths=1.5)
    # ax1.scatter(X_all, Y_all, marker=".", c="k", s=10, alpha=0.25, zorder=2)

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
    # ax1.legend(fontsize=7, loc="lower right", facecolor="w", framealpha=1, frameon=True)
    ax1.set_xlabel(r"X$_{GC}$ [kpc]")
    ax1.set_ylabel(r"Y$_{GC}$ [kpc]", labelpad=1.5)
    ax1.set_xlim(-rmax, rmax)
    ax1.set_ylim(-rmax, rmax)

    # ax2 = fig.add_subplot(gs[3, :])
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

    fig.savefig(
        f"/home/gabriel/Github/UCC/plots/plots_{fname[0]}/gcpos/{fname}.webp",
        format="webp",
        dpi=150,
    )
    plt.close()


def get_KDE(X, Y, UTI, UTI_min=0.5, N_levels=10, bw=0.2, N_grid=100):
    msk = UTI > UTI_min
    x = X[msk].values
    y = Y[msk].values

    # Build grid
    nx = ny = N_grid
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))

    # zz = kde2d_explicit(x, y, xx, yy)
    # Evaluate 2D KDE on grid (xx, yy) with explicit bandwidths (hx, hy).
    # xx, yy: 2D meshgrid arrays
    X = xx[..., None]
    Y = yy[..., None]
    dx = (X - x) / bw
    dy = (Y - y) / bw
    z = np.exp(-0.5 * (dx * dx + dy * dy))
    zz = z.sum(axis=-1)
    zz /= len(x) * 2 * np.pi * bw * bw

    # Get contour levels
    levels = np.linspace(zz.min(), zz.max(), N_levels + 1)[1:]

    return (xx, yy, zz, levels)


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


if __name__ == "__main__":
    main()
