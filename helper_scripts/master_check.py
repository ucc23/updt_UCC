"""
check_ucc.py  –  UCC catalogue consistency checks
===================================================

Available checks
----------------
  1. B_vs_C_pos        – compare B vs C catalogue positions / proper-motions
  2. B_DBs_coords      – cross-DB coordinate consistency within catalogue B
  3. B_DBs_params      – cross-DB parameter consistency within catalogue B
  4. B_vs_membs_coords – compare B center coords with member-file medians

"""

# ---------------------------------------------------------------------------
# Manual exclusions based on previous checks and known issues

# Families to skip in all checks
skip_pfx = ()  # ("hsc", "theia", "dutrabica", "mwsc", "cwnu", "ocsn", "lp")

# OCs to skip in run_coords_bc
skip_names = ()  # ["melotte111", "platais6"]

# OCs to skip in run_dbs_coords
known_bad_oc = {
    "ascc120": ["BICA2019"],
    "ascc123": ["ALFONSO2024"],
    "ascc72": ["BICA2019"],
    "blanco1": ["ALMEIDA2023", "DIAS2021"],
    "graham1": ["BUKOWIECKI2011"],
    "melotte245": ["DIAS2021"],
    "ngc1333": ["HE2022_1"],
    "saurer1": ["FROEBRICH2007"],
    "melotte186": ["DIAS2019"],
    "berkeley58": ["DIAS2021"],
    "berkeley59": ["DIAS2021"],
    "collinder461": ["ALMEIDA2023"],
    "fsr0839": ["GLUSHKOVA2010"],
    "ic5146": ["LADA2003"],
}
# ---------------------------------------------------------------------------


B_cat_path = "../data/UCC_cat_B.csv"
C_cat_path = "../data/UCC_cat_C.csv"
members_path = "../data/zenodo/UCC_members.parquet"
_PARAMS = ["dist", "av", "diff_ext", "age", "met", "mass", "bi_frac", "blue_str"]


def main():
    print(__doc__)

    while True:
        choice = input("Select check [1-4]: ").strip()

        if choice == "1":
            print("Parameters (press Enter to keep default):")
            pos_thr = _prompt_float("position threshold (deg)", 0.5)
            pm_thr = _prompt_float("PM threshold (mas/yr)", 10.0)
            uti_min = _prompt_float("UTI minimum", 0.1)
            run_B_vs_C_pos(pos_thr=pos_thr, pm_thr=pm_thr, uti_min=uti_min)
            break

        elif choice == "2":
            pos_thr = _prompt_float("position threshold (deg)", 1)
            run_B_DBs_coords(pos_thr)
            break

        elif choice == "3":
            _PARAMS_str = ", ".join(_PARAMS)
            while True:
                param = input(f"  Parameter [{_PARAMS_str}]: ").strip().lower()
                if param in _PARAMS:
                    break
                print(f"  Invalid parameter. Choose from: {_PARAMS_str}")
            thr = _prompt_float("fractional deviation threshold", 0.25)
            run_B_DBs_params(param_col=param, median_perc=thr)
            break

        elif choice == "4":
            print("Parameters (press Enter to keep default):")
            dist_thr = _prompt_float("angular separation (deg)", 0.5)
            drad_thr = _prompt_float("normalized separation", 0.25)
            run_B_vs_membs_coords(dist_thr=dist_thr, drad_thr=drad_thr)
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def _prompt_float(label, default):
    raw = input(f"  {label} [{default}]: ").strip()
    return float(raw) if raw else default


def run_B_vs_C_pos(pos_thr, pm_thr, uti_min):
    import webbrowser

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    print(
        f"\nChecking B vs C catalogue consistency "
        f"(pos_thr={pos_thr}°, pm_thr={pm_thr} mas/yr, UTI>{uti_min})\n"
    )

    df1 = pd.read_csv(B_cat_path)
    df1["fname"] = [_.split(";")[0] for _ in df1["fnames"]]
    df2 = pd.read_csv(C_cat_path)
    df = pd.merge(df1, df2, on="fname", suffixes=("_B", "_C"))

    df["dist_2D_x"] = (
        (df["GLON"] - df["GLON_m"]) ** 2 + (df["GLAT"] - df["GLAT_m"]) ** 2
    ) ** 0.5
    df["dist_2D_y"] = (
        (df["pmRA"] - df["pmRA_m"]) ** 2 + (df["pmDE"] - df["pmDE_m"]) ** 2
    ) ** 0.5
    df["dist_plx"] = abs(df["Plx"] - df["Plx_m"])

    # remove manually checked / known-problematic families
    df = df[~df["fname"].isin(skip_names)]
    df = df[~df["fname"].str.startswith(skip_pfx, na=False)]

    msk = (df["UTI"] > uti_min) & (
        (df["dist_2D_x"] > pos_thr) | (df["dist_2D_y"] > pm_thr)
    )
    print(
        f"\n{msk.sum()} clusters flagged "
        f"(pos_thr={pos_thr}°, pm_thr={pm_thr} mas/yr, UTI>{uti_min})\n"
    )

    cols_pos = [
        "fname",
        "GLON",
        "GLAT",
        "GLON_m",
        "GLAT_m",
        "UTI",
        "dist_2D_x",
        # "dist_2D_y",
    ]
    cols_pm = [
        "fname",
        "pmRA",
        "pmDE",
        "pmRA_m",
        "pmDE_m",
        "UTI",
        # "dist_2D_x",
        "dist_2D_y",
    ]

    def fmt(x):
        return f"{x:8.2f}"

    print("\n=== Position differences ===")
    print(
        df.loc[msk, cols_pos]
        .sort_values("dist_2D_x", ascending=False)
        .to_string(index=False, formatters={c: fmt for c in cols_pm[1:]})
    )
    print("\n=== Proper motion differences ===")
    print(
        df.loc[msk, cols_pm]
        .sort_values("dist_2D_y", ascending=False)
        .to_string(index=False, formatters={c: fmt for c in cols_pm[1:]})
    )

    input("\nPress Enter to show interactive plot...")

    names = df["fname"][msk].values
    xp = np.array(df["dist_2D_x"][msk])
    yp = np.array(df["dist_2D_y"][msk])
    color = np.array(df.loc[msk, "UTI"])

    fig, ax = plt.subplots()
    sc = ax.scatter(xp, yp, alpha=0.25, c=color)
    plt.colorbar(sc, label="UTI")
    ax.set_title(f"B vs C  –  N={msk.sum()}")
    ax.set_xlabel("Distance in Position (deg)")
    ax.set_ylabel("Distance in Proper Motion (mas/yr)")

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def update_annot(ind):
        i = ind["ind"][0]
        annot.xy = (float(xp[i]), float(yp[i]))
        annot.set_text(names[i])
        annot.set_visible(True)

    def hover(event):
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                fig.canvas.draw_idle()
            elif annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                i = ind["ind"][0]
                webbrowser.open(f"https://ucc.ar/_clusters/{names[i]}")

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def run_B_DBs_coords(pos_thr):
    import json
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import pandas as pd

    print(
        f"\nChecking cross-DB coordinate consistency within catalogue B (Δ>{pos_thr}°)\n"
    )

    df_B = pd.read_csv(B_cat_path)

    with open("../data/databases_info.json") as f:
        databases_info = json.load(f)

    all_dbs = {}
    for db in Path("../data/databases/").glob("*.csv"):
        db_df = pd.read_csv(db)
        pos = databases_info[db.stem]["pos"]
        if "RA" in pos and "DEC" in pos:
            all_dbs[db.stem] = {
                "RA": db_df[pos["RA"]].values.round(4),
                "DEC": db_df[pos["DEC"]].values.round(4),
            }
        else:
            print(f"[skip] {db.stem}: no position info")

    results = []
    for _, row in df_B.iterrows():
        if row["fnames"].startswith(skip_pfx):
            continue
        fname = row["fnames"].split(";")[0]
        if fname in known_bad_oc:
            continue

        cl_dbs = row["DB"].split(";")
        cl_db_i = [int(x) for x in row["DB_i"].split(";")]
        pairs = [(db, i) for db, i in zip(cl_dbs, cl_db_i) if db in all_dbs]
        if len(pairs) < 2:
            continue
        cl_dbs, cl_db_i = map(list, zip(*pairs))

        ras = [all_dbs[db]["RA"][i] for db, i in zip(cl_dbs, cl_db_i)]
        decs = [all_dbs[db]["DEC"][i] for db, i in zip(cl_dbs, cl_db_i)]

        conflicts = Counter()
        pair_list = []
        max_diff = 0
        for i in range(len(ras)):
            for j in range(i + 1, len(ras)):
                d_ra = abs(ras[i] - ras[j])
                d_ra = min(d_ra, 360 - d_ra)
                d_dec = abs(decs[i] - decs[j])
                max_diff = max(max_diff, d_ra, d_dec)
                if d_ra > pos_thr or d_dec > pos_thr:
                    conflicts[i] += 1
                    conflicts[j] += 1
                    pair_list.append((i, j))

        if conflicts:
            if len(conflicts) == 2 and all(v == 1 for v in conflicts.values()):
                ii, jj = pair_list[0]
                msg = f"{fname:<15} (Δ={max_diff:.2f}°): {cl_dbs[ii]} vs {cl_dbs[jj]}"
            else:
                # offender = max(conflicts, key=conflicts.get)
                offender = np.argmax(list(conflicts.values()))
                msg = (
                    f"{fname:<15} (Δ={max_diff:.2f}°): "
                    f"suspect {cl_dbs[offender]} [{conflicts[offender]} conflicts]"
                )
            results.append((max_diff, msg))

    print(f"\n{len(results)} clusters with coordinate conflicts\n")
    for _, msg in sorted(results, key=lambda x: x[0], reverse=True):
        print(msg)


def run_B_DBs_params(param_col, median_perc, res_max=500):
    import numpy as np
    import pandas as pd

    print(
        f"\nChecking cross-DB consistency for '{param_col}' within catalogue B "
        f"(threshold={100 * median_perc:.0f}% of median, showing top {res_max})\n"
    )

    df_B = pd.read_csv(B_cat_path)

    results = []
    for row in df_B.itertuples(index=False):
        name = str(row.fnames).split(";")[0]
        if name.startswith(skip_pfx):
            continue
        if ";" not in str(row.DB):
            continue

        cl_dbs = str(row.DB).split(";")
        raw_vals = getattr(row, param_col).split(";")

        vals, dbs = [], []
        for db, v in zip(cl_dbs, raw_vals):
            v = v.replace("*", "").strip()
            if v.lower() in ("nan", ""):
                continue
            try:
                vals.append(float(v))
                dbs.append(db)
            except ValueError:
                continue

        if len(vals) < 2:
            continue

        vals = np.asarray(vals)
        med_val = np.median(vals)
        thr = median_perc * abs(med_val)
        diffs = np.abs(vals - med_val)
        bad = np.where(diffs > thr)[0]

        if bad.size:
            max_diff = diffs.max()
            offender = bad[np.argmax(diffs[bad])]
            tag = f"[{bad.size} conflicts]" if bad.size > 1 else ""
            msg = (
                f"{name:<15} (Δ={max_diff:.3g}, "
                f"{100 * max_diff / abs(med_val):.1f}%): "
                f"suspect {dbs[offender]} {tag}".rstrip()
            )
            results.append((bad.size, max_diff, msg))

    results.sort(key=lambda x: (x[1], x[0]), reverse=True)
    print(
        f"\n{len(results)} clusters with '{param_col}' conflicts "
        f"(threshold={100 * median_perc:.0f}% of median)\n"
    )
    for _, _, msg in results[:res_max]:
        print(msg)
    if len(results) > res_max:
        print(f"\n... and {len(results) - res_max} more")


def run_B_vs_membs_coords(dist_thr, drad_thr):
    import numpy as np
    import pandas as pd

    print(
        f"\nChecking B centre coords vs member-file medians "
        f"(dist>{dist_thr}°, d_norm>{drad_thr})\n"
    )

    df_B = pd.read_csv(B_cat_path)
    df_Z = pd.read_parquet(members_path)

    df_B["fname"] = df_B["fnames"].str.split(";").str[0]
    mask_valid = ~df_B["fname"].str.startswith(skip_pfx)

    g = df_Z.groupby("name")

    def circular_span(x):
        x = np.sort(x.to_numpy())
        gaps = np.diff(np.r_[x, x[0] + 360])
        return 360 - gaps.max()

    stats = g.agg(
        RA_Z=("RA_ICRS", "median"),
        DE_Z=("DE_ICRS", "median"),
        n=("GLON", "size"),
    )
    stats["rad"] = g["GLON"].apply(circular_span)
    stats = stats[stats["n"] > 2]

    df = df_B.merge(
        stats[["RA_Z", "DE_Z", "rad"]], left_on="fname", right_index=True, how="left"
    )

    ra1, dec1 = np.deg2rad(df["RA_ICRS"]), np.deg2rad(df["DE_ICRS"])
    ra2, dec2 = np.deg2rad(df["RA_Z"]), np.deg2rad(df["DE_Z"])
    a = (
        np.sin((dec2 - dec1) / 2) ** 2
        + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    )
    df["dist"] = np.rad2deg(2 * np.arcsin(np.sqrt(a)))
    df["d_rad"] = df["dist"] / df["rad"]

    res = df[
        mask_valid & (df["dist"] > dist_thr) & (df["d_rad"] > drad_thr)
    ].sort_values("d_rad", ascending=False)

    print(f"\n{len(res)} clusters flagged  (dist>{dist_thr}°, d_norm>{drad_thr})\n")
    for i, r in res.iterrows():
        print(
            # f"{i:<7}: {r.fname[:14]:<15} | "
            f"{r.fname[:14]:<15} "
            f"dist={r.dist:.2f}° "
            # f"GLON span={r.rad:.2f}° | "
            f"d_norm={r.d_rad:.2f} | "
            f"cat=({r.RA_ICRS:.2f}, {r.DE_ICRS:.2f})  "
            f"memb=({r.RA_Z:.2f}, {r.DE_Z:.2f})"
        )


if __name__ == "__main__":
    main()
