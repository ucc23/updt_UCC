"""
check_ucc.py  –  UCC catalogue consistency checks
===================================================

Available checks
----------------
  1. B_DBs_check       – cross-DB consistency within catalogue B for
                         position, proper motion, or parallax
  2. B_vs_membs_coords – compare B center coords with member-file medians
  3. B_DBs_params      – cross-DB parameter consistency within catalogue B
  4. B_vs_C_pos        – compare B vs C catalogue positions / proper-motions

"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Manual exclusions based on previous checks and known issues

# Families to skip in all checks
skip_pfx = ("hsc", "theia", "cwnu", "ocsn")

oc_cmmts = {
    "melotte186": {"pos": "DIAS2019 (as Collinder 359) bad RA"},
    "melotte111": {"pos": "Very large frame"},
    "graham1": {"pos": "BUKOWIECKI2011 bad DEC"},
    "ns14": {"pos": "JOSHI2016 (as BIP 14) bad RA"},
    "ascc123": {"pos": "ALFONSO2024 bad RA,Plx"},
    "ic5146": {"pos": "LADA2003 bad RA"},
    "fsr0839": {"pos": "GLUSHKOVA2010 (as Koposov 41) bad RA"},
    "ascc120": {"pos": "BICA2019 bad RA"},
    "ruprecht40": {"pos": "ALTER1970 bad RA"},
    "upk655": {"pos": "CHI2023_2 bad RA"},
    "upk367": {"pos": "SIM2019 slightly off center"},
    "dbsb88": {"pos": "MERCER2005 & MORALES2013 bad DEC"},
    "upk287": {"pos": "HUNT2023 and YANG2025 slightly off center"},
    "ads16795": {"pos": "CAVALLO2024 bad RA"},
    "platais6": {"pos": "CAVALLO2024 bad DEC"},
    "upk569": {"pos": "Frame is large"},
    "ngc1981": {"pos": "HE2022_1 bad RA"},
    "fsr1017": {"pos": "CAVALLO2024 bad DEC"},
    "upk160": {"pos": "HE2022_1 bad RA"},
    "ascc93": {"pos": "BICA2019 bad RA,DEC"},
    "rsg7": {"pos": "ROSER2016 slightly off DEC"},
    "teutsch39": {"pos": "HE2022_1 bad DEC"},
    "l1641s": {"pos": "LADA2003 bad RA"},
    "upk303": {"pos": "Frame is large"},
    "kronberger59": {"pos": "TADROSS2014 bad RA"},
    "oc0450": {"pos": "CAVALLO2024 bad DEC"},
    "collinder464": {"pos": "Two groups, probably no OC"},
    "upk612": {"pos": "Frame is large"},
    "eso48901": {"pos": "Frame is large"},
    "collinder132": {"pos": "RICHER2021 bad RA"},
    "upk64": {"pos": "QIN2023 bad RA"},
    "platais3": {"pos": "Frame is large"},
    "upk452": {"pos": "TARRICQ2022 bad RA"},
    "ngc6858": {"pos": "HE2022_1 bad DEC"},
    "loden143": {"pos": "RICHER2021 bad RA"},
    "upk526": {"pos": "CAVALLO2024 bad RA"},
    "oc0470": {"pos": "CHI2023 bad DEC"},
    "rsg8": {"pos": "HE2022_1 bad DEC"},
    "deltacepheicluster": {"pos": "Frame is large"},
    "ubc178": {"pos": "HE2022_1 bad DEC"},
    "ngc6846": {"pos": "ALTER1970 & SULENTIC1973 bad DEC"},
    "teutsch181": {"pos": "CAVALLO2024 bad RA"},
    "mamajek2": {"pos": "CAVALLO2024 bad DEC"},
    "ngc6775": {"pos": "BUKOWIECKI2011 bad DEC"},
    "upk378": {"pos": "CAVALLO2024 bad RA"},
    "bdsb120": {"pos": "BICA2003_1 bad DEC"},
    "loden915": {"pos": "CAVALLO2024 bad RA"},
    "collinder471": {"pos": "Old coords shifted in new articles"},
    "platais9": {"pos": "Frame is large"},
    "ascc21": {"pos": "ALFONSO2024 & LI2023 bad DEC"},
    "chamaleoni": {"pos": "Two sub-clusters"},
    "upk233": {"pos": "Frame is large"},
    "fof2439": {"pos": "CHI2023_2 bad RA"},
    "taurus": {"pos": "HE2022_1 bad DEC"},
    "ngc2184": {"pos": "Theia 1408 mixed up in frame"},
    "ubc21": {"pos": "ALFONSO2024 bad DEC"},
    "upk545": {"pos": "CAVALLO2024 bad DEC"},
    "ngc7438": {"pos": "CAVALLO2024 bad RA"},
    "upk537": {"pos": "Frame is large"},
    "lynga4": {"pos": "VDBH1975 bad RA"},
    "upk24": {"pos": "Frame is large"},
    "upk88": {"pos": "Frame is large"},
    "upk294": {"pos": "Frame is large"},
    "ngc6618": {"pos": "CAVALLO2024 bad DEC"},
    "oc0185": {"pos": "CAVALLO2024 bad DEC"},
    "upk70": {"pos": "HE2022_1 bad RA"},
    "alessiteutsch5": {"pos": "HE2022_1, ALFONSO2024 bad RA"},
    "avenihunter1": {"pos": "HE2022_1, ALFONSO2024, CAVALLO2024 bad DEC"},
    "ubc17b": {"pos": "HE2022_1 bad RA,DEC"},
    "bdsb91": {"pos": "ALFONSO2024 bad DEC"},
    "bdsb125": {"pos": "BICA2003_1 bad DEC"},
    "melotte227": {"pos": "No object probably"},
    "ruprecht167": {"pos": "Dias, Kharchenko bad RA"},
    "ngc5045": {"pos": "Scattered RA and DEC"},
    "fsr0416": {"pos": "DIAS2021 and ALMEIDA2025 bad RA"},
    "alessi145": {"pos": "TARRICQ2022 bad DEC"},
    "platais5": {"pos": "No object probably"},
    "vdbh34": {"pos": "No object probably"},
    "bdsb124": {"pos": "BICA2003_1 bad DEC"},
    "ascc33": {"pos": "Frame is large"},
    "lisciii3668": {"pos": "Frame is large"},
    "ascc19": {"pos": "HE2022_1 bad DEC"},
    "upk72": {"pos": "Frame is large"},
    "upk422": {"pos": "CAVALLO2024 bad DEC"},
    "upk442": {"pos": "Frame is large"},
    "ascc100": {"pos": "CAVALLO2024 bad DEC"},
    "ascc79": {"pos": "HAO2022 bad RA"},
    "ubc10a": {"pos": "Centers are scattered"},
    "pismis22": {"pos": "VDBH1975 bad RA"},
    "upk524": {"pos": "Frame is large"},
    "gulliver10": {"pos": "ALFONSO2024 bad RA"},
    "upk71": {"pos": "HE2022_1 bad DEC"},
    "ruprecht31": {"pos": "QIN2023 bad DEC"},
    "upk533": {"pos": "Frame is large"},
    "ascc24": {"pos": "CAVALLO2024 bad RA"},
    "vdbh56": {"pos": "RICHER2021 bad RA,DEC"},
    "upk41": {"pos": "HAO2022 bad DEC"},
    "upk585": {"pos": "Centers are scattered"},
    "ruprecht53": {"pos": "DIAS2016 bad DEC"},
    "collinder65": {"pos": "No object probably"},
    "gulliver20": {"pos": "TARRICQ2022 bad DEC"},
    "alessi33": {"pos": "HAO2022 bad RA"},
    "upk560": {"pos": "Frame is large"},
    "pismis24": {"pos": "VDBH1975 bad RA"},
    "upk535": {"pos": "Frame is large"},
    "ubc31": {"pos": "Frame is large"},
    "upk260": {"pos": "Frame is large"},
    "mamajek3": {"pos": "Frame is large"},
    "upk78": {"pos": "Frame is large"},
    "upk438": {"pos": "Frame is large"},
    "upk494": {"pos": "Frame is large"},
    "upk305": {"pos": "Frame is large"},
    "fsr1785": {"pos": "Frame is large"},
    "upk307": {"pos": "Frame is large"},
    "lisc3534": {"pos": "Frame is large"},
    "teutsch141": {"pos": "No object probably"},
    "upk241": {"pos": "HUNT2023 bad RA"},
    "upk300": {"pos": "Frame is large"},
    "ngc2039": {"pos": "Teo groups of coords, no object probably"},
    "upk214": {"pos": "Frame is large"},
}

# ---------------------------------------------------------------------------


# Per-check-type config for run_B_DBs_check. `cat_cols` are the keys expected
# under databases_info.json[<db>][<check_type>], e.g. "pos": {"RA": ..., "DEC": ...}.
_CHECK_CONFIG = {
    "pos": {
        "label": "position",
        "unit": "deg",
        "default_thr": 0.5,
        "default_norm_thr": 0.025,
        "cat_cols": ("RA", "DEC"),
    },
    "pm": {
        "label": "proper motion",
        "unit": "mas/yr",
        "default_thr": 1.0,
        "default_norm_thr": 0.25,
        "cat_cols": ("pmra", "pmde"),
    },
    "plx": {
        "label": "parallax",
        "unit": "mas",
        "default_thr": 0.1,
        "default_norm_thr": 0.1,
        "cat_cols": ("plx",),
    },
}

B_cat_path = "../data/UCC_cat_B.csv"
C_cat_path = "../data/UCC_cat_C.csv"
members_path = "../data/zenodo/UCC_members.parquet"
_PARAMS = ["dist", "av", "diff_ext", "age", "met", "mass", "bi_frac", "blue_str"]


def main():
    print(__doc__)

    while True:
        choice = input("Select check [1-4]: ").strip()

        if choice == "1":
            types_str = ", ".join(_CHECK_CONFIG)
            while True:
                check_type = (
                    input(f"  check type [{types_str}] (pos): ").strip().lower()
                    or "pos"
                )
                if check_type in _CHECK_CONFIG:
                    break
                print(f"  Invalid check type. Choose from: {types_str}")
            cfg = _CHECK_CONFIG[check_type]
            thr = _prompt_float(
                f"{cfg['label']} threshold ({cfg['unit']})", cfg["default_thr"]
            )
            drad_thr = _prompt_float("normalized separation", cfg["default_norm_thr"])
            mode = (
                input("  mode [conflicts/groups] (conflicts): ").strip().lower()
                or "conflicts"
            )
            run_B_DBs_check(check_type, thr, drad_thr, mode)
            break

        elif choice == "2":
            print("Parameters (press Enter to keep default):")
            # cat_select = input("  catalog [B/C]: ").strip().upper()
            pos_thr = _prompt_float("angular separation (deg)", 0.5)
            drad_thr = _prompt_float("normalized separation", 0.25)
            run_B_vs_membs_coords(pos_thr=pos_thr, drad_thr=drad_thr)
            break

        elif choice == "3":
            _PARAMS_str = ", ".join(_PARAMS)
            while True:
                param = input(f"  Parameter [{_PARAMS_str}]: ").strip().lower()
                if param in _PARAMS:
                    break
                print(f"  Invalid parameter. Choose from: {_PARAMS_str}")
            def_thresh = 1
            if param == "age":
                def_thresh = 0.25
            thr = _prompt_float("fractional deviation threshold", def_thresh)
            run_B_DBs_params(param_col=param, median_perc=thr)
            break

        elif choice == "4":
            print("Parameters (press Enter to keep default):")
            pos_thr = _prompt_float("position threshold (deg)", 1)
            pm_thr = _prompt_float("PM threshold (mas/yr)", 10.0)
            uti_min = _prompt_float("UTI minimum", 0.1)
            run_B_vs_C_pos(pos_thr=pos_thr, pm_thr=pm_thr, uti_min=uti_min)
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def _prompt_float(label, default):
    raw = input(f"  {label} [{default}]: ").strip()
    return float(raw) if raw else default


def run_B_vs_C_pos(pos_thr, pm_thr, uti_min):
    import webbrowser

    import matplotlib.pyplot as plt

    print(
        f"\nChecking B vs C catalogue consistency "
        f"(pos_thr={pos_thr}°, pm_thr={pm_thr} mas/yr, UTI>{uti_min})\n"
    )

    df1 = pd.read_csv(B_cat_path)
    # df1["fname"] = [_.split(";")[0] for _ in df1["fnames"]]
    df2 = pd.read_csv(C_cat_path)
    df = pd.merge(df1, df2, on="fname", suffixes=("_B", "_C"))

    df["dist_2D_x"] = (
        (df["GLON"] - df["GLON_m"]) ** 2 + (df["GLAT"] - df["GLAT_m"]) ** 2
    ) ** 0.5
    df["dist_2D_y"] = (
        (df["pmRA"] - df["pmRA_m"]) ** 2 + (df["pmDE"] - df["pmDE_m"]) ** 2
    ) ** 0.5
    # df["dist_plx"] = abs(df["Plx"] - df["Plx_m"])

    # remove manually checked / known-problematic families
    df = df[~df["fname"].isin(oc_cmmts)]
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
        bbox={"boxstyle": "round", "fc": "w"},
        arrowprops={"arrowstyle": "->"},
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


def run_B_DBs_check(check_type, thr, drad_thr, mode="conflicts"):
    """Cross-DB consistency check within catalogue B.

    Parameters
    ----------
    check_type : str
        Quantity to compare: "pos", "pm", or "plx".
    thr : float
        Maximum allowed difference between DB values.
    drad_thr : float
        Minimum normalized distance required for an object to be checked.
    mode : str, optional
        Check method:
        - "conflicts": identify individual DBs involved in discrepant pairs.
        - "groups": identify objects split into two or more distinct groups.

    Notes
    -----
    In "groups" mode, measurements separated by <= `thr` are connected.
    Connected components containing at least two DBs are considered
    well-defined groups. Isolated DB measurements are ignored.
    """
    if mode not in {"conflicts", "groups"}:
        raise ValueError("mode must be 'conflicts' or 'groups'")

    cfg = _CHECK_CONFIG[check_type]
    cat_cols = cfg["cat_cols"]
    wrap_ra = check_type == "pos"

    print(
        f"\nChecking cross-DB {cfg['label']} consistency within catalogue B "
        f"(Δ>{thr} {cfg['unit']}, mode={mode})\n"
    )

    df_B = pd.read_csv(B_cat_path)
    df_B = get_norm_dist_rad(df_B)

    with open("../data/databases_info.json") as f:
        databases_info = json.load(f)

    # Load the relevant values from each database.
    all_dbs = {}

    for db in Path("../data/databases/").glob("*.csv"):
        db_df = pd.read_csv(db)
        db_cols = databases_info[db.stem].get(check_type)

        if db_cols and all(c in db_cols for c in cat_cols):
            all_dbs[db.stem] = {
                col: db_df[db_cols[col]].values.round(4) for col in cat_cols
            }
        else:
            print(f"[skip] {db.stem}: no {cfg['label']} info")

    results = []
    for _, row in df_B.iterrows():
        fname = row["fname"]

        if fname.startswith(skip_pfx) or row.dist_norm < drad_thr:
            continue

        # Keep only contributing DBs with the requested information.
        cl_dbs = row["DB"].split(";")
        cl_db_i = [int(x) for x in row["DB_i"].split(";")]

        pairs = [(db, i) for db, i in zip(cl_dbs, cl_db_i) if db in all_dbs]

        if len(pairs) < 2:
            continue

        cl_dbs, cl_db_i = map(list, zip(*pairs))

        vals = {
            col: [all_dbs[db][col][i] for db, i in zip(cl_dbs, cl_db_i)]
            for col in cat_cols
        }

        n_db = len(cl_dbs)

        # Calculate all pairwise differences once.
        pair_diffs = {}
        max_diff = 0.0
        for i in range(n_db):
            for j in range(i + 1, n_db):
                diffs = []

                for col in cat_cols:
                    d = abs(vals[col][i] - vals[col][j])
                    if wrap_ra and col == "RA":
                        d = min(d, 360.0 - d)
                    diffs.append(d)

                pair_diff = max(diffs)
                pair_diffs[(i, j)] = pair_diff
                max_diff = max(max_diff, pair_diff)

        ucc_dist_to_median = ""
        if check_type == "pos":
            ucc_dist_to_median = f", {row.coords_dist:.2f}"

        # -------------------------------------------------------------
        # Pairwise-conflict method.
        # -------------------------------------------------------------
        if mode == "conflicts":
            conflicts = Counter()
            pair_list = []

            for (i, j), pair_diff in pair_diffs.items():
                if pair_diff > thr:
                    conflicts[i] += 1
                    conflicts[j] += 1
                    pair_list.append((i, j))

            if not conflicts:
                continue

            if len(conflicts) == 2 and all(v == 1 for v in conflicts.values()):
                ii, jj = pair_list[0]

                msg = (
                    f"{fname:<15} "
                    f"(Δ={max_diff:.2f}, {row.dist_norm:.2f}"
                    f"{ucc_dist_to_median}): "
                    f"{cl_dbs[ii]} vs {cl_dbs[jj]}"
                )

            else:
                # DB appearing most often in conflicts is the most
                # likely individual offender.
                offender = conflicts.most_common(1)[0][0]

                msg = (
                    f"{fname:<15} "
                    f"(Δ={max_diff:.2f}, {row.dist_norm:.2f}"
                    f"{ucc_dist_to_median}): "
                    f"suspect {cl_dbs[offender]} "
                    f"[{conflicts[offender]} conflicts]"
                )

            if fname in oc_cmmts and check_type in oc_cmmts[fname]:
                msg += f"  # {oc_cmmts[fname][check_type]}"

            results.append((max_diff, ucc_dist_to_median, msg))

        # -------------------------------------------------------------
        # Multiple-group detection method.
        # -------------------------------------------------------------
        else:
            adjacency = [set() for _ in range(n_db)]

            for (i, j), pair_diff in pair_diffs.items():
                if pair_diff <= thr:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

            # Find connected components.
            groups = []
            visited = set()

            for start in range(n_db):
                if start in visited:
                    continue

                stack = [start]
                group = []

                while stack:
                    i = stack.pop()

                    if i in visited:
                        continue

                    visited.add(i)
                    group.append(i)
                    stack.extend(adjacency[i] - visited)

                groups.append(group)

            # Single DBs are treated as outliers, not groups.
            defined_groups = [group for group in groups if len(group) >= 2]

            # Require at least two clearly defined groups.
            if len(defined_groups) < 2:
                continue
            defined_groups.sort(key=len, reverse=True)

            group_txt = " | ".join(str(len(group)) for group in defined_groups)

            total_dbs = sum(len(group) for group in defined_groups)

            # Maximum distance between every pair of defined groups.
            group_diffs = []

            for gi in range(len(defined_groups)):
                for gj in range(gi + 1, len(defined_groups)):
                    group_i = defined_groups[gi]
                    group_j = defined_groups[gj]

                    max_group_diff = max(
                        pair_diffs[tuple(sorted((i, j)))]
                        for i in group_i
                        for j in group_j
                    )

                    group_diffs.append(max_group_diff)

            group_diffs_txt = ", ".join(f"{d:.2f}" for d in group_diffs)

            max_group_diff = max(group_diffs)

            msg = f"{fname[:14]:<15} ({group_diffs_txt}): {group_txt}"

            if fname in oc_cmmts and check_type in oc_cmmts[fname]:
                msg += f"  # {oc_cmmts[fname][check_type]}"

            results.append((total_dbs, max_group_diff, msg))

    # -----------------------------------------------------------------
    # Print results.
    # -----------------------------------------------------------------
    ucc_coords_col = ""
    if check_type == "pos":
        ucc_coords_col = f", UCC dist to median [{cfg['unit']}]"

    if mode == "conflicts":
        print(f"\n{len(results)} clusters with {cfg['label']} conflicts\n")
        print(
            f"{'Name':<15} "
            f"(Δ=max_diff [{cfg['unit']}], "
            f"dist_norm (dist/rad){ucc_coords_col}): "
            f"suspects (...)"
        )
        print("---------------------------------------------------")
        for _, _, msg in sorted(
            results,
            key=lambda x: (x[1], x[0]),
            reverse=True,
        ):
            print(msg)
    else:
        print(f"\n{len(results)} clusters with multiple {cfg['label']} groups\n")
        print(f"{'Name':<15} (Δ_groups [{cfg['unit']}]): groups")
        print("---------------------------------------------------")
        # Largest total number of DBs first, then largest separation.
        for _, _, msg in sorted(
            results,
            key=lambda x: (x[0], x[1]),
            reverse=True,
        ):
            print(msg)


def run_B_DBs_params(param_col, median_perc, res_max=500):
    import numpy as np
    import pandas as pd

    txt = "'" + param_col + "'"
    if param_col == "age":
        txt = "'age' (log)"
    if param_col == "dist":
        txt = "'dist' (dm)"

    print(
        f"\nChecking cross-DB consistency for {txt} within catalogue B "
        f"(threshold={100 * median_perc:.0f}% of median, showing top {res_max})"
    )

    df_B = pd.read_csv(B_cat_path)

    results = []
    for row in df_B.itertuples(index=False):
        name = str(row.fname).split(";")[0]
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

        # Transform parameters before comparison
        if param_col == "age":
            # Myr -> log10(age/yr) (add 1 Myr to avoid errors)
            vals = np.log10((1 + vals) * 1e6)
        elif param_col == "dist":
            # pc -> distance modulus (add 11 pc to avoid errors)
            vals = 5 * np.log10(11 + vals) - 5

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


def run_B_vs_membs_coords(pos_thr, drad_thr, cat_select="B"):
    """
    Compare the cluster center coordinates in catalogue B (or C) with the
    median coordinates of their member stars. Clusters with a separation
    greater than `pos_thr` degrees and a normalized separation greater than
    `drad_thr` are flagged for further inspection.
    """
    print(
        f"\nChecking {cat_select} center coords vs member-file medians "
        f"(dist>{pos_thr}°, dist_norm>{drad_thr})"
    )

    # if cat_select == "B":
    df_B = pd.read_csv(B_cat_path)
    df_C = pd.read_csv(C_cat_path)
    df_B["UTI"] = df_C["UTI"]
    # else:
    #     df_BC = pd.read_csv(C_cat_path)
    #     # Rename RA_ICRS_m, DE_ICRS_m columns
    #     df_BC = df_BC.rename(columns={"RA_ICRS_m": "RA_ICRS", "DE_ICRS_m": "DE_ICRS"})

    mask_valid = ~df_B["fname"].str.startswith(skip_pfx)

    df = get_norm_dist_rad(df_B)

    res = df[
        mask_valid & (df["coords_dist"] > pos_thr) & (df["dist_norm"] > drad_thr)
    ].sort_values("dist_norm", ascending=False)

    print(f"\n{len(res)} clusters flagged  (dist>{pos_thr}° & dist_norm>{drad_thr})\n")
    for i, r in res.iterrows():
        print(
            f"{r.fname[:14]:<15} (UTI={r.UTI:.2f})  "
            f"dist={r.coords_dist:.2f}° "
            # f"GLON span={r.rad:.2f}° | "
            f"dist_norm={r.dist_norm:.2f} | "
            f"cat=({r.RA_ICRS:.2f}, {r.DE_ICRS:.2f})  "
            f"memb=({r.RA_median:.2f}, {r.DE_median:.2f})"
            f" # {oc_cmmts.get(r.fname, {}).get('pos', '')}"
        )


def get_norm_dist_rad(df_B: pd.DataFrame) -> pd.DataFrame:
    """Calculate cluster-center offsets normalized by the member angular span.

    For each cluster, the median member coordinates are used to estimate its
    center, while the angular span in Galactic longitude is used as a
    characteristic radius. The angular separation between this estimated
    center and the catalogue coordinates is then calculated and normalized
    by the cluster radius.

    Clusters with fewer than three members are excluded from the member-based
    statistics because their median positions and angular spans are considered
    unreliable.

    Parameters
    ----------
    df_B : pd.DataFrame
        Cluster catalogue containing at least the columns ``fname``,
        ``RA_ICRS``, and ``DE_ICRS``.

    Returns
    -------
    pd.DataFrame
        Input catalogue augmented with the columns ``RA_median``, ``DE_median``,
        ``rad``, ``dist``, and ``dist_norm``. Clusters without sufficient
        member information have NaN values in these columns.
    """

    # Load the member catalogue and group stars by cluster name.
    df_M = pd.read_parquet(members_path)
    df_M_gr = df_M.groupby("name")

    # Estimate the cluster center from the median member coordinates and
    # count the number of available members for each cluster.
    stats = df_M_gr.agg(
        RA_median=("RA_ICRS", "median"),
        DE_median=("DE_ICRS", "median"),
        nmembs=("GLON", "size"),
    )

    # Calculate the angular span of each cluster, correctly accounting for
    # the circular nature of Galactic longitude.
    stats["rad"] = df_M_gr["GLON"].apply(circular_span)

    # Ignore clusters with fewer than three members, since their median
    # positions and angular spans are poorly constrained.
    stats = stats[stats["nmembs"] > 2]

    # Add the member-derived center coordinates and angular radius to the
    # input cluster catalogue, matching clusters by their names.
    df = df_B.merge(
        stats[["RA_median", "DE_median", "rad"]],
        left_on="fname",
        right_index=True,
        how="left",
    )

    # Convert catalogue and member-derived equatorial coordinates from
    # degrees to radians for the spherical-distance calculation.
    ra1 = np.deg2rad(df["RA_ICRS"])
    dec1 = np.deg2rad(df["DE_ICRS"])
    ra2 = np.deg2rad(df["RA_median"])
    dec2 = np.deg2rad(df["DE_median"])

    # Calculate the angular separation between both cluster centers using
    # the haversine formula for distances on a sphere.
    a = (
        np.sin((dec2 - dec1) / 2) ** 2
        + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    )
    df["coords_dist"] = np.rad2deg(2 * np.arcsin(np.sqrt(a)))

    # Normalize the center separation by the member-derived cluster radius.
    df["dist_norm"] = df["coords_dist"] / df["rad"]

    return df


def circular_span(x):
    if not isinstance(x, np.ndarray):
        x = x.to_numpy()
    x = np.sort(x)
    gaps = np.diff(np.r_[x, x[0] + 360])
    return 360 - gaps.max()


if __name__ == "__main__":
    main()
