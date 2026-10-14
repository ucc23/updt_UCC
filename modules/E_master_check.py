import hashlib
import itertools
import json
import textwrap
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .utils import members_hashes
from .variables import (
    B_VS_C_CONFIG,
    CHECK_CONFIG,
    CMMT_KEYS,
    CMMTS_HASH_DEPS,
    CMMTS_HASH_PARTS,
    DUP_CHECKS,
    DUP_COLS_ORDER,
    DUP_CONFIG,
    DUP_RUN_PARS,
    H23_MEMBS_CONFIG,
    LIT_DIST_CONFIG,
    MEMBS_COORDS_CONFIG,
    PARAMS_CONFIG,
    PROX_DUP_CONFIG,
    SCORE_CONFIG,
    E_B_cat_path,
    E_C_cat_path,
    E_cmmts_record_file,
    E_db_path,
    E_dbs_json_path,
    E_json_path,
    E_members_path,
    E_out_file,
    E_out_width,
    E_record_path,
    N_max_clusters,
    UCC_cat_E_in,
    exclude_DBs,
    include_DBs,
    skip_pfx,
)

# Warnings issued during a run (see `_warn`), also stored in the output file
_WARNINGS = []


def main():
    """Run all the UCC catalogue consistency checks and store the results.

    No user input is required: every check runs with the default parameters
    stored in `CHECK_CONFIG`, `MEMBS_COORDS_CONFIG`, `PARAMS_CONFIG`,
    `B_VS_C_CONFIG`, `DUP_CONFIG`, `PROX_DUP_CONFIG`, `H23_MEMBS_CONFIG`,
    and `LIT_DIST_CONFIG`.

    Checks
    ------
    1. Cross-DB position / proper motion / parallax consistency within
       catalogue B, in "conflicts" and "groups" modes (`run_B_DBs_check`)
    2. Catalogue B center coordinates vs member-file medians
       (`run_B_vs_C_coords`)
    3. Cross-DB parameter consistency within catalogue B, for every parameter
       in `PARAMS_CONFIG` (`run_B_DBs_params`)
    4. Catalogue B vs C proper motions and parallaxes (`run_B_vs_C_pm`,
       `run_B_vs_C_plx`)
    5. Candidate duplicates from shared members (`run_dup_check`), and from
       proximity in position, parallax and proper motion (`run_prox_dup`)
    6. Number of members vs Hunt & Reffert (2023) (`run_H23_membs`)
    7. Literature distance vs member parallax (`run_lit_dist_plx`)

    Every failure is scored (see `_score_flags`), and all the candidate
    duplicates plus the `N_max_clusters` other clusters with the highest total
    scores are written to `E_out_file`, along with their failures and the manual
    comments in `E_json_path`.

    The integrity of catalogues B and C and of the members file is checked
    (see `check_integrity`), with every problem issued as a warning.

    The manual comments are validated (see `load_cmmts`), those whose
    check inputs changed since they were stored are flagged as stale (see
    `check_cmmts_hashes`), and those whose check no longer fails are reported
    (see `check_orphan_cmmts`).

    Every warning issued (see `_warn`), including Python warnings raised by
    the checks, is printed and also stored in the output file.
    """
    _WARNINGS.clear()
    with warnings.catch_warnings(record=True) as py_warns:
        warnings.simplefilter("default")
        checks, clusters, oc_cmmts, stale, UTI = _run_checks()
    for w in py_warns:
        _warn(
            f"{w.category.__name__}: {w.message} ({Path(w.filename).name}:{w.lineno})"
        )

    _write_out_file(checks, clusters, oc_cmmts, stale, UTI)
    dup_cls, other_cls = _split_dup_clusters(clusters)
    print(
        f"\n{_N_flagged(clusters)} clusters flagged (not counting those with "
        f"only minor failures); the {len(dup_cls)} candidate "
        f"duplicates and the {len(other_cls)} other clusters with the highest "
        f"scores were stored in: {E_out_file}"
    )


def _warn(msg):
    """Print a warning and store it for the output file."""
    print(f"WARNING: {msg}")
    _WARNINGS.append(msg)


def _run_checks():
    """Load the data, run every check and score the failures (see `main`).

    Returns the checks performed, the scored clusters, the manual comments,
    the stale comments and the UTI of every cluster.
    """
    print("UCC catalogue consistency checks\n")

    print("Loading catalogues and member files...")
    df_B = pd.read_csv(E_B_cat_path)
    df_C = pd.read_csv(E_C_cat_path)
    df_members = pd.read_parquet(E_members_path)
    with open(E_dbs_json_path) as f:
        databases_info = json.load(f)
    check_DBs_selection(databases_info)
    check_integrity(df_B, df_C, df_members)

    oc_cmmts = load_cmmts(df_B["fname"])
    stale = check_cmmts_hashes(oc_cmmts, df_members, databases_info)

    df_B_norm = get_norm_dist_rad(df_B, df_members)

    # (check IDs, description, parameters, flags) for every check performed
    checks = []

    print("Running check 1: cross-DB position / proper motion / parallax (B)")
    for check_type, cfg in CHECK_CONFIG.items():
        thr, drad_thr = cfg["default_thr"], cfg["default_norm_thr"]
        rel_thr = cfg.get("default_rel_thr")
        flags = run_B_DBs_check(
            df_B_norm, databases_info, check_type, thr, drad_thr, rel_thr
        )
        pars = f"Δ>{thr:g} {cfg['unit']}"
        if rel_thr is not None:
            pars = f"Δ>max({thr:g} {cfg['unit']}, {100 * rel_thr:g}% of median)"
        if drad_thr is not None:
            pars += f", dist_norm>={drad_thr:g}"
        for mode in ("conflicts", "groups"):
            check = f"DBs_{check_type}_{mode}"
            desc = f"Cross-DB {cfg['label']} consistency in B ({mode} mode)"
            checks.append(
                (check, desc, pars, [f for f in flags if f["check"] == check])
            )

    pos_thr, drad_thr = (MEMBS_COORDS_CONFIG[k] for k in ("pos_thr", "drad_thr"))
    print("Running check 2: B center coords vs member-file medians")
    checks.append(
        (
            "B_vs_C_coords",
            "B center coords vs member-file medians",
            f"dist>{pos_thr:g} arcmin, dist_norm>{drad_thr:g}",
            run_B_vs_C_coords(df_B_norm, df_C, pos_thr, drad_thr),
        )
    )

    print("Running check 3: cross-DB parameter consistency (B)")
    for param, pcfg in PARAMS_CONFIG.items():
        thr = pcfg["default_thr"]
        if pcfg["kind"] == "relative":
            pars = f"Δ>{100 * thr:g}% of median ({pcfg['unit']})"
        else:
            pars = f"Δ>{thr:g} {pcfg['unit']}".rstrip()
        checks.append(
            (
                f"params_{param}",
                f"Cross-DB '{param}' consistency in B",
                pars,
                run_B_DBs_params(df_B, param, thr),
            )
        )

    bc = B_VS_C_CONFIG
    pm_thr, uti_min = bc["pm_thr"], bc["uti_min"]
    print("Running check 4: B vs C proper motions and parallaxes")
    checks.append(
        (
            "B_vs_C_pm",
            "B vs C proper motions",
            f"Δ>{pm_thr:g} mas/yr, UTI>{uti_min:g}",
            run_B_vs_C_pm(df_B, df_C, pm_thr, uti_min),
        )
    )
    checks.append(
        (
            "B_vs_C_plx",
            "B vs C parallaxes",
            (
                f"Δ>max({bc['plx_thr']:g} mas, {100 * bc['plx_rel']:g}% of Plx_m), "
                f"UTI>{uti_min:g}"
            ),
            run_B_vs_C_plx(df_B, df_C),
        )
    )

    prob_min = DUP_CONFIG["prob_min"]
    print("Running check 5: candidate duplicates from shared members / proximity")
    dup_flags = run_dup_check(df_B, df_C, databases_info, prob_min)
    for N_run, check in zip(DUP_RUN_PARS, DUP_CHECKS):
        run = DUP_RUN_PARS[N_run]
        multi = {False: "single", True: "multi", None: "all"}[run["multi"]]
        checks.append(
            (
                check,
                (
                    f"Candidate duplicates from shared members (C), run {N_run} "
                    f"({multi}, {run['date']})"
                ),
                f"shared members>={prob_min:g}%",
                [f for f in dup_flags if f["check"] == check],
            )
        )
    pc = PROX_DUP_CONFIG
    checks.append(
        (
            "dup_prox",
            "Candidate duplicates from proximity, without shared members (C)",
            (
                f"sep<{pc['sep_fac']:g}*max(r_50), ΔPlx<=max({pc['plx_thr']:g} mas, "
                f"{100 * pc['plx_rel']:g}%), Δpm<=max({pc['pm_thr']:g} mas/yr, "
                f"{pc['v_thr']:g} km/s)"
            ),
            run_prox_dup(df_B, df_C, databases_info),
        )
    )

    thr, N_bins = H23_MEMBS_CONFIG["thr"], H23_MEMBS_CONFIG["N_bins"]
    print("Running check 6: number of members vs HUNT2023")
    bins_txt = ", ".join(f"{f:g} for max(N)<={n}" for n, f in N_bins)
    checks.append(
        (
            "H23_membs",
            f"Number of members vs Hunt & Reffert (2023); f={bins_txt}, 1 above",
            f"max(N)/min(N)>1+{thr:g}*f",
            run_H23_membs(df_B, df_members),
        )
    )

    lc = LIT_DIST_CONFIG
    print("Running check 7: literature distance vs member parallax")
    checks.append(
        (
            "lit_dist_plx",
            "Median literature distance (B) vs member parallax (C)",
            f"Δμ>{lc['thr']:g} mag, Plx_m+{lc['plx_zp']:g}>{lc['plx_min']:g} mas",
            run_lit_dist_plx(df_B, df_C),
        )
    )

    all_flags = [f for *_, flags in checks for f in flags]
    check_orphan_cmmts(oc_cmmts, all_flags)

    UTI = dict(zip(df_C["fname"], df_C["UTI"]))
    clusters = _score_flags(all_flags, oc_cmmts, stale, UTI)
    return checks, clusters, oc_cmmts, stale, UTI


def check_DBs_selection(databases_info):
    """Raise a ValueError if `exclude_DBs` or `include_DBs` name a check that
    does not accept a DB selection, or a DB not in the databases JSON file."""
    valid = {f"DBs_{t}_{m}" for t in CHECK_CONFIG for m in ("conflicts", "groups")}
    valid |= {"params"} | {f"params_{p}" for p in PARAMS_CONFIG}
    errors = []
    for name, sel in (("exclude_DBs", exclude_DBs), ("include_DBs", include_DBs)):
        for check, dbs in sel.items():
            if check not in valid:
                errors.append(f"{name}: check '{check}' does not accept DB selection")
            errors += [
                f"{name}: {check}: unknown DB '{db}'"
                for db in dbs
                if db not in databases_info
            ]
    if errors:
        raise ValueError(
            "Invalid 'exclude_DBs' / 'include_DBs' (checks accepting a DB "
            f"selection: {', '.join(sorted(valid))}):\n  " + "\n  ".join(errors)
        )


def _skip_DB(*checks):
    """Function telling whether a DB is ignored in a check, given its check
    IDs from the most to the least specific (e.g. "params_dist", "params").
    The first ID with an entry in `include_DBs` keeps only those DBs;
    otherwise the DBs in `exclude_DBs` for any of the IDs are ignored."""
    for check in checks:
        if check in include_DBs:
            incl = set(include_DBs[check])
            return lambda db: db not in incl
    excl = set().union(*(exclude_DBs.get(c, ()) for c in checks))
    return lambda db: db in excl


def load_cmmts(fnames):
    """Load the manual comments in E_json_path and check their format.

    The file must hold a single object {fname: {key: comment}}, with no
    repeated fnames or keys, every key in `CMMT_KEYS`, and every comment a
    non-empty string; a ValueError listing every problem is raised
    otherwise. Comments for clusters not in catalogue B (`fnames`) never
    apply to any check, and are reported with a warning.
    """

    def no_dup_keys(pairs):
        dups = [k for k, n in Counter(k for k, _ in pairs).items() if n > 1]
        if dups:
            errors.append(f"repeated keys: {', '.join(dups)}")
        return dict(pairs)

    errors = []
    with open(E_json_path) as f:
        try:
            oc_cmmts = json.load(f, object_pairs_hook=no_dup_keys)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{E_json_path}': {e}") from None

    if not isinstance(oc_cmmts, dict):
        errors.append("the file must hold a single object {fname: {key: comment}}")
        oc_cmmts = {}
    for fname, cmmts in oc_cmmts.items():
        if not isinstance(cmmts, dict) or not cmmts:
            errors.append(f"{fname}: comments must be a non-empty object")
            continue
        for key, cmmt in cmmts.items():
            if key not in CMMT_KEYS:
                errors.append(f"{fname}: unknown key '{key}'")
            if not isinstance(cmmt, str) or not cmmt.strip():
                errors.append(
                    f"{fname}: comment for '{key}' must be a non-empty string"
                )
    if errors:
        raise ValueError(
            f"Invalid file '{E_json_path}' (valid keys: "
            f"{', '.join(sorted(CMMT_KEYS))}):\n  " + "\n  ".join(errors)
        )

    # Keep the file sorted alphabetically by fname, rewriting it only if the
    # order changed
    if list(oc_cmmts) != sorted(oc_cmmts):
        oc_cmmts = dict(sorted(oc_cmmts.items()))
        with open(E_json_path, "w") as f:
            json.dump(oc_cmmts, f, indent=4, ensure_ascii=False)
        print(f"Sorted '{UCC_cat_E_in}' alphabetically")

    missing = sorted(set(oc_cmmts) - set(fnames))
    if missing:
        _warn(
            f"{len(missing)} clusters with manual comments in "
            f"'{UCC_cat_E_in}' are not in catalogue B: "
            + ", ".join(repr(m) for m in missing)
        )
    return oc_cmmts


def _hash(txt):
    """Short (16 hex digits) SHA-256 hash of a text."""
    return hashlib.sha256(txt.encode()).hexdigest()[:16]


def _cmmts_hashes(oc_cmmts, df_B, df_C, df_members, databases_info, dbs_dir=E_db_path):
    """Current hashes of each manual comment in E_json_path, and of the
    inputs of its checks (see `CMMTS_HASH_DEPS`): one row per (cluster, key).
    Hashes of parts not used by the key are those of an empty text, and all
    of them are empty if the cluster is missing from B.

    `df_B` and `df_C` must be read as text so that the hashes do not depend
    on the inferred dtypes, and so must the DB files in `dbs_dir`. Members are
    hashed as in the D script (see `members_hashes`).
    """
    B = df_B.drop_duplicates("fname").set_index("fname")
    B["DB_1"] = B["DB"].str.split(";").str[0]
    C = df_C.drop_duplicates("fname").set_index("fname")
    df_M = members_hashes(df_members[df_members["name"].isin(oc_cmmts)])
    M_hash = dict(zip(df_M["name"], df_M["hash"]))

    # DB file columns hashed for each key: {key: {DB: (columns)}}
    db_cols = {}
    for key, deps in CMMTS_HASH_DEPS.items():
        db_cols[key] = deps["DB"]
        if isinstance(deps["DB"], str):
            cat_cols = CHECK_CONFIG[deps["DB"]]["cat_cols"]
            db_cols[key] = {
                db: tuple(info["pos"][c] for c in cat_cols)
                for db, info in databases_info.items()
                if all(c in (info.get("pos") or {}) for c in cat_cols)
            }

    # Columns read from each DB file used by the commented clusters
    used = {db for f in oc_cmmts if f in B.index for db in B.at[f, "DB"].split(";")}
    read_cols = {}
    for cols in db_cols.values():
        for db, c in cols.items():
            if db in used:
                read_cols.setdefault(db, set()).update(c)
    db_vals = {
        db: pd.read_csv(
            dbs_dir / f"{db}.csv", usecols=list(c), dtype=str, keep_default_na=False
        )
        for db, c in read_cols.items()
    }

    def db_txt(fname, cols):
        dbs, idxs = B.at[fname, "DB"].split(";"), B.at[fname, "DB_i"].split(";")
        return ";".join(
            f"{db}:" + ",".join(db_vals[db].iloc[int(i)][list(cols[db])])
            for db, i in zip(dbs, idxs)
            if db in cols
        )

    def cols_txt(df, fname, cols):
        if fname not in df.index:
            return ""
        return "\x1f".join(df.loc[fname, list(cols)])

    rows = []
    for fname, cmmts in oc_cmmts.items():
        for key, cmmt in cmmts.items():
            row = {"name": fname, "key": key, "cmmt_hash": _hash(cmmt)}
            row.update(dict.fromkeys((c for c, _ in CMMTS_HASH_PARTS), ""))
            if fname in B.index:
                deps = CMMTS_HASH_DEPS[key]
                row["B_hash"] = _hash(cols_txt(B, fname, deps["B"]))
                row["DB_hash"] = _hash(db_txt(fname, db_cols[key]))
                row["C_hash"] = _hash(cols_txt(C, fname, deps["C"]))
                if deps["membs"]:
                    row["membs_hash"] = M_hash.get(fname, "")
            rows.append(row)
    cols = ["name", "key", "cmmt_hash"] + [c for c, _ in CMMTS_HASH_PARTS]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["name", "key"], ignore_index=True)


def check_cmmts_hashes(oc_cmmts, df_members, databases_info):
    """Flag the manual comments stored for an older version of their cluster.

    `E_record_path` stores, for every manual comment (one row per cluster and
    key), the hash of the comment and of the inputs of the checks of its key
    (see `CMMTS_HASH_DEPS`) when it was stored. Comments that are new or were
    edited since the last run are assumed to refer to the current version of
    the cluster, and take its current hashes. Comments that were not edited,
    but whose check inputs changed, are stale: they keep their stored hashes
    (so they are flagged again in the next run) until they are edited, or
    their row is removed from `E_record_path` to confirm they are still
    valid. Changes in the cluster that the checks of a key do not use never
    make its comments stale.

    Hash columns missing from `E_record_path` (added after it was generated)
    are filled with the current hashes, and columns no longer used are
    dropped. A record without the "key" column (older per-cluster format) is
    regenerated with the current hashes.

    Returns a dict {fname: {key: [changed parts]}} with the stale comments.
    """
    df_B = pd.read_csv(E_B_cat_path, dtype=str, keep_default_na=False)
    df_C = pd.read_csv(E_C_cat_path, dtype=str, keep_default_na=False)
    df_curr = _cmmts_hashes(oc_cmmts, df_B, df_C, df_members, databases_info)
    ids = ["name", "key"]
    if E_record_path.is_file():
        df_old = pd.read_csv(E_record_path, dtype=str, keep_default_na=False)
    else:
        _warn(f"file '{E_cmmts_record_file}' not found, it will be generated")
        df_old = df_curr.iloc[:0]
    if "key" not in df_old.columns:
        _warn(
            f"file '{E_cmmts_record_file}' has the old per-cluster format, it was "
            "regenerated with the current hashes (no comment is flagged as stale "
            "in this run)"
        )
        df_old = df_curr.iloc[:0]

    missing = [c for c in df_curr.columns if c not in df_old.columns]
    if missing:
        curr = df_curr.set_index(ids)
        old_ids = pd.MultiIndex.from_frame(df_old[ids])
        for col in missing:
            df_old[col] = curr[col].reindex(old_ids).fillna("").to_numpy()
        _warn(
            f"columns {missing} added to '{E_cmmts_record_file}' with the current "
            "hashes"
        )
    cols_changed = list(df_old.columns) != list(df_curr.columns)
    df_old = df_old[df_curr.columns]

    df_m = df_curr.merge(df_old, on=ids, how="left", suffixes=("", "_old"))
    stale = {}
    for r in df_m[df_m["cmmt_hash"] == df_m["cmmt_hash_old"]].itertuples():
        changed = [
            lbl
            for col, lbl in CMMTS_HASH_PARTS
            if getattr(r, col) != getattr(r, f"{col}_old")
        ]
        if changed:
            stale.setdefault(r.name, {})[r.key] = changed

    # Comments no longer present are dropped
    is_stale = [k in stale.get(n, {}) for n, k in zip(df_curr["name"], df_curr["key"])]
    old_stale = [k in stale.get(n, {}) for n, k in zip(df_old["name"], df_old["key"])]
    df_new = pd.concat(
        [
            df_curr[~np.array(is_stale, dtype=bool)],
            df_old[np.array(old_stale, dtype=bool)],
        ]
    ).sort_values(ids, ignore_index=True)
    if cols_changed or not df_new.equals(df_old):
        df_new.to_csv(E_record_path, index=False)
        print(f"File '{E_record_path}' updated")

    if stale:
        cls_txt = "; ".join(
            f"{fname} ("
            + ", ".join(f"{k}: {'/'.join(ch)}" for k, ch in keys.items())
            + ")"
            for fname, keys in stale.items()
        )
        N_cmmts = sum(len(keys) for keys in stale.values())
        _warn(
            f"{N_cmmts} manual comments in '{UCC_cat_E_in}' ({len(stale)} clusters) "
            "are for checks whose inputs changed since they were stored (stale "
            "comments). Review them, and edit them or remove their rows from "
            f"'{E_cmmts_record_file}' to confirm them: {cls_txt}"
        )
    return stale


def check_orphan_cmmts(oc_cmmts, flags):
    """Warn about the manual comments whose check no longer fails: comments
    under a key in `CMMT_KEYS` for a cluster with no flag of that
    `cmmt_key`. They no longer apply to any check and can be removed."""
    failed = {(f["fname"], f["cmmt_key"]) for f in flags}
    orphans = [
        f"{fname} ({key})"
        for fname, cmmts in oc_cmmts.items()
        for key in cmmts
        if (fname, key) not in failed
    ]
    if orphans:
        _warn(
            f"{len(orphans)} manual comments in '{UCC_cat_E_in}' are for checks "
            f"that no longer fail (orphan comments): {'; '.join(orphans)}"
        )


def check_integrity(df_B, df_C, df_members):
    """Warn about structural problems in catalogues B and C and the members
    file: duplicated or unmatched cluster names, missing values in the main
    C columns, clusters with no members or a negative member parallax, and
    stars repeated within a cluster's members. No family is skipped."""

    def warn(names, msg):
        names = sorted(set(names))
        if names:
            _warn(f"{len(names)} {msg}: {', '.join(names)}")

    B_names, C_names = set(df_B["fname"]), set(df_C["fname"])
    M_names = set(df_members["name"])
    warn(df_B.loc[df_B["fname"].duplicated(), "fname"], "repeated fnames in B")
    warn(df_C.loc[df_C["fname"].duplicated(), "fname"], "repeated fnames in C")
    warn(B_names - C_names, "clusters in B but not in C")
    warn(C_names - B_names, "clusters in C but not in B")
    warn(C_names - M_names, "clusters in C without members in the members file")
    warn(M_names - C_names, "clusters in the members file but not in C")

    cols = ["RA_ICRS_m", "DE_ICRS_m", "Plx_m", "pmRA_m", "pmDE_m", "r_50", "UTI"]
    warn(
        df_C.loc[df_C[cols].isna().any(axis=1), "fname"],
        f"clusters in C with missing values in some of {cols}",
    )
    warn(df_C.loc[df_C["N_membs"] <= 0, "fname"], "clusters in C with N_membs=0")
    warn(df_C.loc[df_C["Plx_m"] < 0, "fname"], "clusters in C with Plx_m<0")
    rep = df_members.duplicated(["name", "Source"])
    warn(df_members.loc[rep, "name"], "clusters with repeated stars in their members")


def _flag(fname, check, cmmt_key, ratio, details, factor=1.0):
    """A failed check for one cluster.

    `ratio` is the measured value over its threshold (>= 1 for a failure) and
    sets the severity; `cmmt_key` is the key in E_json_path whose manual
    comment applies to this check; `factor` scales the final score.
    """
    return {
        "fname": fname,
        "check": check,
        "cmmt_key": cmmt_key,
        "ratio": ratio,
        "details": details,
        "factor": factor,
    }


def _uti_factor(uti):
    """Score factor for a cluster's UTI: `uti_floor` for UTI=0, growing
    linearly to 1 for UTI=1. Clusters without a UTI value get 1."""
    if not np.isfinite(uti):
        return 1.0
    floor = SCORE_CONFIG["uti_floor"]
    return floor + (1 - floor) * uti


def _score_group(check):
    """Checks in the same group measure the same discrepancy, and only the
    highest score in a group counts: "DBs_<type>_conflicts" and
    "DBs_<type>_groups" share the group "DBs_<type>", and all the "dup_*"
    checks (shared-member runs and "dup_prox") the group "dup"."""
    if check.startswith("DBs_"):
        return check.rsplit("_", 1)[0]
    if check.startswith("dup"):
        return "dup"
    return check


def _is_minor(check):
    """Minor checks (cross-DB parameters: disagreements in the literature
    rather than UCC errors) are scored, but a cluster failing only these is
    not counted as flagged."""
    return check.startswith("params_")


def _N_flagged(clusters):
    """Number of scored clusters with at least one non-minor failure."""
    return sum(any(not _is_minor(c) for c in checks) for _, _, checks in clusters)


def _score_flags(flags, oc_cmmts, stale, UTI):
    """Score the failed checks and group them by cluster.

    Each flag scores

        weight * min(1 + log2(ratio), sev_max) * factor * uti_factor

    with `weight` set by its check ID in SCORE_CONFIG["weights"], and
    `uti_factor` set by the cluster's UTI (see `_uti_factor`), so that the
    failures of the clusters most likely to be real score higher. The score
    is multiplied by `cmmt_factor` if the cluster has a manual comment for the
    flag's `cmmt_key`, unless that comment is stale (see
    `check_cmmts_hashes`). If a check fails more than once for the same cluster
    (e.g. several duplicate pairs), only its highest score counts. A
    cluster's total score is the sum over its failed checks, counting only
    the highest score of each group of checks (see `_score_group`).

    Returns a list of (fname, total_score, {check: (score, [details])}) sorted
    by decreasing total score.
    """
    cfg = SCORE_CONFIG
    per_cl = {}
    for f in flags:
        sev = min(1 + np.log2(max(f["ratio"], 1.0)), cfg["sev_max"])
        score = cfg["weights"][f["check"]] * sev * f["factor"]
        score *= _uti_factor(UTI.get(f["fname"], np.nan))
        cl_cmmts = oc_cmmts.get(f["fname"], {})
        key = f["cmmt_key"]
        if key in cl_cmmts and key not in stale.get(f["fname"], {}):
            score *= cfg["cmmt_factor"]
        score_old, details = per_cl.setdefault(f["fname"], {}).get(
            f["check"], (0.0, [])
        )
        per_cl[f["fname"]][f["check"]] = (max(score, score_old), details)
        details.append(f["details"])

    clusters = []
    for fname, checks in per_cl.items():
        best = {}
        for check, (score, _) in checks.items():
            group = _score_group(check)
            best[group] = max(best.get(group, 0.0), score)
        clusters.append((fname, sum(best.values()), checks))
    return sorted(clusters, key=lambda x: (-x[1], x[0]))


def _split_dup_clusters(clusters):
    """Split the scored clusters into the candidate duplicates (all of them)
    and the `N_max_clusters` other clusters with the highest scores."""
    is_dup = [any(c in cl[2] for c in DUP_CHECKS) for cl in clusters]
    dup_cls = [cl for cl, d in zip(clusters, is_dup) if d]
    other_cls = [cl for cl, d in zip(clusters, is_dup) if not d][:N_max_clusters]
    return dup_cls, other_cls


def _centered_table(header, rows):
    """Markdown table with every column centered, both when rendered and in
    the plain text: cells are padded so that the `|` separators line up."""
    widths = [max(len(r[j]) for r in (header, *rows)) for j in range(len(header))]

    def row(cells):
        return "| " + " | ".join(c.center(w) for c, w in zip(cells, widths)) + " |"

    sep = "|" + "|".join(":" + "-" * w + ":" for w in widths) + "|"
    return [row(header), sep, *(row(r) for r in rows)]


def _write_out_file(checks, clusters, oc_cmmts, stale, UTI):
    """Write the summary of the checks and the flagged clusters: all the
    candidate duplicates first (failed `dup_1` to `dup_5` checks), then the
    `N_max_clusters` other clusters with the highest scores.

    Text is wrapped at `E_out_width` characters; table rows and headings are
    not, since wrapping would break them.
    """
    cfg = SCORE_CONFIG

    def esc(txt):
        return str(txt).replace("|", "\\|")

    def para(txt, first="", rest=""):
        """Wrap a paragraph or list item, with a blank line after it."""
        lines.extend(
            textwrap.wrap(
                txt,
                E_out_width,
                initial_indent=first,
                subsequent_indent=rest,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        lines.append("")

    def item(txt):
        """Wrap a list item (no blank line after it)."""
        para(txt, "- ", "  ")
        lines.pop()

    def heading(txt):
        """Section heading, separated from the previous section by an extra
        blank line."""
        lines.extend(["", txt, ""])

    dup_cls, other_cls = _split_dup_clusters(clusters)
    lines = ["# UCC master check", ""]
    para(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    N_flagged = _N_flagged(clusters)
    para(
        f"Clusters flagged: {N_flagged} (plus {len(clusters) - N_flagged} with "
        "only `params_*` failures, scored but not counted). The "
        f"{len(dup_cls)} candidate "
        f"duplicates (`dup_1` to `dup_5` checks) are always listed first, "
        "followed by the "
        f"{len(other_cls)} other clusters with the highest scores; both sorted "
        "by decreasing score."
    )

    heading("## Warnings")
    if _WARNINGS:
        for msg in _WARNINGS:
            item(msg)
        lines.append("")
    else:
        para("None.")

    heading("## Checks performed")
    # Larger weights first (the order of the checks is kept for equal weights)
    checks = sorted(checks, key=lambda c: -cfg["weights"][c[0]])
    lines += _centered_table(
        ("Check ID", "Parameters", "Weight", "Flagged"),
        [
            (
                f"`{check}`",
                esc(pars),
                f"{cfg['weights'][check]:g}",
                str(len({f["fname"] for f in flags})),
            )
            for check, _, pars, flags in checks
        ],
    )
    lines.append("")
    for check, desc, *_ in checks:
        item(f"`{check}`: {desc}")
    lines.append("")
    para(
        "Families skipped in all checks except `dup_1` to `dup_5`: "
        f"{', '.join(skip_pfx)}"
    )
    if include_DBs:
        para("Only DBs included in some checks (overrides the exclusions):")
        for check, dbs in include_DBs.items():
            item(f"`{check}`: {', '.join(dbs)}")
        lines.append("")
    if exclude_DBs:
        para("DBs excluded from some checks:")
        for check, dbs in exclude_DBs.items():
            item(f"`{check}`: {', '.join(dbs)}")
        lines.append("")

    heading("## Scoring")
    para(
        "Each failed check scores `weight * min(1 + log2(ratio), "
        f"{cfg['sev_max']:g}) * ({cfg['uti_floor']:g} + "
        f"{1 - cfg['uti_floor']:g} * UTI)`, where `ratio` is the measured value "
        "over its threshold (for `dup_1` to `dup_5`: the sum of both shared-member "
        "percentages over the minimum percentage), and `UTI` is the cluster's "
        "UTI (the factor is 1 if it has no UTI). A cluster's score is the sum "
        "over its failed checks (the highest one if a check fails more than "
        "once). The `conflicts` and `groups` checks of the same quantity "
        "measure the same discrepancy, as do all the `dup_*` checks: only the "
        "highest of each group counts."
    )
    item(
        f"Checks with a manual comment in `{UCC_cat_E_in}` are scored "
        f"x{cfg['cmmt_factor']:g}, unless the inputs of the checks of that "
        "comment (B or C columns, DB values or members) changed since it was "
        "stored (stale comments)."
    )
    item(
        f"The original cluster of a candidate duplicate pair (`dup_*`) "
        f"is scored x{cfg['dup_orig_factor']:g}; the duplicate "
        "gets the full score."
    )
    lines.append("")

    heading("## Score distribution")
    para(
        "Clusters flagged (not counting those with only `params_*` failures) "
        "per score range."
    )
    edges = [5, 10, 15, 20, 25]
    flagged_scores = np.array(
        [
            total
            for _, total, cl_checks in clusters
            if any(not _is_minor(c) for c in cl_checks)
        ]
    )
    bin_idx = np.searchsorted(edges, flagged_scores, side="right")
    labels = (
        [f"< {edges[0]}"]
        + [f"{lo}-{hi}" for lo, hi in itertools.pairwise(edges)]
        + [f">= {edges[-1]}"]
    )
    lines += _centered_table(
        ("Score", "Clusters"),
        [(lbl, str(int((bin_idx == i).sum()))) for i, lbl in enumerate(labels)],
    )
    lines.append("")

    heading("## Clusters")

    def add_cluster(i, fname, total, cl_checks):
        lines.extend(
            [
                "",
                f"#### {i}. [{fname}](https://ucc.ar/_clusters/{fname})",
                "",
                f"Score: {total:.1f}, UTI: {UTI.get(fname, np.nan):.2f}",
                "",
            ]
        )
        # One item per check (sorted by decreasing score), with its details
        # inline if there is only one, or as wrapped sub-items otherwise
        for check, (score, details) in sorted(
            cl_checks.items(), key=lambda x: -x[1][0]
        ):
            if len(details) == 1:
                item(f"`{check}` ({score:.1f}): {details[0]}")
                continue
            item(f"`{check}` ({score:.1f}):")
            for d in details:
                para(d, "  - ", "    ")
                lines.pop()
        cmmts = oc_cmmts.get(fname, {})
        if cmmts:
            lines.append("")
            para("Manual comments:")
            for k, v in cmmts.items():
                changed = stale.get(fname, {}).get(k)
                if changed:
                    k += f" (STALE: {', '.join(changed)} changed)"
                item(f"{k}: {v}")

    for title, cls in (
        (f"Candidate duplicates ({len(dup_cls)})", dup_cls),
        (f"Highest scores ({len(other_cls)})", other_cls),
    ):
        heading(f"### {title}")
        for i, cl in enumerate(cls, 1):
            add_cluster(i, *cl)

    # Collapse runs of more than two blank lines left by the helpers
    text = "\n".join(lines).strip("\n")
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    E_out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(E_out_file, "w") as f:
        f.write(text + "\n")


def run_B_DBs_check(df_B, databases_info, check_type, thr, drad_thr, rel_thr=None):
    """Cross-DB consistency check within catalogue B.

    Parameters
    ----------
    df_B : pd.DataFrame
        Catalogue B with the columns added by `get_norm_dist_rad`.
    databases_info : dict
        Contents of the databases JSON file.
    check_type : str
        Quantity to compare: "pos", "pm", or "plx".
    thr : float
        Maximum allowed difference between DB values.
    drad_thr : float or None
        Minimum normalized distance required for an object to be checked
        (None: all objects are checked).
    rel_thr : float or None
        If given, the threshold of each object is max(thr, rel_thr * |median|),
        with the median of its DB values (single-column quantities only).

    The DBs selected for each check ID (see `_skip_DB`) are used in that
    mode.

    Returns
    -------
    list
        Flags (see `_flag`) for two methods, with check IDs
        "DBs_<check_type>_conflicts" and "DBs_<check_type>_groups":
        - conflicts: identify individual DBs involved in discrepant pairs.
        - groups: identify objects split into two or more distinct groups.

    Notes
    -----
    Differences between two DBs are the angular separation for "pos", the
    norm of the difference vector for "pm", and the absolute difference for
    "plx".

    In "groups" mode, measurements separated by <= the threshold are connected.
    Connected components containing at least two DBs are considered
    well-defined groups. Isolated DB measurements are ignored.
    """
    cfg = CHECK_CONFIG[check_type]
    cat_cols = cfg["cat_cols"]
    unit = cfg["unit"]

    # Load the relevant values from each database (DBs without them are
    # skipped).
    all_dbs = {}
    for db in E_db_path.glob("*.csv"):
        db_cols = databases_info[db.stem].get("pos")

        if db_cols and all(c in db_cols for c in cat_cols):
            usecols = [db_cols[col] for col in cat_cols]
            db_df = pd.read_csv(db, usecols=usecols)
            all_dbs[db.stem] = {
                col: np.round(db_df[db_cols[col]].to_numpy(dtype=float), 4)
                for col in cat_cols
            }

    skip = {m: _skip_DB(f"DBs_{check_type}_{m}") for m in ("conflicts", "groups")}

    flags = []
    for fname, cl_db, cl_db_idx, dist_norm, coords_dist in zip(
        df_B["fname"],
        df_B["DB"],
        df_B["DB_i"],
        df_B["dist_norm"],
        df_B["coords_dist"],
    ):
        if fname.startswith(skip_pfx):
            continue
        # Clusters without member-based dist_norm (NaN) are skipped, as in
        # run_B_vs_C_coords.
        if drad_thr is not None and not dist_norm >= drad_thr:
            continue

        # Keep only contributing DBs with finite values for the requested
        # information.
        cl_dbs = cl_db.split(";")
        cl_db_i = [int(x) for x in cl_db_idx.split(";")]

        pairs = [
            (db, i)
            for db, i in zip(cl_dbs, cl_db_i)
            if db in all_dbs
            and all(np.isfinite(all_dbs[db][col][i]) for col in cat_cols)
        ]

        if len(pairs) < 2:
            continue

        cl_dbs, cl_db_i = map(list, zip(*pairs))

        vals = {
            col: [all_dbs[db][col][i] for db, i in zip(cl_dbs, cl_db_i)]
            for col in cat_cols
        }

        n_db = len(cl_dbs)

        # Calculate all pairwise differences once.
        cl_thr = thr
        if rel_thr is not None:
            cl_thr = max(thr, rel_thr * abs(float(np.median(vals[cat_cols[0]]))))
        thr_txt = f", thr={cl_thr:.2f} {unit}" if round(cl_thr, 2) > thr else ""

        pair_diffs = {}
        for i in range(n_db):
            for j in range(i + 1, n_db):
                if check_type == "pos":
                    pair_diff = float(
                        angular_sep(
                            vals["RA"][i], vals["DEC"][i], vals["RA"][j], vals["DEC"][j]
                        )
                    )
                else:
                    pair_diff = float(
                        np.sqrt(sum((vals[c][i] - vals[c][j]) ** 2 for c in cat_cols))
                    )
                pair_diffs[(i, j)] = pair_diff

        ucc_dist_to_median = ""
        if check_type == "pos":
            ucc_dist_to_median = f", UCC dist to median={coords_dist:.2f} {unit}"

        # -------------------------------------------------------------
        # Pairwise-conflict method.
        # -------------------------------------------------------------
        conflicts = Counter()
        pair_list = []
        max_diff = 0.0

        for (i, j), pair_diff in pair_diffs.items():
            if skip["conflicts"](cl_dbs[i]) or skip["conflicts"](cl_dbs[j]):
                continue
            max_diff = max(max_diff, pair_diff)
            if pair_diff > cl_thr:
                conflicts[i] += 1
                conflicts[j] += 1
                pair_list.append((i, j))

        if conflicts:
            if len(conflicts) == 2 and all(v == 1 for v in conflicts.values()):
                ii, jj = pair_list[0]
                suspects = f"{cl_dbs[ii]} vs {cl_dbs[jj]}"
            else:
                # DB appearing most often in conflicts is the most
                # likely individual offender.
                offender = conflicts.most_common(1)[0][0]
                suspects = (
                    f"suspect {cl_dbs[offender]} [{conflicts[offender]} conflicts]"
                )

            flags.append(
                _flag(
                    fname,
                    f"DBs_{check_type}_conflicts",
                    check_type,
                    max_diff / cl_thr,
                    f"Δ={max_diff:.2f} {unit}{thr_txt} (dist_norm={dist_norm:.2f}"
                    f"{ucc_dist_to_median}): {suspects}",
                )
            )

        # -------------------------------------------------------------
        # Multiple-group detection method.
        # -------------------------------------------------------------
        nodes = [i for i in range(n_db) if not skip["groups"](cl_dbs[i])]
        adjacency = [set() for _ in range(n_db)]

        for (i, j), pair_diff in pair_diffs.items():
            if i in nodes and j in nodes and pair_diff <= cl_thr:
                adjacency[i].add(j)
                adjacency[j].add(i)

        # Find connected components.
        groups = []
        visited = set()

        for start in nodes:
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

        # Maximum distance between every pair of defined groups.
        group_diffs = []

        for gi in range(len(defined_groups)):
            for gj in range(gi + 1, len(defined_groups)):
                group_i = defined_groups[gi]
                group_j = defined_groups[gj]

                max_group_diff = max(
                    pair_diffs[tuple(sorted((i, j)))] for i in group_i for j in group_j
                )

                group_diffs.append(max_group_diff)

        group_txt = " / ".join(
            ",".join(cl_dbs[i] for i in group) for group in defined_groups
        )
        group_diffs_txt = ", ".join(f"{d:.2f}" for d in group_diffs)

        flags.append(
            _flag(
                fname,
                f"DBs_{check_type}_groups",
                check_type,
                max(group_diffs) / cl_thr,
                f"Δ_groups=({group_diffs_txt}) {unit}{thr_txt}: "
                f"{len(defined_groups)} groups ({group_txt})",
            )
        )

    return flags


def run_B_vs_C_coords(df_B, df_C, pos_thr, drad_thr):
    """
    Compare the cluster center coordinates in catalogue B with the median
    coordinates of their member stars. Clusters with a separation greater
    than `pos_thr` arcmin and a normalized separation greater than
    `drad_thr` are flagged for further inspection. `df_B` must contain the
    columns added by `get_norm_dist_rad`.
    """
    df = df_B.merge(df_C[["fname", "UTI"]], on="fname", how="left")
    dist_arcmin = df["coords_dist"] * 60

    mask_valid = ~df["fname"].str.startswith(skip_pfx)
    sel = mask_valid & (dist_arcmin > pos_thr) & (df["dist_norm"] > drad_thr)

    flags = []
    for r, dist in zip(df.loc[sel].itertuples(), dist_arcmin[sel]):
        # Both thresholds are exceeded: combine them with a geometric mean
        ratio = np.sqrt((dist / pos_thr) * (r.dist_norm / drad_thr))
        flags.append(
            _flag(
                r.fname,
                "B_vs_C_coords",
                "pos",
                ratio,
                f"dist={dist:.1f} arcmin, dist_norm={r.dist_norm:.2f}, "
                f"UTI={r.UTI:.2f}, cat=({r.RA_ICRS:.2f}, {r.DE_ICRS:.2f}), "
                f"memb=({r.RA_median:.2f}, {r.DE_median:.2f})",
            )
        )
    return flags


def _db_param_vals(cl_db, cl_vals, skip):
    """Per-DB values of a parameter column of catalogue B (';'-separated,
    aligned with the `DB` column; '*' marks are ignored). Returns the DBs and
    their finite values, skipping the DBs for which `skip(db)` is True and the
    missing values."""
    vals, dbs = [], []
    for db, v in zip(str(cl_db).split(";"), str(cl_vals).split(";")):
        v = v.replace("*", "").strip()
        if skip(db) or v.lower() in ("nan", ""):
            continue
        try:
            v = float(v)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
            dbs.append(db)
    return dbs, vals


def run_B_DBs_params(df_B, param_col, thr):
    """Cross-DB parameter consistency within catalogue B.

    `thr` is a fraction of the median for "relative" parameters and a value
    in the parameter's units for "absolute" ones (see PARAMS_CONFIG). Only
    the DBs selected for the check ID, or for all the parameters ("params"
    key), are used (see `_skip_DB`).
    """
    pcfg = PARAMS_CONFIG[param_col]
    skip = _skip_DB(f"params_{param_col}", "params")
    relative = pcfg["kind"] == "relative"

    flags = []
    for fname, cl_db, cl_vals in zip(df_B["fname"], df_B["DB"], df_B[param_col]):
        if fname.startswith(skip_pfx):
            continue
        dbs, vals = _db_param_vals(cl_db, cl_vals, skip)
        if len(vals) < 2:
            continue

        vals = np.asarray(vals)

        # Transform parameters before comparison
        if param_col == "age":
            # Myr -> log10(age/yr) (ages below 1 Myr are set to 1 Myr)
            vals = np.log10(np.maximum(vals, 1.0) * 1e6)
        elif param_col == "dist":
            # kpc -> distance modulus (distances below 10 pc are set to 10 pc)
            vals = 5 * np.log10(np.maximum(vals, 0.01) * 1e3) - 5

        med_val = np.median(vals)
        if relative and med_val == 0:
            # No relative difference can be defined
            continue
        cl_thr = thr * abs(med_val) if relative else thr
        diffs = np.abs(vals - med_val)
        bad = np.where(diffs > cl_thr)[0]

        if bad.size:
            max_diff = diffs.max()
            offender = bad[np.argmax(diffs[bad])]
            tag = f" [{bad.size} conflicts]" if bad.size > 1 else ""
            perc = f", {100 * max_diff / abs(med_val):.1f}%" if relative else ""
            ratio = max_diff / cl_thr
            flags.append(
                _flag(
                    fname,
                    f"params_{param_col}",
                    "params",
                    ratio,
                    f"Δ={max_diff:.3g} {pcfg['unit']}{perc}: "
                    f"suspect {dbs[offender]}{tag}",
                )
            )

    return flags


def run_B_vs_C_pm(df_B, df_C, pm_thr, uti_min):
    """Compare the proper motions in catalogue B with those derived from the
    members in catalogue C. Clusters with UTI > `uti_min` and a difference
    larger than `pm_thr` are flagged.

    Positions are not compared here: C's member-based center is the median of
    the members, already compared with B's center in `run_B_vs_C_coords`.
    """
    df = pd.merge(df_B, df_C, on="fname", suffixes=("_B", "_C"))
    df["dist_pm"] = np.hypot(df["pmRA"] - df["pmRA_m"], df["pmDE"] - df["pmDE_m"])

    # remove known-problematic families
    df = df[~df["fname"].str.startswith(skip_pfx, na=False)]
    df = df[df["UTI"] > uti_min]

    flags = []
    for r in df[df["dist_pm"] > pm_thr].itertuples():
        flags.append(
            _flag(
                r.fname,
                "B_vs_C_pm",
                "pm",
                r.dist_pm / pm_thr,
                f"Δ={r.dist_pm:.2f} mas/yr, UTI={r.UTI:.2f}, "
                f"B=({r.pmRA:.2f}, {r.pmDE:.2f}), C=({r.pmRA_m:.2f}, {r.pmDE_m:.2f})",
            )
        )
    return flags


def run_B_vs_C_plx(df_B, df_C):
    """Compare the parallaxes in catalogue B with those derived from the
    members in catalogue C. Clusters with UTI > `uti_min` and a difference
    larger than max(plx_thr, plx_rel * |Plx_m|) are flagged (see
    `B_VS_C_CONFIG`): the members may belong to a foreground or background
    group rather than to the catalogued cluster."""
    cfg = B_VS_C_CONFIG
    df = pd.merge(df_B, df_C, on="fname")
    df = df[~df["fname"].str.startswith(skip_pfx, na=False)]
    df = df[(df["UTI"] > cfg["uti_min"]) & df["Plx"].notna()]

    diff = (df["Plx"] - df["Plx_m"]).abs()
    thr = np.maximum(cfg["plx_thr"], cfg["plx_rel"] * df["Plx_m"].abs())

    flags = []
    for r, d, t in zip(df.itertuples(), diff, thr):
        if d > t:
            flags.append(
                _flag(
                    r.fname,
                    "B_vs_C_plx",
                    "plx",
                    d / t,
                    f"Δ={d:.3f} mas (thr={t:.2f}), UTI={r.UTI:.2f}, "
                    f"B={r.Plx:.3f}, C={r.Plx_m:.3f}",
                )
            )
    return flags


def _pm_tol(plx, pm_thr, v_thr):
    """Proper motion tolerance (mas/yr): the largest of `pm_thr` and the
    proper motion of a `v_thr` km/s velocity at parallax `plx` (mas)."""
    return max(pm_thr, v_thr * max(plx, 0.0) / 4.74)


def run_prox_dup(df_B, df_C, databases_info):
    """Flag candidate duplicates from their proximity in position, parallax
    and proper motion, for the pairs that do not share members (those are
    covered by `run_dup_check`): e.g. one cluster split in two by fastMP.

    Uses the member-based values in catalogue C (see `PROX_DUP_CONFIG`).
    Pairs where both clusters are in a skipped family (`skip_pfx`) are
    ignored. The cluster published later ('received' of its first DB; on
    ties, the one with fewer members) is the candidate duplicate and gets the
    full score, the other one is scored x`dup_orig_factor`. The ratio is the
    geometric mean of threshold/value for the separation, parallax and
    proper motion differences, each capped at 8.
    """
    cfg = PROX_DUP_CONFIG
    fnames = df_C["fname"].to_numpy()
    ra, de = (np.deg2rad(df_C[c].to_numpy(float)) for c in ("RA_ICRS_m", "DE_ICRS_m"))
    r50 = df_C["r_50"].to_numpy(float) / 60  # deg
    plx = df_C["Plx_m"].to_numpy(float)
    pmra, pmde = df_C["pmRA_m"].to_numpy(float), df_C["pmDE_m"].to_numpy(float)
    N_membs = df_C["N_membs"].to_numpy(int)
    skip = df_C["fname"].str.startswith(skip_pfx).to_numpy()
    shared = {f: set(str(s).split(";")) for f, s in zip(fnames, df_C["shared_members"])}
    first_db = {f: str(db).split(";")[0] for f, db in zip(df_B["fname"], df_B["DB"])}
    received = [int(databases_info[first_db[f]]["received"]) for f in fnames]

    # Pairs closer than sep_fac * r_50 of either cluster (chord distance on
    # the unit sphere)
    xyz = np.c_[np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]
    tree = KDTree(xyz)
    chord = 2 * np.sin(np.deg2rad(cfg["sep_fac"] * r50) / 2)
    pairs = set()
    for i, js in enumerate(tree.query_ball_point(xyz, chord)):
        pairs.update((min(i, j), max(i, j)) for j in js if j != i)

    flags = []
    for i, j in sorted(pairs):
        if (skip[i] and skip[j]) or fnames[j] in shared[fnames[i]]:
            continue
        if fnames[i] in shared[fnames[j]]:
            continue
        plx_mean = (plx[i] + plx[j]) / 2
        sep = float(angular_sep(*np.rad2deg([ra[i], de[i], ra[j], de[j]])))
        d_plx = abs(plx[i] - plx[j])
        d_pm = float(np.hypot(pmra[i] - pmra[j], pmde[i] - pmde[j]))
        thrs = (
            cfg["sep_fac"] * max(r50[i], r50[j]),
            max(cfg["plx_thr"], cfg["plx_rel"] * abs(plx_mean)),
            _pm_tol(plx_mean, cfg["pm_thr"], cfg["v_thr"]),
        )
        vals = (sep, d_plx, d_pm)
        if any(v > t for v, t in zip(vals, thrs)):
            continue
        ratio = np.prod([t / max(v, t / 8) for v, t in zip(vals, thrs)]) ** (1 / 3)

        # Duplicate: published later, or with fewer members on ties
        dup, orig = (
            (j, i)
            if (received[j], -N_membs[j]) > (received[i], -N_membs[i])
            else (i, j)
        )
        pair = (
            f"{fnames[dup]} ({received[dup]}, N={N_membs[dup]}) --> "
            f"{fnames[orig]} ({received[orig]}, N={N_membs[orig]}): "
            f"sep={60 * sep:.1f} arcmin (r_50={60 * r50[dup]:.1f}, "
            f"{60 * r50[orig]:.1f}), ΔPlx={d_plx:.3f} mas, Δpm={d_pm:.2f} mas/yr"
        )
        flags.append(_flag(fnames[dup], "dup_prox", "dup", ratio, f"dup of: {pair}"))
        flags.append(
            _flag(
                fnames[orig],
                "dup_prox",
                "dup",
                ratio,
                f"orig of: {pair}",
                SCORE_CONFIG["dup_orig_factor"],
            )
        )
    return flags


def run_lit_dist_plx(df_B, df_C):
    """Compare the median literature distance of each cluster (catalogue B,
    `dist` column, kpc) with the distance from its members' parallax
    (catalogue C, plus the Gaia zero point), as distance moduli; see
    `LIT_DIST_CONFIG`. Only the DBs selected for "params" are used (see
    `_skip_DB`).
    Unlike `params_dist`, which compares the literature values among
    themselves, this flags members that may belong to another object."""
    cfg = LIT_DIST_CONFIG
    skip = _skip_DB("params")
    plx_m = dict(zip(df_C["fname"], df_C["Plx_m"] + cfg["plx_zp"]))

    flags = []
    for fname, cl_db, cl_vals in zip(df_B["fname"], df_B["DB"], df_B["dist"]):
        plx = plx_m.get(fname, np.nan)
        if fname.startswith(skip_pfx) or not plx > cfg["plx_min"]:
            continue
        _, vals = _db_param_vals(cl_db, cl_vals, skip)
        vals = [v for v in vals if v > 0]
        if not vals:
            continue
        d_lit = float(np.median(vals))
        d_plx = 1 / plx  # kpc
        d_mu = 5 * np.log10(d_lit / d_plx)
        if abs(d_mu) > cfg["thr"]:
            flags.append(
                _flag(
                    fname,
                    "lit_dist_plx",
                    "params",
                    abs(d_mu) / cfg["thr"],
                    f"Δμ={d_mu:+.2f} mag: d_lit={d_lit:.2f} kpc ({len(vals)} DBs), "
                    f"d_plx={d_plx:.2f} kpc (Plx_m+zp={plx:.3f} mas)",
                )
            )
    return flags


def run_H23_membs(df_B, df_members):
    """Compare the number of members of each cluster with the number given by
    the `H23_MEMBS_CONFIG["db"]` database (Hunt & Reffert 2023).

    The UCC count is the number of rows of the cluster in the members file;
    the HUNT2023 count is the `N` column of its database file (identical to
    the number of members in its members catalogue), matched to the UCC
    entries through the `DB` and `DB_i` columns of catalogue B.

    The difference is measured as |log2(N_UCC / N_H23)|, symmetric in both
    counts, against the threshold log2(1 + thr * factor), with `factor`
    decreasing with max(N_UCC, N_H23) (see `H23_MEMBS_CONFIG`), so that the
    same relative difference gives a larger ratio (and score) for larger
    clusters.
    """
    cfg = H23_MEMBS_CONFIG
    db, thr0 = cfg["db"], cfg["thr"]
    N_H23 = pd.read_csv(E_db_path / f"{db}.csv", usecols=["N"])["N"].to_numpy()
    N_UCC = df_members.groupby("name").size().to_dict()

    flags = []
    for fname, cl_db, cl_db_idx in zip(df_B["fname"], df_B["DB"], df_B["DB_i"]):
        cl_dbs = str(cl_db).split(";")
        if fname.startswith(skip_pfx) or db not in cl_dbs or fname not in N_UCC:
            continue
        N_h = int(N_H23[int(str(cl_db_idx).split(";")[cl_dbs.index(db)])])
        N_u = int(N_UCC[fname])

        N_max = max(N_u, N_h)
        factor = next((f for N_bin, f in cfg["N_bins"] if N_max <= N_bin), 1.0)
        thr = 1 + thr0 * factor
        N_ratio = N_max / max(min(N_u, N_h), 1)
        if N_ratio > thr:
            flags.append(
                _flag(
                    fname,
                    "H23_membs",
                    "N_membs",
                    np.log2(N_ratio) / np.log2(thr),
                    f"N_UCC={N_u}, N_H23={N_h} "
                    f"(max(N)/min(N)={N_ratio:.2f}, thr={thr:g})",
                )
            )
    return flags


def run_dup_check(df_B, df_C, databases_info, prob_min=50.0):
    """Flag candidate duplicates from the clusters' shared members.

    Catalogue C's `shared_members` column lists, for each cluster, the
    clusters that share members with it, and `shared_members_p` the
    percentage ([0, 100]) of the cluster's own members shared with each one.

    Each run tests pairs (D, O), where D is the cluster under examination and
    O is a cluster sharing members with D. In runs 1-4 the publication date
    ('received' of its first DB) of O is strictly earlier: D is the candidate
    duplicate, O the original. Run 5 covers pairs with equal dates.

        p_D : percentage of D's members also in O   (shared_p_dup)
        p_O : percentage of O's members also in D   (shared_p_orig)

    For each D, the candidates are the older clusters O for which the run's
    metric (p_D for new-->old, p_O for old-->new) is >= `prob_min`. Newer
    overlapping clusters are ignored. N_cand is the number of candidates.

    Run 1 (single, new-->old): exactly one older O with p_D >= prob_min. D is
        largely contained in O: likely a re-detection of O or a substructure
        of it.

    Run 2 (multi, new-->old): two or more older O with p_D >= prob_min, all
        reported (one row per pair). Since D's members are mostly inside each
        O, these O must overlap each other: flags chains of mutual duplicates.

    Run 3 (single, old-->new): exactly one older O with p_O >= prob_min. O is
        largely contained in D: D likely recovers O at a larger scale or
        merges it with neighbouring structure.

    Run 4 (multi, old-->new): two or more older O with p_O >= prob_min, all
        reported. D absorbs several previously catalogued clusters.

    Run 5 (same date): O has the same publication date as D (typically the
        same DB), and max(p_D, p_O) >= prob_min. No direction of precedence
        exists, so each pair is reported once, oriented so that D is the more
        contained cluster (p_D >= p_O; ties broken by name). All candidates
        are reported, with no single/multi split. Flags internal duplicates
        within a catalogue.

    In runs 1-4, pairs with both p_D and p_O above the threshold appear in
    both a new-->old and an old-->new run; these are the strongest duplicate
    candidates.

    Each pair flags both D (full score) and O (scored x`dup_orig_factor`),
    with ratio (p_D + p_O) / prob_min, so that mutually contained pairs
    score higher. The flags of each run have check ID "dup_<run>", weighted
    less from run 1 to run 5 (see SCORE_CONFIG["weights"]).

    Parameters
    ----------
    prob_min : float
        Minimum shared-member percentage for a pair to be a candidate.
    """
    data = _dup_prepare_data(df_B, df_C, databases_info)

    flags = []
    for N_run in DUP_RUN_PARS:
        df_run = _dup_check_run(*data, N_run, prob_min)
        for r in df_run.itertuples():
            # ratio = np.nansum([r.shared_p_dup, r.shared_p_orig]) / prob_min
            ratio = (
                np.nansum(np.array([r.shared_p_dup, r.shared_p_orig], dtype=float))
                / prob_min
            )
            pair = (
                f"(run {N_run}, N_cand={r.N_cand}) {r.fname_dup} "
                f"({r.date_cl_dup}, N={r.N_membs_dup}, {r.shared_p_dup:.0f}%, "
                f"UTI={r.UTI_dup:.2f}) --> {r.fname_orig} ({r.date_cl_orig}, "
                f"N={r.N_membs_orig}, {r.shared_p_orig:.0f}%, "
                f"UTI={r.UTI_orig:.2f}), N_ratio={r.N_ratio:.2f}, "
                f"dist={r.dist_arcmin:.1f} arcmin"
            )
            check = f"dup_{N_run}"
            flags.append(_flag(r.fname_dup, check, "dup", ratio, f"dup of: {pair}"))
            flags.append(
                _flag(
                    r.fname_orig,
                    check,
                    "dup",
                    ratio,
                    f"orig of: {pair}",
                    SCORE_CONFIG["dup_orig_factor"],
                )
            )
    return flags


def _dup_prepare_data(df_B, df_C, current_JSON):
    # Per-cluster lookups keyed by 'fname'
    RA_C = dict(zip(df_C["fname"], df_C["RA_ICRS_m"]))
    DE_C = dict(zip(df_C["fname"], df_C["DE_ICRS_m"]))
    N_membs = dict(zip(df_C["fname"], df_C["N_membs"]))
    UTI = dict(zip(df_C["fname"], df_C["UTI"]))
    DBs = {k: str(v).split(";")[0] for k, v in zip(df_B["fname"], df_B["DB"])}
    DBs_i = {k: int(str(v).split(";")[0]) for k, v in zip(df_B["fname"], df_B["DB_i"])}

    # shared_perc_vals[A][B] = percentage of B's members shared with A
    # (i.e. the reverse of A's own 'shared_members_p' entry for B)
    shared_perc_vals = {f: {} for f in df_C["fname"]}
    sub = df_C.loc[
        df_C["shared_members"].notna(), ["fname", "shared_members", "shared_members_p"]
    ]
    for fname, members, perc in zip(
        sub["fname"], sub["shared_members"], sub["shared_members_p"]
    ):
        for name, p in zip(members.split(";"), str(perc).split(";")):
            shared_perc_vals.setdefault(name, {})[fname] = float(p)

    # Publication date of each cluster's first DB
    received = {f: int(current_JSON[DBs[f]]["received"]) for f in df_C["fname"]}

    # Initial coordinates of each cluster, taken from its first DB (DBs
    # without coordinate column info use the UCC values)
    coord_cols = {}
    for db, info in current_JSON.items():
        try:
            coord_cols[db] = (info["pos"]["RA"], info["pos"]["DEC"])
        except KeyError:
            pass

    df_cache = {}
    init_coords = {}
    for fname in df_C["fname"]:
        db = DBs[fname]
        # if fname in _UCC_COORDS_OVERRIDE or db not in coord_cols:
        if db not in coord_cols:
            init_coords[fname] = (RA_C[fname], DE_C[fname])
            continue
        ra_col, dec_col = coord_cols[db]
        if db not in df_cache:
            df_cache[db] = pd.read_csv(
                E_db_path / f"{db}.csv", usecols=[ra_col, dec_col]
            )
        df_db = df_cache[db]
        init_coords[fname] = (
            df_db.at[DBs_i[fname], ra_col],
            df_db.at[DBs_i[fname], dec_col],
        )

    return df_C, N_membs, UTI, shared_perc_vals, received, init_coords


def _dup_select_orig(
    fname, date_cl, shared_n, shared_p, shared_perc_vals, received, pars, prob_min
):
    """Return all clusters that 'fname' may duplicate.

    "new-->old" / "old-->new": candidates are the shared clusters published
    strictly before 'fname' whose selection metric reaches `prob_min`. The
    metric is the percentage of the duplicate's members shared with the
    candidate ("new-->old") or of the candidate's members shared with the
    duplicate ("old-->new").

    "same": candidates are the shared clusters with the same publication date,
    with p_dup >= p_orig and p_dup >= prob_min, so that each pair is returned
    from one side only (the more contained cluster; ties broken by name).

    multi=False keeps clusters with exactly one candidate, multi=True with two
    or more, multi=None applies no filter. Returns a list of
    (name, p_dup, p_orig) tuples sorted by decreasing metric, empty if nothing
    qualifies.
    """
    date = pars["date"]
    if date not in ("new-->old", "old-->new", "same"):
        raise ValueError(f"Unknown direction: {date}")
    # Position of the selection metric in each (name, p_dup, p_orig) tuple
    i_metric = 2 if date == "old-->new" else 1

    cands = []
    for name, p in zip(shared_n, shared_p):
        date_o = received[name]
        if (date_cl != date_o) if date == "same" else (date_cl <= date_o):
            continue
        cand = (name, p, shared_perc_vals[fname].get(name, np.nan))
        if not cand[i_metric] >= prob_min:  # also rejects NaN
            continue
        if date == "same" and (cand[2] > p or (cand[2] == p and name < fname)):
            continue  # pair is reported from the other cluster's side
        cands.append(cand)

    if pars["multi"] is not None and (len(cands) > 1) != pars["multi"]:
        return []
    return sorted(cands, key=lambda c: c[i_metric], reverse=True)


def _dup_check_run(
    df_C, N_membs, UTI, shared_perc_vals, received, init_coords, N_run, prob_min
):
    pars = DUP_RUN_PARS[N_run]
    records = []
    for fname_dup, shared_members, shared_members_p, uti_dup in zip(
        df_C["fname"], df_C["shared_members"], df_C["shared_members_p"], df_C["UTI"]
    ):
        if pd.isna(shared_members):
            continue

        date_cl = received[fname_dup]
        shared_n = shared_members.split(";")
        shared_p = [float(p) for p in str(shared_members_p).split(";")]

        cands = _dup_select_orig(
            fname_dup,
            date_cl,
            shared_n,
            shared_p,
            shared_perc_vals,
            received,
            pars,
            prob_min,
        )

        N_membs_dup = int(N_membs[fname_dup])
        for fname_orig, shared_p_dup, shared_p_orig in cands:
            N_membs_orig = int(N_membs[fname_orig])
            sep = angular_sep(*init_coords[fname_dup], *init_coords[fname_orig])
            records.append(
                {
                    "run": N_run,
                    "N_cand": len(cands),
                    "UTI_dup": uti_dup,
                    "fname_dup": fname_dup,
                    "date_cl_dup": date_cl,
                    "N_membs_dup": N_membs_dup,
                    "shared_p_dup": shared_p_dup,
                    "fname_orig": fname_orig,
                    "date_cl_orig": received[fname_orig],
                    "N_membs_orig": N_membs_orig,
                    "shared_p_orig": shared_p_orig,
                    "UTI_orig": UTI[fname_orig],
                    "N_ratio": round(N_membs_dup / max(1, N_membs_orig), 2),
                    "dist_arcmin": round(float(sep) * 60, 2),
                }
            )

    return pd.DataFrame(records, columns=DUP_COLS_ORDER)


def get_norm_dist_rad(df_B: pd.DataFrame, df_members: pd.DataFrame) -> pd.DataFrame:
    """Calculate cluster-center offsets normalized by the member radius.

    For each cluster, the median member coordinates are used to estimate its
    center, and the median angular distance of the members to this center
    (r50, deg) as its characteristic radius. The angular separation between
    this center and the catalogue coordinates (`coords_dist`, deg) is then
    normalized by r50 (`dist_norm`).

    Clusters with fewer than three members are excluded from the member-based
    statistics because their median positions and radii are considered
    unreliable.

    Parameters
    ----------
    df_B : pd.DataFrame
        Cluster catalogue containing at least the columns ``fname``,
        ``RA_ICRS``, and ``DE_ICRS``.
    df_members : pd.DataFrame
        Members file, with at least the columns ``name``, ``RA_ICRS`` and
        ``DE_ICRS``.

    Returns
    -------
    pd.DataFrame
        Input catalogue augmented with the columns ``RA_median``,
        ``DE_median``, ``r50``, ``coords_dist``, and ``dist_norm``. Clusters
        without sufficient member information have NaN values in these
        columns.
    """
    names = df_members["name"].to_numpy()
    ra = df_members["RA_ICRS"].to_numpy(dtype=float)
    de = df_members["DE_ICRS"].to_numpy(dtype=float)

    # Clusters crossing RA=0 have their RA shifted by 180 deg before taking
    # the median, so that it is not pulled towards RA=180.
    gr_ra = pd.Series(ra).groupby(names)
    wraps = (gr_ra.max() - gr_ra.min()) > 180
    ra_shift = np.where(wraps.reindex(names).to_numpy(), (ra + 180) % 360, ra)

    stats = (
        pd.DataFrame({"ra_shift": ra_shift, "DE_median": de, "nmembs": 1})
        .groupby(names)
        .agg({"ra_shift": "median", "DE_median": "median", "nmembs": "size"})
    )
    stats["RA_median"] = np.where(
        wraps.reindex(stats.index), (stats["ra_shift"] - 180) % 360, stats["ra_shift"]
    )

    # Median angular distance of the members to the cluster center
    center = stats.loc[names, ["RA_median", "DE_median"]].to_numpy()
    d_memb = angular_sep(ra, de, center[:, 0], center[:, 1])
    stats["r50"] = pd.Series(d_memb).groupby(names).median()

    # Ignore clusters with fewer than three members, since their median
    # positions and radii are poorly constrained.
    stats = stats[stats["nmembs"] > 2]

    df = df_B.merge(
        stats[["RA_median", "DE_median", "r50"]],
        left_on="fname",
        right_index=True,
        how="left",
    )

    # Angular separation between both cluster centers, normalized by r50
    df["coords_dist"] = angular_sep(
        df["RA_ICRS"], df["DE_ICRS"], df["RA_median"], df["DE_median"]
    )
    df["dist_norm"] = df["coords_dist"] / df["r50"]

    return df


def angular_sep(lon1, lat1, lon2, lat2):
    """Angular separation in degrees (haversine) between two sets of
    spherical coordinates given in degrees."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, (lon1, lat1, lon2, lat2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return np.rad2deg(2 * np.arcsin(np.sqrt(a)))


if __name__ == "__main__":
    main()
