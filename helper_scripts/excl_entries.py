import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modules.B_update_UCC_DB import check_duplicated_fnames
from modules.utils import get_fnames, normalize_name
from modules.variables import (
    GCs_cat,
    UCC_cat_B_out,
    UCC_cmmts_folder,
    UCC_excl_folder,
    all_OC_names,
    data_folder,
    name_DBs_json,
    temp_folder,
)

excl_cols = ["row", "name", "category", "comment"]
categories = ["duplicate", "GC", "not_cluster", "other"]
md_titles = {
    "duplicate": "Removed duplicates",
    "GC": "Removed GCs",
    "not_cluster": "Removed",
    "other": "Removed",
}


def main():
    """
    Interactive builder of the exclusion file of a database.

    The B script skips the entries of a DB listed in its exclusion file
    ('data/databases/excluded/<DB>.csv', or the same path under 'temp_updt/' for a
    DB not yet added to the UCC). This script helps to generate that file without
    editing the DB itself:

    - Review probable GCs (name match or close to an entry in the GCs catalogue)
    - Review groups of entries sharing a name (fnames) within the DB, also through
      the aliases in 'all_names'
    - Review close pairs of entries with similar names within the DB
    - Exclude all entries with given values in a column (e.g. a 'type' column)
    - Search entries by name/row and exclude them manually

    Each excluded entry is stored with its row (0-based position in the DB file, as
    used in the 'DB_i' column), its name (to validate the row), a category
    (duplicate, GC, not_cluster, other) and a comment.
    """
    db = ExclDB(select_DB(sys.argv[1] if len(sys.argv) > 1 else None))

    options = [
        ("Review probable GCs", db.review_GCs),
        ("Review entries sharing names within the DB", db.review_dup_names),
        ("Review close entries with similar names within the DB", db.review_close),
        ("Exclude entries by column value", db.excl_by_value),
        ("Search and exclude entries (by name or row)", db.excl_manual),
        ("Show excluded entries", db.show),
        ("Remove entries from the exclusion list", db.remove),
    ]
    while True:
        print(f"\n=== {db.name}: {len(db.df)} rows, {len(db.excl)} excluded ===")
        for i, (desc, _) in enumerate(options, 1):
            print(f"  {i}) {desc}")
        print("  s) save   q) quit")
        try:
            ans = input("> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            ans = "q"

        if ans == "s":
            db.save()
        elif ans == "q":
            try:
                if db.unsaved and ask_yn(
                    "Unsaved changes, save before quitting?", True
                ):
                    db.save()
            except (KeyboardInterrupt, EOFError):
                pass
            break
        elif ans.isdigit() and 1 <= int(ans) <= len(options):
            try:
                options[int(ans) - 1][1]()
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled")
        else:
            print("Invalid option")


class ExclDB:
    """DB being processed and its list of excluded entries"""

    def __init__(self, path: str):
        fpath = Path(path).expanduser().resolve()
        if not fpath.is_file():
            raise FileNotFoundError(fpath)
        self.name = fpath.stem
        self.df = pd.read_csv(fpath)

        # DBs not yet added to the UCC are stored under the temp folder
        in_temp = (ROOT / temp_folder) in fpath.parents
        base = ROOT / temp_folder if in_temp else ROOT
        self.excl_path = base / UCC_excl_folder / f"{self.name}.csv"

        self.names_col, self.ra_col, self.dec_col = self.get_cols(
            [base / name_DBs_json, ROOT / name_DBs_json]
        )
        self.names = self.df[self.names_col].astype(str).str.strip().to_numpy()
        self.fnames = []
        for row, name in enumerate(self.names):
            try:
                self.fnames.append(get_fnames([name])[0])
            except ValueError as e:
                # Names not yet sanitized: use the normalized names only
                print(f"WARNING: row {row} '{name}' could not be standardized ({e})")
                self.fnames.append([normalize_name(_) for _ in name.split(",")])

        self.excl: dict[int, list[str]] = {}
        if self.excl_path.is_file():
            df_excl = pd.read_csv(self.excl_path, keep_default_na=False)
            for row, name, cat, cmmt in df_excl[excl_cols].itertuples(index=False):
                if self.names[row] != str(name).strip():
                    print(f"WARNING: row {row} '{name}' != '{self.names[row]}' (DB)")
                self.excl[int(row)] = [str(name), cat, cmmt]
            print(f"Loaded {len(self.excl)} excluded entries from {self.excl_path}")
        else:
            print(f"No exclusion file found, it will be stored in {self.excl_path}")
        self.unsaved = False
        # Normalized names, computed on the first search
        self.names_norm = None

    def get_cols(self, json_paths: list[Path]) -> tuple[str, str | None, str | None]:
        """Names and (RA, DEC) columns from the DBs JSON file, or from the user"""
        for json_path in json_paths:
            if not json_path.is_file():
                continue
            with open(json_path) as f:
                db_json = json.load(f)
            if self.name in db_json:
                vals = db_json[self.name]
                pos = vals["pos"]
                print(f"Columns read from {json_path}")
                return vals["names"], pos.get("RA"), pos.get("DEC")

        print(f"{self.name} not found in the JSON file. Columns:")
        print_list([f"  {i}: {c}" for i, c in enumerate(self.df.columns)], 500)
        print("Select columns by name or number (Enter accepts the suggestion)")
        names_col = ask_col(self.df, "Names column", guess_names_col(self.df))
        ra_col = ask_col(
            self.df,
            "RA column [deg] ('-' if none)",
            guess_coord_col(self.df, "RA"),
            True,
        )
        dec_col = None
        if ra_col is not None:
            dec_col = ask_col(
                self.df,
                "DEC column [deg] ('-' if none)",
                guess_coord_col(self.df, "DEC", ra_col),
                True,
            )
            if dec_col is None:
                ra_col = None
        return names_col, ra_col, dec_col

    # ------------------------------------------------------------------------
    # Candidates review

    def review_GCs(self):
        """Entries that match a GC by name, or are close to one"""
        df_GCs = pd.read_csv(ROOT / GCs_cat)
        gc_names = (
            df_GCs["Name"].str.strip() + "," + df_GCs["OName"].fillna("").str.strip()
        ).str.strip(",")
        gc_fnames = {}
        for i, fnames in enumerate(get_fnames(gc_names)):
            for fname in fnames:
                if fname:
                    gc_fnames.setdefault(fname, i)

        # {row: (GC index, distance in arcmin)}
        cands = {}
        for row, fnames in enumerate(self.fnames):
            for fname in fnames:
                if fname in gc_fnames:
                    cands[row] = (gc_fnames[fname], 0.0)
                    break

        if self.ra_col is not None:
            rad = ask_float("Search radius around GCs [arcmin]", 10.0)
            d = ang_dist(
                self.df[self.ra_col].to_numpy(float),
                self.df[self.dec_col].to_numpy(float),
                df_GCs["RAJ2000"].to_numpy(float),
                df_GCs["DEJ2000"].to_numpy(float),
            )
            gc_idx = np.argmin(d, axis=1)
            gc_dist = d[np.arange(len(d)), gc_idx]
            for row in np.flatnonzero(gc_dist < rad):
                if row not in cands:
                    cands[int(row)] = (int(gc_idx[row]), float(gc_dist[row]))
        else:
            print("No (RA, DEC) columns, only names are matched")

        items = []
        for row, (i_gc, dist) in sorted(cands.items(), key=lambda x: x[1][1]):
            gc_name = gc_names[i_gc]
            info = f"--> GC {gc_name} (d={dist:.1f}')"
            items.append((row, info, "GC", f"{gc_name} (d={dist:.1f}')"))
        self.review(items, "probable GCs")

    def review_dup_names(self):
        """Groups of entries sharing a name, also through the 'all_names' aliases"""
        all_names = pd.read_csv(ROOT / data_folder / all_OC_names)
        alias_to_canon = {}
        for fnames in all_names["fnames"]:
            fnames = fnames.split(";")
            for fname in fnames:
                alias_to_canon[fname] = fnames[0]

        # Add the canonical fname of each entry (if any), as in the B script
        fnames_ext = []
        for fnames in self.fnames:
            canon = next(
                (alias_to_canon[f] for f in fnames if f in alias_to_canon), None
            )
            fnames_ext.append(fnames + ([canon] if canon else []))

        groups = {}
        for fname, rows in check_duplicated_fnames(fnames_ext).items():
            rows = tuple(sorted(r for r in rows if r not in self.excl))
            if len(rows) > 1:
                groups.setdefault(rows, fname)
        if not groups:
            print("No groups of entries sharing names found")
            return

        print(f"\nFound {len(groups)} groups of entries sharing names")
        for N_g, (rows, fname) in enumerate(groups.items(), 1):
            print(f"\nGroup {N_g}/{len(groups)}: '{fname}'")
            for row in rows:
                print(f"  {row:<6} {self.names[row]:<40} {self.coords_str(row)}")
            keep = ask(
                "Row to keep (the others are excluded), 'n' skip, 'q' stop",
                str(rows[0]),
            )
            if keep == "q":
                break
            if not keep.isdigit() or int(keep) not in rows:
                continue
            keep = int(keep)
            for row in rows:
                if row != keep:
                    self.add(
                        row, "duplicate", f"{self.names[row]} → {self.names[keep]}"
                    )

    def review_close(self):
        """Pairs of entries close to each other with similar names"""
        if self.ra_col is None:
            print("No (RA, DEC) columns in this DB")
            return
        rad = ask_float("Search radius [arcmin]", 10.0)
        L_min = ask_float("Minimum name similarity [0, 1]", 0.9)
        ra = np.deg2rad(self.df[self.ra_col].to_numpy(float))
        dec = np.deg2rad(self.df[self.dec_col].to_numpy(float))
        xyz = np.array(
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
        ).T
        msk = np.isfinite(xyz).all(axis=1)
        idx = np.flatnonzero(msk)
        # Chord length equivalent to the angular radius
        chord = 2 * np.sin(np.deg2rad(rad / 60) / 2)

        pairs = []
        for a, b in KDTree(xyz[msk]).query_pairs(chord):
            i, j = sorted((int(idx[a]), int(idx[b])))
            if i in self.excl or j in self.excl:
                continue
            dist = np.rad2deg(2 * np.arcsin(np.linalg.norm(xyz[i] - xyz[j]) / 2)) * 60
            names_i = normalize_name(self.names[i]).split(",")
            names_j = normalize_name(self.names[j]).split(",")
            L = max(
                process.extractOne(q, names_j, scorer=fuzz.ratio)[1] for q in names_i
            )
            if L / 100 >= L_min:
                pairs.append((i, j, dist, L / 100))
        if not pairs:
            print("No close pairs with similar names found")
            return

        pairs.sort(key=lambda x: x[2])
        print(f"\nFound {len(pairs)} close pairs with similar names")
        for N_p, (i, j, dist, L) in enumerate(pairs, 1):
            if i in self.excl or j in self.excl:
                continue
            print(f"\nPair {N_p}/{len(pairs)}: d={dist:.2f}', L={L:.2f}")
            print(f"  {i:<6} {self.names[i]:<40} {self.coords_str(i)}")
            print(f"  {j:<6} {self.names[j]:<40} {self.coords_str(j)}")
            rm = ask("Row to exclude, 'n' skip, 'q' stop", "n")
            if rm == "q":
                break
            if rm.isdigit() and int(rm) in (i, j):
                rm = int(rm)
                keep = j if rm == i else i
                self.add(rm, "duplicate", f"{self.names[rm]} → {self.names[keep]}")

    def review(self, items: list[tuple[int, str, str, str]], desc: str):
        """Ask for each candidate (row, info, category, comment) to be excluded"""
        items = [_ for _ in items if _[0] not in self.excl]
        if not items:
            print(f"No {desc} found (excluding those already excluded)")
            return
        print(f"\nFound {len(items)} {desc}:")
        for row, info, _, _ in items:
            print(f"  {row:<6} {self.names[row]:<40} {info}")
        mode = ask("Exclude: 'a' all, 'r' review one by one, 'n' none", "r")
        if mode == "a":
            for row, _, cat, cmmt in items:
                self.add(row, cat, cmmt)
        elif mode == "r":
            for row, info, cat, cmmt in items:
                print(f"\n  {row:<6} {self.names[row]:<40} {info}")
                ans = ask("Exclude? 'y' yes, 'n' no, 'e' edit comment, 'q' stop", "n")
                if ans == "q":
                    break
                if ans == "e":
                    cat = ask_category(cat)
                    cmmt = ask("Comment", cmmt)
                if ans in ("y", "e"):
                    self.add(row, cat, cmmt)

    # ------------------------------------------------------------------------
    # Manual exclusions

    def excl_by_value(self):
        """Exclude all entries with given values in a column"""
        col = ask_col(self.df, "Column")
        counts = self.df[col].astype(str).str.strip().value_counts()
        print_list([f"{v!r}: {n}" for v, n in counts.items()])
        values = [
            _.strip() for _ in ask("Values to exclude (comma separated)").split(",")
        ]
        col_vals = self.df[col].astype(str).str.strip().to_numpy()

        # Rows, category and comment for each value
        rows_cat = {}
        cat = "not_cluster"
        for v in dict.fromkeys(values):
            rows = [int(_) for _ in np.flatnonzero(col_vals == v) if _ not in self.excl]
            if not rows:
                print(f"'{v}': no (new) entries found")
                continue
            print(f"'{v}': {len(rows)} entries")
            # The previous value's category is the default
            cat = ask_category(cat)
            cmmt = ask(f"Comment for '{v}'", f"{col}={v}")
            rows_cat[v] = (rows, cat, cmmt)
        if not rows_cat:
            return

        N_rows = sum(len(rows) for rows, _, _ in rows_cat.values())
        if not ask_yn(f"Exclude {N_rows} entries?", True):
            return
        for rows, cat, cmmt in rows_cat.values():
            for row in rows:
                self.add(row, cat, cmmt)

    def excl_manual(self):
        """
        Search entries by (part of their) name or row, and exclude them. An input
        with the format of the B script's duplicates output, i.e.
        'Bica 27 --> Dutra Bica 42 (214)', excludes row 214 as a duplicate of
        'Dutra Bica 42'
        """
        while True:
            q = ask(
                "Name (part of it), row number, or 'name --> original (row)'; "
                "blank to stop",
                "",
            )
            if q == "":
                return
            m = re.match(r"^\s*-?\s*(.*?)\s*-->\s*(.*?)\s*\((\d+)\)\s*$", q)
            if m:
                self.excl_dup(*m.groups())
                continue
            if q.isdigit() and int(q) < len(self.df):
                rows = [int(q)]
            else:
                rows = self.search(q)
            if not rows:
                print("  No entries found")
                continue
            for row in rows[:50]:
                flag = " (excluded)" if row in self.excl else ""
                print(f"  {row:<6} {self.names[row]:<40} {self.coords_str(row)}{flag}")
            if len(rows) > 50:
                print(f"  ... ({len(rows)} entries found, only first 50 shown)")
            sel = ask("Rows to exclude (comma separated), blank for none", "")
            sel_rows = []
            for r in sel.split(","):
                r = r.strip()
                if not r.isdigit() or int(r) not in rows:
                    if r:
                        print(f"  Row {r} not in the list above, skipped")
                    continue
                r = int(r)
                if r in self.excl:
                    _, cat, cmmt = self.excl[r]
                    print(f"  Row {r} already excluded ({cat}: {cmmt})")
                    if not ask_yn("  Replace its category and comment?"):
                        continue
                sel_rows.append(r)
            sel = sel_rows
            if not sel:
                continue
            cat = ask_category("other")
            cmmt = ask("Comment", "")
            for row in sel:
                self.add(row, cat, cmmt)

    def excl_dup(self, name: str, orig: str, row: str):
        """Exclude 'row' (named 'name') as a duplicate of 'orig'"""
        row = int(row)
        if row >= len(self.df):
            print(f"  Row {row} out of range")
            return
        print(f"  {row:<6} {self.names[row]:<40} {self.coords_str(row)}")
        if row in self.excl:
            print(f"  Row {row} already excluded: {self.excl[row][1:]}")
            return
        if name and row not in self.search(name):
            print(f"  WARNING: '{name}' not found in the names of row {row}")
            if not ask_yn("Exclude anyway?"):
                return
        self.add(row, "duplicate", orig)

    def search(self, q: str) -> list[int]:
        """
        Rows whose names contain any of the (comma separated) names in 'q'. Each
        name is searched normalized, and standardized as in the B script (e.g.
        'FSR 111' --> 'fsr0111'), also in uppercase since some standardizations
        are case sensitive. The names of each row are compared both standardized
        and only normalized (as written in the DB), so that the search does not
        depend on the case or on zero padding. Exact matches are listed first.
        """
        terms = set()
        for name in q.strip().strip("'\"").split(","):
            name = name.strip()
            if not name:
                continue
            terms.add(normalize_name(name))
            for n in (name, name.upper()):
                try:
                    terms.add(get_fnames([n])[0][0])
                except ValueError:
                    pass
        terms.discard("")

        if self.names_norm is None:
            self.names_norm = [
                [normalize_name(_) for _ in str(names).split(",")]
                for names in self.names
            ]

        exact, partial = [], []
        for i, (fnames, norms) in enumerate(zip(self.fnames, self.names_norm)):
            all_names = fnames + norms
            if any(t in all_names for t in terms):
                exact.append(i)
            elif any(t in f for t in terms for f in all_names):
                partial.append(i)
        return exact + partial

    def show(self):
        if not self.excl:
            print("No excluded entries")
            return
        print_list(
            [
                f"{row:<6} {name:<40} {cat:<12} {cmmt}"
                for row, (name, cat, cmmt) in sorted(self.excl.items())
            ]
        )

    def remove(self):
        self.show()
        sel = ask("Rows to remove from the exclusion list (comma separated)", "")
        for r in sel.split(","):
            if r.strip().isdigit() and int(r) in self.excl:
                del self.excl[int(r)]
                self.unsaved = True
                print(f"  Row {r.strip()} removed")

    # ------------------------------------------------------------------------
    # Helpers

    def add(self, row: int, cat: str, cmmt: str):
        self.excl[row] = [self.names[row], cat, cmmt]
        self.unsaved = True
        print(f"  + {row:<6} {self.names[row]:<40} {cat}: {cmmt}")

    def coords_str(self, row: int) -> str:
        if self.ra_col is None:
            return ""
        return f"({self.df.at[row, self.ra_col]}, {self.df.at[row, self.dec_col]})"

    def save(self):
        if not self.unsaved:
            print("No changes to save")
            return
        df_excl = pd.DataFrame(
            [[row, *vals] for row, vals in sorted(self.excl.items())],
            columns=excl_cols,
        )
        self.excl_path.parent.mkdir(parents=True, exist_ok=True)
        df_excl.to_csv(self.excl_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        self.unsaved = False
        print(f"{len(df_excl)} excluded entries saved to {self.excl_path}")

        # Summary for the 'DBs_edits.md' file (first N_max entries per section)
        N_max = 50
        print("\nSummary (DBs_edits.md):")
        for title in dict.fromkeys(md_titles.values()):
            lines = []
            for name, cat, cmmt in self.excl.values():
                if md_titles[cat] != title:
                    continue
                if cat == "duplicate" and "→" in cmmt:
                    # Comment already in the 'name → original' format
                    lines.append(f"- {cmmt}")
                else:
                    lines.append(f"- {name}" + (f" → {cmmt}" if cmmt else ""))
            if lines:
                print(f"\n{title}:\n")
                print("\n".join(lines[:N_max]))
                if len(lines) > N_max:
                    print(f"(and {len(lines) - N_max} more)")


def select_DB(path: str | None = None) -> str:
    """
    Select the DB file to load. If no path is given, list the CSV files in the
    temp folder (most recently modified first) as candidates. A file name that is
    not a valid path ('BICA2019', 'BICA2019.csv') is searched in the data and temp
    folders.
    """
    if path is not None:
        found = find_DB(path)
        if found is not None:
            return found

    temp_path = ROOT / temp_folder
    excl_dir = temp_path / UCC_excl_folder
    # Files generated by the scripts, not DBs
    skip = {UCC_cat_B_out, all_OC_names}
    cands = sorted(
        (
            f
            for f in temp_path.rglob("*.csv")
            if excl_dir not in f.parents
            and f.name not in skip
            and not f.name.startswith("B_cat_diff_")
        ),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if cands:
        print(f"CSV files found in {temp_path}:")
        for i, f in enumerate(cands, 1):
            print(f"  {i}) {f.relative_to(temp_path)}")
        msg, default = "Select a file by number, or enter a path or file name", "1"
    else:
        msg, default = "Path or name of the DB file", None

    while True:
        ans = ask(msg, default)
        if cands and ans.isdigit() and 1 <= int(ans) <= len(cands):
            return str(cands[int(ans) - 1])
        found = find_DB(ans)
        if found is not None:
            return found


def find_DB(ans: str) -> str | None:
    """
    Return 'ans' if it is an existing file. Otherwise search the data and temp
    folders for CSV files matching the name (exact matches, or else files whose
    name contains it) and ask the user to select one
    """
    if Path(ans).expanduser().is_file():
        return ans
    name = Path(ans).name.lower().removesuffix(".csv")
    if not name:
        return None

    # Folders with files that are not DBs
    skip_dirs = [ROOT / UCC_excl_folder, ROOT / UCC_cmmts_folder]
    skip_dirs += [ROOT / temp_folder / _ for _ in (UCC_excl_folder, UCC_cmmts_folder)]
    files = []
    for folder in (ROOT / data_folder, ROOT / temp_folder):
        for f in folder.rglob("*.csv"):
            if not any(d in f.parents for d in skip_dirs):
                files.append(f)
    matches = [f for f in files if f.stem.lower() == name]
    if not matches:
        matches = [f for f in files if name in f.stem.lower()]
    if not matches:
        print(f"  No file matching '{ans}' found in {data_folder} or {temp_folder}")
        return None

    print(f"Files matching '{ans}':")
    for i, f in enumerate(matches, 1):
        print(f"  {i}) {f.relative_to(ROOT)}")
    sel = ask(
        "Select a file by number, blank to cancel", "1" if len(matches) == 1 else ""
    )
    if sel.isdigit() and 1 <= int(sel) <= len(matches):
        return str(matches[int(sel) - 1])
    return None


def ang_dist(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Angular distance matrix [arcmin] between two sets of (RA, DEC) [deg]"""
    ra1, dec1, ra2, dec2 = map(np.deg2rad, (ra1, dec1, ra2, dec2))
    hav = (
        np.sin((dec2[None, :] - dec1[:, None]) / 2) ** 2
        + np.cos(dec1[:, None])
        * np.cos(dec2[None, :])
        * np.sin((ra2[None, :] - ra1[:, None]) / 2) ** 2
    )
    d = np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))) * 60
    # NaN coordinates never match
    return np.nan_to_num(d, nan=np.inf)


# ----------------------------------------------------------------------------
# Input helpers


def ask(msg: str, default: str | None = None) -> str:
    txt = f"{msg} [{default}]: " if default not in (None, "") else f"{msg}: "
    ans = input(txt).strip()
    if ans == "" and default is not None:
        return default
    return ans


def ask_yn(msg: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    ans = input(f"{msg} ({d}): ").strip().lower()
    if ans == "":
        return default
    return ans in ("y", "yes")


def ask_float(msg: str, default: float) -> float:
    while True:
        try:
            return float(ask(msg, str(default)))
        except ValueError:
            print("  Not a number")


def ask_col(
    df: pd.DataFrame,
    msg: str,
    default: str | None = None,
    allow_none: bool = False,
) -> str | None:
    """Ask for a column by name or number. With allow_none, '-' returns None"""
    if default is None and allow_none:
        default = "-"
    while True:
        col = ask(msg, default)
        if col == "-" and allow_none:
            return None
        if col in df.columns:
            return col
        if col.isdigit() and int(col) < len(df.columns):
            return df.columns[int(col)]
        print(f"  Column not found. Columns: {list(df.columns)}")


def norm_col(col: str) -> str:
    """Column name in lowercase, without non alphanumeric characters"""
    return "".join(_ for _ in col.lower() if _.isalnum())


def guess_names_col(df: pd.DataFrame) -> str | None:
    """
    Suggest the names column: a known name, or else the text column with the
    largest fraction of unique values
    """
    known = ["name", "names", "cluster", "clname", "designation", "id", "oname"]
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    by_norm = {norm_col(c): c for c in text_cols}
    for k in known:
        if k in by_norm:
            return by_norm[k]
    if not text_cols:
        return None
    return max(text_cols, key=lambda c: df[c].nunique() / len(df))


def guess_coord_col(df: pd.DataFrame, coord: str, ra_col: str | None = None):
    """
    Suggest the RA or DEC column [deg]: numeric, with values in the valid range
    (negative RA values are allowed, the B script wraps them) and a known name.
    For DEC, the column paired with the selected RA column is preferred (e.g.
    'RA_ICRS' --> 'DE_ICRS'). Columns computed by Vizier ('_RA.icrs') and columns
    with mixed case names are suggested only if no other column is found.
    """
    vmin, vmax = (-180, 360) if coord == "RA" else (-90, 90)
    valid = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = df[c].dropna()
        if len(v) > 0 and v.between(vmin, vmax).all():
            valid.append(c)

    if coord == "RA":
        known = ["ra", "raicrs", "raj2000", "radeg", "radegj2000", "alpha", "alphadeg"]
    else:
        known = ["dec", "de", "deicrs", "decicrs", "dej2000", "decj2000", "decdeg"]
        known += ["dedeg", "delta", "deltadeg"]
        if ra_col is not None:
            # Paired names first: 'RA_ICRS' --> 'DE_ICRS', 'DEC_ICRS', ...
            ra_norm = norm_col(ra_col)
            for p_ra, p_dec in (("ra", "de"), ("ra", "dec"), ("alpha", "delta")):
                if ra_norm.startswith(p_ra):
                    known.insert(0, p_dec + ra_norm[len(p_ra) :])

    def uniform_case(c: str) -> bool:
        letters = "".join(_ for _ in c if _.isalpha())
        return letters.isupper() or letters.islower()

    # Mixed case names ('Ra': apocentre, 'rA': reference) are suggested last
    for cols in (
        [c for c in valid if not c.startswith("_") and uniform_case(c)],
        [c for c in valid if c.startswith("_")],
        [c for c in valid if not c.startswith("_") and not uniform_case(c)],
    ):
        by_norm = {}
        for c in cols:
            by_norm.setdefault(norm_col(c), c)
        for k in known:
            if k in by_norm:
                return by_norm[k]
    return None


def ask_category(default: str) -> str:
    """Ask for a category by number or name"""
    opts = ", ".join(f"{c} [{i}]" for i, c in enumerate(categories, 1))
    while True:
        cat = ask(f"Category: {opts}", default)
        if cat.isdigit() and 1 <= int(cat) <= len(categories):
            return categories[int(cat) - 1]
        if cat in categories:
            return cat
        print("  Unknown category")


def print_list(lines: list[str], N_max: int = 50):
    for i in range(0, len(lines), N_max):
        print("\n".join(lines[i : i + N_max]))
        if i + N_max < len(lines) and not ask_yn("Show more?", True):
            break


if __name__ == "__main__":
    main()
