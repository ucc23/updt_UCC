"""
Interactive sanitizer for new databases before they are added to the UCC.

Generalizes the one-off fixes collected in 'DBs_edits.py'. Load a CSV, apply
any number of operations from the menu (each one asks for its parameters),
inspect/undo as needed, and store the result with the UCC CSV format
(na_rep="nan", QUOTE_NONNUMERIC).

Usage:
    python edit_DBs.py [path/to/DB.csv]
"""

import csv
import json
import re
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from modules.utils import normalize_name, rename_standard  # noqa: E402

UCC_names_file = ROOT / "data" / "all_names.csv"
coords_round = 5

def main():
    df, path = load_csv(path=sys.argv[1] if len(sys.argv) > 1 else None)
    history: list[tuple[pd.DataFrame, str]] = []
    unsaved = False

    flat = [f for _, fs in OPERATIONS for f in fs]
    while True:
        print(f"\n=== {path.name}: {len(df)} rows, {len(df.columns)} cols ===")
        i = 1
        for group, fs in OPERATIONS:
            print(f"\n{group}")
            for f in fs:
                print(f"  {i:2d}) {f.__doc__.strip().splitlines()[0]}")
                i += 1
        print("\nOther options:   c) list columns   u) undo   l) log   s) save   q) quit")
        try:
            ans = input("> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            ans = "q"

        if ans == "c":
            print_list([f"{j}: {c}" for j, c in enumerate(df.columns)], 500)
        elif ans == "u":
            if history:
                df, desc = history.pop()
                print(f"Undone: {desc}")
                unsaved = True
            else:
                print("Nothing to undo")
        elif ans == "l":
            print_list([d for _, d in history] or ["(no operations)"], 500)
        elif ans == "s":
            if save(df, path) is not None:
                unsaved = False
        elif ans == "q":
            try:
                if unsaved and ask_yn("Unsaved changes, save before quitting?", True):
                    save(df, path)
            except (KeyboardInterrupt, EOFError):
                pass
            break
        elif ans.isdigit() and 1 <= int(ans) <= len(flat):
            f = flat[int(ans) - 1]
            try:
                new_df, desc = f(df.copy())
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled")
                continue
            except Exception as e:
                print(f"Error: {e!r} (no changes applied)")
                continue
            if desc is not None:
                history.append((df, desc))
                df = new_df
                unsaved = True
                print(f"Applied: {desc}")
        else:
            print("Invalid option")


# ----------------------------------------------------------------------------
# Input helpers


def ask(msg: str, default: str | None = None, strip: bool = True) -> str:
    """
    Ask for a string, returning 'default' on empty input. Use strip=False for
    separators where surrounding spaces matter (e.g. ', ')
    """
    txt = f"{msg} [{default!r}]: " if not strip else (
        f"{msg} [{default}]: " if default not in (None, "") else f"{msg}: "
    )
    raw = input(txt)
    ans = raw.strip()
    if ans == "" and default is not None:
        return default
    return ans if strip else raw


def ask_yn(msg: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    ans = input(f"{msg} ({d}): ").strip().lower()
    if ans == "":
        return default
    return ans in ("y", "yes")


def ask_int(msg: str, default: int | None = None) -> int:
    while True:
        ans = ask(msg, None if default is None else str(default))
        try:
            return int(ans)
        except ValueError:
            print("  Invalid integer")


def ask_float(msg: str, default: float | None = None) -> float:
    while True:
        ans = ask(msg, None if default is None else str(default))
        try:
            return float(ans)
        except ValueError:
            print("  Invalid number")


def ask_choice(msg: str, options: list[str], default: str | None = None) -> str:
    """Ask the user to pick one of 'options' (by value or 1-based index)"""
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        ans = ask(msg, default)
        if ans in options:
            return ans
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1]
        print("  Invalid option")


def ask_list(msg: str, default: str | None = None, sep: str = ",") -> list[str]:
    """Ask for a list of items. If 'default' is None, at least one is required"""
    while True:
        ans = ask(msg, default)
        items = [_.strip() for _ in ans.split(sep) if _.strip() != ""]
        if items or default is not None:
            return items
        print("  Input required")


def eval_mask(df: pd.DataFrame, query: str) -> pd.Series:
    """Evaluate a boolean query over the DataFrame"""
    msk = df.eval(query, engine="python")
    if not isinstance(msk, pd.Series) or msk.dtype != bool:
        raise ValueError(f"Query '{query}' does not return a boolean mask")
    return msk


def ask_cols(
    df: pd.DataFrame,
    msg: str,
    single: bool = False,
    allow_new: bool = False,
    default: str | None = None,
    allow_empty: bool = False,
) -> list[str]:
    """
    Ask for one or more (comma separated) column names. Integers are accepted
    as column indexes. If 'allow_new' is True, unknown names are accepted
    """
    cols = list(df.columns)
    while True:
        ans = ask(msg, default)
        if ans == "":
            if allow_empty:
                return []
            print("  Input required")
            continue
        tokens = [ans.strip()] if single else [_.strip() for _ in ans.split(",")]
        out, ok = [], True
        for t in tokens:
            if t in cols:
                out.append(t)
            elif t.isdigit() and int(t) < len(cols):
                out.append(cols[int(t)])
            elif allow_new:
                out.append(t)
            else:
                print(f"  Column '{t}' not found")
                ok = False
        if ok and out:
            return out


def ask_col(df: pd.DataFrame, msg: str, **kwargs) -> str:
    return ask_cols(df, msg, single=True, **kwargs)[0]


def read_lines_or_file(msg: str) -> list[str]:
    """
    Read a list of items either from a file path or pasted, one per line
    (or comma separated), ending with an empty line
    """
    ans = input(f"{msg} (file path, or paste items and end with empty line):\n")
    if ans.strip() and Path(ans.strip()).expanduser().is_file():
        txt = Path(ans.strip()).expanduser().read_text()
        lines = txt.splitlines()
    else:
        lines = [ans]
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
    items = []
    for line in lines:
        items += [_.strip().strip('"').strip("'") for _ in line.split(",")]
    return [_ for _ in items if _ != ""]


def read_csv_from_input(msg: str) -> pd.DataFrame:
    """Read a CSV either from a file path or pasted text (end with empty line)"""
    ans = input(f"{msg} (file path, or paste CSV with header and end with empty line):\n")
    path = Path(ans.strip()).expanduser()
    if ans.strip() and path.is_file():
        return pd.read_csv(path)
    from io import StringIO

    lines = [ans]
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return pd.read_csv(StringIO("\n".join(lines)))


def load_csv(msg: str = "Path to CSV file", path: str | None = None) -> tuple:
    while True:
        if path is None:
            path = ask(msg)
        fpath = Path(path).expanduser()
        if fpath.is_file():
            break
        print(f"  File not found: {fpath}")
        path = None
    str_cols = ask_list(
        "Columns to read as strings (e.g. 'hhmmss' coords with leading zeros, blank for none)",
        "",
    )
    df = pd.read_csv(fpath, dtype={c: "string" for c in str_cols} or None)
    print(f"Loaded '{fpath}': {len(df)} rows, {len(df.columns)} columns")
    return df, fpath


# ----------------------------------------------------------------------------
# Generic helpers


def fmt_val(v) -> str:
    """Format a value for use inside a name (1.0 -> '1', nan -> '')"""
    if pd.isna(v):
        return ""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def join_values(vals, sep: str, unique: bool = True):
    """
    Join values with 'sep'. With unique=True NaNs and repeated values are
    dropped; otherwise NaNs are kept as 'nan' so positions stay aligned.
    A single remaining value is returned as is (keeps numeric types)
    """
    vals = list(vals)
    if all(pd.isna(v) for v in vals):
        return np.nan
    if unique:
        seen = {}
        for v in vals:
            if pd.notna(v):
                seen.setdefault(fmt_val(v), v)
        vals = list(seen.values())
    if len(vals) == 1:
        return vals[0]
    return sep.join(fmt_val(v) if pd.notna(v) else "nan" for v in vals)


def insert_after(df: pd.DataFrame, col: str, new_col: str, values) -> None:
    """Insert (or replace) 'new_col' right after 'col'"""
    if new_col in df.columns:
        df.drop(columns=[new_col], inplace=True)
    df.insert(df.columns.get_loc(col) + 1, new_col, values)


def place_column(df: pd.DataFrame, col: str, values, position: str) -> None:
    """
    Add/replace column 'col' with 'values' at 'position': 'end', 'first',
    an integer index, or 'after:<col>'
    """
    if col in df.columns:
        if position == "keep":
            df[col] = values
            return
        df.drop(columns=[col], inplace=True)
    if position == "end" or position == "keep":
        df[col] = values
    elif position == "first":
        df.insert(0, col, values)
    elif position.startswith("after:"):
        ref = position.split(":", 1)[1]
        df.insert(df.columns.get_loc(ref) + 1, col, values)
    else:
        df.insert(min(int(position), len(df.columns)), col, values)


def ask_position(df: pd.DataFrame, default: str = "end") -> str:
    while True:
        pos = ask(
            "Position for new column(s): 'end', 'first', 'keep' (if replacing), "
            "an index, or 'after:<col>'",
            default,
        )
        if pos in ("end", "first", "keep") or pos.isdigit():
            return pos
        if pos.startswith("after:") and pos.split(":", 1)[1] in df.columns:
            return pos
        print("  Invalid position")


def split_names(val, sep: str = ",") -> list[str]:
    if pd.isna(val):
        return []
    return [_.strip() for _ in str(val).split(sep) if _.strip() != ""]


def norm(name: str) -> str:
    """Normalize a name the same way the UCC does it"""
    return normalize_name(rename_standard(str(name).strip()))


def preview(df: pd.DataFrame, cols: list[str], n: int = 5) -> None:
    cols = [_ for _ in cols if _ in df.columns]
    if cols:
        print(df[cols].head(n).to_string())


def print_list(items, max_n: int = 50) -> None:
    items = list(items)
    for it in items[:max_n]:
        print(f"  {it}")
    if len(items) > max_n:
        print(f"  ... ({len(items) - max_n} more)")


# ----------------------------------------------------------------------------
# Operations. Each one receives the DataFrame and returns the (new) DataFrame
# plus a short description stored in the session log


def op_info(df):
    """Show summary: columns, dtypes, NaN counts, head, duplicated names"""
    print(f"\n{len(df)} rows, {len(df.columns)} columns\n")
    info = pd.DataFrame(
        {"dtype": df.dtypes.astype(str), "N_nan": df.isna().sum()},
        index=df.columns,
    )
    info.index = [f"{i}: {c}" for i, c in enumerate(df.columns)]
    print(info.to_string())
    print()
    print(df.head(int(ask("Rows to show", "5"))).to_string())
    col = ask("Column to check for duplicated values (blank to skip)", "")
    if col in df.columns:
        dups = df[df[col].duplicated(keep=False)].sort_values(col)
        print(f"{dups[col].nunique()} duplicated values ({len(dups)} rows)")
        print_list(dups[col].astype(str).unique())
    return df, None


def op_drop_columns(df):
    """Drop columns"""
    cols = ask_cols(df, "Columns to drop (comma separated)")
    return df.drop(columns=cols), f"drop columns {cols}"


def op_rename_columns(df):
    """Rename columns"""
    pairs = ask_list("Renames as 'old:new' (comma separated)")
    mapping = {}
    for p in pairs:
        old, new = [_.strip() for _ in p.split(":", 1)]
        if old not in df.columns:
            print(f"  Column '{old}' not found, skipped")
            continue
        mapping[old] = new
    return df.rename(columns=mapping), f"rename columns {mapping}"


def op_add_nan_columns(df):
    """Add empty (NaN) columns"""
    cols = ask_list("New column names (comma separated)")
    pos = ask_position(df)
    for i, c in enumerate(cols):
        p = pos
        if pos.isdigit():
            p = str(int(pos) + i)
        elif pos.startswith("after:") and i > 0:
            p = f"after:{cols[i - 1]}"
        place_column(df, c, np.nan, p)
    return df, f"add NaN columns {cols}"


def op_reorder_columns(df):
    """Move columns to a position, or set the full column order"""
    if ask_yn("Provide the full column order?"):
        cols = ask_cols(df, "All columns in the new order (comma separated)")
        rest = [c for c in df.columns if c not in cols]
        if rest:
            print(f"  Columns not listed (appended at the end): {rest}")
        return df[cols + rest], f"reorder columns {cols}"
    cols = ask_cols(df, "Columns to move (comma separated, kept in this order)")
    pos = ask_position(df, "first")
    for i, c in enumerate(cols):
        vals = df.pop(c)
        p = pos
        if pos.isdigit():
            p = str(int(pos) + i)
        elif i > 0:
            p = f"after:{cols[i - 1]}"
        place_column(df, c, vals, p)
    return df, f"move columns {cols} to {pos}"


def op_round(df):
    """Round float columns"""
    if ask_yn("Round ALL float columns?", True):
        cols = list(df.select_dtypes(include=["float"]).columns)
    else:
        cols = ask_cols(df, "Columns to round")
    nd = ask_int("Decimals", 5)
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").round(nd)
    return df, f"round {cols} to {nd} decimals"


def op_sort(df):
    """Sort rows by column(s)"""
    cols = ask_cols(df, "Columns to sort by")
    asc = ask_yn("Ascending?", True)
    df = df.sort_values(cols, ascending=asc).reset_index(drop=True)
    return df, f"sort by {cols}"


def op_gal_to_equ(df):
    """Galactic (l, b) -> equatorial ICRS (RA, DEC) in degrees"""
    glon = ask_col(df, "Galactic longitude column", default=_guess(df, ["GLON", "l", "_Glon"]))
    glat = ask_col(df, "Galactic latitude column", default=_guess(df, ["GLAT", "b", "_Glat"]))
    ra_c, de_c = _ask_out_coords(df, "RA", "DEC")
    c = SkyCoord(
        l=pd.to_numeric(df[glon], errors="coerce").values * u.deg,
        b=pd.to_numeric(df[glat], errors="coerce").values * u.deg,
        frame="galactic",
    ).icrs
    pos = ask_position(df, "keep" if ra_c in df.columns else "end")
    place_column(df, ra_c, c.ra.deg.round(coords_round), pos)
    place_column(df, de_c, c.dec.deg.round(coords_round), f"after:{ra_c}")
    preview(df, [glon, glat, ra_c, de_c])
    return df, f"({glon}, {glat}) -> ({ra_c}, {de_c})"


def op_equ_to_gal(df):
    """Equatorial (RA, DEC) degrees -> galactic (GLON, GLAT)"""
    ra = ask_col(df, "RA column (deg)", default=_guess(df, ["RA_ICRS", "RAJ2000", "RA", "_RA"]))
    de = ask_col(df, "DEC column (deg)", default=_guess(df, ["DE_ICRS", "DEJ2000", "DEC", "_DE"]))
    l_c, b_c = _ask_out_coords(df, "GLON", "GLAT")
    c = SkyCoord(
        ra=pd.to_numeric(df[ra], errors="coerce").values * u.deg,
        dec=pd.to_numeric(df[de], errors="coerce").values * u.deg,
        frame="icrs",
    ).galactic
    pos = ask_position(df, "keep" if l_c in df.columns else "end")
    place_column(df, l_c, c.l.deg.round(coords_round), pos)
    place_column(df, b_c, c.b.deg.round(coords_round), f"after:{l_c}")
    preview(df, [ra, de, l_c, b_c])
    return df, f"({ra}, {de}) -> ({l_c}, {b_c})"


def _guess(df, names) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _ask_out_coords(df, d1, d2) -> tuple[str, str]:
    c1 = ask("Output column name (1st coordinate)", d1)
    c2 = ask("Output column name (2nd coordinate)", d2)
    for c in (c1, c2):
        if c in df.columns and not ask_yn(f"Column '{c}' exists, overwrite?", True):
            raise RuntimeError("Aborted")
    return c1, c2


def _ask_frame() -> tuple[str, str | None]:
    frame = ask_choice(
        "Frame/equinox of the input coordinates",
        ["icrs", "fk5 (J2000)", "fk4 (B1950)", "fk4 (other)"],
        "1",
    )
    if frame == "icrs":
        return "icrs", None
    if frame == "fk5 (J2000)":
        return "fk5", "J2000"
    if frame == "fk4 (B1950)":
        return "fk4", "B1950"
    return "fk4", ask("Equinox (e.g. B1900)")


def _skycoord(ra, dec, frame: str, equinox: str | None, unit) -> SkyCoord:
    kw = {"frame": frame, "unit": unit}
    if equinox is not None:
        kw["equinox"] = equinox
    return SkyCoord(ra=ra, dec=dec, **kw).icrs


def op_sexa_to_deg(df):
    """Sexagesimal / other-frame coordinates -> ICRS degrees"""
    mode = ask_choice(
        "Input format",
        [
            "two string columns (e.g. '12 34 56.7' / '12:34:56.7' / '12h34m56.7s')",
            "two compact columns (hhmmss.s / +ddmmss.s)",
            "six columns (h, m, s, d, m, s)",
            "two decimal-degree columns in another frame (e.g. B1950)",
        ],
        "1",
    )
    src_cols = []
    if mode.startswith("two string"):
        ra = ask_col(df, "RA column")
        de = ask_col(df, "DEC column")
        src_cols = [ra, de]
        ra_v, de_v = df[ra].astype(str).str.strip(), df[de].astype(str).str.strip()
        unit = (u.hourangle, u.deg)
        if ask_yn("Is RA given in degrees (not hours)?"):
            unit = (u.deg, u.deg)
    elif mode.startswith("two compact"):
        ra = ask_col(df, "RA column (hhmmss.s)")
        de = ask_col(df, "DEC column (+ddmmss.s)")
        src_cols = [ra, de]

        def f_ra(x):
            x = str(x).strip()
            return f"{x[0:2]}h{x[2:4]}m{x[4:]}s"

        def f_de(x):
            x = str(x).strip()
            if x[0] not in "+-":
                x = "+" + x
            return f"{x[0]}{x[1:3]}d{x[3:5]}m{x[5:]}s"

        ra_v, de_v = df[ra].apply(f_ra), df[de].apply(f_de)
        unit = (u.hourangle, u.deg)
        print("  (read these columns as strings to keep leading zeros)")
    elif mode.startswith("six"):
        cs = ask_cols(df, "The six columns in order: RAh,RAm,RAs,DEd,DEm,DEs")
        if len(cs) != 6:
            raise RuntimeError("Six columns required")
        src_cols = cs
        h, m, s = (df[c].map(fmt_val) for c in cs[:3])
        d, dm, ds = (df[c].map(fmt_val) for c in cs[3:])
        if ask_yn("Is the DEC sign stored in a separate column?"):
            sign_c = ask_col(df, "DEC sign column")
            src_cols.append(sign_c)
            d = df[sign_c].astype(str).str.strip() + d.str.lstrip("+-")
        ra_v = h + "h" + m + "m" + s + "s"
        de_v = d + "d" + dm + "m" + ds + "s"
        unit = (u.hourangle, u.deg)
    else:
        ra = ask_col(df, "RA column (deg)")
        de = ask_col(df, "DEC column (deg)")
        src_cols = [ra, de]
        ra_v = pd.to_numeric(df[ra], errors="coerce").values
        de_v = pd.to_numeric(df[de], errors="coerce").values
        unit = (u.deg, u.deg)

    frame, equinox = _ask_frame()
    ra_c, de_c = _ask_out_coords(df, "RA", "DEC")

    # Parse row by row only if the vectorized call fails (bad/missing entries)
    try:
        c = _skycoord(ra_v.values if hasattr(ra_v, "values") else ra_v,
                      de_v.values if hasattr(de_v, "values") else de_v,
                      frame, equinox, unit)
        ra_d, de_d = c.ra.deg, c.dec.deg
    except Exception:
        ra_d, de_d = [], []
        for r, d in zip(ra_v, de_v):
            try:
                ci = _skycoord(r, d, frame, equinox, unit)
                ra_d.append(ci.ra.deg)
                de_d.append(ci.dec.deg)
            except Exception:
                print(f"  Could not parse: {r}, {d}")
                ra_d.append(np.nan)
                de_d.append(np.nan)
        ra_d, de_d = np.array(ra_d), np.array(de_d)

    pos = ask_position(df, "keep" if ra_c in df.columns else "end")
    place_column(df, ra_c, np.round(ra_d, coords_round), pos)
    place_column(df, de_c, np.round(de_d, coords_round), f"after:{ra_c}")
    preview(df, src_cols + [ra_c, de_c])
    drop = [c for c in src_cols if c not in (ra_c, de_c)]
    if drop and ask_yn(f"Drop source columns {drop}?"):
        df = df.drop(columns=drop)
    return df, f"{src_cols} ({frame} {equinox or ''}) -> ICRS ({ra_c}, {de_c})"


def op_compare_coords(df):
    """Compare two coordinate pairs; list/replace large separations"""
    ra1 = ask_col(df, "RA column of pair A (to check)")
    de1 = ask_col(df, "DEC column of pair A")
    ra2 = ask_col(df, "RA column of pair B (reference)")
    de2 = ask_col(df, "DEC column of pair B")
    c1 = SkyCoord(ra=pd.to_numeric(df[ra1], errors="coerce").values * u.deg,
                  dec=pd.to_numeric(df[de1], errors="coerce").values * u.deg)
    c2 = SkyCoord(ra=pd.to_numeric(df[ra2], errors="coerce").values * u.deg,
                  dec=pd.to_numeric(df[de2], errors="coerce").values * u.deg)
    sep = c1.separation(c2).arcmin
    print(f"Separation [arcmin]: median={np.nanmedian(sep):.3f}, max={np.nanmax(sep):.3f}")
    thr = ask_float("Threshold [arcmin]", 60)
    msk = sep > thr
    name_c = ask_col(df, "Name column to display", default=str(df.columns[0]))
    print(f"{msk.sum()} rows above threshold:")
    show = df.loc[msk, [name_c, ra1, de1, ra2, de2]].copy()
    show["sep_arcmin"] = sep[msk].round(2)
    print(show.to_string())
    nan_msk = np.isnan(sep) & df[ra2].notna().values & df[de2].notna().values
    if nan_msk.sum():
        print(f"{nan_msk.sum()} rows with missing pair A and available pair B")
    log = None
    if msk.sum() and ask_yn("Replace pair A with pair B for these rows?"):
        df.loc[msk, [ra1, de1]] = df.loc[msk, [ra2, de2]].values
        log = f"replace ({ra1}, {de1}) with ({ra2}, {de2}) where sep > {thr} arcmin"
    if nan_msk.sum() and ask_yn("Fill missing pair A with pair B?"):
        df.loc[nan_msk, [ra1, de1]] = df.loc[nan_msk, [ra2, de2]].values
        log = (log or "") + f"; fill NaN ({ra1}, {de1}) from ({ra2}, {de2})"
    return df, log


def op_split_value_error(df):
    """Split 'value±error' columns into value and error columns"""
    cols = ask_cols(df, "Columns with 'value±error' entries")
    sep = ask("Separator", "±")
    prefix = ask("Prefix for the error column", "e_")
    for col in cols:
        parts = df[col].astype(str).str.strip().str.split(sep, n=1, expand=True)
        if parts.shape[1] == 1:
            parts[1] = np.nan
        df[col] = pd.to_numeric(parts[0], errors="coerce")
        insert_after(df, col, f"{prefix}{col}", pd.to_numeric(parts[1], errors="coerce"))
        preview(df, [col, f"{prefix}{col}"], 3)
    return df, f"split value{sep}error in {cols}"


def op_split_paren_error(df):
    """Split 'value(err)' columns (error in last significant digits)"""
    cols = ask_cols(df, "Columns with 'value(err)' entries, e.g. '0.012(3)'")
    prefix = ask("Prefix for the error column", "e_")
    pat = re.compile(r"\s*([+-]?\d*\.?\d+)\s*\((\d+)\)\s*")
    for col in cols:
        vals, errs = [], []
        for s in df[col].astype(str):
            m = pat.fullmatch(s)
            if m:
                v, e = m.groups()
                n_dec = len(v.split(".")[1]) if "." in v else 0
                vals.append(float(v))
                errs.append(round(float(e) * 10 ** (-n_dec), n_dec))
            else:
                vals.append(pd.to_numeric(s, errors="coerce"))
                errs.append(np.nan)
        df[col] = vals
        insert_after(df, col, f"{prefix}{col}", errs)
        preview(df, [col, f"{prefix}{col}"], 3)
    return df, f"split value(err) in {cols}"


def op_derived_column(df):
    """New/replaced column from an expression (e.g. sqrt(X**2+Y**2+Z**2))"""
    print("  Examples: 'sqrt(X**2 + Y**2 + Z**2)', 'log10(Age * 1e6)', 'dist / 1000'")
    print("  Use backticks for names with special characters: `E(B-V)` * 3.1")
    expr = ask("Expression")
    col = ask("Output column name")
    res = df.eval(expr, engine="python")
    nd = ask("Round to N decimals (blank for no rounding)", "")
    if nd:
        res = pd.to_numeric(res, errors="coerce").round(int(nd))
    pos = ask_position(df, "keep" if col in df.columns else "end")
    place_column(df, col, res, pos)
    preview(df, [col])
    return df, f"{col} = {expr}"


def op_combine_columns(df):
    """Combine two columns into one (e.g. 'Names' + 'ONames', 'Cluster' + 'cn')"""
    c1 = ask_col(df, "Main column")
    c2 = ask_col(df, "Column to add")
    sep = ask("Separator (use '_' for 'Name_num', ', ' for an alternative name)", ", ", strip=False)
    out = ask("Output column", c1)

    def comb(row):
        a, b = fmt_val(row[c1]), fmt_val(row[c2])
        if b == "" or b == a:
            return row[c1] if a != "" else np.nan
        if a == "":
            return b
        # Do not repeat a name that is already listed
        if sep.strip() == "," and b in split_names(a):
            return a
        return f"{a}{sep}{b}"

    vals = df.apply(comb, axis=1)
    place_column(df, out, vals, "keep" if out in df.columns else f"after:{c1}")
    preview(df, [c1, c2, out] if out != c1 else [c1, c2])
    if c2 != out and ask_yn(f"Drop column '{c2}'?"):
        df = df.drop(columns=[c2])
    return df, f"{out} = {c1} + '{sep}' + {c2}"


def op_name_template(df):
    """Build names from a template using other columns (e.g. 'VDBH_{No}')"""
    print("  Use {col} to insert values of a column, e.g.: 'VDBH_{No}', 'MWSC_{MWSC}'")
    tmpl = ask("Template")
    refs = re.findall(r"{([^{}]+)}", tmpl)
    for r in refs:
        if r not in df.columns:
            raise RuntimeError(f"Column '{r}' not found")
    col = ask_col(df, "Names column", allow_new=True)
    mode = ask_choice(
        "Mode",
        [
            "append as alternative name",
            "prepend as main name",
            "fill only empty names",
            "replace names",
        ],
        "1",
    )
    sep = ", "

    def make(row):
        vals = {r: fmt_val(row[r]) for r in refs}
        if any(v == "" for v in vals.values()):
            return None
        new = tmpl
        for r, v in vals.items():
            new = new.replace("{" + r + "}", v)
        return new

    if col not in df.columns:
        df[col] = np.nan
    out = []
    for _, row in df.iterrows():
        new, cur = make(row), row[col]
        names = split_names(cur)
        if new is None:
            out.append(cur)
        elif mode == "fill only empty names":
            out.append(cur if names else new)
        elif mode == "replace names":
            out.append(new)
        elif norm(new) in [norm(_) for _ in names]:
            out.append(cur)
        elif mode == "append as alternative name":
            out.append(sep.join(names + [new]))
        else:
            out.append(sep.join([new] + names))
    df[col] = out
    preview(df, [col] + refs)
    if ask_yn(f"Drop columns {refs}?"):
        df = df.drop(columns=[r for r in refs if r != col])
    return df, f"{mode} '{tmpl}' in {col}"


def op_expand_prefix(df):
    """Expand/replace abbreviated name prefixes (e.g. 'RUP12' -> 'Ruprecht_12')"""
    print("  Pairs as 'abbrev:full', e.g.: 'RUP:Ruprecht, CR:Collinder, N:NGC, I:IC'")
    pairs = ask_list("Prefix pairs")
    mapping = [tuple(_.strip() for _ in p.split(":", 1)) for p in pairs]
    # Longest prefixes first so that 'NGC' is not caught by 'N'
    mapping.sort(key=lambda x: -len(x[0]))
    col = ask_col(df, "Names column")
    joiner = ask("Joiner between new prefix and number", "_", strip=False)
    case = ask_yn("Case insensitive match?", True)
    keep = ask_yn("Keep original name as an alternative name?", False)
    sep = ask("Separator for multiple names in a cell", ",")
    flags = re.IGNORECASE if case else 0
    pats = [
        (re.compile(rf"^{re.escape(a)}[\s_\-]*(\d.*)$", flags), full)
        for a, full in mapping
    ]

    def conv(name):
        for pat, full in pats:
            m = pat.match(name)
            if m:
                return f"{full}{joiner}{m.group(1).strip()}"
        return None

    n_ch = 0
    out = []
    for val in df[col]:
        names = split_names(val, sep)
        if not names:
            out.append(val)
            continue
        new_names = []
        for n in names:
            new = conv(n)
            if new is None or new == n:
                new_names.append(n)
                continue
            n_ch += 1
            new_names.append(new)
            if keep:
                new_names.append(n)
        # Remove duplicates keeping order
        new_names = list(dict.fromkeys(new_names))
        out.append(f"{sep.strip()} ".join(new_names))
    df[col] = out
    print(f"{n_ch} names changed")
    preview(df, [col], 10)
    return df, f"expand prefixes {mapping} in {col}"


def op_split_letters_digits(df):
    """Insert a separator between leading letters and digits ('NGC123' -> 'NGC_123')"""
    col = ask_col(df, "Names column")
    jn = ask("Separator", "_", strip=False)
    only_first = ask_yn("Only apply to the first name in each cell?", False)
    pat = re.compile(r"^([A-Za-z][A-Za-z\-\.]*?)(\d.*)$")

    def conv(name):
        m = pat.match(name)
        return f"{m.group(1)}{jn}{m.group(2)}" if m else name

    out = []
    for val in df[col]:
        names = split_names(val)
        if not names:
            out.append(val)
            continue
        if only_first:
            names[0] = conv(names[0])
        else:
            names = [conv(_) for _ in names]
        out.append(", ".join(names))
    df[col] = out
    preview(df, [col], 10)
    return df, f"split letters/digits in {col} with '{jn}'"


def op_remove_substrings(df):
    """Remove words/substrings from a column (e.g. 'IR Cluster', 'Cluster')"""
    col = ask_col(df, "Column")
    subs = ask_list("Substrings to remove, in order (comma separated)")
    s = df[col].astype("string")
    for sub in subs:
        s = s.str.replace(sub, "", regex=False)
    df[col] = s.str.replace(r"\s+", " ", regex=True).str.strip().str.strip(",").str.strip()
    preview(df, [col], 10)
    return df, f"remove {subs} from {col}"


def op_regex_replace(df):
    """Regex/text replace in a column (e.g. whitespace -> '_')"""
    col = ask_col(df, "Column")
    print("  Example: pattern '\\s+' with replacement '_' replaces spaces by underscores")
    pat = ask("Pattern (regex)")
    rep = ask("Replacement", "")
    before = df[col].copy()
    df[col] = df[col].astype("string").str.replace(pat, rep, regex=True)
    print(f"{(before.astype('string') != df[col]).sum()} entries changed")
    preview(df, [col], 10)
    return df, f"regex '{pat}' -> '{rep}' in {col}"


def op_first_name_column(df):
    """New column with the first name from a list column (e.g. 'Names')"""
    src = ask_col(df, "Source column with multiple names")
    sep = ask("Separator", ",")
    new = ask("New column name", "Name")
    vals = df[src].astype("string").str.split(sep).str[0].str.strip()
    pos = ask_position(df, "first")
    place_column(df, new, vals, pos)
    preview(df, [new, src])
    print("  (use 'remove substrings' to clean the new column if needed)")
    return df, f"{new} = first name in {src}"


def op_map_codes(df):
    """Map code columns to names (e.g. 'ClSeq'=3 + 'ClNum'=12 -> 'Berkeley_12')"""
    code_c = ask_col(df, "Column with codes")
    print("  Mapping as JSON file path, or 'code:name' pairs (e.g. '1:NGC, 2:IC, 3:Berkeley')")
    ans = ask("Mapping")
    if Path(ans).expanduser().is_file():
        mapping = {str(k): v for k, v in json.loads(Path(ans).expanduser().read_text()).items()}
    else:
        mapping = {}
        for p in ans.split(","):
            k, v = p.split(":", 1)
            mapping[k.strip()] = v.strip()
    num_c = ask("Column with numbers to append (blank for none)", "")
    if num_c and num_c not in df.columns:
        raise RuntimeError(f"Column '{num_c}' not found")
    jn = ask("Joiner", "_", strip=False) if num_c else ""
    unknown = ask("Name for unknown codes", "Unknown")
    out_c = ask("Output column", "Cluster")

    def make(row):
        base = mapping.get(fmt_val(row[code_c]), unknown)
        if num_c:
            n = fmt_val(row[num_c])
            return f"{base}{jn}{n}" if n else base
        return base

    vals = df.apply(make, axis=1)
    missing = sorted({fmt_val(_) for _ in df[code_c]} - set(mapping))
    if missing:
        print(f"  Codes without mapping: {missing}")
    pos = ask_position(df, "first")
    place_column(df, out_c, vals, pos)
    preview(df, [out_c, code_c] + ([num_c] if num_c else []))
    return df, f"{out_c} from codes in {code_c}"


def op_filter_values(df):
    """Keep/remove rows where a column matches a list of values"""
    col = ask_col(df, "Column")
    print(f"  Unique values (first 30): {[fmt_val(_) for _ in df[col].unique()[:30]]}")
    vals = ask_list("Values (comma separated)")
    keep = ask_choice("Action", ["keep", "remove"], "keep") == "keep"
    msk = df[col].map(fmt_val).isin(vals)
    df = df[msk] if keep else df[~msk]
    print(f"{'Kept' if keep else 'Removed'} {msk.sum()} rows, {len(df)} remain")
    return df.reset_index(drop=True), f"{'keep' if keep else 'remove'} rows {col} in {vals}"


def op_filter_query(df):
    """Keep/remove rows matching a pandas query (e.g. 'Type == \"OC\" and N > 10')"""
    print("  Examples: 'Type == \"OC\"', 'Type in [1, 6, 16]', '`E(B-V)` > 2', 'Ref.isna()'")
    q = ask("Query")
    msk = eval_mask(df, q)
    print(f"{msk.sum()} rows match")
    keep = ask_choice("Action", ["keep", "remove"], "keep") == "keep"
    df = df[msk] if keep else df[~msk]
    print(f"{len(df)} rows remain")
    return df.reset_index(drop=True), f"{'keep' if keep else 'remove'} rows where {q}"


def op_filter_names(df):
    """Keep/remove rows whose names are in a given list (optionally + a condition)"""
    col = ask_col(df, "Names column")
    names = {norm(_) for _ in read_lines_or_file("Names to match")}
    first_only = ask_yn("Match only the first name in each cell?", False)
    cond = ask("Extra condition that must also hold (query, blank for none)", "")
    msk = []
    for val in df[col]:
        ns = split_names(val)
        ns = ns[:1] if first_only else ns
        msk.append(any(norm(_) in names for _ in ns))
    msk = pd.Series(msk, index=df.index)
    if cond:
        msk &= eval_mask(df, cond)
    print(f"{msk.sum()} rows match:")
    print_list(df.loc[msk, col])
    keep = ask_choice("Action", ["keep", "remove"], "remove") == "keep"
    df = df[msk] if keep else df[~msk]
    print(f"{len(df)} rows remain")
    return df.reset_index(drop=True), f"{'keep' if keep else 'remove'} {msk.sum()} rows by name list"


def op_filter_ucc(df):
    """Keep/remove rows whose names are (not) in the UCC"""
    col = ask_col(df, "Names column")
    ucc = pd.read_csv(UCC_names_file)
    ucc_fnames = {f for fn in ucc["fnames"] for f in str(fn).split(";")}
    msk = pd.Series(
        [any(norm(n) in ucc_fnames for n in split_names(v)) for v in df[col]],
        index=df.index,
    )
    print(f"{msk.sum()} rows in the UCC, {(~msk).sum()} not in the UCC")
    if ask_yn("List entries NOT in the UCC?"):
        print_list(df.loc[~msk, col])
    act = ask_choice("Action", ["keep in UCC", "keep not in UCC", "nothing"], "nothing")
    if act == "nothing":
        return df, None
    df = df[msk] if act == "keep in UCC" else df[~msk]
    print(f"{len(df)} rows remain")
    return df.reset_index(drop=True), f"{act} ({col})"


def op_set_nan_where(df):
    """Set columns to NaN where a condition holds (e.g. 'Ref.isna()')"""
    q = ask("Condition (query)")
    cols = ask_cols(df, "Columns to set to NaN")
    msk = eval_mask(df, q)
    df.loc[msk, cols] = np.nan
    print(f"{msk.sum()} rows modified")
    return df, f"set {cols} to NaN where {q}"


def op_coalesce(df):
    """Fill NaNs in a column from other column(s); or merge '_x/_y' column pairs"""
    if ask_yn("Auto-coalesce column pairs with suffixes (e.g. 'X_x'/'X_y')?"):
        s1 = ask("Suffix of preferred columns", "_x")
        s2 = ask("Suffix of secondary columns", "_y")
        pairs = [
            c[: -len(s1)]
            for c in df.columns
            if c.endswith(s1) and c[: -len(s1)] + s2 in df.columns
        ]
        print(f"  Pairs found: {pairs}")
        for base in pairs:
            a, b = df[base + s1], df[base + s2]
            a = a.mask(a.astype(str).isin(["nan", ""]))
            place_column(df, base, a.combine_first(b), f"after:{base + s1}")
            df = df.drop(columns=[base + s1, base + s2])
        return df, f"coalesce {pairs} ({s1} over {s2})"
    tgt = ask_col(df, "Target column (NaNs are filled)", allow_new=True)
    srcs = ask_cols(df, "Source column(s), in order of priority")
    if tgt not in df.columns:
        df[tgt] = np.nan
    n0 = df[tgt].isna().sum()
    for s in srcs:
        df[tgt] = df[tgt].combine_first(df[s])
    print(f"{n0 - df[tgt].isna().sum()} values filled")
    if ask_yn(f"Drop source columns {srcs}?"):
        df = df.drop(columns=[s for s in srcs if s != tgt])
    return df, f"fill NaN in {tgt} from {srcs}"


def op_update_rows(df):
    """Update values in specific rows from a key-matched table (CSV file or pasted)"""
    upd = read_csv_from_input("Table with a key column + columns to update")
    print(upd.head().to_string())
    key_u = ask_col(upd, "Key column in this table", default=str(upd.columns[0]))
    key_d = ask_col(df, "Key column in the DB", default=key_u if key_u in df.columns else None)
    match = ask_choice("Match keys", ["exact", "normalized (UCC style)", "any name in cell"], "1")

    def keyf(v):
        return norm(v) if match != "exact" else fmt_val(v)

    upd_map = {keyf(k): i for i, k in enumerate(upd[key_u])}
    cols = [c for c in upd.columns if c != key_u]
    for c in cols:
        if c not in df.columns:
            print(f"  Adding new column '{c}'")
            df[c] = np.nan
    found = set()
    for idx, val in df[key_d].items():
        cands = split_names(val) if match == "any name in cell" else [val]
        for cnd in cands:
            j = upd_map.get(keyf(cnd))
            if j is not None:
                for c in cols:
                    df.at[idx, c] = upd.at[j, c]
                found.add(j)
                break
    print(f"{len(found)} of {len(upd)} rows updated")
    miss = [upd[key_u].iloc[j] for j in range(len(upd)) if j not in found]
    if miss:
        print("Keys not found in the DB:")
        print_list(miss)
    return df, f"update {cols} for {len(found)} rows from table"


def _report_keys(a: pd.Series, b: pd.Series, la: str, lb: str) -> None:
    sa, sb = set(a.dropna()), set(b.dropna())
    for x, y, lx, ly in ((sa, sb, la, lb), (sb, sa, lb, la)):
        miss = sorted(map(str, x - y))
        print(f"{len(miss)} keys in {lx} not in {ly}")
        if miss and ask_yn("  List them?"):
            print_list(miss)


def op_merge(df):
    """Merge with another CSV on a key column (e.g. two tables from the same paper)"""
    df2, p2 = load_csv("Path to the CSV to merge")
    print(f"  Columns: {list(df2.columns)}")
    key1 = ask_col(df, "Key column in the current DB")
    key2 = ask_col(df2, "Key column in the other DB", default=key1 if key1 in df2.columns else None)
    drop = ask_cols(df2, "Columns to drop from the other DB before merging (blank for none)",
                    allow_empty=True, default="")
    df2 = df2.drop(columns=[c for c in drop if c != key2])
    if key2 != key1:
        df2 = df2.rename(columns={key2: key1})
    for d, lbl in ((df, "current"), (df2, "other")):
        n_dup = d[key1].duplicated().sum()
        if n_dup:
            print(f"  WARNING: {n_dup} duplicated keys in the {lbl} DB")
    _report_keys(df[key1], df2[key1], "current", "other")

    how = ask_choice("Merge type", ["outer", "left", "inner", "right"], "outer")
    shared = [c for c in df.columns if c in df2.columns and c != key1]
    strategy = "suffixes"
    if shared:
        print(f"  Shared columns: {shared}")
        strategy = ask_choice(
            "Shared columns",
            ["suffixes", "prefer current", "prefer other", "join values"],
            "suffixes",
        )
    s1, s2 = "_1", "_2"
    if strategy == "suffixes":
        s1 = ask("Suffix for current DB columns", "_1")
        s2 = ask("Suffix for other DB columns", "_2")
    join_cols = shared
    if strategy == "join values":
        join_cols = ask_cols(df, "Columns to join (others prefer the other DB)",
                             default=",".join(shared))
    merged = pd.merge(df, df2, on=key1, how=how, suffixes=(s1, s2))

    if strategy != "suffixes":
        for c in shared:
            a, b = merged[c + s1], merged[c + s2]
            if strategy == "prefer current":
                v = a.combine_first(b)
            elif strategy == "prefer other" or c not in join_cols:
                v = b.combine_first(a)
            else:
                jsep = ","

                v = a.combine(b, lambda x, y: join_values((x, y), jsep))
            place_column(merged, c, v, f"after:{c + s1}")
            merged = merged.drop(columns=[c + s1, c + s2])
    print(f"Merged DB: {len(merged)} rows, {len(merged.columns)} columns")
    return merged, f"merge with '{p2.name}' on {key1} ({how}, {strategy})"


def op_append_rows(df):
    """Append the rows of another CSV (missing columns are filled with NaN)"""
    df2, p2 = load_csv("Path to the CSV to append")
    only1 = [c for c in df.columns if c not in df2.columns]
    only2 = [c for c in df2.columns if c not in df.columns]
    if only1 or only2:
        print(f"  Columns only in current: {only1}\n  Columns only in other: {only2}")
    df = pd.concat([df, df2], ignore_index=True, sort=False)
    print(f"{len(df)} rows")
    return df, f"append rows from '{p2.name}'"


def op_group_duplicates(df):
    """Combine rows with the same key into a single row (first/median/join...)"""
    key = ask_col(df, "Key column")
    n_dup = df[key].duplicated(keep=False).sum()
    print(f"{df[key].duplicated().sum()} duplicated keys ({n_dup} rows involved)")
    if n_dup == 0:
        return df, None
    print("  Aggregations: first, last, median, mean, min, max, join, join_unique")
    default = ask("Default aggregation", "first")
    over = ask_list("Per-column overrides as 'col:agg' (comma separated, blank for none)", "")
    jsep = ask("Separator for 'join'", ";", strip=False)
    aggs = {c: default for c in df.columns if c != key}
    for o in over:
        c, a = [_.strip() for _ in o.split(":", 1)]
        aggs[c] = a

    def mk(a):
        if a == "join":
            return lambda s: join_values(s, jsep, unique=False)
        if a == "join_unique":
            return lambda s: join_values(s, jsep)
        if a in ("median", "mean"):
            return lambda s: getattr(pd.to_numeric(s, errors="coerce"), a)()
        return a

    out = df.groupby(key, as_index=False, sort=False).agg({c: mk(a) for c, a in aggs.items()})
    out = out[df.columns]
    print(f"{len(df)} -> {len(out)} rows")
    return out, f"group by {key} (default {default}, overrides {over})"


def op_compare_columns(df):
    """Print differences between pairs of columns (e.g. after a merge)"""
    pairs = ask_list("Pairs as 'colA:colB' (comma separated)")
    for p in pairs:
        a, b = [_.strip() for _ in p.split(":", 1)]
        x = pd.to_numeric(df[a], errors="coerce")
        y = pd.to_numeric(df[b], errors="coerce")
        d = (x - y).abs()
        print(f"  {a} - {b}: N={d.notna().sum()}, median={d.median():.5g}, max={d.max():.5g}")
    return df, None


# ----------------------------------------------------------------------------
# Session


OPERATIONS = [
    ("Inspect", [op_info, op_compare_columns]),
    ("Columns", [op_drop_columns, op_rename_columns, op_add_nan_columns,
                 op_reorder_columns, op_round, op_derived_column]),
    ("Coordinates", [op_gal_to_equ, op_sexa_to_deg, op_equ_to_gal, op_compare_coords]),
    ("Values", [op_split_value_error, op_split_paren_error, op_set_nan_where,
                op_coalesce, op_update_rows]),
    ("Names", [op_combine_columns, op_name_template, op_expand_prefix,
               op_split_letters_digits, op_remove_substrings, op_regex_replace,
               op_first_name_column, op_map_codes]),
    ("Rows", [op_filter_values, op_filter_query, op_filter_names, op_filter_ucc,
              op_group_duplicates, op_sort]),
    ("Combine DBs", [op_merge, op_append_rows]),
]


def save(df: pd.DataFrame, in_path: Path) -> Path | None:
    stem = in_path.stem
    default = in_path.with_name(stem + ("" if stem.endswith("_2") else "_2") + ".csv")
    out = Path(ask("Output path", str(default))).expanduser()
    if out.exists() and not ask_yn(f"'{out}' exists, overwrite?"):
        return None
    df.to_csv(out, na_rep="nan", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Saved {len(df)} rows to '{out}'")
    return out


if __name__ == "__main__":
    main()
