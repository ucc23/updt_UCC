"""
Rewrite the Gaia DR3 parquet files so that each file is split into row groups of
consecutive HEALPix level 8 (nested) pixels.

The original files hold a single row group, so reading any part of a file means
reading and decompressing all of it. Rows are already sorted by 'source_id' (and
thus by HEALPix pixel), so no re-sorting is needed: only the row group boundaries
are added. A reader can then load just the row groups of the pixels it needs.
Each row group holds whole pixels, merging consecutive pixels until it has at least
'min_rows' rows (sparse regions have ~100 stars per pixel, and very small row groups
make full reads much slower). In dense regions a row group holds a single pixel.

Columns not in 'keep_float64' are stored as float32 to reduce the size of the
files. The coordinates 'ra', 'dec', 'l', 'b' are kept as float64: float32 resolves
only ~3e-5 deg at ~300 deg, which moves stars across the edges of the frames
(these are cut in RA/Dec and l/b). 'source_id' stays int64.

The range of HEALPix pixels in each row group, as [first, last] (inclusive) and in
the order of the row groups, is stored in the file's custom metadata under the
'hp8_ranges' key.

Files are written to a temporary name and renamed when complete, so the script
can be stopped and re-run: files already present in the output folder are skipped.

Usage:
    python helper_scripts/gaia_files_to_rowgroups.py [N_max_files]
"""

import json
import os
import sys
import time

import fastparquet
import numpy as np
import pandas as pd

path_in = "/media/gabriel/backup/gabriel/GaiaDR3/datafiles_G20/"
path_out = "/media/gabriel/backup/gabriel/GaiaDR3/datafiles_G20_rg/"

# Minimum number of rows per row group
min_rows = 5000

# Columns that keep their original type
keep_float64 = ("ra", "dec", "l", "b")

# Number of bits to shift 'source_id' to obtain the HEALPix level 12 index, and
# factor to go from level 12 to level 8 (4**4)
sid_shift, lvl12_to_lvl8 = 35, 256


def main():
    N_max = int(sys.argv[1]) if len(sys.argv) > 1 else None

    os.makedirs(path_out, exist_ok=True)
    files = sorted(f for f in os.listdir(path_in) if f.endswith(".parquet"))
    done = set(os.listdir(path_out))
    pending = [f for f in files if f not in done]
    print(
        f"{len(files)} files, {len(files) - len(pending)} done, {len(pending)} pending"
    )
    if N_max is not None:
        pending = pending[:N_max]
        print(f"Processing {len(pending)} files")

    size_in, size_out = 0, 0
    t0 = time.time()
    for i, fname in enumerate(pending):
        s_in, s_out = convert_file(fname)
        size_in += s_in
        size_out += s_out

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(pending) - i - 1)
        print(
            f"{i + 1}/{len(pending)} {fname}  {s_in / 1e6:.1f} -> {s_out / 1e6:.1f} MB"
            + f"  (ETA {eta / 60:.0f} min)",
            flush=True,
        )

    if pending:
        print(
            f"\nTotal: {size_in / 1e9:.2f} -> {size_out / 1e9:.2f} GB "
            + f"in {(time.time() - t0) / 60:.1f} min"
        )


def convert_file(fname: str) -> tuple[int, int]:
    """
    Convert a single file, returning the input and output sizes in bytes.
    """
    file_in = path_in + fname
    file_out = path_out + fname
    file_tmp = file_out + ".tmp"

    data = pd.read_parquet(file_in)

    sid = data["source_id"].values
    if not np.all(np.diff(sid) > 0):
        raise ValueError(f"{fname}: rows are not sorted by 'source_id'")

    # Check that the HEALPix pixels match the range in the file name
    hp8 = (sid >> sid_shift) // lvl12_to_lvl8
    hp_min, hp_max = map(int, fname.split("_")[1].split("-"))
    if hp8[0] < hp_min or hp8[-1] > hp_max:
        raise ValueError(f"{fname}: HEALPix pixels outside of the file name's range")

    row_group_offsets, hp8_ranges = get_row_groups(hp8)

    for col in data.columns:
        if col not in keep_float64 and data[col].dtype == np.float64:
            data[col] = data[col].astype(np.float32)

    fastparquet.write(
        file_tmp,
        data,
        row_group_offsets=row_group_offsets,
        compression="SNAPPY",
        write_index=False,
        custom_metadata={"hp8_ranges": json.dumps(hp8_ranges)},
    )
    check_file(file_tmp, data, hp8_ranges)
    os.rename(file_tmp, file_out)

    return os.path.getsize(file_in), os.path.getsize(file_out)


def get_row_groups(hp8: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """
    Group consecutive pixels into row groups of at least 'min_rows' rows.

    Returns the first row of each row group and the [first, last] pixels in it.
    """
    pixels, first_row, counts = np.unique(hp8, return_index=True, return_counts=True)

    row_group_offsets, hp8_ranges = [], []
    N_rows = 0
    for pix, row, N in zip(pixels.tolist(), first_row.tolist(), counts.tolist()):
        if not row_group_offsets or N_rows >= min_rows:
            # Start a new row group
            row_group_offsets.append(row)
            hp8_ranges.append([pix, pix])
            N_rows = 0
        hp8_ranges[-1][1] = pix
        N_rows += N

    # Merge a small last row group into the previous one
    if len(row_group_offsets) > 1 and N_rows < min_rows:
        row_group_offsets.pop()
        last = hp8_ranges.pop()
        hp8_ranges[-1][1] = last[1]

    return row_group_offsets, hp8_ranges


def check_file(file_tmp: str, data: pd.DataFrame, hp8_ranges: list) -> None:
    """
    Check that the written file has the expected row groups, metadata and data.
    """
    pf = fastparquet.ParquetFile(file_tmp)
    if len(pf.row_groups) != len(hp8_ranges):
        raise ValueError(f"{file_tmp}: expected {len(hp8_ranges)} row groups")
    if json.loads(pf.key_value_metadata["hp8_ranges"]) != hp8_ranges:
        raise ValueError(f"{file_tmp}: wrong 'hp8_ranges' metadata")
    if not pf.to_pandas().equals(data):
        raise ValueError(f"{file_tmp}: written data does not match")


if __name__ == "__main__":
    main()
