import json
from collections import OrderedDict

import fastparquet
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

# Gaia EDR3 zero points. Sigmas are already squared here.
Zp_G, sigma_ZG_2 = 25.6873668671, 0.00000759
Zp_BP, sigma_ZBP_2 = 25.3385422158, 0.000007785
Zp_RP, sigma_ZRP_2 = 24.7478955012, 0.00001428

# The Gaia DR3 files are split by ranges of HEALPix level 8 (nested) indexes, given
# in the file names as 'GaiaSource_XXXXXX-YYYYYY_p.parquet' (inclusive range)
hp_lvl8 = HEALPix(nside=2**8, order="nested")
# Approximate size of a level 8 HEALPix pixel (in degrees), used as padding
hp_lvl8_pad = 0.25

# Cache of the most recently loaded Gaia data, shared across queries. Each entry is
# a row group of a file, or a full file if it was not split into row groups by
# HEALPix pixels (see 'helper_scripts/gaia_files_to_rowgroups.py'). The cache is
# limited by its total number of rows (~1.5 GB in memory)
cache_max_rows = 15_000_000
_data_cache: OrderedDict[tuple[str, int | None], pd.DataFrame] = OrderedDict()
_cache_rows = 0
# Opened parquet files (only their metadata is loaded)
_pq_files: dict[str, fastparquet.ParquetFile] = {}


def query_run(
    logging,
    frames_path: str,
    fdata: pd.DataFrame,
    box_s_eq: float,
    plx_min: float,
    max_mag: float,
    c_ra: float,
    c_dec: float,
    frame_limit_dict: dict,
) -> pd.DataFrame:
    """
    Queries Gaia data frames based on specified parameters and returns a combined
    DataFrame.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    frames_path : str
        Path to the directory containing Gaia data frames.
    fdata : pd.DataFrame
        DataFrame containing information about Gaia data frames.
    box_s_eq : float
        Size of the box to query (in degrees).
    plx_min : float
        Minimum parallax value for data retrieval.
    max_mag : float
        Maximum magnitude for data retrieval.
    c_ra : float
        Central right ascension for the query.
    c_dec : float
        Central declination for the query.
    frame_limit_dict : dict
        Dictionary specifying manual frame limits in the format 'limit_type: value'.

    Returns
    -------
    gaia_frame: pd.DataFrame
        DataFrame containing the combined Gaia data.
    """
    txt_flim = ""
    if frame_limit_dict:
        txt_flim = "; "
        for fl, v in frame_limit_dict.items():
            txt_flim += f"{fl}: {v}, "
        txt_flim = txt_flim[:-2]
    logging.info(
        f"  cent=({c_ra:.3f}, {c_dec:.3f}); Box size: {box_s_eq:.2f}, "
        + f"Plx min: {plx_min:.2f}{txt_flim}"
    )

    # Check for presence of a manual plx_min value
    if "plxl" in frame_limit_dict:
        plx_min = frame_limit_dict["plxl"]
        logging.info(f"  Using manual plx_min={plx_min:.2f}")

    # Check if box size should be modified
    max_lon_lat_length = np.nan
    if "l" in frame_limit_dict and "r" in frame_limit_dict:
        lon_range = abs(frame_limit_dict["r"] - frame_limit_dict["l"]) % 360
        lon_range = min(lon_range, 360 - lon_range)
        max_lon_lat_length = np.fmax(max_lon_lat_length, lon_range)
    if "b" in frame_limit_dict and "t" in frame_limit_dict:
        lat_range = abs(frame_limit_dict["t"] - frame_limit_dict["b"])
        max_lon_lat_length = np.fmax(max_lon_lat_length, lat_range)
    if not np.isnan(max_lon_lat_length) and (max_lon_lat_length > box_s_eq):
        # WARNING: this compares length in lon/lat with box size in equatorial
        # coordinates, which is not strictly correct
        box_s_eq = max_lon_lat_length
        logging.info(f"  Box size modified to {box_s_eq:.2f} due to frame limits")

    # Check if the cluster region wraps around the RA=0/360 boundary
    c_ra_wrapped = [c_ra]
    if c_ra - box_s_eq < 0:
        logging.info("  Split frame, c_ra + 360")
        c_ra_wrapped.append(c_ra + 360)
    if c_ra + box_s_eq > 360:
        logging.info("  Split frame, c_ra - 360")
        c_ra_wrapped.append(c_ra - 360)

    dicts = []
    for c_ra_w in c_ra_wrapped:
        xmin_cl, xmax_cl, ymin_cl, ymax_cl = findFrames(c_ra_w, c_dec, box_s_eq)
        data_in_files = findFiles(c_ra, c_dec, box_s_eq, fdata)

        if len(data_in_files) == 0:
            continue

        all_frames = query(
            logging,
            Zp_G,
            c_ra,
            c_dec,
            box_s_eq,
            frames_path,
            max_mag,
            data_in_files,
            xmin_cl,
            xmax_cl,
            ymin_cl,
            ymax_cl,
            plx_min,
        )

        dicts.append(all_frames)

    if len(dicts) > 1:
        # Combine
        all_frames = (
            pd.concat([dicts[0], dicts[1]]).drop_duplicates().reset_index(drop=True)
        )
    else:
        all_frames = dicts[0]

    # Apply manual frame limits if any
    if frame_limit_dict:
        for fk, val in frame_limit_dict.items():
            if fk == "b":
                msk = all_frames["GLAT"] > val
                all_frames = all_frames[msk]
            elif fk == "t":
                msk = all_frames["GLAT"] < val
                all_frames = all_frames[msk]
            elif fk == "l":
                msk = all_frames["GLON"] > val
                all_frames = all_frames[msk]
            elif fk == "r":
                msk = all_frames["GLON"] < val
                all_frames = all_frames[msk]

            elif fk == "plxl":
                msk = all_frames["Plx"] > val
                all_frames = all_frames[msk]
            elif fk == "plxr":
                msk = all_frames["Plx"] < val
                all_frames = all_frames[msk]

            elif fk == "pmb":
                msk = all_frames["pmDE"] > val
                all_frames = all_frames[msk]
            elif fk == "pmt":
                msk = all_frames["pmDE"] < val
                all_frames = all_frames[msk]
            elif fk == "pml":
                msk = all_frames["pmRA"] > val
                all_frames = all_frames[msk]
            elif fk == "pmr":
                msk = all_frames["pmRA"] < val
                all_frames = all_frames[msk]
            else:
                raise ValueError("Unknown frame limit: " + str(fk))

        all_frames = pd.DataFrame(all_frames)

    all_frames = uncertMags(
        Zp_G, Zp_BP, Zp_RP, sigma_ZG_2, sigma_ZBP_2, sigma_ZRP_2, all_frames
    )
    gaia_frame = all_frames.drop(columns=["FG", "e_FG", "FBP", "e_FBP", "FRP", "e_FRP"])

    # Round all values ('Source' can be included here since it is all ints)
    for col in gaia_frame.columns:
        gaia_frame[col] = gaia_frame[col].round(5)

    return pd.DataFrame(gaia_frame)


def findFrames(
    c_ra: float, c_dec: float, box_s_eq: float
) -> tuple[float, float, float, float]:
    """
    Obtains the equatorial limits of the region used to pre-filter the Gaia files.

    Parameters
    ----------
    c_ra : float
        Central right ascension of the region.
    c_dec : float
        Central declination of the region.
    box_s_eq : float
        Size of the box to query (in degrees).

    Returns
    -------
    tuple
        A tuple containing:
        - xmin_cl: Minimum RA of the cluster region.
        - xmax_cl: Maximum RA of the cluster region.
        - ymin_cl: Minimum Dec of the cluster region.
        - ymax_cl: Maximum Dec of the cluster region.
    """
    # frame == 'galactic':
    box_s_eq = np.sqrt(2) * box_s_eq
    # Correct size in RA
    box_s_x = box_s_eq / np.cos(np.deg2rad(c_dec))

    xl, yl = box_s_x * 0.5, box_s_eq * 0.5

    # Limits of the cluster's region in Equatorial
    xmin_cl, xmax_cl = c_ra - xl, c_ra + xl
    ymin_cl, ymax_cl = c_dec - yl, c_dec + yl

    return xmin_cl, xmax_cl, ymin_cl, ymax_cl


def findFiles(
    c_ra: float, c_dec: float, box_s_eq: float, fdata: pd.DataFrame
) -> dict[str, np.ndarray]:
    """
    Identifies the Gaia data files that contain stars within the specified region,
    using the HEALPix level 8 ranges stored in the file names.

    The final frame is a box of side 'box_s_eq' in galactic coordinates, so every
    star in it lies within a radius of box_s_eq/sqrt(2) of the center. The cone is
    padded by the size of a HEALPix pixel because the cone search only returns
    pixels whose centers fall inside the radius.

    Parameters
    ----------
    c_ra : float
        Central right ascension of the region.
    c_dec : float
        Central declination of the region.
    box_s_eq : float
        Size of the box to query (in degrees).
    fdata : pd.DataFrame
        DataFrame containing information about Gaia data frames.

    Returns
    -------
    dict[str, np.ndarray]
        Filenames of the files that overlap the region, and the HEALPix pixels of
        the region contained in each one.
    """
    fnames = fdata["filename"].values
    hp_ranges = np.array(
        [f.split("_")[1].split("-") for f in fnames], dtype=int
    )  # (N, 2): first and last HEALPix index in each file
    i_sort = np.argsort(hp_ranges[:, 0])
    hp_min, hp_max = hp_ranges[i_sort, 0], hp_ranges[i_sort, 1]

    radius = box_s_eq / np.sqrt(2) + hp_lvl8_pad
    pix = hp_lvl8.cone_search_lonlat(
        c_ra * u.deg,  # pyright: ignore
        c_dec * u.deg,  # pyright: ignore
        radius=min(radius, 180) * u.deg,  # pyright: ignore
    )

    # Index of the file whose range could contain each pixel
    idx = np.searchsorted(hp_min, pix, side="right") - 1
    msk = (idx >= 0) & (pix <= hp_max[idx.clip(0)])
    pix, idx = pix[msk], idx[msk]

    fnames_sort = fnames[i_sort]
    data_in_files = {fnames_sort[i]: pix[idx == i] for i in np.unique(idx)}

    return data_in_files


def read_file(file_path: str, pixels: np.ndarray) -> pd.DataFrame:
    """
    Reads the data of the given HEALPix pixels from a Gaia data file.

    If the file was split into row groups by HEALPix pixels, only the row groups
    that contain the pixels are read. Otherwise the full file is read. The returned
    data can contain stars from other pixels.
    """
    if file_path not in _pq_files:
        _pq_files[file_path] = fastparquet.ParquetFile(file_path)
    pf = _pq_files[file_path]

    hp8_ranges = pf.key_value_metadata.get("hp8_ranges")
    if hp8_ranges is None:
        return read_row_group(file_path, pf, None)

    # [first, last] pixels in each row group
    ranges = np.array(json.loads(hp8_ranges))
    i_rg = np.searchsorted(ranges[:, 0], pixels, side="right") - 1
    i_rg = np.unique(i_rg[(i_rg >= 0) & (pixels <= ranges[i_rg.clip(0), 1])])
    if len(i_rg) == 0:
        return pd.DataFrame(columns=pf.columns)

    return pd.concat(
        [read_row_group(file_path, pf, int(i)) for i in i_rg], ignore_index=True
    )


def read_row_group(
    file_path: str, pf: fastparquet.ParquetFile, i_rg: int | None
) -> pd.DataFrame:
    """
    Reads a row group of a file (or the full file if 'i_rg' is None), keeping the
    most recently used ones in memory.
    """
    global _cache_rows

    key = (file_path, i_rg)
    if key in _data_cache:
        _data_cache.move_to_end(key)
        return _data_cache[key]

    data = pf.to_pandas() if i_rg is None else pf[i_rg].to_pandas()

    _data_cache[key] = data
    _cache_rows += len(data)
    while _cache_rows > cache_max_rows and len(_data_cache) > 1:
        _, old = _data_cache.popitem(last=False)
        _cache_rows -= len(old)

    return data


def radec2lonlat(
    ra: float | list | np.ndarray, dec: float | list | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Converts equatorial coordinates (RA, Dec) to galactic coordinates (lon, lat).

    Parameters
    ----------
    ra : float or list
        Right ascension in degrees.
    dec : float or list
        Declination in degrees.

    Returns
    -------
    tuple
        A tuple containing the galactic longitude and latitude in degrees.
    """
    gc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)  # pyright: ignore
    lb = gc.transform_to("galactic")
    return lb.l.value, lb.b.value  # pyright: ignore


def query(
    logging,
    Zp_G: float,
    c_ra: float,
    c_dec: float,
    box_s_eq: float,
    frames_path: str,
    max_mag: float,
    data_in_files: dict[str, np.ndarray],
    xmin_cl: float,
    xmax_cl: float,
    ymin_cl: float,
    ymax_cl: float,
    plx_min: float,
) -> pd.DataFrame:
    """
    Queries individual Gaia data frames and combines the results.

    Parameters
    ----------
    logging : logging.Logger
        Logger object for outputting information.
    Zp_G : float
        Zero point for the G band.
    c_ra : float
        Central right ascension of the region.
    c_dec : float
        Central declination of the region.
    box_s_eq : float
        Size of the box to query (in degrees).
    frames_path : str
        Path to the directory containing Gaia data frames.
    max_mag : float
        Maximum magnitude for data retrieval.
    data_in_files : dict
        Filenames of overlapping frames and the HEALPix pixels needed from each.
    xmin_cl : float
        Minimum RA of the cluster region.
    xmax_cl : float
        Maximum RA of the cluster region.
    ymin_cl : float
        Minimum Dec of the cluster region.
    ymax_cl : float
        Maximum Dec of the cluster region.
    plx_min : float
        Minimum parallax value for data retrieval.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the combined Gaia data from the queried frames.
    """
    # Mag (flux) filter
    min_G_flux = 10 ** ((max_mag - Zp_G) / (-2.5))

    all_frames = []
    for file, pixels in data_in_files.items():
        data = read_file(frames_path + file, pixels)

        mx = (data["ra"] >= xmin_cl) & (data["ra"] <= xmax_cl)
        my = (data["dec"] >= ymin_cl) & (data["dec"] <= ymax_cl)
        m_plx = data["parallax"] > plx_min
        m_gmag = data["phot_g_mean_flux"] > min_G_flux
        msk = mx & my & m_plx & m_gmag

        if msk.sum() == 0:
            continue
        # logging.info(f"N={msk.sum()} stars in {file}")

        all_frames.append(data[msk])
    all_frames = pd.concat(all_frames)
    # Files converted with 'gaia_files_to_rowgroups.py' store most columns as float32
    all_frames = all_frames.astype(
        {c: np.float64 for c, t in all_frames.dtypes.items() if t == np.float32}
    )

    # c_ra, c_dec = c_ra, c_dec
    box_s_h = box_s_eq * 0.5
    gal_cent = radec2lonlat(c_ra, c_dec)

    if all_frames["l"].max() - all_frames["l"].min() > 180:
        logging.info("  Fix frame that wraps around 360 in longitude")

        lon = np.array(all_frames["l"])
        if gal_cent[0] > 180:
            lon[lon < 180] += 360
        else:
            lon[lon > 180] -= 360
        all_frames["l"] = lon

    # Filter the stars in the cluster box (in galactic coordinates)
    xmin_cl, xmax_cl = gal_cent[0] - box_s_h, gal_cent[0] + box_s_h
    ymin_cl, ymax_cl = gal_cent[1] - box_s_h, gal_cent[1] + box_s_h
    mx = (all_frames["l"] >= xmin_cl) & (all_frames["l"] <= xmax_cl)
    my = (all_frames["b"] >= ymin_cl) & (all_frames["b"] <= ymax_cl)
    msk = mx & my
    all_frames = pd.DataFrame(all_frames[msk])

    all_frames = all_frames.rename(
        columns={
            "source_id": "Source",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "parallax": "Plx",
            "parallax_error": "e_Plx",
            "pmra": "pmRA",
            "pmra_error": "e_pmRA",
            "b": "GLAT",
            "pmdec": "pmDE",
            "pmdec_error": "e_pmDE",
            "l": "GLON",
            "phot_g_mean_flux": "FG",
            "phot_g_mean_flux_error": "e_FG",
            "phot_bp_mean_flux": "FBP",
            "phot_bp_mean_flux_error": "e_FBP",
            "phot_rp_mean_flux": "FRP",
            "phot_rp_mean_flux_error": "e_FRP",
            "radial_velocity": "RV",
            "radial_velocity_error": "e_RV",
        }
    )

    logging.info(f"  N_final={len(all_frames)}")
    return all_frames


def uncertMags(
    Zp_G: float,
    Zp_BP: float,
    Zp_RP: float,
    sigma_ZG_2: float,
    sigma_ZBP_2: float,
    sigma_ZRP_2: float,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates magnitudes and uncertainties in the G, BP, and RP bands.

    Gaia DR3 zero points: https://www.cosmos.esa.int/web/gaia/dr3-passbands
    "The GBP (blue curve), G (green curve) and GRP (red curve) passbands are
    applicable to both Gaia Early Data Release 3 as well as to the full Gaia
    Data Release 3"

    Parameters
    ----------
    Zp_G : float
        Zero point for the G band.
    Zp_BP : float
        Zero point for the BP band.
    Zp_RP : float
        Zero point for the RP band.
    sigma_ZG_2 : float
        Variance of the zero point for the G band.
    sigma_ZBP_2 : float
        Variance of the zero point for the BP band.
    sigma_ZRP_2 : float
        Variance of the zero point for the RP band.
    data : pd.DataFrame
        DataFrame containing Gaia data with flux measurements.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns for Gmag, BP-RP, e_Gmag, and e_BP-RP.
    """
    I_G, e_IG = np.array(data["FG"]), np.array(data["e_FG"])
    I_BP, e_IBP = np.array(data["FBP"]), np.array(data["e_FBP"])
    I_RP, e_IRP = np.array(data["FRP"]), np.array(data["e_FRP"])

    data["Gmag"] = Zp_G + -2.5 * np.log10(I_G)
    BPmag = Zp_BP + -2.5 * np.log10(I_BP)
    RPmag = Zp_RP + -2.5 * np.log10(I_RP)
    data["BP-RP"] = BPmag - RPmag

    e_G = np.sqrt(sigma_ZG_2 + 1.179 * (e_IG / I_G) ** 2)
    data["e_Gmag"] = e_G
    e_BP = np.sqrt(sigma_ZBP_2 + 1.179 * (e_IBP / I_BP) ** 2)
    e_RP = np.sqrt(sigma_ZRP_2 + 1.179 * (e_IRP / I_RP) ** 2)
    data["e_BP-RP"] = np.sqrt(e_BP**2 + e_RP**2)

    return data
