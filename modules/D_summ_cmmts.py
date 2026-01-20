import datetime
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

from .utils import get_fnames, logger
from .variables import (
    UCC_cmmts_file,
    UCC_cmmts_folder,
    data_folder,
    fpars_order,
    merged_dbs_file,
    name_DBs_json,
    temp_folder,
    ucc_cat_file,
)

HTML_WARN = '⚠️ <span style="color: #99180f; font-weight: bold;">Warning: </span>'
HTML_UTI = """<a href="/faq#what-is-the-uti-parameter" title="UTI parameter" target="_blank"><b>UTI</b></a>"""
HTML_C3 = """<a href="/faq#what-is-the-c3-parameter" title="C3 classification" target="_blank">C3 quality</a>"""
HTML_BAD_OC = """<a href="/faq#how-are-objects-flagged-as-likely-not-real" title="Not real open cluster" target="_blank"><u>not a real open cluster</u></a>"""
HTML_TABLE = (
    'See table with <a href="#tab_obj_shared" '
    "onclick=\"activateTabById(event, 'tab_obj_shared', 'obj_shared')\">"
    "shared members information</a>."
)
tsp = "    "


def main():
    """ """
    logging = logger()

    df_UCC, fnames_B, B_lookup, DBs_JSON, cmmts_JSONS = load_data()

    # fnames_check(logging, fnames_B, UCC_summ_cmmts)

    # from pyinstrument import Profiler
    # profiler = Profiler()
    # profiler.start()

    logging.info("\nGenerate all summaries")
    fpars_medians, summaries, descriptors, fpars_badges = get_summaries(
        df_UCC, DBs_JSON
    )

    logging.info("\nAdd summaries to file")
    UCC_summ_cmmts = update_summary(fpars_medians, summaries, descriptors, fpars_badges)

    logging.info("\nAdd all comments")
    fnames_lst = list(UCC_summ_cmmts.keys())
    for art_ID, cmmt_json_dict in cmmts_JSONS.items():
        # if fname_json is not None:
        #
        art_name, art_year, art_url, art_cmmts, fnames_found = get_comments(
            logging, B_lookup, art_ID, cmmt_json_dict
        )
        #
        UCC_summ_cmmts = update_comments(
            fnames_lst,
            UCC_summ_cmmts,
            fnames_B,
            art_ID,
            art_name,
            art_year,
            art_url,
            art_cmmts,
            fnames_found,
        )

    # profiler.stop()
    # profiler.open_in_browser()

    # Check the 'fnames' columns in df_UCC_B and df_UCC_C_final dataframes are equal
    if not df_UCC["fname"].to_list() == list(UCC_summ_cmmts.keys()):
        raise ValueError("The 'fname' columns in B/C and final JSON differ")

    temp_UCC_cmmts_file = temp_folder + data_folder + UCC_cmmts_file
    os.makedirs(os.path.dirname(temp_folder + data_folder), exist_ok=True)
    update_json_file(UCC_summ_cmmts, temp_UCC_cmmts_file)
    logging.info(f"Updated {temp_UCC_cmmts_file} file.")

    if input("\nMove file to final location? (y/n): ").lower() == "y":
        os.replace(
            temp_UCC_cmmts_file,
            f"{data_folder}{UCC_cmmts_file}",
        )
        logging.info(f"Moved file to {data_folder}{UCC_cmmts_file}.")


def load_data() -> tuple[pd.DataFrame, list, dict, dict, dict]:
    """ """
    df_B = pd.read_csv(f"{data_folder}{merged_dbs_file}")
    # Extract fnames from B file
    fnames_B = df_B["fnames"].tolist()
    B_lookup = {}
    for i, s in enumerate(fnames_B):
        for token in s.split(";"):
            B_lookup[token] = i

    cols_from_B_to_C = ["Names", "DB", "DB_i", "fnames", "fund_pars"]
    df_C = pd.read_csv(f"{data_folder}{ucc_cat_file}")
    df_UCC = df_C.copy()
    df_UCC[cols_from_B_to_C] = df_B[cols_from_B_to_C]

    # with open(f"{data_folder}/{UCC_cmmts_file}", "r") as f:
    #     UCC_summ_cmmts = json.load(f)

    # Load clusters data in JSON file
    with open(name_DBs_json) as f:
        DBs_JSON = json.load(f)

    cmmts_JSONS = {}
    json_cmmts_path = f"{data_folder}{UCC_cmmts_folder}"
    for fpath_json in os.listdir(json_cmmts_path):
        # Check for duplicate clusters keys in JSON and raise error if any.
        # 'json.load()' silently drops duplicates, hence the check here
        with open(f"{json_cmmts_path}{fpath_json}", "r") as f:
            raw_json = json.load(f, object_pairs_hook=list)

        clusters = dict(raw_json).get("clusters")
        if isinstance(clusters, list):
            seen = set()
            dups = {k for k, _ in clusters if k in seen or seen.add(k)}
            if dups:
                raise ValueError(
                    f'Duplicate keys in "clusters": {", ".join(sorted(dups))}'
                )

        # with open(f"{json_cmmts_path}{fpath_json}", "r") as f:
        #     new_JSON = json.load(f)

        cmmts_JSONS[fpath_json.replace(".json", "")] = dict(raw_json)

    return df_UCC, fnames_B, B_lookup, DBs_JSON, cmmts_JSONS


def fnames_check(logging, fnames_B, UCC_summ_cmmts):
    """Check if there are entries in UCC_summ_cmmts that are not in df_B"""
    fnames_in_cmmts = set(UCC_summ_cmmts.keys())
    fnames_in_B = set([s.split(";")[0] for s in fnames_B])
    fnames_not_in_B = fnames_in_cmmts - fnames_in_B
    if fnames_not_in_B:
        logging.info(
            f"{len(fnames_not_in_B)} entries in {UCC_cmmts_file} not found in {merged_dbs_file}:"
        )
        for fname in fnames_not_in_B:
            logging.info(f"- {fname}")
        if input(f"Remove entries not found in {merged_dbs_file}") == "y":
            for fname in fnames_not_in_B:
                del UCC_summ_cmmts[fname]
            logging.info("Removed entries not in merged DBs file.")
        else:
            sys.exit(f"Please update {UCC_cmmts_file} accordingly and rerun.")


def get_comments(
    logging, B_lookup, art_ID, cmmt_json_dict
) -> tuple[str, str, str, list, dict]:
    """Read JSON file with comments from article"""

    art_name = cmmt_json_dict["art_name"]
    art_year = cmmt_json_dict["art_year"]
    art_url = cmmt_json_dict["art_url"]
    # art_clusters = cmmt_json_dict["clusters"].keys()
    # art_cmmts = list(cmmt_json_dict["clusters"].values())
    art_clusters, art_cmmts = zip(*cmmt_json_dict["clusters"])

    # Convert all names to fnames
    jsonf_fnames = get_fnames(art_clusters)

    # Check which objects in the JSON file are in the UCC
    not_found = []
    fnames_found = {}
    for i, fnames in enumerate(jsonf_fnames):
        not_found_in = []
        for fname in fnames:
            idx = B_lookup.get(fname)
            if idx is None:
                not_found_in.append(fname)
            else:
                fnames_found[i] = idx
        if not_found_in:
            not_found.append(not_found_in)
    if not_found:
        logging.info(f"\n{art_ID}: {len(not_found)} objects not found in UCC")
        for cl_not_found in not_found:
            logging.info(f"  {','.join(cl_not_found)}")

    return art_name, art_year, art_url, list(art_cmmts), fnames_found


def get_summaries(df_UCC, DBs_JSON):
    """
    Generate a textual summary and UTI descriptors for an astronomical object.
    """
    current_year = datetime.datetime.now().year

    fpars_medians, summaries, descriptors, fpars_badges = {}, {}, {}, {}
    # Iterate trough each entry in the UCC database
    cols = df_UCC.columns
    for UCC_cl in df_UCC.itertuples(index=False, name=None):
        UCC_cl = dict(zip(cols, UCC_cl))
        fname0 = UCC_cl["fname"]

        (
            members,
            density,
            quality,
            literature,
            duplicate,
            dup_warn,
            plx_dist,
            z_position,
        ) = get_C_txt(
            UCC_cl["C_N"],
            UCC_cl["C_dens"],
            UCC_cl["C_C3"],
            UCC_cl["C_lit"],
            UCC_cl["C_dup"],
            UCC_cl["Plx_m"],
            UCC_cl["Z_GC"],
        )

        # Generate fundamental parameters table
        recent_year_i, fpars_dict = fpars_in_lit(
            current_year,
            DBs_JSON,
            UCC_cl["DB"],
            UCC_cl["fund_pars"],
        )

        medians, large_spread = fpars_medians_spread(recent_year_i, fpars_dict)
        fpars_medians[fname0] = medians

        fpars_summ, fpars_note, fpars_badges_lst = fpars_summary(
            medians, large_spread, UCC_cl["Plx_m"], plx_dist, z_position
        )

        cl_name0 = UCC_cl["Names"].split(";")[0]
        summaries[fname0] = get_summary(
            cl_name0,
            UCC_cl["DB"],
            UCC_cl["shared_members_p"],
            UCC_cl["C_dup"],
            UCC_cl["C_dup_same_db"],
            UCC_cl["bad_oc"],
            members,
            density,
            quality,
            fpars_summ,
            current_year,
            literature,
            duplicate,
            dup_warn,
            fpars_note,
        )

        # Descriptors, for back of summary card
        UTI_C_N_desc = members.capitalize()
        UTI_C_dens_desc = density.capitalize()
        UTI_C_C3_desc = quality.capitalize() + " quality"
        UTI_C_lit_desc = literature.replace("</u>", "").replace("<u>", "").capitalize()
        UTI_C_dup_desc = (
            duplicate.replace("</u>", "")
            .replace("<u>", "")
            .replace("a ", "")
            .capitalize()
        )
        descriptors[fname0] = [
            UTI_C_N_desc,
            UTI_C_dens_desc,
            UTI_C_C3_desc,
            UTI_C_lit_desc,
            UTI_C_dup_desc,
        ]

        fpars_badges[fname0] = fpars_badges_lst

    return fpars_medians, summaries, descriptors, fpars_badges


def level(value, thresholds, labels):
    """Map a numeric value to a qualitative label."""
    for t, lbl in zip(thresholds, labels):
        if value >= t:
            return lbl
    return labels[-1]


def get_C_txt(C_N, C_dens, C_C3, C_lit, C_dup, plx, Z_GC):
    """ """
    members = level(
        C_N,
        [0.9, 0.75, 0.5, 0.25],
        ["very rich", "rich", "moderately populated", "poorly populated", "sparse"],
    )
    density = level(
        C_dens,
        [0.9, 0.75, 0.5, 0.25],
        ["very dense", "dense", "moderately dense", "loose", "very loose"],
    )
    quality = level(
        C_C3,
        [0.99, 0.7, 0.5, 0.2],
        ["very high", "high", "intermediate", "low", "very low"],
    )

    plx_dist = level(
        plx,
        [2, 1, 0.5, 0.2, 0.1],
        ["very close", "close", "relatively close", "moderate", "large", "very large"],
    )
    z_position = level(
        Z_GC,
        [0.5, 0.05, -0.05, -0.5],
        [
            "well above the mid-plane",
            "above the mid-plane",
            "near the mid-plane",
            "below the mid-plane",
            "well below the mid-plane",
        ],
    )

    literature = level(
        C_lit,
        [0.9, 0.75, 0.5, 0.25],
        [
            "very well-studied",
            "well-studied",
            "moderately studied",
            "poorly studied",
            "<u>rarely</u> studied",
        ],
    )

    duplicate, dup_warn = level(
        C_dup,
        [0.95, 0.75, 0.5, 0.25, 0.1],
        [
            ("a unique", ""),
            ("very likely a unique", ""),
            ("likely a unique", ""),
            ("possibly a duplicate", HTML_WARN),
            ("<u>likely a duplicate</u>", HTML_WARN),
            ("<u>very likely a duplicate</u>", HTML_WARN),
        ],
    )

    return (
        members,
        density,
        quality,
        literature,
        duplicate,
        dup_warn,
        plx_dist,
        z_position,
    )


def get_summary(
    name,
    cl_DB,
    shared_members_p,
    C_dup,
    C_dup_same_db,
    bad_oc,
    members,
    density,
    quality,
    fpars_summ,
    current_year,
    literature,
    duplicate,
    dup_warn,
    fpars_note,
):
    """ """
    summary = (
        f"{tsp}<b>{name}</b> is a {members}, {density} object of {quality} {HTML_C3}."
    )

    summary += " " + fpars_summ
    summary += " " + lit_summary(current_year, cl_DB, literature)

    dup_summ, dup_note = dupl_summary(
        shared_members_p, C_dup, C_dup_same_db, duplicate, dup_warn
    )
    summary += f" {dup_summ}{fpars_note}{dup_note}"

    # Bad OC warning
    if bad_oc == "y":
        summary += (
            f"<p>{HTML_WARN}the low {HTML_UTI} value and no obvious signs of "
            f"duplication (<i>C<sub>dup</sub>={C_dup}</i>) indicate that this "
            f"is quite probably an asterism, moving group, or artifact, and {HTML_BAD_OC}.</p>"
        )

    return summary


def fpars_in_lit(
    current_year: int,
    DBs_json: dict,
    DBs: str,
    fund_pars: str,
    recent_years=5,
) -> tuple[int, dict]:
    """
    DBs_json: JSON that contains the data for all DBs
    DBs: DBs where this cluster is present
    """

    # Reverse years order to show the most recent first
    DBs_lst, fund_pars_lst = DBs.split(";")[::-1], fund_pars.split(";")[::-1]

    # Index of last value in pars_years that is equal to (current_year - recent_years)
    recent_year_i = -1
    fpars_dict = {par: [] for par in fpars_order}

    for i, (db, pars) in enumerate(zip(DBs_lst, fund_pars_lst)):
        info = DBs_json[db]

        if int(info["year"]) >= current_year - recent_years:
            recent_year_i = i

        values = pars.split(",")
        if all(v == "--" for v in values):
            continue

        for x, par in zip(values, fpars_order):
            fpars_dict[par].append(np.nan if x == "--" else float(x.replace("*", "")))

    return recent_year_i, fpars_dict


def fpars_medians_spread(recent_year_i, fpars_dict) -> tuple[dict, list]:
    """ """
    labels = {
        "dist": "distance",
        "ext": "absorption",
        "diff_ext": "differential extinction",
        "age": "age",
        "met": "metallicity",
        "mass": "mass",
        "bi_frac": "binary fraction",
        "blue_str": "blue stragglers",
    }

    medians, large_spread = {}, []
    for name, vals in fpars_dict.items():
        # if name in ("diff_ext", "bi_frac"):
        #     # Not used in the parameters summary, skip median and spread check
        #     continue

        arr = np.array(vals, dtype=float)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            continue

        if valid.size == 1:
            medians[name] = valid[0]
            continue

        if name == "blue_str":
            # BSS values can be either fractions or total number, skip spread check
            continue

        median = np.median(valid)
        medians[name] = median

        # Estimate large spread for recent sources
        arr_recent = np.array(vals[:recent_year_i], dtype=float)
        valid_recent = arr_recent[np.isfinite(arr_recent)]
        if valid_recent.size == 0:
            continue

        mx, mn = valid_recent.max(), valid_recent.min()
        if mx == 0:
            mx += 0.01
        rel_range = mn / mx

        flag_spread = False

        if name == "met":
            flag_spread = (mx - mn) > 0.3
        elif name == "age":
            if mx < 50:  # very young clusters
                flag_spread = rel_range < 0.1
            if mx < 100:  # young clusters
                flag_spread = rel_range < 0.15
            elif mx < 250:  # young clusters
                flag_spread = rel_range < 0.2
            else:
                flag_spread = rel_range < 0.3
        elif name == "dist":
            if mx < 0.5:  # very nearby clusters
                flag_spread = rel_range < 0.1
            elif mx < 1.0:  # nearby clusters
                flag_spread = rel_range < 0.2
            else:
                flag_spread = rel_range < 0.3
        elif name == "ext":
            if mx < 0.5:  # very low extinction
                flag_spread = rel_range < 0.05
            elif mx < 1:  # low extinction
                flag_spread = rel_range < 0.15
            else:
                flag_spread = rel_range < 0.3
        elif name == "mass":
            if mx < 500:  # very low-mass clusters / associations
                flag_spread = rel_range < 0.1
            elif mx < 1e3:  # low-mass clusters / associations
                flag_spread = rel_range < 0.2
            else:
                flag_spread = rel_range < 0.3

        if flag_spread:
            large_spread.append(labels[name])

    return medians, large_spread


def fpars_summary(
    medians, large_spread, plx, plx_dist, plx_z_dist
) -> tuple[str, str, dict]:
    """Summarize fundamental astrophysical parameters."""
    fpars_badges = {}

    dist_flag, fpars_note, dist_txt = "", "", ""
    if "dist" in medians:
        plx_kpc = 1 / plx
        ratio_lim = 0.3
        if plx_kpc < 0.5 and medians["dist"] < 0.5:
            ratio_lim = 0.5
        if abs(plx_kpc / medians["dist"] - 1) > ratio_lim:
            dist_flag = "<sup><b>*</b></sup>"
            fpars_note = (
                '<p class="note"><strong>(*):</strong> '
                f"The parallax distance estimate (~{plx_kpc:.2f} kpc) "
                "differs significantly from the median photometric distance "
                f"(~{medians['dist']:.2f} kpc).</p>"
            )

        if medians["dist"] < 0.5:
            dist_txt = "Very close"
        elif medians["dist"] < 1:
            dist_txt = "Close"
        elif medians["dist"] < 3:
            dist_txt = "Relatively close"
        elif medians["dist"] < 5:
            dist_txt = "Distant"
        elif medians["dist"] < 10:
            dist_txt = "Very distant"
        else:
            dist_txt = "Extremely distant"

    fpars_badges["dist"] = dist_txt

    ext_txt = ""
    if "ext" in medians:
        Av = medians["ext"]
        if Av > 10:
            ext_txt = ", affected by <u>extremely</u> high extinction"
        elif Av > 5:
            ext_txt = ", affected by very high extinction"
        elif Av > 3:
            ext_txt = ", affected by high extinction"
        elif Av > 1:
            ext_txt = ", affected by moderate extinction"
        else:
            ext_txt = ", affected by low extinction"

        fpars_badges["ext"] = (
            ext_txt.replace(", affected by ", "")
            .replace("<u>", "")
            .replace("</u>", "")
            .capitalize()
        )

    fpars_summ = (
        f"Its parallax locates it at a {plx_dist}{dist_flag} distance, "
        f"{plx_z_dist}{ext_txt}."
    )

    if not medians:
        fpars_summ += " No fundamental parameter values are available for this object."
        return fpars_summ, fpars_note, {}

    descriptors = []

    if (mass := medians.get("mass")) is not None:
        if mass > 10000:
            mass_txt = "<u>extremely</u> massive"
        elif mass > 5000:
            mass_txt = "very massive"
        elif mass > 1000:
            mass_txt = "massive"
        elif mass < 50:
            mass_txt = "low-mass"
        else:
            mass_txt = ""

        if mass_txt != "":
            descriptors.append(mass_txt)
            fpars_badges["mass"] = (
                mass_txt.replace("<u>", "")
                .replace("</u>", "")
                .replace("-", " ")
                .capitalize()
            )

    if (feh := medians.get("met")) is not None:
        if feh > 1:
            feh_txt = "very metal-rich"
        elif feh >= 0.5:
            feh_txt = "metal-rich"
        elif feh > -0.5:
            feh_txt = "near-solar metallicity"
        elif feh > -1:
            feh_txt = "metal-poor"
        elif feh > -2:
            feh_txt = "very metal-poor"
        else:
            feh_txt = "<u>extremely</u> metal-poor"

        descriptors.append(feh_txt)
        fpars_badges["feh"] = (
            feh_txt.replace("<u>", "").replace("</u>", "").capitalize()
        )

    if (age := medians.get("age")) is not None:
        if age < 20:
            age_txt = "very young"
        elif age < 100:
            age_txt = "young"
        elif age < 1000:
            age_txt = "intermediate-age"
        elif age < 5000:
            age_txt = "old"
        elif age < 10000:
            age_txt = "very old"
        else:
            age_txt = "<u>extremely</u> old"

        descriptors.append(age_txt)
        fpars_badges["age"] = (
            age_txt.replace("-", " ")
            .replace("<u>", "")
            .replace("</u>", "")
            .capitalize()
        )

    if medians.get("blue_str", 0) > 0:
        fpars_note += (
            '<p class="note"><strong>Note:</strong> '
            "This object contains blue stragglers according to at least one source.</p>"
        )
        fpars_badges["bss"] = "Contains BSS"

    spread_txt = ""
    if large_spread:
        if len(large_spread) == 1:
            joined = large_spread[0]
        elif len(large_spread) == 2:
            joined = " and ".join(large_spread)
        else:
            joined = ", ".join(large_spread[:-1]) + f", and {large_spread[-1]}"
        spread_txt = (
            ", but with a <u>large variance across recent sources</u> "
            f"for the {joined} parameter{'s' if len(large_spread) > 1 else ''}"
        )

    article = "an" if descriptors and descriptors[0][0] in "aeiou<" else "a"
    desc_txt = ", ".join(descriptors)

    fpars_summ += (
        f" It is catalogued as {article} {desc_txt} cluster{spread_txt} "
        '(see <a href="#parameters" '
        "onclick=\"activateTabById(event, 'tab_parameters', 'parameters')\">"
        "Parameters</a>)."
    )

    return fpars_summ, fpars_note, fpars_badges


def lit_summary(current_year, cl_DB, literature, year_gap=3) -> str:
    """ """
    parts = cl_DB.split(";")
    first_lit_year = int(parts[0].split("_")[0][-4:])
    last_lit_year = int(parts[-1].split("_")[0][-4:])

    years_gap_first = current_year - first_lit_year
    years_gap_last = current_year - last_lit_year

    if years_gap_first <= year_gap:
        # The FIRST article associated to this entry is recent
        modifier = ""
        if literature in ("well-studied", "moderately studied"):
            modifier = f"but it is {literature} "
        return f"It was recently reported {modifier}in the literature."
    elif years_gap_last > 5:
        # The FIRST article associated to this entry is NOT recent and
        # this is an OC that has not been revisited in a while
        return (
            f"It is {literature} in the literature, with no articles "
            f"listed in the last {years_gap_last} years."
        )

    return f"It is {literature} in the literature."


def dupl_summary(shared_members_p, C_dup, C_dup_same_db, duplicate, dup_warn):
    """ """

    def member_level(v):
        return level(
            v,
            [0.9, 0.75, 0.5, 0.25],
            ["very small", "small", "moderate", "significant", "large"],
        )

    # If this unique object contains shared members with other entries
    if C_dup == 1.0 and C_dup_same_db == 1.0:
        if str(shared_members_p) == "nan":
            # Return with no duplication info added
            return "", ""
        shared_members_p = list(map(float, shared_members_p.split(";")))
        max_p = max(shared_members_p) / 100.0
        n = len(shared_members_p)
        dupl_note = (
            '<p class="note"><strong>Note:</strong> '
            f"This object shares a {member_level(1 - max_p)} percentage of members "
            f"with {'a' if n == 1 else n} later reported "
            f"{'entry' if n == 1 else 'entries'}. {HTML_TABLE}</p>"
        )
        return "", dupl_note

    if C_dup == 1.0 and C_dup_same_db < 1.0:
        # Only same DB duplicates found
        dupl_note = (
            '<p class="note"><strong>Note:</strong> '
            f"This object shares a {member_level(C_dup_same_db)} percentage "
            "of members with at least one entry reported in the same catalogue. "
            f"{HTML_TABLE}</p>"
        )
        return "", dupl_note

    # Duplicates found in different DBs only (C_dup<1.0 & C_dup_same_db==1.0)
    prev_or_same_db = "previously reported entry"
    if C_dup_same_db < 1.0:
        prev_or_same_db += (
            f", and a {member_level(C_dup_same_db)} percentage with at least "
            "one entry reported in the same catalogue"
        )

    text = (
        'This is <a href="/faq#how-is-the-duplicate-probability-estimated" '
        'target="_blank" title="How is the duplicate probability estimated?">'
        f"{duplicate}</a> object, which shares a "
        f"{member_level(C_dup)} percentage of members with at least one "
        f"{prev_or_same_db}. {HTML_TABLE}"
    )

    if dup_warn:
        dup_summary, dupl_note = f"<p>{dup_warn}{text}</p>", ""
    else:
        dup_summary, dupl_note = (
            "",
            f'<p class="note"><strong>Note:</strong> {text}</p>',
        )

    return dup_summary, dupl_note


def update_summary(fpars_medians, summaries, descriptors, fpars_badges):
    """ """
    # Round fundamental parameters medians to 4 decimal places
    fpars_round = {
        k: {kk: round(vv, 4) if isinstance(vv, float) else vv
            for kk, vv in v.items()}
        for k, v in fpars_medians.items()
    }

    UCC_summ_cmmts = {}
    for fname0, summary in summaries.items():
        # if fname0 in UCC_summ_cmmts:
        #     # Update summary
        #     if summary != UCC_summ_cmmts[fname0]["summary"]:
        #         UCC_summ_cmmts[fname0]["summary"] = summary
        #         N_summ_updt += 1
        #     if descriptors[fname0] != UCC_summ_cmmts[fname0]["descriptors"]:
        #         UCC_summ_cmmts[fname0]["descriptors"] = descriptors[fname0]
        #         N_desc_updt += 1
        #     if fpars_badges[fname0] != UCC_summ_cmmts[fname0]["fpars_badges"]:
        #         UCC_summ_cmmts[fname0]["fpars_badges"] = fpars_badges[fname0]
        #         N_fbadges_updt += 1
        # else:
        # Create new entry
        UCC_summ_cmmts[fname0] = {
            "summary": summary,
            "descriptors": descriptors[fname0],
            "fpars_badges": fpars_badges[fname0],
            "fpars_medians": fpars_round[fname0],
        }
        # N_new += 1
    # logging.info(f"Updated summaries for {N_summ_updt} objects")
    # logging.info(f"Updated descriptors for {N_desc_updt} objects")
    # logging.info(f"Updated parameters badges for {N_fbadges_updt} objects")
    # logging.info(f"Added summaries for {N_new} new objects\n")

    return UCC_summ_cmmts


def update_comments(
    fnames_lst,
    UCC_summ_cmmts,
    fnames_B,
    art_ID,
    art_name,
    art_year,
    art_url,
    art_cmmts,
    fnames_found,
):
    """ """
    for i, idx in fnames_found.items():
        fname0 = fnames_B[idx].split(";")[0]
        if fname0 not in fnames_lst:
            continue

        comment = art_cmmts[i]
        comment_entry = {
            "ID": art_ID,
            "name": art_name,
            "url": art_url,
            "year": art_year,
            "comment": comment,
        }

        if "comments" not in UCC_summ_cmmts[fname0]:
            UCC_summ_cmmts[fname0]["comments"] = [comment_entry]
            # UCC_summ_cmmts[fname0]["comments"].append(comment_entry)
        else:
            # # Check if comment from this article already exists
            # flag_ID = False
            # for cmt in UCC_summ_cmmts[fname0]["comments"]:
            #     if cmt["ID"] == art_ID:
            #         flag_ID = True
            #         cmt["comment"] = comment
            #         break
            # # If not, append new comment
            # if flag_ID is False:
            UCC_summ_cmmts[fname0]["comments"].append(comment_entry)

    return UCC_summ_cmmts


def update_json_file(UCC_summ_cmmts, temp_UCC_cmmts_file):
    """Update UCC_cmmts json file"""

    # Sort comments by year (descending) for each entry
    for obj in UCC_summ_cmmts.values():
        if "comments" in obj and isinstance(obj["comments"], list):
            obj["comments"].sort(key=lambda c: int(c.get("year", 0)), reverse=True)

    # Store compressed
    with gzip.open(temp_UCC_cmmts_file, "wt", encoding="utf-8") as f:
        json.dump(UCC_summ_cmmts, f)


if __name__ == "__main__":
    main()
