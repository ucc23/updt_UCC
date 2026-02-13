import numpy as np

from ..variables import (
    HTML_BAD_OC,
    HTML_C3,
    HTML_TABLE,
    HTML_UTI,
    HTML_WARN,
    fpars_order,
    tsp,
)


def run(current_year, UCC_cl, DBs_JSON, cmmts_JSONS_lst):
    """ """
    # logging.info("\nGenerate all summaries")
    summary, descriptors, fpars_badges, badges_url = get_summary(
        current_year, DBs_JSON, UCC_cl
    )

    fnames0 = UCC_cl["fnames"].split(";")

    comments_lst = []
    for cmmt_json_dict in cmmts_JSONS_lst:
        for fname0 in fnames0:
            if fname0 in cmmt_json_dict["clusters"]:
                art_name = cmmt_json_dict["art_name"]
                art_year = cmmt_json_dict["art_year"]
                art_url = cmmt_json_dict["art_url"]

                comments_lst.append(
                    {
                        "name": art_name,
                        "url": art_url,
                        "year": art_year,
                        "comment": cmmt_json_dict["clusters"][fname0],
                    }
                )
                break

    # # Check the 'fnames' columns in df_UCC_B and df_UCC_C_final dataframes are equal
    # if not df_UCC["fname"].to_list() == list(UCC_summ_cmmts.keys()):
    #     raise ValueError("The 'fname' columns in B/C and final JSON differ")

    # # Sort comments by year (descending) for each entry
    # for obj in UCC_summ_cmmts.values():
    #     if "comments" in obj and isinstance(obj["comments"], list):
    #         obj["comments"].sort(key=lambda c: int(c.get("year", 0)), reverse=True)

    return summary, descriptors, fpars_badges, badges_url, comments_lst


def get_summary(current_year, DBs_JSON, UCC_cl):
    """
    Generate a textual summary and UTI descriptors for an astronomical object.
    """
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

    large_spread = fpars_spread(current_year, DBs_JSON, UCC_cl)

    # Extract medians
    medians = {}
    for par in fpars_order:
        par_v = UCC_cl[par + "_median"]
        if not np.isnan(par_v):
            medians[par] = par_v

    fpars_summ, fpars_note, fpars_badges_lst, fpars_badges_url_lst = fpars_summary(
        medians, large_spread, UCC_cl["Plx_m"], plx_dist, z_position
    )

    cl_name0 = UCC_cl["Names"].split(";")[0]
    summary = f"{tsp}<b>{cl_name0}</b> is a {members}, {density} object of {quality} {HTML_C3}."

    summary += " " + fpars_summ
    summary += " " + lit_summary(current_year, UCC_cl["DB"], literature)

    dup_summ, dup_note = dupl_summary(
        UCC_cl["shared_members_p"],
        UCC_cl["C_dup"],
        UCC_cl["C_dup_same_db"],
        duplicate,
        dup_warn,
    )
    summary += f" {dup_summ}{fpars_note}{dup_note}"

    # Bad OC warning
    if UCC_cl["bad_oc"] == "y":
        summary += (
            f"<p>{HTML_WARN}the low {HTML_UTI} value and no obvious signs of "
            f"duplication (<i>C<sub>dup</sub>={UCC_cl['C_dup']}</i>) indicate that this "
            f"is quite probably an asterism, moving group, or artifact, and {HTML_BAD_OC}.</p>"
        )

    # Descriptors, for back of summary card
    UTI_C_N_desc = members.capitalize()
    UTI_C_dens_desc = density.capitalize()
    UTI_C_C3_desc = quality.capitalize() + " quality"
    UTI_C_lit_desc = literature.replace("</u>", "").replace("<u>", "").capitalize()
    UTI_C_dup_desc = (
        duplicate.replace("</u>", "").replace("<u>", "").replace("a ", "").capitalize()
    )
    descriptors = [
        UTI_C_N_desc,
        UTI_C_dens_desc,
        UTI_C_C3_desc,
        UTI_C_lit_desc,
        UTI_C_dup_desc,
    ]

    return summary, descriptors, fpars_badges_lst, fpars_badges_url_lst


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
    # Add click event that takes you to the tab with the plot
    z_position = f"""<a href="#tab_gcpos" onclick="activateTabById(event, 'tab_gcpos', 'gcpos')">{z_position}</a>"""

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


def fpars_spread(
    current_year: int,
    DBs_json: dict,
    UCC_cl_row: dict,
    max_years_gap=5,
) -> list:
    """ """
    labels = {
        "dist": "distance",
        "av": "absorption",
        "diff_ext": "differential extinction",
        "age": "age",
        "met": "metallicity",
        "mass": "mass",
        "bi_frac": "binary fraction",
        "blue_str": "blue stragglers",
    }

    max_i = -1
    for i, db in enumerate(UCC_cl_row["DB"].split(";")):
        # Check if the year of this DB is recent
        if current_year - int(DBs_json[db]["year"]) <= max_years_gap:
            max_i = i
            break
    if max_i < 0:
        return []

    large_spread = []
    for par in fpars_order:
        if isinstance(UCC_cl_row[par], float):
            # Single value
            continue

        # Use only recent values
        cl_vals = UCC_cl_row[par].split(";")[max_i:]
        # Use only non-nan values
        cl_vals = [x for x in cl_vals if x != "nan"]

        if not cl_vals:
            # No non-nan values
            continue

        cl_vals = [_.replace("*", "") for _ in cl_vals]
        cl_vals = [float(x) for x in cl_vals]
        mn, mx = min(cl_vals), max(cl_vals)

        if mx == mn:
            continue

        if mx == 0:
            mx += 0.01
        rel_range = mn / mx

        flag_spread = False

        if par == "met":
            flag_spread = (mx - mn) > 0.3
        elif par == "age":
            if mx < 50:  # very young clusters
                flag_spread = rel_range < 0.1
            if mx < 100:  # young clusters
                flag_spread = rel_range < 0.15
            elif mx < 250:  # young clusters
                flag_spread = rel_range < 0.2
            else:
                flag_spread = rel_range < 0.3
        elif par == "dist":
            if mx < 0.5:  # very nearby clusters
                flag_spread = rel_range < 0.1
            elif mx < 1.0:  # nearby clusters
                flag_spread = rel_range < 0.2
            else:
                flag_spread = rel_range < 0.3
        elif par == "av":
            if mx < 0.5:  # very low extinction
                flag_spread = rel_range < 0.05
            elif mx < 1:  # low extinction
                flag_spread = rel_range < 0.15
            else:
                flag_spread = rel_range < 0.3
        elif par == "mass":
            if mx < 500:  # very low-mass clusters / associations
                flag_spread = rel_range < 0.1
            elif mx < 1e3:  # low-mass clusters / associations
                flag_spread = rel_range < 0.2
            else:
                flag_spread = rel_range < 0.3

        if flag_spread:
            large_spread.append(labels[par])

    return large_spread


def fpars_summary(
    medians, large_spread, plx, plx_dist, z_position
) -> tuple[str, str, dict, dict]:
    """Summarize fundamental astrophysical parameters."""

    if not medians:
        fpars_summ = (
            f"Its parallax locates it at a {plx_dist} distance, {z_position}."
            + " No fundamental parameter values are available for this object."
        )
        return fpars_summ, "", {}, {}

    fpars_badges, fpars_badges_url = {}, {}

    dist_flag, fpars_note, dist_txt, dist_range = "", "", "", ""
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
            dist_range = (0, 0.5)
        elif medians["dist"] < 1:
            dist_txt = "Close"
            dist_range = (0.5, 1)
        elif medians["dist"] < 3:
            dist_txt = "Relatively close"
            dist_range = (1, 3)
        elif medians["dist"] < 5:
            dist_txt = "Distant"
            dist_range = (3, 5)
        elif medians["dist"] < 10:
            dist_txt = "Very distant"
            dist_range = (5, 10)
        else:
            dist_txt = "Extremely distant"
            dist_range = (10, 1e6)

        fpars_badges["dist"] = dist_txt
        fpars_badges_url["dist"] = dist_range

    av_txt = ""
    if "av" in medians:
        Av = medians["av"]
        if Av > 10:
            av_txt = ", affected by <u>extremely</u> high extinction"
            av_range = (10, 1e6)
        elif Av > 5:
            av_txt = ", affected by very high extinction"
            av_range = (5, 10)
        elif Av > 3:
            av_txt = ", affected by high extinction"
            av_range = (3, 5)
        elif Av > 1:
            av_txt = ", affected by moderate extinction"
            av_range = (1, 3)
        else:
            av_txt = ", affected by low extinction"
            av_range = (0, 1)

        fpars_badges["av"] = (
            av_txt.replace(", affected by ", "")
            .replace("<u>", "")
            .replace("</u>", "")
            .capitalize()
        )
        fpars_badges_url["av"] = av_range

    fpars_summ = (
        f"Its parallax locates it at a {plx_dist}{dist_flag} distance, "
        f"{z_position}{av_txt}."
    )

    descriptors = []

    if (mass := medians.get("mass")) is not None:
        if mass > 10000:
            mass_txt = "<u>extremely</u> massive"
            mass_range = (10000, 1e6)
        elif mass > 5000:
            mass_txt = "very massive"
            mass_range = (5000, 10000)
        elif mass > 1000:
            mass_txt = "massive"
            mass_range = (1000, 5000)
        elif mass < 50:
            mass_txt = "low-mass"
            mass_range = (0, 50)
        else:
            mass_txt = ""
            mass_range = (0, 1e6)

        if mass_txt != "":
            descriptors.append(mass_txt)
            fpars_badges["mass"] = (
                mass_txt.replace("<u>", "")
                .replace("</u>", "")
                .replace("-", " ")
                .capitalize()
            )
            fpars_badges_url["mass"] = mass_range

    if (feh := medians.get("met")) is not None:
        if feh > 1:
            feh_txt = "very metal-rich"
            feh_range = (1, 100)
        elif feh >= 0.5:
            feh_txt = "metal-rich"
            feh_range = (0.5, 1)
        elif feh > -0.5:
            feh_txt = "near-solar metallicity"
            feh_range = (-0.5, 0.5)
        elif feh > -1:
            feh_txt = "metal-poor"
            feh_range = (-1, -0.5)
        elif feh > -2:
            feh_txt = "very metal-poor"
            feh_range = (-2, -1)
        else:
            feh_txt = "<u>extremely</u> metal-poor"
            feh_range = (-100, -2)

        descriptors.append(feh_txt)
        fpars_badges["feh"] = (
            feh_txt.replace("<u>", "").replace("</u>", "").capitalize()
        )
        fpars_badges_url["feh"] = feh_range

    if (age := medians.get("age")) is not None:
        if age < 20:
            age_txt = "very young"
            age_range = (0, 20)
        elif age < 100:
            age_txt = "young"
            age_range = (20, 100)
        elif age < 1000:
            age_txt = "intermediate-age"
            age_range = (100, 1000)
        elif age < 5000:
            age_txt = "old"
            age_range = (1000, 5000)
        elif age < 10000:
            age_txt = "very old"
            age_range = (5000, 10000)
        else:
            age_txt = "<u>extremely</u> old"
            age_range = (10000, 1e6)

        descriptors.append(age_txt)
        fpars_badges["age"] = (
            age_txt.replace("-", " ")
            .replace("<u>", "")
            .replace("</u>", "")
            .capitalize()
        )
        fpars_badges_url["age"] = age_range

    if medians.get("blue_str", 0) > 0:
        fpars_note += (
            '<p class="note"><strong>Note:</strong> '
            "This object contains blue stragglers according to at least one source.</p>"
        )
        fpars_badges["bss"] = "Contains BSS"
        fpars_badges_url["bss"] = (0, 1e6)

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

    return fpars_summ, fpars_note, fpars_badges, fpars_badges_url


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


# def update_summary(
#     fpars_medians, summaries, descriptors, fpars_badges, fpars_badges_url
# ):
#     """ """
#     # Round fundamental parameters medians to 4 decimal places
#     fpars_round = {
#         k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
#         for k, v in fpars_medians.items()
#     }

#     UCC_summ_cmmts = {}
#     for fname0, summary in summaries.items():
#         # if fname0 in UCC_summ_cmmts:
#         #     # Update summary
#         #     if summary != UCC_summ_cmmts[fname0]["summary"]:
#         #         UCC_summ_cmmts[fname0]["summary"] = summary
#         #         N_summ_updt += 1
#         #     if descriptors[fname0] != UCC_summ_cmmts[fname0]["descriptors"]:
#         #         UCC_summ_cmmts[fname0]["descriptors"] = descriptors[fname0]
#         #         N_desc_updt += 1
#         #     if fpars_badges[fname0] != UCC_summ_cmmts[fname0]["fpars_badges"]:
#         #         UCC_summ_cmmts[fname0]["fpars_badges"] = fpars_badges[fname0]
#         #         N_fbadges_updt += 1
#         # else:
#         # Create new entry
#         UCC_summ_cmmts[fname0] = {
#             "summary": summary,
#             "descriptors": descriptors[fname0],
#             "fpars_badges": fpars_badges[fname0],
#             "fpars_badges_url": fpars_badges_url[fname0],
#             "fpars_medians": fpars_round[fname0],
#         }
#         # N_new += 1
#     # logging.info(f"Updated summaries for {N_summ_updt} objects")
#     # logging.info(f"Updated descriptors for {N_desc_updt} objects")
#     # logging.info(f"Updated parameters badges for {N_fbadges_updt} objects")
#     # logging.info(f"Added summaries for {N_new} new objects\n")

#     return UCC_summ_cmmts


# def get_comments(cmmt_json_dict) -> tuple[str, str, str, list, dict]:
#     """Read JSON file with comments from article"""

#     art_name = cmmt_json_dict["art_name"]
#     art_year = cmmt_json_dict["art_year"]
#     art_url = cmmt_json_dict["art_url"]
#     # art_clusters = cmmt_json_dict["clusters"].keys()
#     # art_cmmts = list(cmmt_json_dict["clusters"].values())
#     # art_clusters, art_cmmts = zip(*cmmt_json_dict["clusters"])

#     # Convert all names to fnames
#     # jsonf_fnames = get_fnames(art_clusters)

#     # Check which objects in the JSON file are in the UCC
#     not_found = []
#     fnames_found = {}
#     for i, fnames in enumerate(jsonf_fnames):
#         not_found_in = []
#         for fname in fnames:
#             idx = B_lookup.get(fname)
#             if idx is None:
#                 not_found_in.append(fname)
#             else:
#                 fnames_found[i] = idx
#         if not_found_in:
#             not_found.append(not_found_in)
#     # if not_found:
#     #     logging.info(f"{art_ID}: {len(not_found)} objects not found in UCC")
#     # for cl_not_found in not_found:
#     #     logging.info(f"  {','.join(cl_not_found)}")

#     return art_name, art_year, art_url, list(art_cmmts), fnames_found


# def update_comments(
#     fnames_lst,
#     UCC_summ_cmmts,
#     fnames_B,
#     art_ID,
#     art_name,
#     art_year,
#     art_url,
#     art_cmmts,
#     fnames_found,
# ):
#     """ """
#     for i, idx in fnames_found.items():
#         fname0 = fnames_B[idx].split(";")[0]
#         if fname0 not in fnames_lst:
#             continue

#         comment = art_cmmts[i]
#         comment_entry = {
#             "ID": art_ID,
#             "name": art_name,
#             "url": art_url,
#             "year": art_year,
#             "comment": comment,
#         }

#         if "comments" not in UCC_summ_cmmts[fname0]:
#             UCC_summ_cmmts[fname0]["comments"] = [comment_entry]
#             # UCC_summ_cmmts[fname0]["comments"].append(comment_entry)
#         else:
#             # # Check if comment from this article already exists
#             # flag_ID = False
#             # for cmt in UCC_summ_cmmts[fname0]["comments"]:
#             #     if cmt["ID"] == art_ID:
#             #         flag_ID = True
#             #         cmt["comment"] = comment
#             #         break
#             # # If not, append new comment
#             # if flag_ID is False:
#             UCC_summ_cmmts[fname0]["comments"].append(comment_entry)

#     return UCC_summ_cmmts
