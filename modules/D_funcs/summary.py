import numpy as np

HTML_WARN = '<span style="color: #99180f; font-weight: bold;">Warning: </span>'
HTML_UTI = """<a href="/faq#what-is-the-uti-parameter" title="UTI parameter" target="_blank"><b>UTI</b></a>"""
HTML_C3 = """<a href="/faq#what-is-the-c3-parameter" title="C3 classification" target="_blank">C3 quality</a>"""
HTML_BAD_OC = """<a href="/faq#how-are-objects-flagged-as-likely-not-real" title="Not real open cluster" target="_blank"><u>not a real open cluster</u></a>"""
HTML_TABLE = (
    'See table with <a href="#tab_obj_shared" '
    "onclick=\"activateTabById(event, 'tab_obj_shared', 'obj_shared')\">"
    "shared members information</a>."
)


def level(value, thresholds, labels):
    """Map a numeric value to a qualitative label."""
    for t, lbl in zip(thresholds, labels):
        if value >= t:
            return lbl
    return labels[-1]


def summarize_object(
    current_year,
    name,
    cl_DB,
    plx,
    shared_members_p,
    Z_GC,
    C_N,
    C_dens,
    C_C3,
    C_lit,
    C_dup,
    C_dup_same_db,
    bad_oc,
    recent_year_i,
    fpars_dict,
    tsp,
) -> tuple[str, str, str, str, str, str]:
    """
    Generate a textual summary and UTI descriptors for an astronomical object.
    """

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

    summary = (
        f"{tsp}<b>{name}</b> is a {members}, {density} object of {quality} {HTML_C3}."
    )

    fpars_summ, fpars_note = fpars_summary(
        recent_year_i, fpars_dict, plx, plx_dist, z_position
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

    # For back of summary card
    UTI_C_N_desc = members.capitalize()
    UTI_C_dens_desc = density.capitalize()
    UTI_C_C3_desc = quality.capitalize() + " quality"
    UTI_C_lit_desc = literature.replace("</u>", "").replace("<u>", "").capitalize()
    UTI_C_dup_desc = (
        duplicate.replace("</u>", "").replace("<u>", "").replace("a ", "").capitalize()
    )

    return (
        summary,
        UTI_C_N_desc,
        UTI_C_dens_desc,
        UTI_C_C3_desc,
        UTI_C_lit_desc,
        UTI_C_dup_desc,
    )


def fpars_summary(
    recent_year_i: int, params: dict, plx, plx_dist, plx_z_dist
) -> tuple[str, str]:
    """Summarize fundamental astrophysical parameters."""

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
    for name, vals in params.items():
        if name in ("diff_ext", "bi_frac"):
            # Not used in the parameters summary, skip median and spread check
            continue

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

    dist_flag, fpars_note = "", ""
    if "dist" in medians:
        plx_kpc = 1 / plx
        if abs(plx_kpc / medians["dist"] - 1) > 0.3:
            dist_flag = "<sup><b>*</b></sup>"
            fpars_note = (
                '<p class="note"><strong>(*):</strong> '
                f"The parallax distance estimate (~{plx_kpc:.2f} kpc) "
                "differs significantly from the median photometric distance "
                f"(~{medians['dist']:.2f} kpc).</p>"
            )

    ext_txt = ""
    if "ext" in medians:
        Av = medians["ext"]
        if Av > 10:
            ext_txt = ", affected by <u>extremely</u> high extinction"
        elif Av > 5:
            ext_txt = ", affected by very high extinction"
        elif Av > 3:
            ext_txt = ", affected by high extinction"

    fpars_summ = (
        f"Its parallax locates it at a {plx_dist}{dist_flag} distance, "
        f"{plx_z_dist}{ext_txt}."
    )

    if not medians:
        fpars_summ += " No fundamental parameter values are available for this object."
        return fpars_summ, fpars_note

    descriptors = []

    if (mass := medians.get("mass")) is not None:
        if mass > 20000:
            descriptors.append("<u>extremely</u> massive")
        elif mass > 5000:
            descriptors.append("very massive")
        elif mass > 2000:
            descriptors.append("massive")

    if (feh := medians.get("met")) is not None:
        if feh > 1:
            descriptors.append("very metal-rich")
        elif feh >= 0.5:
            descriptors.append("metal-rich")
        elif feh > -0.5:
            descriptors.append("near-solar metallicity")
        elif feh > -1:
            descriptors.append("metal-poor")
        elif feh > -2:
            descriptors.append("very metal-poor")
        else:
            descriptors.append("<u>extremely</u> metal-poor")

    if (age := medians.get("age")) is not None:
        if age < 20:
            descriptors.append("very young")
        elif age < 100:
            descriptors.append("young")
        elif age < 1000:
            descriptors.append("intermediate-age")
        elif age < 5000:
            descriptors.append("old")
        elif age < 10000:
            descriptors.append("very old")
        else:
            descriptors.append("<u>extremely</u> old")

    if medians.get("blue_str", 0) > 0:
        fpars_note += (
            '<p class="note"><strong>Note:</strong> '
            "This object contains blue stragglers according to at least one source.</p>"
        )

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

    return fpars_summ, fpars_note


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
