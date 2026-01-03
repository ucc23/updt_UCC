import json
import os

import pandas as pd

from .utils import get_fnames, logger
from .variables import (
    data_folder,
    merged_dbs_file,
)

# json_struct = {
#     "ngc9999": {
#         "summary": "NGC 9999 is ...",
#         "comments": [
#             {
#                 "name": "Smith et al.",
#                 "url": "htps://example.com",
#                 "year": "2023",
#                 "comment": "This object...",
#             },
#             {
#                 "name": "Doe et al.",
#                 "url": "htps://example2.com",
#                 "year": "2024",
#                 "comment": "Further studies...",
#             },
#         ],
#     }
# }


def main():
    """ """
    logging = logger()

    df_B = pd.read_csv(f"{data_folder}{merged_dbs_file}")
    
    with open(f"{data_folder}/UCC_cmmts.json", "r") as f:
        UCC_cmmts = json.load(f)

    # Read path to JSON file with comments from article
    while True:
        fpath_json = input(
            f"Enter name of JSON file in {data_folder}/json_cmmts/ folder: "
        )
        fpath_json = f"{data_folder}/json_cmmts/" + fpath_json.replace(".json", "")
        if os.path.exists(f"{fpath_json}.json"):
            break
        else:
            logging.info("File does not exist. Try again.")
    with open(f"{fpath_json}.json", "r") as f:
        jsonf = json.load(f)

    art_name = jsonf["art_name"]
    art_year = jsonf["art_year"]
    art_url =  jsonf["art_url"]

    art_clusters = jsonf['clusters'].keys()
    art_cmmts = list(jsonf['clusters'].values())

    # Convert all names to fnames
    jsonf_fnames = get_fnames(art_clusters)

    # Extract fnames from B file
    fnames_B = df_B["fnames"].tolist()
    lookup = {}
    for i, s in enumerate(fnames_B):
        for token in s.split(";"):
            lookup[token] = i

    # Check which objects in the JSON file are in the UCC
    not_found = []
    fnames_found = {}
    for i, fnames in enumerate(jsonf_fnames):
        not_found_in = []
        for fname in fnames:
            idx = lookup.get(fname)
            if idx is None:
                not_found_in.append(fname)
            else:
                fnames_found[i] = idx
        if not_found_in:
            not_found.append(not_found_in)
    if not_found:
        for cl_not_found in not_found:
            logging.info(f"{','.join(cl_not_found)} not found in UCC")

    for i, idx in fnames_found.items():
        fname0 = fnames_B[idx].split(';')[0]

        summary = "text" #jsonf[list(jsonf.keys())[i]]["summary"]
        comment = art_cmmts[i]

        comment_entry = {
            "name": art_name,
            "url": art_url,
            "year": art_year,
            "comment": comment
        }

        # Check if fname0 exists in UCC_cmmts
        if fname0 in UCC_cmmts:
            # Append new comments
            existing_comments = UCC_cmmts[fname0]["comments"]
            existing_comments.append(comment_entry)
            # Optionally update summary if needed
            UCC_cmmts[fname0]["summary"] = summary
        else:
            # Create new entry
            UCC_cmmts[fname0] = {
                "summary": summary,
                "comments": []
            }
            UCC_cmmts[fname0]["comments"].append(comment_entry)

    # Update UCC_cmmts json file
    with open(f"{data_folder}/UCC_cmmts.json", "w") as f:
        json.dump(UCC_cmmts, f, indent=2)

    breakpoint()




if __name__ == "__main__":
    main()
