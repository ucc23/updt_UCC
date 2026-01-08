import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    """
    """
    import json

    with open("/home/gabriel/Github/UCC/updt_UCC/data/databases_info.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry, vals in data.items():
        # print(entry)
        for par in vals["pars"].items():
            print(par)
        # for err in vals["e_pars"].items():
        #     k, v = list(err[1].keys()), list(err[1].values())
        #     print(f"    {err[0]}: {k} --> {v}")




if __name__ == '__main__':
    main()
