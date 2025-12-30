import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("../UCC_cat_B.csv")

# "dist", "ext", "diff_ext", "age", "met", "mass", "bi_frac", "blue_str"
par_idx, par_min, par_max = 0, 0.1, 10  # dist [kpc]
par_idx, par_min, par_max = 1, 0.0, 10  # ext
# par_idx, par_min, par_max = 2, 0., 5  # diff_ext
# par_idx, par_min, par_max = 3, 1., 10000  # age
par_idx, par_min, par_max = 4, -2, 1  # met
# par_idx, par_min, par_max = 5, 0, 5000  # mass
# par_idx, par_min, par_max = 7, 0, 10  # blue_str


all_par_vals = []
min_par_val, max_par_val = np.inf, -np.inf
append = all_par_vals.append

for i, cell in enumerate(df["fund_pars"]):
    for val in cell.split(";"):
        par = val.split(",")[par_idx]
        if "--" in par:
            continue
        if "*" in par:
            par = par[:-1]
        par = float(par)
        append(par)

        if par > par_max:
            print(f"(max) {df['Names'][i].split(';')[0]}: {par}")
            max_par_val = max(max_par_val, par)
        if par < par_min:
            print(f"(min) {df['Names'][i].split(';')[0]}: {par}")
            min_par_val = min(min_par_val, par)

print("")
print(f"Min par val: {min_par_val}")
print(f"Max par val: {max_par_val}")

all_par_vals = np.clip(all_par_vals, a_min=par_min, a_max=par_max)

plt.hist(all_par_vals, bins=30)
plt.show()
breakpoint()
