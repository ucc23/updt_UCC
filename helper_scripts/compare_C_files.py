import pandas as pd
import matplotlib.pyplot as plt

old_C = pd.read_csv("../UCC_cat_C.csv")
new_C = pd.read_csv("../../temp_updt/UCC_cat_C.csv")

col = "C_lit"  #"UTI"

plt.subplot(121)
plt.scatter(old_C[col], old_C[col]-new_C[col], alpha=0.5)
plt.xlabel(f"Old {col}")
plt.ylabel(f"Old-New {col}")

plt.subplot(122)
plt.scatter(old_C[col], new_C[col], alpha=0.5)
plt.xlabel(f"Old {col}")
plt.ylabel(f"New {col}")
plt.show()


