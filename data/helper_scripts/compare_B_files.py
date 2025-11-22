import pandas as pd

old_B = pd.read_csv("../UCC_cat_B.csv")
new_B = pd.read_csv("../../temp_updt/UCC_cat_B.csv")

fname0_old_all = [_.split(";")[0] for _ in old_B["fnames"]]
fname0_new_all = [_.split(";")[0] for _ in new_B["fnames"]]

print("fnames in old B that changed in new:\n")
for fname0_old in fname0_old_all:
    if fname0_old not in fname0_new_all:
        for i, fname_new in enumerate(new_B["fnames"]):
            if fname0_old in fname_new.split(";"):
                print(f"{fname0_old} --> {fname_new}, {new_B['DB'][i].split(';')[0]}")
                break

print("\nfnames in new B that are not in old:\n")
for fname0_new in fname0_new_all:
    if fname0_new not in fname0_old_all:
        found = False
        for i, fname_old in enumerate(old_B["fnames"]):
            if fname0_new in fname_old.split(";"):
                found = True
                # print(f"{fname0_new}")
                break
        if not found:
            print(f"{fname0_new}")
            pass
