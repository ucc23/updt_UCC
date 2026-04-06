import pandas as pd

df = pd.read_csv("../data/UCC_cat_C.csv")

# Create new dictionary with 'fname' as key and 'N_50' values
N_membs = dict(zip(df["fname"], df["N_50"]))


for row in df.itertuples():
    shared_n_vals, s_prob_vals, s_n_membs = [], [], []
    if str(row.shared_members) != "nan":
        shared_n = str(row.shared_members).split(";")
        shared_p = [float(_) for _ in str(row.shared_members_p).split(";")]
        for i, s_prob in enumerate(shared_p):
            if s_prob > 50 and float(row.C_dup) < 0.25:
                shared_n_vals.append(shared_n[i])
                s_prob_vals.append(str(s_prob))
                s_n_membs.append(N_membs.get(shared_n[i], "N/A"))

    if shared_n_vals:
        # Intertwine shared_n_vals, s_prob_vals, s_n_membs lists
        intertwined = list(zip(shared_n_vals, s_prob_vals, s_n_membs))
        txt = ", ".join(
            [
                f"{n} (p={p}%, N={nm:.0f})"
                for n, p, nm in intertwined
            ]
        )
        print(
            f"{row.fname} (C_dup={row.C_dup}, N={row.N_50:.0f}) [dup of]--> {txt}"
        )

