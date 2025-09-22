import pandas as pd


def main():
    df_new = pd.read_csv("data/DBs_merged.csv")

    check_fnames = True
    check_fnames =  False

    if check_fnames:
        df_ucc = pd.read_csv("/home/gabriel/Descargas/DBs_merged_OLD.csv")
    else:
        df_ucc = pd.read_csv("data/UCC_cat_25092211.csv")

    fnames_ucc = [_.split(";") for _ in df_ucc["fnames"]]
    fnames_new = [_.split(";") for _ in df_new["fnames"]]

    N = 0
    for i, fname_u in enumerate(fnames_ucc):
        fname_u0 = fname_u[0]
        for j, fname_n in enumerate(fnames_new):
            if fname_u0 in fname_n:
                if fname_u0 != fname_n[0]:
                    if check_fnames:
                        print(f"{fname_u} != {fname_n}")
                # elif not fname_u0.startswith("hsc"):
                if check_fnames is False:
                    # Calculate the distance between both entries
                    lon_o, lat_o = df_ucc.loc[i, ["GLON_m", "GLAT_m"]]
                    lon_n, lat_n = df_new.loc[j, ["GLON", "GLAT"]]
                    dist = (((lon_o - lon_n) ** 2 + (lat_o - lat_n) ** 2) ** 0.5) * 60
                    r_50 = df_ucc.loc[i, "r_50"]
                    if dist > r_50 and dist > 15:
                        print(f"{fname_n[0]}, {dist:.2f}, {r_50}")
                    N += 1
    print(N)
    breakpoint()


if __name__ == "__main__":
    main()
