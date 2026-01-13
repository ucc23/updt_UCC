import webbrowser

import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv("/home/gabriel/Github/UCC/updt_UCC/data/UCC_cat_B.csv")
df1["fname"] = [_.split(";")[0] for _ in df1["fnames"]]
df2 = pd.read_csv("/home/gabriel/Github/UCC/updt_UCC/data/UCC_cat_C.csv")

df_merged = pd.merge(df1, df2, left_on="fname", right_on="fname", suffixes=("_B", "_C"))


# norm = df_merged["Plx_m"] #df_merged["r_50"] / 60
df_merged["dist_2D_x"] = (
    (df_merged["GLON"] - df_merged["GLON_m"]) ** 2
    + (df_merged["GLAT"] - df_merged["GLAT_m"]) ** 2
) ** 0.5  # / norm

norm = 0.5 * (abs(df_merged["pmRA_m"]) + abs(df_merged["pmDE_m"]))
df_merged["dist_2D_y"] = (
    (df_merged["pmRA"] - df_merged["pmRA_m"]) ** 2
    + (df_merged["pmDE"] - df_merged["pmDE_m"]) ** 2
) ** 0.5 / norm
df_merged["dist_plx"] = abs(df_merged["Plx"] - df_merged["Plx_m"])

# Remove OCs checked manually
checked_manually = [
    "melotte111",
    "platais6",
]
df_merged = df_merged[~df_merged["fname"].isin(checked_manually)]

# Remove known problematic OCs
df_merged = df_merged[~df_merged["fname"].str.startswith("hsc", na=False)]
df_merged = df_merged[~df_merged["fname"].str.startswith("theia", na=False)]
df_merged = df_merged[~df_merged["fname"].str.startswith("dutrabica", na=False)]
df_merged = df_merged[~df_merged["fname"].str.startswith("mwsc", na=False)]
df_merged = df_merged[~df_merged["fname"].str.startswith("cwnu", na=False)]
df_merged = df_merged[~df_merged["fname"].str.startswith("ocsn", na=False)]
df_merged = df_merged[~df_merged["fname"].str.startswith("lp", na=False)]


N_top = 20
x = df_merged.sort_values(by="dist_2D_x", ascending=False)
# x["dist_2D_x"] = (x["dist_2D_x"]).round(3)
print(
    x[["fname", "GLON", "GLAT", "GLON_m", "GLAT_m", "UTI", "dist_2D_x", "dist_2D_y"]]
    .iloc[:N_top]
    .to_string(index=False)
)
y = df_merged.sort_values(by="dist_2D_y", ascending=False)
print(
    y[["fname", "pmRA", "pmDE", "pmRA_m", "pmDE_m", "UTI", "dist_2D_x", "dist_2D_y"]]
    .iloc[:N_top]
    .to_string(index=False)
)

msk = (df_merged["UTI"] > 0.0) & (
    (df_merged["dist_2D_x"] > 0.5) | (df_merged["dist_2D_y"] > 10)
)


names = df_merged["fname"][msk].values
xp = df_merged["dist_2D_x"][msk].values
yp = df_merged["dist_2D_y"][msk].values
# color = df_merged.loc[msk, "dist_plx"].fillna(df_merged.loc[msk, "Plx_m"]).values
color = df_merged.loc[msk, "UTI"]


fig, ax = plt.subplots()
sc = ax.scatter(xp, yp, alpha=0.25, c=color)
plt.colorbar(sc)
plt.title(msk.sum())
plt.xlabel("Distance in Position (deg)")
plt.ylabel("Distance in Proper Motion (mas/yr)")

annot = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(10, 10),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
)
annot.set_visible(False)


def update_annot(ind):
    i = ind["ind"][0]
    annot.xy = (xp[i], yp[i])
    annot.set_text(names[i])
    annot.set_visible(True)


def hover(event):
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()


def onclick(event):
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            i = ind["ind"][0]
            url = f"https://ucc.ar/_clusters/{names[i]}"
            webbrowser.open(url)


fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
