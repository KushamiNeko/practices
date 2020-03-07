# %%
##################################################

# WITH / WITHOUT OUTLIERS

REMOVE_OUTLIERS = False
# REMOVE_OUTLIERS = True

# %%
##########################################################################

# SETUP

import math
import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt
import fun.plot.utils as fu

font_src = "fonts/Kosugi/Kosugi-Regular.ttf"

df = pd.read_csv("data.csv")

df = utils.clean_data(df)
utils.test_data_na(df)


drop_1 = []
drop_2 = []
drop_3 = []

cols = []

for i, col in enumerate(df.columns[17:]):
    if i != 0 and i % 3 == 0:
        utils.test_column_pair(cols)
        utils.drop_append(cols, drop_1, drop_2, drop_3)

        cols = [col]
    else:
        cols.append(col)

utils.test_column_pair(cols)
utils.drop_append(cols, drop_1, drop_2, drop_3)

base_set = df.drop(drop_1, axis=1)
middle_set = df.drop(drop_2, axis=1)
final_set = df.drop(drop_3, axis=1)


utils.rename_columns(base_set)
utils.rename_columns(middle_set)
utils.rename_columns(final_set)

assert len(base_set.columns) == len(middle_set.columns)
assert len(base_set.columns) == len(final_set.columns)

for col in base_set.columns:
    if col not in middle_set.columns:
        print(f"middle missing: {col}")
    if col not in final_set.columns:
        print(f"final missing: {col}")

for col in middle_set.columns:
    if col not in base_set.columns:
        print(f"base missing: {col}")
    if col not in final_set.columns:
        print(f"final missing: {col}")

for col in final_set.columns:
    if col not in base_set.columns:
        print(f"base missing: {col}")
    if col not in middle_set.columns:
        print(f"middle missing: {col}")


base_set["measure"] = "0w"
middle_set["measure"] = "6w"
final_set["measure"] = "12w"

drops = ["ID", "measure"]

if REMOVE_OUTLIERS:
    print("REMOVING OUTLIERS.....")
    for col in base_set.columns[17:]:
        if col in drops:
            continue

        utils.remove_outliers_dataframe(base_set, col)
        utils.remove_outliers_dataframe(middle_set, col)
        utils.remove_outliers_dataframe(final_set, col)

df = base_set.append(middle_set).append(final_set)
df = df.reset_index(drop=True)


mri = pd.read_csv("mri.csv")
mri = mri.drop("No.", axis=1)

if REMOVE_OUTLIERS:
    print("REMOVING OUTLIERS.....")
    for col in mri.columns:
        if col in ["ID"]:
            continue

        utils.remove_outliers_dataframe(mri, col)

mri.loc[:, "delta"] = mri["MRI_Neutral-2"] - mri["MRI_Neutral-1"]

t = df.set_index("ID")
t.loc[
    (t["measure"] == "0w") & (t.index.isin(mri["ID"])), "MRI_Neutral"
] = mri.set_index("ID")["MRI_Neutral-1"]

t.loc[t["measure"] == "6w", "MRI_Neutral"] = np.nan

t.loc[
    (t["measure"] == "12w") & (t.index.isin(mri["ID"])), "MRI_Neutral"
] = mri.set_index("ID")["MRI_Neutral-2"]

df = t.reset_index()

for i in df[df["measure"] == "0w"]["ID"]:
    if i not in mri["ID"].values:
        assert (
            math.isnan(df.loc[(df["ID"] == i) & (df["measure"] == "0w"), "MRI_Neutral"])
            == True
        )

        assert (
            math.isnan(df.loc[(df["ID"] == i) & (df["measure"] == "6w"), "MRI_Neutral"])
            == True
        )

        assert (
            math.isnan(
                df.loc[(df["ID"] == i) & (df["measure"] == "12w"), "MRI_Neutral"]
            )
            == True
        )
    else:
        assert (
            df.loc[(df["ID"] == i) & (df["measure"] == "0w"), "MRI_Neutral"].iloc[0]
            == mri.loc[mri["ID"] == i, "MRI_Neutral-1"].iloc[0]
        )

        assert (
            df.loc[(df["ID"] == i) & (df["measure"] == "12w"), "MRI_Neutral"].iloc[0]
            == mri.loc[mri["ID"] == i, "MRI_Neutral-2"].iloc[0]
        )

        assert (
            math.isnan(df.loc[(df["ID"] == i) & (df["measure"] == "6w"), "MRI_Neutral"])
            == True
        )
        pass


f = None

# %%
##########################################################################

# YOUR WORKS STARTS HERE

colx = utils.find_col(
    df,
    [
        ###,
        "total",
        "lac",
        "",
        "",
        "",
        ###,
    ],
)

coly = utils.find_col(
    df,
    [
        ###,
        "hamd",
        "",
        "",
        "",
        "",
        ###,
    ],
)

print(f"X: {colx}")
print(f"Y: {coly}")


# %%
##########################################################################

# wilcoxon group

if f is not None:
    plt.close(f)

f, ax = plt.subplots(figsize=(7, 7))
utils.plot_group(
    ###,
    df,
    colx,
    ax=ax,
    adjust=False,
    title=colx,
)

plt.tight_layout()
plt.show()


# %%
##########################################################################

# direct correlation

if f is not None:
    plt.close(f)

f, ax = plt.subplots(figsize=(7, 7))
fu.plot_correlation(
    ###,
    xs=df[colx],
    ys=df[coly],
    ax=ax,
    xlabel=colx,
    ylabel=coly,
    font_src=font_src,
)

plt.tight_layout()
plt.show()


# %%
##########################################################################

# 12W - 0W correlation

if f is not None:
    plt.close(f)

f, ax = plt.subplots(figsize=(7, 7))
fu.plot_correlation(
    xs=utils.week_delta(df, colx, op="fb"),
    ys=utils.week_delta(df, coly, op="fb"),
    ax=ax,
    xlabel=f"{colx} (12W - 0W)",
    ylabel=f"{coly} (12W - 0W)",
    font_src=font_src,
)

plt.tight_layout()
plt.show()

# %%
##########################################################################

# AUC(X) vs 12W - 0W(Y) correlation

if f is not None:
    plt.close(f)

f, ax = plt.subplots(figsize=(7, 7))
fu.plot_correlation(
    xs=utils.auc(df, colx, dx=6),
    ys=utils.week_delta(df, coly, op="fb"),
    ax=ax,
    xlabel=f"{colx} (AUC)",
    ylabel=f"{coly} (12W - 0W)",
    font_src=font_src,
)

plt.tight_layout()
plt.show()

# %%
##########################################################################

# 6W - 0W correlation

if f is not None:
    plt.close(f)

f, ax = plt.subplots(figsize=(7, 7))
fu.plot_correlation(
    xs=utils.week_delta(df, colx, op="mb"),
    ys=utils.week_delta(df, coly, op="mb"),
    ax=ax,
    xlabel=f"{colx} (6W - 0W)",
    ylabel=f"{coly} (6W - 0W)",
    font_src=font_src,
)

plt.tight_layout()
plt.show()

# %%
##########################################################################
