##################################################

import importlib
import math
import os

import fun.plot.utils as pu
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import utils
from utils import clean_data, test_data_na

matplotlib.use("gtk3agg")

font_src = "fonts/Kosugi/Kosugi-Regular.ttf"

##################################################

REMOVE_OUTLIERS = False
# REMOVE_OUTLIERS = True

output_root = "with_outliers"
if REMOVE_OUTLIERS == True:
    output_root = "remove_outliers"

print(output_root)

##################################################

df = pd.read_csv("data.csv")

##################################################

df = clean_data(df)
test_data_na(df)

##################################################


drop_1 = []
drop_2 = []
drop_3 = []

cols = []

# start of the 3-set columns
# df.columns[17]

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

##################################################

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


##################################################

base_set["measure"] = "0w"
middle_set["measure"] = "6w"
final_set["measure"] = "12w"

##################################################

drops = ["ID", "measure"]

##################################################

if REMOVE_OUTLIERS:
    print("REMOVING OUTLIERS.....")
    for col in base_set.columns[17:]:
        if col in drops:
            continue

        utils.remove_outliers_dataframe(base_set, col)
        utils.remove_outliers_dataframe(middle_set, col)
        utils.remove_outliers_dataframe(final_set, col)

##################################################

df = base_set.append(middle_set).append(final_set)
df = df.reset_index(drop=True)


##################################################

hamd_delta = df[df["measure"] == "12w"]["HAMD"].reset_index(drop=True) - df[
    df["measure"] == "0w"
]["HAMD"].reset_index(drop=True)

hamd_delta


##################################################

auc = []
col = utils.find_col(df, ["total", "lac"])

# print(col)

# print(
# [
# df[df["measure"] == "0w"][col].iloc[0],
# df[df["measure"] == "6w"][col].iloc[0],
# df[df["measure"] == "12w"][col].iloc[0],
# ]
# )

for i in range(len(df[df["measure"] == "0w"])):
    ys = [
        df[df["measure"] == "0w"][col].iloc[i],
        df[df["measure"] == "6w"][col].iloc[i],
        df[df["measure"] == "12w"][col].iloc[i],
    ]

    auc.append(np.trapz(ys, dx=6.0))

auc = pd.Series(auc)
auc


##################################################


num_rows = 2
num_cols = 2

f, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))

col = utils.find_col(df, ["weight"])

utils.plot_group(df, col, ax=axes[0, 0], adjust=False, title=col)

col = utils.find_col(df, ["crp"])

utils.plot_group(df, col, ax=axes[0, 1], adjust=False, title=col)

col = utils.find_col(df, ["total", "lac"])

pu.plot_correlation(
    xs=df[df["measure"] == "0w"][col],
    ys=hamd_delta,
    ax=axes[1, 0],
    xlabel=f"{col} (0W)",
    ylabel=f"HAMD (12W - 0W)",
    font_src=font_src,
)

pu.plot_correlation(
    xs=auc,
    ys=hamd_delta,
    ax=axes[1, 1],
    xlabel=f"{col} (AUC)",
    ylabel=f"HAMD (12W - 0W)",
    font_src=font_src,
)


plt.tight_layout()

f.savefig(
    f"charts/0.png", facecolor="w",
)

plt.close(f)


##################################################


# max_col_length = max([len(col) for col in df.columns])
#
# time_significant_features_multiple = []
# time_significant_features_pair = []
#
# report_multiple = []
# report_pair_bm = []
# report_pair_bf = []
# report_pair_both = []
#
# for col in df.columns:
#    if col in drops:
#        continue
#
#    p = utils.significant_test_multiple(df, col)
#
#    if p <= 0.05:
#        time_significant_features_multiple.append(col)
#        report_multiple.append([col, p])
#
#    bmp, bfp = utils.significant_test_pair(df, col)
#
#    if bmp <= 0.05:
#        report_pair_bm.append([col, bmp])
#    if bfp <= 0.05:
#        report_pair_bf.append([col, bfp])
#
#    if bmp <= 0.05 and bfp <= 0.05:
#        report_pair_both.append([col, bmp, bfp])
#
#    if bmp <= 0.05 or bfp <= 0.05:
#        time_significant_features_pair.append(col)


# report_multiple.sort(key=lambda x: x[1])
# report_pair_bm.sort(key=lambda x: x[1])
# report_pair_bf.sort(key=lambda x: x[1])

##################################################

mri = pd.read_csv("mri.csv")
mri = mri.drop("No.", axis=1)

##################################################

if REMOVE_OUTLIERS:
    print("REMOVING OUTLIERS.....")
    for col in mri.columns:
        if col in ["ID"]:
            continue

        utils.remove_outliers_dataframe(mri, col)

##################################################

mri

##################################################

mri.loc[:, "delta"] = mri["MRI_Neutral-2"] - mri["MRI_Neutral-1"]
mri.head()

##################################################

mask = ~np.isnan(mri["MRI_Neutral-2"]) & ~np.isnan(mri["MRI_Neutral-1"])
_, p_mri = stats.wilcoxon(
    # mri["MRI_Neutral-2"], mri["MRI_Neutral-1"]
    mri.loc[mask, "MRI_Neutral-2"],
    mri.loc[mask, "MRI_Neutral-1"],
)
p_mri

##################################################

# wilcoxon_bonferroni_report = []
wilcoxon_report = []

if p_mri <= 0.05:
    wilcoxon_report.append(["MRI_Neutral", p_mri])
    # wilcoxon_bonferroni_report.append(["MRI_Neutral", p_mri])

for col in df.columns:
    if col in drops:
        continue

    # bmp, bfp = utils.significant_test_pair(df, col, adjust=True)
    # if bfp <= 0.05:
    # wilcoxon_bonferroni_report.append([col, bfp])

    bmp, bfp = utils.significant_test_pair(df, col, adjust=False)
    if bfp <= 0.05:
        wilcoxon_report.append([col, bfp])

##################################################

# utils.write_time_significant_report(
# "Wilcoxon_Bonferroni_0W_vs_12W.txt",
# "Wilcoxon X Bonferroni (0W vs 12W)",
# wilcoxon_bonferroni_report,
# )

utils.write_time_significant_report(
    f"{output_root}/reports/Wilcoxon_0W_vs_12W.txt",
    "Wilcoxon (0W vs 12W)",
    wilcoxon_report,
)

# utils.plot_group_set(
# df,
# [r[0] for r in wilcoxon_bonferroni_report],
# "charts/Wilcoxon_Bonferroni/Wilcoxon_Bonferroni_0W_vs_12W",
# )

utils.plot_group_set(
    df,
    cols=[r[0] for r in wilcoxon_report],
    output=f"{output_root}/charts/Wilcoxon_0W_vs_12W",
    adjust=False,
)

##################################################


additional_cols = [
    utils.find_col(df, ["coccoides"]),
    utils.find_col(df, ["leptum"]),
    utils.find_col(df, ["b", "fragilis"]),
    utils.find_col(df, ["bifidobacte"]),
    utils.find_col(df, ["atopobium"]),
    utils.find_col(df, ["prevotella"]),
    utils.find_col(df, ["total", "lacto"]),
    utils.find_col(df, ["enterob"]),
    utils.find_col(df, ["enterococcu"]),
    utils.find_col(df, ["staphylo"]),
    utils.find_col(df, ["c", "difficile"]),
    utils.find_col(df, ["c", "perfri"]),
    utils.find_col(df, ["reuteri"]),
]

additional_cols.sort(key=lambda x: list(df.columns).index(x))
additional_cols

##################################################

significant_cols = set(r[0] for r in wilcoxon_report).union(set(additional_cols))
significant_cols = list(significant_cols)
significant_cols.sort(key=lambda x: list(df.columns).index(x))
significant_cols

##################################################

# col = utils.find_col(df, ["total", "lacto"])


r = 0
c = 0

f = None
plotted = 0

i = 0

num_rows = 3
num_cols = 2

plotted_page = 0
plotted_chart = 0

output = f"{output_root}/charts/MRI_Neutral"

font_src = "fonts/Kosugi/Kosugi-Regular.ttf"

if not os.path.exists(output):
    os.makedirs(output, exist_ok=True)


for col in additional_cols:
    mask = df["ID"].isin(mri["ID"])

    xs = df[mask & (df["measure"] == "12w")][col].reset_index(drop=True) - df[
        mask & (df["measure"] == "0w")
    ][col].reset_index(drop=True)

    ys = mri["delta"].reset_index(drop=True)

    mask = ~np.isnan(xs) & ~np.isnan(ys)

    tau, p = stats.kendalltau(
        # xs.astype(np.float), ys.astype(np.float), nan_policy="omit"
        xs[mask].astype(np.float),
        ys[mask].astype(np.float),
    )

    if p > 0.05 or abs(tau) < 0.2:
        continue
    if math.isnan(p) or math.isnan(tau):
        continue

    if i % int(num_rows * num_cols) == 0:
        if i != 0:
            print(f"save at {i}")
            plt.tight_layout()
            f.savefig(
                os.path.join(output, f"{os.path.basename(output)}_{plotted}.png",),
                facecolor="w",
            )
            plt.close(f)

            plotted_page += 1

        f, axes = plt.subplots(num_rows, num_cols, figsize=(20, 30))
        r = 0
        c = 0

    pu.plot_correlation(
        xs, ys, ax=axes[r, c], xlabel=col, ylabel="MRI_Neutral", font_src=font_src
    )

    plotted_chart += 1

    c += 1
    if c % num_cols == 0:
        r += 1
        c = 0

    i += 1

if plotted_chart > (num_rows * num_cols) * plotted_page:
    plt.tight_layout()
    f.savefig(
        os.path.join(output, f"{os.path.basename(output)}_{plotted}.png",),
        facecolor="w",
    )
    plt.close(f)
    plotted_page += 1

print(f"total {plotted_chart} charts, {plotted_page} pages")

##################################################


# significant_cols = set(r[0] for r in report_multiple).intersection(
#     set(r[0] for r in report_pair_bm).union(set(r[0] for r in report_pair_bf))
# )
#
# significant_cols = significant_cols.union(set(additional_cols))
#
# targets = []
#
# for col in df.columns:
#     if col in significant_cols:
#         targets.append(col)
#
# significant_cols = targets


##################################################

# utils.write_friedman_wilcoxon_report(
#     file_name="Friedman_X_Wilcoxon_0W_vs_6W_report.txt",
#     title="Friedman X Wilcoxon(0W vs 6W)",
#     friedman_report=report_multiple,
#     wilcoxon_report=report_pair_bm,
# )
#
# utils.write_friedman_wilcoxon_report(
#     file_name="Friedman_X_Wilcoxon_0W_vs_12W_report.txt",
#     title="Friedman X Wilcoxon(0W vs 12W)",
#     friedman_report=report_multiple,
#     wilcoxon_report=report_pair_bf,
# )

##################################################

# utils.bake_correlation_statistic(
#     df,
#     drops,
#     lambda df, x, y: utils.correlation(df, x, y, delta=None),
#     "statistic_simple.json",
# )
#
# utils.bake_correlation_statistic(
#     df,
#     drops,
#     lambda df, x, y: utils.correlation(df, x, y, delta="fb"),
#     "statistic_delta_12W.json",
# )
#
# utils.bake_correlation_statistic(
#     df,
#     drops,
#     lambda df, x, y: utils.correlation(df, x, y, delta="mb"),
#     "statistic_delta_6W.json",
# )
#
# utils.bake_correlation_statistic(
#     df,
#     drops,
#     lambda df, x, y: utils.correlation(df, x, y, delta="mbfb"),
#     "statistic_delta_6W_12W.json",
# )
#
# utils.bake_correlation_statistic(
#     df,
#     drops,
#     lambda df, x, y: utils.correlation(df, x, y, delta="bbmbfb"),
#     "statistic_delta_0W_6W_12W.json",
# )

##################################################

# delta_corr_6 = {}
delta_corr_12 = {}
# delta_corr_6_12 = {}
# delta_corr_0_6_12 = {}

# repeated = set()

p_threshold = 0.05
tau_threshold = 0.2

for colx in df.columns:
    if colx in drops:
        continue

    if colx not in significant_cols:
        continue

    # if delta_corr_6.get(colx, None) is None:
    # delta_corr_6[colx] = {}

    if delta_corr_12.get(colx, None) is None:
        delta_corr_12[colx] = {}

    # if delta_corr_6_12.get(colx, None) is None:
    # delta_corr_6_12[colx] = {}

    # if delta_corr_0_6_12.get(colx, None) is None:
    # delta_corr_0_6_12[colx] = {}

    for coly in df.columns:

        if coly in drops:
            continue

        if coly not in significant_cols:
            continue

        if coly == colx:
            continue

        # if f"{colx}_{coly}" in repeated or f"{coly}_{colx}" in repeated:
        # continue

        # tau, p = utils.correlation(df, colx, coly, delta="mb")
        # if p <= p_threshold and abs(tau) >= tau_threshold:
        # delta_corr_6[colx][coly] = {
        # "p": p,
        # "tau": tau,
        # }

        tau, p = utils.correlation(df, colx, coly, delta="fb")
        if p <= p_threshold and abs(tau) >= tau_threshold:
            delta_corr_12[colx][coly] = {
                "p": p,
                "tau": tau,
            }

        # tau, p = utils.correlation(df, colx, coly, delta="mbfb")
        # if p <= p_threshold and abs(tau) >= tau_threshold:
        # delta_corr_6_12[colx][coly] = {
        # "p": p,
        # "tau": tau,
        # }

        # tau, p = utils.correlation(df, colx, coly, delta="bbmbfb")
        # if p <= p_threshold and abs(tau) >= tau_threshold:
        # delta_corr_0_6_12[colx][coly] = {
        # "p": p,
        # "tau": tau,
        # }

##################################################

delta_corr_bfb = []

for col in df.columns:
    if col in drops:
        continue

    if col not in significant_cols:
        continue

    tau, p = utils.correlation(df, col, col, delta="bfb")
    if p <= p_threshold and abs(tau) >= tau_threshold:
        delta_corr_bfb.append([col, p, tau])

##################################################


def write_kendall_report(filepath, title, kendall_report):
    spaces = "  "
    multiplier = 2

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{title}: {len(kendall_report)} sets\n")
        f.write("\n")
        for r in kendall_report:
            f.write(f"{r[0].strip()}\n")
            f.write(f"{spaces * multiplier * 1}P:   {r[1]:.19f}\n")
            f.write(f"{spaces * multiplier * 1}TAU: {r[2]:.19f}\n")
            f.write("\n\n")


##################################################


write_kendall_report(
    f"{output_root}/reports/Kendall_Delta_0W_VS_12W_0W.txt",
    "Kendall (0W, 12W - 0W)",
    delta_corr_bfb,
)

##################################################


# utils.write_kendall_delta_report(
# "Kendall_Delta_6W.txt", "Kendall (6W - 0W)", delta_corr_6
# )

utils.write_kendall_delta_report(
    f"{output_root}/reports/Kendall_Delta_12W.txt", "Kendall (12W - 0W)", delta_corr_12
)

# utils.write_kendall_delta_report(
# "Kendall_Delta_6W_12W.txt", "Kendall (6W - 0W + 12W - 0W)", delta_corr_6_12
# )

# utils.write_kendall_delta_report(
# "Kendall_Delta_0W_6W_12W.txt",
# "Kendall (0W - 0W + 6W - 0W + 12W - 0W)",
# delta_corr_0_6_12,
# )

##################################################

# intersection = set(r[0] for r in report_multiple).intersection(
#     set(r[0] for r in report_pair_bm)
# )
#
# intersection = utils.make_intersection(df, intersection)
#
# utils.plot_group_set(
#     df,
#     cols=intersection,
#     output="charts/Friedman_X_Wilcoxon/Friedman_X_Wilcoxon_0W_vs_6W",
# )
#
#
# intersection = set(r[0] for r in report_multiple).intersection(
#     set(r[0] for r in report_pair_bf)
# )
#
# intersection = utils.make_intersection(df, intersection)
#
#
# utils.plot_group_set(
#     df,
#     cols=intersection,
#     output="charts/Friedman_X_Wilcoxon/Friedman_X_Wilcoxon_0W_vs_12W",
#     adjust=False,
# )

##################################################

# utils.plot_correlation_set(
# df,
# colxs=significant_cols,
# colys=significant_cols,
# delta="mb",
# output="charts/Kendall_Delta_6W",
# same_column=False,
# )

utils.plot_correlation_set(
    df,
    colxs=significant_cols,
    colys=significant_cols,
    delta="fb",
    output=f"{output_root}/charts/Kendall_Delta_12W",
    same_column=False,
)

##################################################


utils.plot_correlation_set(
    df,
    colxs=significant_cols,
    colys=significant_cols,
    delta="bfb",
    output=f"{output_root}/charts/Kendall_Delta_0W_VS_12W_0W",
    same_column=True,
)

##################################################

# plt.close()
# utils.plot_group(df, "Staphylococcus", title="Staphylococcus")
# plt.show()


##################################################


# print(
# df[df["measure"] == "12w"]["Staphylococcus"].reset_index(drop=True)
# - df[df["measure"] == "0w"]["Staphylococcus"].reset_index(drop=True)
# )

# print(
# df[df["measure"] == "12w"]["HAMD"].reset_index(drop=True)
# - df[df["measure"] == "0w"]["HAMD"].reset_index(drop=True)
# )

##################################################
