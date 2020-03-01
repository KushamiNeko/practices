##################################################

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

import utils

from preprocess import clean_data, test_data_na


matplotlib.use("gtk3agg")


##################################################

df = pd.read_csv("data.csv")

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

df = base_set.append(middle_set).append(final_set)
df = df.reset_index(drop=True)


##################################################


max_col_length = max([len(col) for col in df.columns])

time_significant_features_multiple = []
time_significant_features_pair = []

drops = ["ID", "measure"]

report_multiple = []
report_pair_bm = []
report_pair_bf = []
report_pair_both = []

for col in df.columns:
    if col in drops:
        continue

    p = utils.significant_test_multiple(df, col)

    if p <= 0.05:
        time_significant_features_multiple.append(col)
        report_multiple.append([col, p])

    bmp, bfp = utils.significant_test_pair(df, col)
    if bmp <= 0.05:
        report_pair_bm.append([col, bmp])
    if bfp <= 0.05:
        report_pair_bf.append([col, bfp])

    if bmp <= 0.05 and bfp <= 0.05:
        report_pair_both.append([col, bmp, bfp])

    if bmp <= 0.05 or bfp <= 0.05:
        time_significant_features_pair.append(col)


# report_multiple.sort(key=lambda x: x[1])
# report_pair_bm.sort(key=lambda x: x[1])
# report_pair_bf.sort(key=lambda x: x[1])

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
]

additional_cols


significant_cols = set(r[0] for r in report_multiple).intersection(
    set(r[0] for r in report_pair_bm).union(set(r[0] for r in report_pair_bf))
)

significant_cols = significant_cols.union(set(additional_cols))

targets = []

for col in df.columns:
    if col in significant_cols:
        targets.append(col)

significant_cols = targets


##################################################


utils.bake_correlation_statistic(
    df,
    drops,
    lambda df, x, y: utils.correlation(df, x, y, delta=None),
    "statistic_simple.json",
)

utils.bake_correlation_statistic(
    df,
    drops,
    lambda df, x, y: utils.correlation(df, x, y, delta="fb"),
    "statistic_delta_12W.json",
)

utils.bake_correlation_statistic(
    df,
    drops,
    lambda df, x, y: utils.correlation(df, x, y, delta="mb"),
    "statistic_delta_6W.json",
)

utils.bake_correlation_statistic(
    df,
    drops,
    lambda df, x, y: utils.correlation(df, x, y, delta="mbfb"),
    "statistic_delta_6W_12W.json",
)

utils.bake_correlation_statistic(
    df,
    drops,
    lambda df, x, y: utils.correlation(df, x, y, delta="bbmbfb"),
    "statistic_delta_0W_6W_12W.json",
)

##################################################

delta_corr_6 = {}
delta_corr_12 = {}
delta_corr_6_12 = {}
delta_corr_0_6_12 = {}

# repeated = set()

p_threshold = 0.05
tau_threshold = 0.2

for colx in df.columns:
    if colx in drops:
        continue

    if colx not in significant_cols:
        continue

    if delta_corr_6.get(colx, None) is None:
        delta_corr_6[colx] = {}

    if delta_corr_12.get(colx, None) is None:
        delta_corr_12[colx] = {}

    if delta_corr_6_12.get(colx, None) is None:
        delta_corr_6_12[colx] = {}

    if delta_corr_0_6_12.get(colx, None) is None:
        delta_corr_0_6_12[colx] = {}

    for coly in df.columns:

        if coly in drops:
            continue

        if coly not in significant_cols:
            continue

        if coly == colx:
            continue

        # if f"{colx}_{coly}" in repeated or f"{coly}_{colx}" in repeated:
        # continue

        tau, p = utils.correlation(df, colx, coly, delta="mb")
        if p <= p_threshold and abs(tau) >= tau_threshold:
            delta_corr_6[colx][coly] = {
                "p": p,
                "tau": tau,
            }

        tau, p = utils.correlation(df, colx, coly, delta="fb")
        if p <= p_threshold and abs(tau) >= tau_threshold:
            delta_corr_12[colx][coly] = {
                "p": p,
                "tau": tau,
            }

        tau, p = utils.correlation(df, colx, coly, delta="mbfb")
        if p <= p_threshold and abs(tau) >= tau_threshold:
            delta_corr_6_12[colx][coly] = {
                "p": p,
                "tau": tau,
            }

        tau, p = utils.correlation(df, colx, coly, delta="bbmbfb")
        if p <= p_threshold and abs(tau) >= tau_threshold:
            delta_corr_0_6_12[colx][coly] = {
                "p": p,
                "tau": tau,
            }

##################################################


utils.write_kendall_delta_report(
    "Kendall_Delta_6W.txt", "Kendall (6W - 0W)", delta_corr_6
)
utils.write_kendall_delta_report(
    "Kendall_Delta_12W.txt", "Kendall (12W - 0W)", delta_corr_12
)

utils.write_kendall_delta_report(
    "Kendall_Delta_6W_12W.txt", "Kendall (6W - 0W + 12W - 0W)", delta_corr_6_12
)

utils.write_kendall_delta_report(
    "Kendall_Delta_0W_6W_12W.txt",
    "Kendall (0W - 0W + 6W - 0W + 12W - 0W)",
    delta_corr_0_6_12,
)

##################################################


utils.plot_correlation_set(df, "charts/Kendall_Delta_6W", significant_cols, delta="mb")
utils.plot_correlation_set(df, "charts/Kendall_Delta_12W", significant_cols, delta="fb")
utils.plot_correlation_set(
    df, "charts/Kendall_Delta_6W_12W", significant_cols, delta="mbfb"
)
utils.plot_correlation_set(
    df, "charts/Kendall_Delta_0W_6W_12W", significant_cols, delta="bbmbfb"
)

##################################################

plt.close()
utils.plot_group(df, "Staphylococcus", title="Staphylococcus")
plt.show()


##################################################


print(
    df[df["measure"] == "12w"]["Staphylococcus"].reset_index(drop=True)
    - df[df["measure"] == "0w"]["Staphylococcus"].reset_index(drop=True)
)

print(
    df[df["measure"] == "12w"]["HAMD"].reset_index(drop=True)
    - df[df["measure"] == "0w"]["HAMD"].reset_index(drop=True)
)

##################################################
