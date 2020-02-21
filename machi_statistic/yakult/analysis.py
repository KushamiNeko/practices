# %%
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import font_manager as fm
from scipy import linalg
from statsmodels.stats import multitest

from preprocess import clean_data, test_data_na

# %%
df = pd.read_csv("data.csv")

df = clean_data(df)
test_data_na(df)


# %%


def test_column_pair(cols):
    assert len(cols) == 3
    for j in cols:
        for k in cols:
            assert j.startswith(k[:3])


def drop_append(cols, drop_1, drop_2, drop_3):
    drop_1.append(cols[1])
    drop_1.append(cols[2])

    drop_2.append(cols[0])
    drop_2.append(cols[2])

    drop_3.append(cols[0])
    drop_3.append(cols[1])


drop_1 = []
drop_2 = []
drop_3 = []

cols = []

# start of the 3-set columns
# df.columns[17]

for i, col in enumerate(df.columns[17:]):
    if i != 0 and i % 3 == 0:
        test_column_pair(cols)
        drop_append(cols, drop_1, drop_2, drop_3)

        cols = [col]
    else:
        cols.append(col)

test_column_pair(cols)
drop_append(cols, drop_1, drop_2, drop_3)

base_set = df.drop(drop_1, axis=1)
middle_set = df.drop(drop_2, axis=1)
final_set = df.drop(drop_3, axis=1)


def rename_columns(data):
    regex = re.compile(r"(^.+)([-_]\d{1})(.*)$", re.MULTILINE)
    new_cols = []

    for col in data.columns:
        match = regex.match(col)
        if match is not None:
            new_cols.append(f"{match.group(1)}{match.group(3)}")
        else:
            new_cols.append(col)

    data.columns = new_cols


rename_columns(base_set)
rename_columns(middle_set)
rename_columns(final_set)

# %%

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


# %%


def significant_test_pair(col):

    try:
        _, bmp = stats.wilcoxon(
            base_set[col].astype(np.float), middle_set[col].astype(np.float),
        )
    except ValueError as e:
        print(f"{col}: {e}")
        bmp = np.nan

    try:
        _, bfp = stats.wilcoxon(
            base_set[col].astype(np.float), final_set[col].astype(np.float),
        )
    except ValueError as e:
        print(f"{col}: {e}")
        bfp = np.nan

    try:
        _, mfp = stats.wilcoxon(
            middle_set[col].astype(np.float), final_set[col].astype(np.float),
        )
    except ValueError as e:
        print(f"{col}: {e}")
        mfp = np.nan

    _, adj_p, _, _ = multitest.multipletests(
        pvals=[bmp, bfp, mfp], alpha=0.05, method="bonferroni",
    )

    return adj_p[0], adj_p[1]


def significant_test_multiple(col):
    _, p = stats.friedmanchisquare(
        base_set[col].astype(np.float),
        middle_set[col].astype(np.float),
        final_set[col].astype(np.float),
    )

    return p


def correlation(x, y, calculate_middle=True):
    if calculate_middle:
        xs = base_set[x].append(middle_set[x]).append(final_set[x])
        ys = base_set[y].append(middle_set[y]).append(final_set[y])
    else:
        xs = base_set[x].append(final_set[x])
        ys = base_set[y].append(final_set[y])

    tau, p = stats.kendalltau(xs.values, ys.values)

    return tau, p


def partial_correlation(cols, calculate_middle=True):
    if calculate_middle:
        cs = base_set[cols].append(middle_set[cols]).append(final_set[cols])
    else:
        cs = base_set[cols].append(final_set[cols])

    return partial_corr(cs)


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


# %%

max_col_length = max([len(col) for col in base_set.columns])

time_significant_features_multiple = []
time_significant_features_pair = []

drops = ["ID"]

report_multiple = []
report_pair_bm = []
report_pair_bf = []

for col in base_set.columns:
    if col in drops:
        continue

    p = significant_test_multiple(col)

    if p <= 0.05:
        time_significant_features_multiple.append(col)

        report_multiple.append([col, p])

    bmp, bfp = significant_test_pair(col)
    if bmp <= 0.05:
        report_pair_bm.append([col, bmp])
    if bfp <= 0.05:
        report_pair_bf.append([col, bfp])
    if bmp <= 0.05 or bfp <= 0.05:
        time_significant_features_pair.append(col)


report_multiple.sort(key=lambda x: x[1])
report_pair_bm.sort(key=lambda x: x[1])
report_pair_bf.sort(key=lambda x: x[1])

# %%

corr_significant_features = []
corr_significant_features_strict = []

repeated = set()

for x in base_set.drop(drops, axis=1).columns:
    for y in base_set.drop(drops, axis=1).columns:

        if x == y:
            continue

        if f"{x}_{y}" in repeated or f"{y}_{x}" in repeated:
            continue

        tau, p = correlation(x, y)
        if p <= 0.05:
            corr_significant_features.append((x, y, p, tau))
            if abs(tau) >= 0.2:
                corr_significant_features_strict.append((x, y, p, tau))

        repeated.add(f"{x}_{y}")

corr_significant_features.sort(key=lambda x: x[2])
corr_significant_features_strict.sort(key=lambda x: x[2])

# %%


def write_time_significant_report(file_name, title, report):
    with open(f"reports/{file_name}", "w", encoding="utf-8") as f:
        f.write(f"{title}: {len(report)} features\n")
        f.write("\n")
        for r in report:
            f.write(f"{r[0].strip()}\n")
            f.write(f"{r[1]:.30f}\n")
            f.write("\n")


write_time_significant_report(
    "Friedman_report.txt", "Friedman Test", report_multiple,
)

write_time_significant_report(
    "Wilcoxon_0W_vs_6W_report.txt", "Wilcoxon Test(0W vs 6W)", report_pair_bm,
)

write_time_significant_report(
    "Wilcoxon_0W_vs_12W_report.txt", "Wilcoxon Test(0W vs 12W)", report_pair_bf,
)


def write_friedman_wilcoxon_report(file_name, title, friedman_report, wilcoxon_report):
    with open(f"reports/{file_name}", "w", encoding="utf-8",) as f:
        intersection = set(r[0] for r in friedman_report).intersection(
            set(r[0] for r in wilcoxon_report)
        )

        f.write(f"{title}: {len(intersection)} features\n")
        f.write("\n")
        for r in friedman_report:
            if r[0].strip() in intersection:
                f.write(f"{r[0].strip()}\n")
                f.write(f"  Friedman P: {r[1]:.30f}\n")
            for jr in wilcoxon_report:
                if jr[0].strip() == r[0].strip():
                    f.write(f"  Wilcoxon P: {jr[1]:.30f}\n")
                    f.write("\n")


write_friedman_wilcoxon_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_6W_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 6W)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bm,
)

write_friedman_wilcoxon_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_12W_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 12W)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bf,
)


def write_kendall_report(file_name, title, kendall_report):
    with open(f"reports/{file_name}", "w", encoding="utf-8") as f:
        f.write(f"{title}: {len(kendall_report)} sets\n")
        f.write("\n")
        for r in kendall_report:
            f.write(f"{r[0].strip()}\n")
            f.write(f"{r[1].strip()}\n")
            f.write(f"  P:   {r[2]:.30f}\n")
            f.write(f"  TAU: {r[3]:.30f}\n")
            f.write("\n")


write_kendall_report(
    "Kendall_report.txt", "Kendall Correlation(P <= 0.05)", corr_significant_features
)

write_kendall_report(
    "Kendall_strict_report.txt",
    "Kendall Correlation(P <= 0.05 X |TAU| >= 0.2)",
    corr_significant_features_strict,
)

# %%


def write_friedman_wilcoxon_kendall_report(
    file_name, title, friedman_report, wilcoxon_report, kendall_report
):
    with open(f"reports/{file_name}", "w", encoding="utf-8") as f:
        intersection = (
            set(r[0] for r in friedman_report)
            .intersection(set(r[0] for r in wilcoxon_report))
            .intersection(
                set(r[0] for r in kendall_report).union(
                    set(r[1] for r in kendall_report)
                )
            )
        )

        f.write(f"{title}: {len(intersection)} sets\n")
        f.write("\n")
        for r in friedman_report:
            if r[0].strip() in intersection:
                f.write(f"{r[0].strip()}\n")
                f.write(f"  Friedman P: {r[1]:.30f}\n")

                for jr in wilcoxon_report:
                    if jr[0].strip() == r[0].strip():
                        f.write(f"  Wilcoxon P: {jr[1]:.30f}\n")

                f.write("\n")

                for jr in kendall_report:
                    if jr[0].strip() == r[0].strip() or jr[1].strip() == r[0].strip():
                        if jr[0].strip() == r[0].strip():
                            if jr[1] not in intersection:
                                continue

                            f.write(f"    {jr[1].strip()}\n")
                        else:
                            if jr[0] not in intersection:
                                continue

                            f.write(f"    {jr[0].strip()}\n")

                        f.write(f"      Kendall P:   {jr[2]:.30f}\n")
                        f.write(f"      Kendall TAU: {jr[3]:.30f}\n")
                        f.write("\n")

                f.write("\n")


write_friedman_wilcoxon_kendall_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_6W_X_Kendall_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 6W) X Kendall Correlation(P <= 0.05)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bm,
    kendall_report=corr_significant_features,
)

write_friedman_wilcoxon_kendall_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_6W_X_Kendall_strict_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 6W) X Kendall Correlation(P <= 0.05 X |TAU| >= 0.2)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bm,
    kendall_report=corr_significant_features_strict,
)

write_friedman_wilcoxon_kendall_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_12W_X_Kendall_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 12W) X Kendall Correlation(P <= 0.05)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bf,
    kendall_report=corr_significant_features,
)

write_friedman_wilcoxon_kendall_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_12W_X_Kendall_strict_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 12W) X Kendall Correlation(P <= 0.05 X |TAU| >= 0.2)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bf,
    kendall_report=corr_significant_features_strict,
)


def write_friedman_kendall_report(file_name, title, friedman_report, kendall_report):
    with open(f"reports/{file_name}", "w", encoding="utf-8") as f:
        intersection = set(r[0] for r in friedman_report).intersection(
            set(r[0] for r in kendall_report).union(set(r[1] for r in kendall_report))
        )

        f.write(f"{title}: {len(intersection)} sets\n")
        f.write("\n")
        for r in friedman_report:
            if r[0].strip() in intersection:
                f.write(f"{r[0].strip()}\n")
                f.write(f"  Friedman P: {r[1]:.30f}\n")

                f.write("\n")

                for jr in kendall_report:
                    if jr[0].strip() == r[0].strip() or jr[1].strip() == r[0].strip():
                        if jr[0].strip() == r[0].strip():
                            if jr[1] not in intersection:
                                continue

                            f.write(f"    {jr[1].strip()}\n")
                        else:
                            if jr[0] not in intersection:
                                continue

                            f.write(f"    {jr[0].strip()}\n")

                        f.write(f"      Kendall P:   {jr[2]:.30f}\n")
                        f.write(f"      Kendall TAU: {jr[3]:.30f}\n")
                        f.write("\n")

                f.write("\n")


write_friedman_kendall_report(
    file_name="Friedman_X_Kendall_report.txt",
    title="Friedman Test X Kendall Correlation(P <= 0.05)",
    friedman_report=report_multiple,
    kendall_report=corr_significant_features,
)

write_friedman_kendall_report(
    file_name="Friedman_X_Kendall_strict_report.txt",
    title="Friedman Test X Kendall Correlation(P <= 0.05 X |TAU| >= 0.2)",
    friedman_report=report_multiple,
    kendall_report=corr_significant_features_strict,
)


# %%

with open("reports/significant_summary.txt", "w") as f:
    spaces = "  "
    new_lines = "\n\n\n"

    f.write(f"Friedman: {len(report_multiple)} sets\n")
    for r in report_multiple:
        f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    f.write(f"Wilcoxon(0W vs 6W): {len(report_pair_bm)} sets\n")
    for r in report_pair_bm:
        f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    f.write(f"Wilcoxon(0W vs 12W): {len(report_pair_bf)} sets\n")
    for r in report_pair_bf:
        f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = set(r[0] for r in report_multiple).intersection(
        set(r[0] for r in report_pair_bm)
    )
    f.write(f"Friedman X Wilcoxon(0W vs 6W): {len(intersection)} sets\n")
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = set(r[0] for r in report_multiple).intersection(
        set(r[0] for r in report_pair_bf)
    )
    f.write(f"Friedman X Wilcoxon(0W vs 12W): {len(intersection)} sets\n")
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = set(r[0] for r in report_multiple).intersection(
        set(r[0] for r in report_pair_bm).union(set(r[0] for r in report_pair_bf))
    )
    f.write(f"Friedman X Wilcoxon(0W vs 6W + 0W vs 12W): {len(intersection)} sets\n")
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = set(r[0] for r in report_multiple).intersection(
        set(r[0] for r in corr_significant_features).union(
            set(r[1] for r in corr_significant_features)
        )
    )
    f.write(f"Friedman X Kendall(P <= 0.05): {len(intersection)} sets\n")
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = set(r[0] for r in report_multiple).intersection(
        set(r[0] for r in corr_significant_features_strict).union(
            set(r[1] for r in corr_significant_features_strict)
        )
    )
    f.write(f"Friedman X Kendall(P <= 0.05 X |TAU| >= 0.2): {len(intersection)} sets\n")
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = (
        set(r[0] for r in report_multiple)
        .intersection(set(r[0] for r in report_pair_bm))
        .intersection(
            set(r[0] for r in corr_significant_features).union(
                set(r[1] for r in corr_significant_features)
            )
        )
    )
    f.write(
        f"Friedman X Wilcoxon(0W vs 6W) X Kendall(P <= 0.05): {len(intersection)} sets\n"
    )
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = (
        set(r[0] for r in report_multiple)
        .intersection(set(r[0] for r in report_pair_bm))
        .intersection(
            set(r[0] for r in corr_significant_features_strict).union(
                set(r[1] for r in corr_significant_features_strict)
            )
        )
    )
    f.write(
        f"Friedman X Wilcoxon(0W vs 6W) X Kendall(P <= 0.05 X |TAU| >= 0.2): {len(intersection)} sets\n"
    )
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = (
        set(r[0] for r in report_multiple)
        .intersection(set(r[0] for r in report_pair_bf))
        .intersection(
            set(r[0] for r in corr_significant_features).union(
                set(r[1] for r in corr_significant_features)
            )
        )
    )
    f.write(
        f"Friedman X Wilcoxon(0W vs 12W) X Kendall(P <= 0.05): {len(intersection)} sets\n"
    )
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = (
        set(r[0] for r in report_multiple)
        .intersection(set(r[0] for r in report_pair_bf))
        .intersection(
            set(r[0] for r in corr_significant_features_strict).union(
                set(r[1] for r in corr_significant_features_strict)
            )
        )
    )
    f.write(
        f"Friedman X Wilcoxon(0W vs 12W) X Kendall(P <= 0.05 X |TAU| >= 0.2): {len(intersection)} sets\n"
    )
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = (
        set(r[0] for r in report_multiple)
        .intersection(
            set(r[0] for r in report_pair_bm).union(set(r[0] for r in report_pair_bf))
        )
        .intersection(
            set(r[0] for r in corr_significant_features).union(
                set(r[1] for r in corr_significant_features)
            )
        )
    )
    f.write(
        f"Friedman X Wilcoxon(0W vs 6W + 0W vs 12W) X Kendall(P <= 0.05): {len(intersection)} sets\n"
    )
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)

    intersection = (
        set(r[0] for r in report_multiple)
        .intersection(
            set(r[0] for r in report_pair_bm).union(set(r[0] for r in report_pair_bf))
        )
        .intersection(
            set(r[0] for r in corr_significant_features_strict).union(
                set(r[1] for r in corr_significant_features_strict)
            )
        )
    )
    f.write(
        f"Friedman X Wilcoxon(0W vs 6W + 0W vs 12W) X Kendall(P <= 0.05 X |TAU| >= 0.2): {len(intersection)} sets\n"
    )
    for r in report_multiple:
        if r[0] in intersection:
            f.write(f"{spaces}{r[0].strip()}\n")

    f.write(new_lines)


# %%


def plot_group(
    col,
    ax=None,
    point_size=6,
    linelength=0.7,
    linewidth=2,
    title="",
    title_len_limit=50,
    title_size=14,
):

    prop = fm.FontProperties(fname="fonts/Kosugi/Kosugi-Regular.ttf", size=title_size)

    vs = pd.DataFrame(
        {
            "0W": base_set[col].astype(np.float),
            "6W": middle_set[col].astype(np.float),
            "12W": final_set[col].astype(np.float),
        }
    )

    if title != "":
        if len(title) <= title_len_limit:
            plt.title(title, fontproperties=prop)
        else:
            plt.title(title[:title_len_limit] + "...", fontproperties=prop)

    if ax is None:
        plt.plot(
            [0 - (linelength / 2.0), 0 + (linelength / 2.0)],
            [vs["0W"].mean(), vs["0W"].mean()],
            color="k",
            linewidth=linewidth,
        )
        plt.plot(
            [1 - (linelength / 2.0), 1 + (linelength / 2.0)],
            [vs["6W"].mean(), vs["6W"].mean()],
            color="k",
            linewidth=linewidth,
        )
        plt.plot(
            [2 - (linelength / 2.0), 2 + (linelength / 2.0)],
            [vs["12W"].mean(), vs["12W"].mean()],
            color="k",
            linewidth=linewidth,
        )
        sns.swarmplot(data=vs, color="k", ax=ax, size=point_size)
    else:
        ax.plot(
            [0 - (linelength / 2.0), 0 + (linelength / 2.0)],
            [vs["0W"].mean(), vs["0W"].mean()],
            color="k",
            linewidth=linewidth,
        )
        ax.plot(
            [1 - (linelength / 2.0), 1 + (linelength / 2.0)],
            [vs["6W"].mean(), vs["6W"].mean()],
            color="k",
            linewidth=linewidth,
        )
        ax.plot(
            [2 - (linelength / 2.0), 2 + (linelength / 2.0)],
            [vs["12W"].mean(), vs["12W"].mean()],
            color="k",
            linewidth=linewidth,
        )
        sns.swarmplot(data=vs, color="k", size=point_size)


def plot_corr(
    colx, coly, ax=None, pointsize=20, linewidth=2, labelsize=14, label_len_limit=50
):
    prop = fm.FontProperties(fname="fonts/Kosugi/Kosugi-Regular.ttf", size=labelsize)

    xs = base_set[colx].append(middle_set[colx]).append(final_set[colx])
    ys = base_set[coly].append(middle_set[coly]).append(final_set[coly])

    s, i, _, _, _ = stats.linregress(xs.values, ys.values)
    xl = np.linspace(xs.min(), xs.max())

    if ax is None:
        plt.scatter(xs.values, ys.values, s=pointsize, color="k")
        plt.plot(xl, xl * s + i, color="k", linewidth=linewidth)

        if len(colx) > label_len_limit:
            plt.xlabel(colx[:label_len_limit] + "...", fontproperties=prop)
        else:
            plt.xlabel(colx, fontproperties=prop)

        if len(coly) > label_len_limit:
            plt.ylabel(coly[:label_len_limit] + "...", fontproperties=prop)
        else:
            plt.ylabel(coly, fontproperties=prop)

    else:
        ax.scatter(xs.values, ys.values, s=pointsize, color="k")
        ax.plot(xl, xl * s + i, color="k", linewidth=linewidth)

        if len(colx) > label_len_limit:
            ax.set_xlabel(colx[:label_len_limit] + "...", fontproperties=prop)
        else:
            ax.set_xlabel(colx, fontproperties=prop)

        if len(coly) > label_len_limit:
            ax.set_ylabel(coly[:label_len_limit] + "...", fontproperties=prop)
        else:
            ax.set_ylabel(coly, fontproperties=prop)


# %%

# statistic = {}

# progress = 0.0
# works = float(len(base_set.columns) * len(base_set.columns))

# progress_report = 2

# for x in base_set.columns:

#     if statistic.get(x, None) is None:
#         statistic[x] = {}

#     for y in base_set.columns:

#         tau, p_value = correlation(x, y)

#         if math.isnan(tau) or math.isnan(p_value):
#             pass

#         progress += 1

#         if progress % progress_report == 0:
#             print("{}%.....".format(round((progress / works) * 100.0, 4)))

#         if statistic[x].get(y, None) is None:
#             statistic[x][y] = {}

#         statistic[x][y] = {
#             "p": str(p_value),
#             "tau": str(tau),
#         }


# with open("statistic_new.json", "w") as output:
#     json.dump(statistic, output, indent=2)


# %%

# plot_w("HAMD")
# plot_corr("Weight", "BMI")

intersection = (
    set(r[0] for r in report_multiple)
    .intersection(set(r[0] for r in report_pair_bf))
    .intersection(
        set(r[0] for r in corr_significant_features).union(
            set(r[1] for r in corr_significant_features)
        )
    )
)

for col in intersection:
    f, ax = plt.subplots(figsize=(10, 10))
    plot_group(col, ax=ax, title=col)

    plt.tight_layout()
    f.savefig(f"charts/group/{col.replace('/', '_')}.png", facecolor="w")
    plt.close(f)


repeated = set()
for colx in intersection:
    for coly in intersection:
        if colx == coly:
            continue

        if f"{colx}_{coly}" in repeated or f"{coly}_{colx}" in repeated:
            continue

        f, ax = plt.subplots(figsize=(10, 10))
        plot_corr(colx, coly, ax=ax, pointsize=35)

        plt.tight_layout()
        f.savefig(
            f"charts/corr/{colx.replace('/', '_')}_X_{coly.replace('/', '_')}.png",
            facecolor="w",
        )
        plt.close(f)

        repeated.add(f"{colx}_{coly}")

# %%

for r in set(r[0] for r in report_pair_bm):
    if r not in set(r[0] for r in report_multiple):
        print(r)

print()

for r in set(r[0] for r in report_pair_bf):
    if r not in set(r[0] for r in report_multiple):
        print(r)

# %%

# ndf = base_set.append(middle_set).append(final_set)
# ndf = ndf.reset_index(drop=True)
# ndf
# base_set["HAMD"]
# middle_set["HAMD"]
# final_set["HAMD"]
# %%
Y = [
    "HAMD",
    # "BDI",
]

significant_features = []
for col in base_set.columns:
    bmp, mfp, bfp = significant_test(col, method="friedman")
    if bfp <= 0.05 or bmp <= 0.05:
        print(col)
        if col not in significant_features:
            significant_features.append(col)

    bmp, mfp, bfp = significant_test(col, method="tr")
    if bfp <= 0.05 or bmp <= 0.05:
        print(col)
        if col not in significant_features:
            significant_features.append(col)


# %%


final_features = []
for feature in significant_features:
    for tau, p in kendall(feature, calculate_middle=True):
        # for tau, p in kendall(feature, calculate_middle=False):
        print(feature)
        print(f"p: {p}")
        print(f"tau: {tau}")
        print()

        if p <= 0.05:
            final_features.append(feature)


# %%


# %%

# for feature in final_features:
#     print(feature)

# len(final_features)
final_features

# plot_diff("k__Bacteria;p__Firmicutes;c__Bacilli;o__Turicibacterales", point_size=6)
# for col in base_set.columns:
#     f, axes = plt.subplots(figsize=(20, 16))
#     plot_diff(col, point_size=8)
#     plt.tight_layout()
#     f.savefig(f"pics/{col.replace('/', '_', -1)}.png", facecolor="w")
#     plt.close(f)


# %%

# num_rows = 3
num_rows = 4
# num_cols = 2
num_cols = 3

title_len_limit = 50

prop = fm.FontProperties(fname="fonts/Kosugi/Kosugi-Regular.ttf", size=14)

# f, axes = plt.subplots(num_rows, num_cols, figsize=(20, 25))
# f, axes = plt.subplots(num_rows, num_cols, figsize=(25, 25))
f, axes = plt.subplots(num_rows, num_cols, figsize=(25, 30))

plot_diff("HAMD", point_size=10, ax=axes[0, 0])
axes[0, 0].set_title("HAMD", fontproperties=prop)

plot_diff("BDI", point_size=10, ax=axes[0, 1])
axes[0, 1].set_title("BDI", fontproperties=prop)

# row = 1
# col = 0
row = 0
col = 2

# for feature in significant_features:
for feature in final_features:
    plot_diff(feature, point_size=8, ax=axes[row, col])

    if len(feature) <= title_len_limit:
        axes[row, col].set_title(feature, fontproperties=prop)
    else:
        axes[row, col].set_title(feature[:title_len_limit] + "...", fontproperties=prop)

    col += 1
    if col % num_cols == 0:
        col = 0
        row += 1

plt.tight_layout()
f.savefig("analysis.png", facecolor="w")

# %%
