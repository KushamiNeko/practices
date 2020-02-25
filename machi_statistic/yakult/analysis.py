# %%
import gc
import math
import os
import re
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# import pingouin
import seaborn as sns
from matplotlib import font_manager as fm
from preprocess import clean_data, test_data_na

# from scipy import linalg
from statsmodels.stats import multitest

matplotlib.use("gtk3agg")


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

base_set["measure"] = "0w"
middle_set["measure"] = "6w"
final_set["measure"] = "12w"

df = base_set.append(middle_set).append(final_set)
df = df.reset_index(drop=True)

df

# %%

# testing mulipletests bonferroni

col = "HAMD"

try:
    _, p_base_middle = stats.wilcoxon(
        df[df["measure"] == "0w"][col].astype(np.float),
        df[df["measure"] == "6w"][col].astype(np.float),
    )
except ValueError as e:
    print(f"{col}: {e}")
    p_base_middle = np.nan

try:
    _, p_base_final = stats.wilcoxon(
        df[df["measure"] == "0w"][col].astype(np.float),
        df[df["measure"] == "12w"][col].astype(np.float),
    )
except ValueError as e:
    print(f"{col}: {e}")
    p_base_final = np.nan

try:
    _, p_middle_final = stats.wilcoxon(
        df[df["measure"] == "6w"][col].astype(np.float),
        df[df["measure"] == "12w"][col].astype(np.float),
    )
except ValueError as e:
    print(f"{col}: {e}")
    p_middle_final = np.nan

_, adj_p_t, _, _ = multitest.multipletests(
    pvals=[p_base_middle, p_base_final],
    # pvals=[p_base_middle, p_base_final],
    alpha=0.05,
    method="bonferroni",
)

_, adj_p, _, _ = multitest.multipletests(
    pvals=[p_base_middle, p_base_final, p_middle_final],
    # pvals=[p_base_middle, p_base_final],
    alpha=0.05,
    method="bonferroni",
)

print(p_base_middle)
print(p_base_final)
print(adj_p_t)
print(adj_p)

# %%


def significant_test_pair(df, col):

    try:
        _, p_base_middle = stats.wilcoxon(
            df[df["measure"] == "0w"][col].astype(np.float),
            df[df["measure"] == "6w"][col].astype(np.float),
        )
    except ValueError as e:
        print(f"{col}: {e}")
        p_base_middle = np.nan

    try:
        _, p_base_final = stats.wilcoxon(
            df[df["measure"] == "0w"][col].astype(np.float),
            df[df["measure"] == "12w"][col].astype(np.float),
        )
    except ValueError as e:
        print(f"{col}: {e}")
        p_base_final = np.nan

    try:
        _, p_middle_final = stats.wilcoxon(
            df[df["measure"] == "6w"][col].astype(np.float),
            df[df["measure"] == "12w"][col].astype(np.float),
        )
    except ValueError as e:
        print(f"{col}: {e}")
        p_middle_final = np.nan

    _, adj_p, _, _ = multitest.multipletests(
        pvals=[p_base_middle, p_base_final, p_middle_final],
        # pvals=[p_base_middle, p_base_final],
        alpha=0.05,
        method="bonferroni",
    )

    return adj_p[0], adj_p[1]


def significant_test_multiple(df, col):
    _, p = stats.friedmanchisquare(
        df[df["measure"] == "0w"][col].astype(np.float),
        df[df["measure"] == "6w"][col].astype(np.float),
        df[df["measure"] == "12w"][col].astype(np.float),
    )

    return p


def correlation(df, x, y):
    tau, p = stats.kendalltau(df[x].astype(np.float), df[y].astype(np.float))
    return tau, p


# %%

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

    p = significant_test_multiple(df, col)

    if p <= 0.05:
        time_significant_features_multiple.append(col)
        report_multiple.append([col, p])

    bmp, bfp = significant_test_pair(df, col)
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

# %%

corr_significant_features = []
corr_significant_features_strict = []

repeated = set()

for x in df.drop(drops, axis=1).columns:
    for y in df.drop(drops, axis=1).columns:

        if x == y:
            continue

        if f"{x}_{y}" in repeated or f"{y}_{x}" in repeated:
            continue

        tau, p = correlation(df, x, y)
        if p <= 0.05:
            corr_significant_features.append((x, y, p, tau))
            if abs(tau) >= 0.2:
                corr_significant_features_strict.append((x, y, p, tau))

        repeated.add(f"{x}_{y}")

# corr_significant_features.sort(key=lambda x: x[2])
# corr_significant_features_strict.sort(key=lambda x: x[2])

# %%


def write_time_significant_report(file_name, title, report):
    with open(f"reports/{file_name}", "w", encoding="utf-8") as f:
        f.write(f"{title}: {len(report)} features\n")
        f.write("\n")
        for r in report:
            f.write(f"{r[0].strip()}\n")
            f.write(f"{r[1]:.19f}\n")
            f.write("\n")


def write_friedman_wilcoxon_report(file_name, title, friedman_report, wilcoxon_report):
    spaces = "  "
    multiplier = 2

    with open(f"reports/{file_name}", "w", encoding="utf-8",) as f:
        intersection = set(r[0] for r in friedman_report).intersection(
            set(r[0] for r in wilcoxon_report)
        )

        f.write(f"{title}: {len(intersection)} features\n")
        f.write("\n")
        for r in friedman_report:
            if r[0].strip() in intersection:
                f.write(f"{r[0].strip()}\n")
                # f.write(f"  Friedman P: {r[1]:.19f}\n")
            for jr in wilcoxon_report:
                if jr[0].strip() == r[0].strip():
                    if len(jr) == 2:
                        # f.write(f"  Wilcoxon P: {jr[1]:.19f}\n")
                        f.write(f"{spaces * multiplier * 1}P: {jr[1]:.19f}\n")
                        f.write("\n")
                    elif len(jr) == 3:
                        f.write(
                            f"{spaces * multiplier * 1}P(0W vs 6W):  {jr[1]:.19f}\n"
                        )
                        f.write(
                            f"{spaces * multiplier * 1}P(0W vs 12W): {jr[2]:.19f}\n"
                        )
                        f.write("\n")


write_friedman_wilcoxon_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_6W_report.txt",
    title="Friedman X Wilcoxon(0W vs 6W)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bm,
)

write_friedman_wilcoxon_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_12W_report.txt",
    title="Friedman X Wilcoxon(0W vs 12W)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bf,
)

write_friedman_wilcoxon_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_6W_X_0W_vs_12W_report.txt",
    title="Friedman X Wilcoxon(0W vs 6W X 0W vs 12W)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_both,
)


def write_kendall_report(file_name, title, kendall_report):
    with open(f"reports/{file_name}", "w", encoding="utf-8") as f:
        f.write(f"{title}: {len(kendall_report)} sets\n")
        f.write("\n")
        for r in kendall_report:
            f.write(f"{r[0].strip()}\n")
            f.write(f"{r[1].strip()}\n")
            f.write(f"  P:   {r[2]:.19f}\n")
            f.write(f"  TAU: {r[3]:.19f}\n")
            f.write("\n")


# %%


def write_friedman_wilcoxon_kendall_report(
    file_name, title, friedman_report, wilcoxon_report, kendall_report
):
    spaces = "  "
    multiplier = 2

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
                # f.write(f"  Friedman P: {r[1]:.19f}\n")

                for jr in wilcoxon_report:
                    if jr[0].strip() == r[0].strip():
                        # f.write(f"  Wilcoxon P: {jr[1]:.19f}\n")
                        if len(jr) == 2:
                            # f.write(f"  Wilcoxon P: {jr[1]:.19f}\n")
                            f.write(
                                f"{spaces * multiplier * 1}Wilcoxon P: {jr[1]:.19f}\n"
                            )
                            f.write("\n")
                        elif len(jr) == 3:
                            f.write(
                                f"{spaces * multiplier * 1}Wilcoxon P(0W vs 6W):  {jr[1]:.19f}\n"
                            )
                            f.write(
                                f"{spaces * multiplier * 1}Wilcoxon P(0W vs 12W): {jr[2]:.19f}\n"
                            )
                            f.write("\n")

                # f.write("\n")

                for jr in kendall_report:
                    if jr[0].strip() == r[0].strip() or jr[1].strip() == r[0].strip():
                        if jr[0].strip() == r[0].strip():
                            if jr[1] not in intersection:
                                continue

                            f.write(f"{spaces * multiplier * 2}{jr[1].strip()}\n")
                        else:
                            if jr[0] not in intersection:
                                continue

                            f.write(f"{spaces * multiplier * 2}{jr[0].strip()}\n")

                        f.write(f"{spaces * multiplier * 3}Kendall P:   {jr[2]:.19f}\n")
                        f.write(f"{spaces * multiplier * 3}Kendall TAU: {jr[3]:.19f}\n")
                        f.write("\n")

                f.write("\n")


write_friedman_wilcoxon_kendall_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_6W_X_Kendall_strict_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 6W) X Kendall Correlation(P <= 0.05 X |TAU| >= 0.2)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bm,
    kendall_report=corr_significant_features_strict,
)


write_friedman_wilcoxon_kendall_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_12W_X_Kendall_strict_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 12W) X Kendall Correlation(P <= 0.05 X |TAU| >= 0.2)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_bf,
    kendall_report=corr_significant_features_strict,
)

write_friedman_wilcoxon_kendall_report(
    file_name="Friedman_X_Wilcoxon_0W_vs_6W_X_0W_vs_12W_X_Kendall_strict_report.txt",
    title="Friedman Test X Wilcoxon(0W vs 6W X 0W vs 12W) X Kendall Correlation(P <= 0.05 X |TAU| >= 0.2)",
    friedman_report=report_multiple,
    wilcoxon_report=report_pair_both,
    kendall_report=corr_significant_features_strict,
)


def write_friedman_kendall_report(file_name, title, friedman_report, kendall_report):
    spaces = "  "
    multiplier = 2

    with open(f"reports/{file_name}", "w", encoding="utf-8") as f:
        intersection = set(r[0] for r in friedman_report).intersection(
            set(r[0] for r in kendall_report).union(set(r[1] for r in kendall_report))
        )

        f.write(f"{title}: {len(intersection)} sets\n")
        f.write("\n")
        for r in friedman_report:
            if r[0].strip() in intersection:
                f.write(f"{r[0].strip()}\n")
                f.write(f"{spaces * multiplier * 1}Friedman P: {r[1]:.19f}\n")

                f.write("\n")

                for jr in kendall_report:
                    if jr[0].strip() == r[0].strip() or jr[1].strip() == r[0].strip():
                        if jr[0].strip() == r[0].strip():
                            if jr[1] not in intersection:
                                continue

                            f.write(f"{spaces * multiplier * 2}{jr[1].strip()}\n")
                        else:
                            if jr[0] not in intersection:
                                continue

                            f.write(f"{spaces * multiplier * 2}{jr[0].strip()}\n")

                        f.write(f"{spaces * multiplier * 3}Kendall P:   {jr[2]:.19f}\n")
                        f.write(f"{spaces * multiplier * 3}Kendall TAU: {jr[3]:.19f}\n")
                        f.write("\n")

                f.write("\n")


# %%


def plot_group(
    df,
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
            "0W": df[df["measure"] == "0w"][col].astype(np.float),
            "6W": df[df["measure"] == "6w"][col].astype(np.float),
            "12W": df[df["measure"] == "12w"][col].astype(np.float),
        }
    )

    p_base_middle, p_base_final = significant_test_pair(df, col)

    maxy = df[col].max()
    miny = df[col].min()

    ry = maxy - miny

    ryratio = 0.3

    if ax is None:
        ax = plt.gca()

    if title != "":
        if len(title) <= title_len_limit:
            ax.set_title(title, fontproperties=prop)
        else:
            ax.set_title(
                title[: int(title_len_limit / 2.0)]
                + "....."
                + title[-int(title_len_limit / 2.0) :],
                fontproperties=prop,
            )

    ax.set_ylim(top=maxy + (ry * ryratio), bottom=miny - (ry * (ryratio / 2.0)))

    ty = maxy + (ry * 0.1)

    ax.plot(
        [0, 1], [ty, ty], color="k", linewidth=linewidth / 1.5,
    )

    ax.text(
        0.5,
        ty,
        s=f"P: {round(p_base_middle, 3):.3f}",
        color="k",
        ha="center",
        va="bottom",
        fontproperties=prop,
    )

    ty = maxy + (ry * 0.2)

    ax.plot(
        [0, 2], [ty, ty], color="k", linewidth=linewidth / 1.5,
    )

    ax.text(
        1,
        ty,
        s=f"P: {round(p_base_final, 3):.3f}",
        color="k",
        ha="center",
        va="bottom",
        fontproperties=prop,
    )

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

    sns.swarmplot(data=vs, color="k", ax=ax, size=point_size)


def plot_corr(
    df, colx, coly, ax=None, pointsize=20, linewidth=2, labelsize=14, label_len_limit=50
):
    prop = fm.FontProperties(fname="fonts/Kosugi/Kosugi-Regular.ttf", size=labelsize)

    xs = df[colx]
    ys = df[coly]

    maxy = ys.max()
    miny = ys.min()

    ry = maxy - miny
    ryratio = 0.2

    tau, p = correlation(df, colx, coly)

    tx = xs.max()
    ha = "right"
    if (
        ys[xs < ((xs.max() + xs.min()) / 2.0)].max()
        < ys[xs > ((xs.max() + xs.min()) / 2.0)].max()
    ):
        tx = xs.min()
        ha = "left"

    s, i, _, _, _ = stats.linregress(xs.values, ys.values)
    xl = np.linspace(xs.min(), xs.max())

    if ax is None:
        ax = plt.gca()

    ax.set_ylim(
        top=maxy + (ry * ryratio),
        bottom=min(miny, (xs.min() * s) + i) - (ry * (ryratio / 2.0)),
    )

    ax.text(
        tx,
        maxy + (ry * (ryratio / 2.0)),
        s=f"P: {round(p, 3):.3f}\nTAU: {round(tau, 3):.3f}",
        color="k",
        ha=ha,
        va="bottom",
        fontproperties=prop,
    )

    ax.scatter(xs.values, ys.values, s=pointsize, color="k")
    ax.plot(xl, xl * s + i, color="k", linewidth=linewidth)

    if len(colx) > label_len_limit:
        ax.set_xlabel(
            colx[: int(label_len_limit / 2.0)]
            + "....."
            + colx[-int(label_len_limit / 2.0) :],
            fontproperties=prop,
        )
    else:
        ax.set_xlabel(colx, fontproperties=prop)

    if len(coly) > label_len_limit:
        ax.set_ylabel(
            coly[: int(label_len_limit / 2.0)]
            + "....."
            + coly[-int(label_len_limit / 2.0) :],
            fontproperties=prop,
        )
    else:
        ax.set_ylabel(coly, fontproperties=prop)


# %%

gc.collect()


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


def plot_group_set(df, output, intersection, num_rows=3, num_cols=2, point_size=10):
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    num_plots = len(intersection)
    num_pages = int(math.ceil(float(num_plots) / float(num_rows * num_cols)))

    r = 0
    c = 0

    f = None
    plotted = 0

    print(num_plots)

    for i, col in enumerate(list(intersection)):
        if i % int(num_rows * num_cols) == 0:
            if i != 0:
                print(f"save at {i}")
                plt.tight_layout()
                f.savefig(
                    os.path.join(output, f"{os.path.basename(output)}_{plotted}.png",),
                    facecolor="w",
                )
                plt.close(f)

                plotted += 1

            f, axes = plt.subplots(num_rows, num_cols, figsize=(20, 30))
            r = 0
            c = 0

        plot_group(df, col, ax=axes[r, c], title=col, point_size=point_size)

        c += 1
        if c % num_cols == 0:
            r += 1
            c = 0

    if plotted != num_pages:
        plt.tight_layout()
        f.savefig(
            os.path.join(output, f"{os.path.basename(output)}_{plotted}.png",),
            facecolor="w",
        )
        plt.close(f)
        plotted += 1

    print(f"total {plotted} charts")


def plot_corr_set(df, output, intersection, num_rows=3, num_cols=2, point_size=50):
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    num_plots = len(list(combinations(list(intersection), 2)))
    num_pages = int(math.ceil(float(num_plots) / float(num_rows * num_cols)))

    r = 0
    c = 0

    f = None
    plotted = 0

    i = 0

    print(num_plots)

    repeated = set()
    for colx in intersection:
        for coly in intersection:
            if colx == coly:
                continue

            if f"{colx}_{coly}" in repeated or f"{coly}_{colx}" in repeated:
                continue

            tau, p = correlation(df, colx, coly)
            if p > 0.05 or abs(tau) < 0.2:
                continue

            if i % int(num_rows * num_cols) == 0:
                if i != 0:
                    print(f"save at {i}")
                    plt.tight_layout()
                    f.savefig(
                        os.path.join(
                            output, f"{os.path.basename(output)}_{plotted}.png",
                        ),
                        facecolor="w",
                    )
                    plt.close(f)

                    plotted += 1

                f, axes = plt.subplots(num_rows, num_cols, figsize=(20, 30))
                r = 0
                c = 0

            # plot_corr(df, colx, coly, ax=axes[r, c], pointsize=point_size)
            plot_corr(df, coly, colx, ax=axes[r, c], pointsize=point_size)

            c += 1
            if c % num_cols == 0:
                r += 1
                c = 0

            i += 1
            repeated.add(f"{colx}_{coly}")

    if plotted != num_pages:
        plt.tight_layout()
        f.savefig(
            os.path.join(output, f"{os.path.basename(output)}_{plotted}.png",),
            facecolor="w",
        )
        plt.close(f)
        plotted += 1

    print(f"total {plotted} charts")


# %%


def make_intersection(df, reports):
    intersection = []
    for col in df.columns:
        if col in reports:
            intersection.append(col)

    return intersection


intersection = set(r[0] for r in report_multiple).intersection(
    set(r[0] for r in report_pair_bm)
)

intersection = make_intersection(df, intersection)

plot_group_set(
    df, "charts/Friedman_X_Wilcoxon/Friedman_X_Wilcoxon_0W_vs_6W", intersection
)

gc.collect()

intersection = set(r[0] for r in report_multiple).intersection(
    set(r[0] for r in report_pair_bf)
)

intersection = make_intersection(df, intersection)

plot_group_set(
    df, "charts/Friedman_X_Wilcoxon/Friedman_X_Wilcoxon_0W_vs_12W", intersection
)

gc.collect()

intersection = set(r[0] for r in report_multiple).intersection(
    set(r[0] for r in report_pair_both)
)

intersection = make_intersection(df, intersection)

plot_group_set(
    df,
    "charts/Friedman_X_Wilcoxon/Friedman_X_Wilcoxon_0W_vs_6W_X_0W_vs_12W",
    intersection,
)

gc.collect()

# %%

intersection = set(r[0] for r in report_multiple).intersection(
    set(r[0] for r in report_pair_bm)
)

intersection = make_intersection(df, intersection)

plot_corr_set(
    df,
    "charts/Friedman_X_Wilcoxon_X_Kendall/Friedman_X_Wilcoxon_0W_vs_6W",
    intersection,
)

gc.collect()

intersection = set(r[0] for r in report_multiple).intersection(
    set(r[0] for r in report_pair_bf)
)

intersection = make_intersection(df, intersection)

plot_corr_set(
    df,
    "charts/Friedman_X_Wilcoxon_X_Kendall/Friedman_X_Wilcoxon_0W_vs_12W",
    intersection,
)

gc.collect()

intersection = set(r[0] for r in report_multiple).intersection(
    set(r[0] for r in report_pair_both)
)

intersection = make_intersection(df, intersection)

plot_corr_set(
    df,
    "charts/Friedman_X_Wilcoxon_X_Kendall/Friedman_X_Wilcoxon_0W_vs_6W_X_0W_vs_12W",
    intersection,
)

gc.collect()

# %%

for r in set(r[0] for r in report_pair_bm):
    if r not in set(r[0] for r in report_multiple):
        print(r)

print()

for r in set(r[0] for r in report_pair_bf):
    if r not in set(r[0] for r in report_multiple):
        print(r)

# %%

# testing partial correlation

# base_set["measure"] = "0W"
# middle_set["measure"] = "6W"
# final_set["measure"] = "12W"
# ndf = base_set.append(middle_set).append(final_set)
# ndf = ndf.reset_index(drop=True)
# ndf

# %%

# significant_columns = list(
# set(r[0] for r in report_multiple).intersection(
# set(r[0] for r in report_pair_bm).union(set(r[0] for r in report_pair_bf))
# )
# )

# %%

# sdf = ndf[significant_columns]

# x = "HAMD"
# y = "L.casei-sg."

# cols = set(significant_columns)
# cols.remove(x)
# cols.remove(y)
# cols

# sdf.corr(method="spearman")["HAMD"]

# print(sdf.corr("kendall")["HAMD"])
# for col in significant_columns:
# if col == x or col == y:
# continue
# corr = pingouin.partial_corr(data=sdf, x=x, y=y, covar=col, method="spearman")
# if (corr["p-val"] <= 0.05).any():
# print(col[:75])
# print(corr)
# print()


# X	Y	Z
# 2	1	0
# 4	2	0
# 15	3	1
# 20	4	1

# test = pd.DataFrame({"X":[2,4,15,20], "Y":[1,2,3,4], "Z": [0,0,1,1]})
# print(pingouin.partial_corr(data=test, x="X", y="Y", covar="Z"))
# print(partial_corr(test))

# print(pingouin.partial_corr(data=sdf, x=x, y=y, covar=list(cols)[:5], method="kendall"))

# len(ndf[significant_columns].columns)

# kendall_corr = ndf[significant_columns].corr(method="kendall")
# porr = pd.DataFrame(data=partial_corr(sdf), index=sdf.columns)
# porr = partial_corr(sdf)
# porr

# prop = fm.FontProperties(fname="fonts/Kosugi/Kosugi-Regular.ttf", size=14)
# f, ax = plt.subplots(figsize=(15, 15))
# sns.set(font=prop)
# sns.heatmap(porr, annot=True, fmt=".3f", linewidths=0.5, ax=ax)
# plt.tight_layout()
# plt.show()

# %%

plt.close()
gc.collect()

# %%
