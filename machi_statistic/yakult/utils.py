import json
import math
import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import font_manager as fm
from statsmodels.stats import multitest


def clean_data(df):
    regex = re.compile(r"^.+[-_](\d{1}).*$", re.MULTILINE)

    for i, col in enumerate(df.columns):
        m = regex.match(col)
        na = df[col][df[col].isnull()]
        if len(na) > 0:
            if m is not None:

                pcol = (
                    col[: m.start(1)] + str(int(m.group(1)) - 1) + col[m.start(1) + 1 :]
                )

                df[col][df[col].isnull()] = df[pcol][df[col].isnull()]

            else:
                raise Exception("\n{}\n{}\n".format(na, m))

    return df


def test_data_na(df):
    for col in df.columns:
        for i, d in enumerate(df[col]):
            if math.isnan(d):
                raise Exception("data should not be nan, col: {}, {}".format(col, i))


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


def significant_test_multiple(df, col):
    _, p = stats.friedmanchisquare(
        df[df["measure"] == "0w"][col].astype(np.float),
        df[df["measure"] == "6w"][col].astype(np.float),
        df[df["measure"] == "12w"][col].astype(np.float),
    )

    return p


def significant_test_pair(df, col, all_pair=False):

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

    if all_pair:
        _, adj_p, _, _ = multitest.multipletests(
            pvals=[p_base_middle, p_base_final, p_middle_final],
            alpha=0.05,
            method="bonferroni",
        )

    else:
        _, adj_p, _, _ = multitest.multipletests(
            pvals=[p_base_middle, p_base_final], alpha=0.05, method="bonferroni",
        )

    return adj_p[0], adj_p[1]


def correlation(df, colx, coly, delta):

    base = df[df["measure"] == "0w"]
    middle = df[df["measure"] == "6w"]
    final = df[df["measure"] == "12w"]

    assert delta in (None, "fb", "mb", "mbfb", "bbmbfb")

    if delta is None:
        tau, p = stats.kendalltau(df[colx].astype(np.float), df[coly].astype(np.float))
        return tau, p

    elif delta == "fb":

        fbx = final[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        fby = final[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)

        tau, p = stats.kendalltau(fbx.astype(np.float), fby.astype(np.float))

        return tau, p

    elif delta == "mb":

        mbx = middle[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        mby = middle[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)

        tau, p = stats.kendalltau(mbx.astype(np.float), mby.astype(np.float))

        return tau, p

    elif delta == "mbfb":

        mbx = middle[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        fbx = final[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)

        x = mbx.append(fbx).reset_index(drop=True)

        mby = middle[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)
        fby = final[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)

        y = mby.append(fby).reset_index(drop=True)

        tau, p = stats.kendalltau(x.astype(np.float), y.astype(np.float))
        return tau, p

    elif delta == "bbmbfb":

        bbx = base[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        mbx = middle[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        fbx = final[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)

        x = bbx.append(mbx).append(fbx).reset_index(drop=True)

        bby = base[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)
        mby = middle[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)
        fby = final[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)

        y = bby.append(mby).append(fby).reset_index(drop=True)

        tau, p = stats.kendalltau(x.astype(np.float), y.astype(np.float))
        return tau, p

    else:
        raise ValueError("invalid op")


def find_col(df, key_words):
    targets = []

    for col in df.columns:
        found = True
        for word in key_words:
            if word not in col.lower():
                found = False
                break

        if found:
            targets.append(col)

    targets.sort(key=lambda x: len(x))

    return targets[0]


def bake_correlation_statistic(df, drops, correlation_func, output_file):

    if not os.path.exists("statistics"):
        os.makedirs("statistics", exist_ok=True)

    statistic = {}

    progress = 0.0
    works = float(len(df.columns) * len(df.columns))

    progress_report = 2

    for x in df.columns:
        if x in drops:
            continue

        if statistic.get(x, None) is None:
            statistic[x] = {}

        for y in df.columns:
            if y in drops:
                continue

            tau, p_value = correlation_func(df, x, y)

            if math.isnan(tau) or math.isnan(p_value):
                pass

            progress += 1

            if progress % progress_report == 0:
                print(f"{round((progress / works) * 100.0, 4)}%.....")

            statistic[x][y] = {
                "p": str(p_value),
                "tau": str(tau),
            }

    with open(f"statistics/{output_file}", "w") as output:
        json.dump(statistic, output, indent=2)


def write_friedman_wilcoxon_report(file_name, title, friedman_report, wilcoxon_report):
    spaces = "  "
    multiplier = 2

    if not os.path.exists("reports"):
        os.makedirs("reports", exist_ok=True)

    with open(f"reports/{file_name}", "w", encoding="utf-8",) as f:
        intersection = set(r[0] for r in friedman_report).intersection(
            set(r[0] for r in wilcoxon_report)
        )

        f.write(f"{title}: {len(intersection)} features\n")
        f.write("\n")
        for r in friedman_report:
            if r[0].strip() in intersection:
                f.write(f"{r[0].strip()}\n")
            for jr in wilcoxon_report:
                if jr[0].strip() == r[0].strip():
                    if len(jr) == 2:
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


def write_friedman_wilcoxon_kendall_report(
    file_name, title, friedman_report, wilcoxon_report, kendall_report
):
    spaces = "  "
    multiplier = 2

    if not os.path.exists("reports"):
        os.makedirs("reports", exist_ok=True)

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

                for jr in wilcoxon_report:
                    if jr[0].strip() == r[0].strip():
                        if len(jr) == 2:
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


def write_kendall_delta_report(file_name, title, kendall_report):
    spaces = "  "
    multiplier = 2

    if not os.path.exists("reports"):
        os.makedirs("reports", exist_ok=True)

    with open(f"reports/{file_name}", "w", encoding="utf-8",) as f:
        f.write(f"{title}: {len(kendall_report)} features")
        f.write("\n\n")

        for x in kendall_report.keys():
            f.write(f"{x.strip()}")
            f.write("\n")

            for y in kendall_report[x].keys():
                f.write(f"{spaces * multiplier * 1}{y.strip()}")
                f.write("\n")

                f.write(
                    f"{spaces * multiplier * 2}P:    {kendall_report[x][y]['p']:.19f}"
                )
                f.write("\n")
                f.write(
                    f"{spaces * multiplier * 2}TAU:  {kendall_report[x][y]['tau']:.19f}"
                )
                f.write("\n")

                f.write("\n")

            f.write("\n")


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


def plot_correlation(
    df,
    colx,
    coly,
    delta,
    ax=None,
    pointsize=50,
    linewidth=2,
    labelsize=14,
    label_len_limit=50,
):
    prop = fm.FontProperties(fname="fonts/Kosugi/Kosugi-Regular.ttf", size=labelsize)

    base = df[df["measure"] == "0w"]
    middle = df[df["measure"] == "6w"]
    final = df[df["measure"] == "12w"]

    assert delta in (None, "fb", "mb", "mbfb", "bbmbfb")

    if delta is None:
        xs = df[colx]
        ys = df[coly]

    elif delta == "fb":

        xs = final[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        ys = final[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)

    elif delta == "mb":

        xs = middle[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        ys = middle[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)

    elif delta == "mbfb":

        mbx = middle[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        fbx = final[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)

        xs = mbx.append(fbx).reset_index(drop=True)

        mby = middle[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)
        fby = final[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)

        ys = mby.append(fby).reset_index(drop=True)

    elif delta == "bbmbfb":

        bbx = base[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        mbx = middle[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)
        fbx = final[colx].reset_index(drop=True) - base[colx].reset_index(drop=True)

        xs = bbx.append(mbx).append(fbx).reset_index(drop=True)

        bby = base[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)
        mby = middle[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)
        fby = final[coly].reset_index(drop=True) - base[coly].reset_index(drop=True)

        ys = bby.append(mby).append(fby).reset_index(drop=True)

    maxy = ys.max()
    miny = ys.min()

    ry = maxy - miny
    ryratio = 0.2

    tau, p = stats.kendalltau(xs, ys)

    s, i, _, _, _ = stats.linregress(xs.values, ys.values)
    xl = np.linspace(xs.min(), xs.max())

    if ax is None:
        ax = plt.gca()

    ax.set_ylim(
        top=maxy + (ry * ryratio),
        bottom=min(miny, (xs.min() * s) + i) - (ry * (ryratio / 2.0)),
    )

    ax.scatter(xs.values, ys.values, s=pointsize, color="k")
    ax.plot(xl, xl * s + i, color="k", linewidth=linewidth)

    minx, maxx = ax.get_xlim()
    rx = maxx - minx
    rxratio = 0.05

    tx = maxx - (rx * rxratio)
    ha = "right"
    if (
        ys[xs < ((xs.max() + xs.min()) / 2.0)].max()
        < ys[xs > ((xs.max() + xs.min()) / 2.0)].max()
    ):
        tx = minx + (rx * rxratio)
        ha = "left"

    ax.text(
        tx,
        maxy + (ry * (ryratio / 2.0)),
        s=f"P: {round(p, 3):.3f}\nTAU: {round(tau, 3):.3f}",
        color="k",
        ha=ha,
        va="bottom",
        fontproperties=prop,
    )

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


def plot_correlation_set(
    df, output, cols, delta, num_rows=3, num_cols=2, point_size=60
):
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    assert delta in (None, "fb", "mb", "mbfb", "bbmbfb")

    num_plots = len(list(combinations(list(cols), 2)))
    num_pages = int(math.ceil(float(num_plots) / float(num_rows * num_cols)))

    r = 0
    c = 0

    f = None
    plotted = 0

    i = 0

    print(num_plots)

    repeated = set()
    for colx in cols:
        for coly in cols:
            if colx == coly:
                continue

            if f"{colx}_{coly}" in repeated or f"{coly}_{colx}" in repeated:
                continue

            tau, p = correlation(df, colx, coly, delta)
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

            plot_correlation(df, coly, colx, delta, ax=axes[r, c], pointsize=point_size)

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
