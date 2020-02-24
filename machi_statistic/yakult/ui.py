import gc
import json
import math
import re
import tkinter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
from statsmodels.stats import multitest

from preprocess import clean_data, test_data_na

matplotlib.use("tkagg")


class SearchField:
    def __init__(self, window, column, row, tag, width=50):
        self._entry = self._new_search_entry(
            window, column=column, row=row, width=width
        )

        self._tag = tag

        self._entry.bind("<KeyRelease>", lambda event: self._onkey(event))

        self._listbox = self._new_list_box(window, column=column, row=row + 1)

        self._listbox.bind("<<ListboxSelect>>", lambda event: self._onselect(event))
        self._listbox.bind("<KeyRelease>", lambda event: self._arrowKey(event))

        self._items = None
        self._filtered_items = None

        self._selected = None

        self._selection_cb = None
        self._significant_cb = None
        self._significant_coloring_cb = None

    @staticmethod
    def _new_search_entry(window, column, row, width=25):
        frame = tkinter.Frame(window)
        frame.grid(
            column=column,
            row=row,
            sticky=tkinter.W + tkinter.E + tkinter.S + tkinter.N,
        )

        entry = tkinter.Entry(frame, width=width)
        entry.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=tkinter.YES)

        return entry

    @staticmethod
    def _new_list_box(window, column, row, height=35):
        frame = tkinter.Frame(window)
        frame.grid(
            column=column,
            row=row,
            sticky=tkinter.W + tkinter.E + tkinter.S + tkinter.N,
        )

        scrollbarX = tkinter.Scrollbar(frame, orient=tkinter.HORIZONTAL)
        scrollbarY = tkinter.Scrollbar(frame, orient=tkinter.VERTICAL)

        listbox = tkinter.Listbox(
            frame,
            height=height,
            selectmode=tkinter.SINGLE,
            exportselection=False,
            xscrollcommand=scrollbarX.set,
            yscrollcommand=scrollbarY.set,
        )

        scrollbarX.config(command=listbox.xview)
        scrollbarX.pack(side=tkinter.BOTTOM, fill=tkinter.X)

        scrollbarY.config(command=listbox.yview)
        scrollbarY.pack(side=tkinter.RIGHT, fill=tkinter.Y)

        listbox.pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=tkinter.YES)

        return listbox

    def _arrowKey(self, event):
        # print(event.keysym)

        w = event.widget
        selection = w.curselection()
        if len(selection) >= 1:
            index = int(w.curselection()[0])

            if event.keysym == "Up":
                new_index = (index - 1) % len(self._filtered_items)
            elif event.keysym == "Down":
                new_index = (index + 1) % len(self._filtered_items)

            self._listbox.selection_clear(index)
            self._listbox.selection_set(new_index)
            self._listbox.activate(new_index)

            self._selected = w.get(new_index)

            if self._selection_cb is not None:
                self._selection_cb(self._tag)

    def _onkey(self, event):
        text = str(self._entry.get()).strip()

        self._listbox.delete(0, tkinter.END)

        if text == "":
            self._filtered_items = self._items

            for item in self._items:
                self._listbox.insert(tkinter.END, str(item))

        else:
            # if text == "~":
            if text.startswith("~"):
                if self._significant_cb is not None:
                    self._filtered_items = self._significant_cb()

                else:
                    self._filtered_items = self._items

                    # for item in self._filtered_items:
                    #     self._listbox.insert(tkinter.END, str(item))

            else:
                self._filtered_items = self._items

            if text == "~":
                # for item in self._filtered_items:
                #     self._listbox.insert(tkinter.END, str(item))
                pass
            else:
                filtered = []
                # for item in self._items:
                for item in self._filtered_items:
                    found = False
                    ts = text.replace("~", "", -1).split(",", -1)

                    for t in ts:
                        if t.strip() == "":
                            continue

                        if t.strip() in str(item).lower():
                            found = True
                        else:
                            found = False
                            break

                    if found:
                        filtered.append(item)

                self._filtered_items = filtered

            # for item in filtered:
            for item in self._filtered_items:
                self._listbox.insert(tkinter.END, str(item))

        if self._significant_coloring_cb is not None:
            self._significant_coloring_cb()

    def _onselect(self, event):
        w = event.widget
        selection = w.curselection()

        if len(selection) >= 1:
            index = int(w.curselection()[0])

            self._selected = w.get(index)

            if self._selection_cb is not None:
                self._selection_cb(self._tag)

    def set_listbox_items(self, items):
        self._items = items
        self._filtered_items = items

        for item in items:
            self._listbox.insert(tkinter.END, str(item))

    def set_listbox_items_significant(self, indexes):
        for i in range(len(self._filtered_items)):
            self._listbox.itemconfig(i, {"fg": "black", "bg": "white"})

        for i in indexes:
            self._listbox.itemconfig(i, {"fg": "white", "bg": "blue"})

    def init_search(self):
        self._onkey(None)

    @property
    def tag(self):
        return self._tag

    @property
    def filtered_items(self):
        return self._filtered_items

    @property
    def entry(self):
        return self._entry

    @property
    def listbox(self):
        return self._listbox

    @property
    def selected(self):
        return self._selected

    def set_selection_cb(self, cb):
        self._selection_cb = cb

    def set_significant_cb(self, cb):
        self._significant_cb = cb

    def set_significant_coloring_cb(self, cb):
        self._significant_coloring_cb = cb


class MyApp:
    def __init__(self, statistic, df, title="Statistic"):

        self._statistic = statistic
        self._df = df

        self._window = tkinter.Tk()
        self._window.title(title)

        self._root = tkinter.Frame(self._window)
        self._root.pack(fill=tkinter.BOTH, expand=tkinter.YES)
        self._root.grid_columnconfigure(0, weight=1)
        self._root.grid_columnconfigure(1, weight=1)

        self._root.grid_rowconfigure(0, weight=1)
        self._root.grid_rowconfigure(1, weight=35)
        self._root.grid_rowconfigure(2, weight=1)
        self._root.grid_rowconfigure(3, weight=1)
        self._root.grid_rowconfigure(4, weight=1)

        self._src = SearchField(self._root, 0, 0, "src")
        self._tar = SearchField(self._root, 1, 0, "tar")

        self._src.set_selection_cb(self._selection_cb)
        self._tar.set_selection_cb(self._selection_cb)
        self._tar.set_significant_cb(self._significant_cb)
        self._tar.set_significant_coloring_cb(self._significant_coloring_cb)

        self._src.set_listbox_items(self._statistic.keys())
        self._tar.set_listbox_items(self._statistic.keys())

        self._info = tkinter.Frame(self._root)
        self._info.grid(
            column=0,
            row=2,
            columnspan=2,
            sticky=tkinter.W + tkinter.E + tkinter.S + tkinter.N,
        )

        self._info.grid_rowconfigure(0, weight=1)
        self._info.grid_rowconfigure(1, weight=1)
        self._info.grid_rowconfigure(2, weight=1)
        self._info.grid_rowconfigure(3, weight=1)
        # self._info.grid_rowconfigure(4, weight=1)

        self._a_var = tkinter.StringVar()
        self._a_label = tkinter.Label(self._info, textvariable=self._a_var)
        self._a_label.grid(column=0, row=0, columnspan=2, sticky=tkinter.W)

        self._a_var.set("A:")

        self._b_var = tkinter.StringVar()
        self._b_label = tkinter.Label(self._info, textvariable=self._b_var)
        self._b_label.grid(column=0, row=1, columnspan=2, sticky=tkinter.W)

        self._b_var.set("B:")

        self._p_var = tkinter.StringVar()
        self._p_label = tkinter.Label(self._info, textvariable=self._p_var)
        self._p_label.grid(column=0, row=2, columnspan=2, sticky=tkinter.W)

        self._p_var.set("P:")

        self._tau_var = tkinter.StringVar()
        self._tau_label = tkinter.Label(self._info, textvariable=self._tau_var)
        self._tau_label.grid(column=0, row=3, columnspan=2, sticky=tkinter.W)

        self._tau_var.set("TAU:")

        self._plot_button = tkinter.Button(
            self._root,
            text="plot group",
            height=2,
            command=lambda: self._plot_chart(chart_type="g"),
        )

        self._plot_button.grid(
            column=0,
            row=3,
            columnspan=2,
            sticky=tkinter.W + tkinter.E + tkinter.S + tkinter.N,
        )

        self._plot_button = tkinter.Button(
            self._root,
            text="plot correlation",
            height=2,
            command=lambda: self._plot_chart(chart_type="c"),
        )

        self._plot_button.grid(
            column=0,
            row=4,
            columnspan=2,
            sticky=tkinter.W + tkinter.E + tkinter.S + tkinter.N,
        )

    def _significant_cb(self):
        src = self._src.selected
        if src is not None and src != "":
            items = []

            for k, v in self._statistic[src].items():
                if float(v["p"]) <= 0.05:
                    items.append(k)

            return items

    def _significant_coloring_cb(self):
        src = self._src.selected
        if src is not None and src != "":
            indexes = []
            for i, v in enumerate(self._tar.filtered_items):
                if float(self._statistic[src][v]["p"]) <= 0.05:
                    indexes.append(i)

            self._tar.set_listbox_items_significant(indexes)

    def _selection_cb(self, tag):
        if tag == "src":
            self._tar.init_search()
            self._significant_coloring_cb()

        src = self._src.selected
        tar = self._tar.selected

        self._a_var.set("A: {}".format(src))
        self._b_var.set("B: {}".format(tar))

        if self._statistic is None:
            return

        sk = self._statistic.get(src, None)
        if sk is not None:
            vk = sk.get(tar, None)

            if vk is not None:
                p = vk.get("p", math.nan)
                tau = vk.get("tau", math.nan)

                self._p_var.set("P: {}".format(str(p)))
                self._tau_var.set("TAU: {}".format(str(tau)))

    def _plot_chart(self, chart_type="g"):

        assert chart_type in ("g", "c")

        win = tkinter.Toplevel()
        win.wm_title("plot")

        fig, ax = plt.subplots(figsize=(7, 7))
        win.protocol("WM_DELETE_WINDOW", lambda: self._clear_plot(win, fig, ax))
        win.bind("<Escape>", lambda event: self._clear_plot(win, fig, ax))

        if chart_type == "g":
            self._plot_group(ax=ax)
        elif chart_type == "c":
            self._plot_corr(ax=ax)

        fig.tight_layout()

        # if chart_type == "g":
        #     fig.savefig(
        #         "{}.png".format(self._src.selected.replace("/", "_", -1)), facecolor="w"
        #     )
        # elif chart_type == "c":
        #     fig.savefig(
        #         "{}_X_{}.png".format(
        #             self._src.selected.replace("/", "_", -1),
        #             self._tar.selected.replace("/", "_", -1),
        #         ),
        #         facecolor="w",
        #     )
        # else:
        #     fig.savefig("plot.png", facecolor="w")
        fig.savefig("plot.png", facecolor="w")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().grid(row=0, column=0)

        canvas.draw()

    def _clear_plot(self, win, fig, ax):
        win.destroy()
        plt.cla()
        plt.clf()

        gc.collect()

    @property
    def window(self):
        return self._window

    def _correlation(self, x, y):
        xs = self._df[x]
        ys = self._df[y]

        tau, p = stats.kendalltau(xs.values, ys.values)

        return tau, p

    def _significant_test_pair(self, col):

        try:
            _, bmp = stats.wilcoxon(
                self._df[self._df["measure"] == "0W"][col].astype(np.float),
                self._df[self._df["measure"] == "6W"][col].astype(np.float),
            )
        except ValueError as e:
            print(f"{col}: {e}")
            bmp = np.nan

        try:
            _, bfp = stats.wilcoxon(
                self._df[self._df["measure"] == "0W"][col].astype(np.float),
                self._df[self._df["measure"] == "12W"][col].astype(np.float),
            )
        except ValueError as e:
            print(f"{col}: {e}")
            bfp = np.nan

        try:
            _, mfp = stats.wilcoxon(
                self._df[self._df["measure"] == "6W"][col].astype(np.float),
                self._df[self._df["measure"] == "12W"][col].astype(np.float),
            )
        except ValueError as e:
            print(f"{col}: {e}")
            mfp = np.nan

        _, adj_p, _, _ = multitest.multipletests(
            pvals=[bmp, bfp, mfp], alpha=0.05, method="bonferroni",
        )

        return adj_p[0], adj_p[1]

    def _plot_group(
        self,
        ax=None,
        point_size=6,
        linelength=0.7,
        linewidth=2,
        title_len_limit=50,
        title_size=14,
    ):

        col = self._src.selected
        title = self._src.selected

        prop = fm.FontProperties(
            fname="fonts/Kosugi/Kosugi-Regular.ttf", size=title_size
        )

        vs = pd.DataFrame(
            {
                "0W": self._df[self._df["measure"] == "0W"][col].astype(np.float),
                "6W": self._df[self._df["measure"] == "6W"][col].astype(np.float),
                "12W": self._df[self._df["measure"] == "12W"][col].astype(np.float),
            }
        )

        bmp, bfp = self._significant_test_pair(col)

        maxy = max(
            base_set[col].astype(np.float).max(),
            middle_set[col].astype(np.float).max(),
            final_set[col].astype(np.float).max(),
        )

        miny = min(
            base_set[col].astype(np.float).min(),
            middle_set[col].astype(np.float).min(),
            final_set[col].astype(np.float).min(),
        )

        ry = maxy - miny

        ryratio = 0.3

        if title != "":
            if len(title) <= title_len_limit:
                plt.title(title, fontproperties=prop)
            else:
                plt.title(title[:title_len_limit] + "...", fontproperties=prop)

        if ax is None:
            ax = plt.gca()

        ax.set_ylim(top=maxy + (ry * ryratio), bottom=miny - (ry * (ryratio / 2.0)))

        ty = maxy + (ry * 0.1)

        ax.plot(
            [0, 1], [ty, ty], color="k", linewidth=linewidth / 1.5,
        )

        ax.text(
            0.5,
            ty,
            # s=f"P: {round(bmp, 3):.3f}",
            s="P: {:.3f}".format(round(bmp, 3)),
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
            # s=f"P: {round(bfp, 3):.3f}",
            s="P: {:.3f}".format(round(bfp, 3)),
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
        sns.swarmplot(data=vs, color="k", size=point_size)

    def _plot_corr(
        self, ax=None, pointsize=20, linewidth=2, labelsize=14, label_len_limit=50
    ):
        prop = fm.FontProperties(
            fname="fonts/Kosugi/Kosugi-Regular.ttf", size=labelsize
        )

        coly = self._src.selected
        colx = self._tar.selected

        xs = self._df[colx]
        ys = self._df[coly]

        maxy = ys.max()
        miny = ys.min()

        ry = maxy - miny
        ryratio = 0.2

        tau, p = self._correlation(colx, coly)

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
            # s=f"P: {round(p, 3):.3f}\nTAU: {round(tau, 3):.3f}",
            s="P: {:.3f}\nTAU: {:.3f}".format(round(p, 3), round(tau, 3)),
            color="k",
            ha=ha,
            va="bottom",
            fontproperties=prop,
        )

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
            # new_cols.append(f"{match.group(1)}{match.group(3)}")
            new_cols.append("{}{}".format(match.group(1), match.group(3)))
        else:
            new_cols.append(col)

    data.columns = new_cols


def test_column_pair(cols):
    assert len(cols) == 3
    for j in cols:
        for k in cols:
            assert j.startswith(k[:3])


if __name__ == "__main__":
    stat = None

    print("reading statistic.json")

    with open("statistic.json", "r") as f:
        stat = json.load(f)

    print("reading data.csv")

    df = pd.read_csv("data.csv")

    df = df[:19]
    df = df.astype(np.float64)

    df = clean_data(df)

    print("testing dataframe.....")

    test_data_na(df)

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

    rename_columns(base_set)
    rename_columns(middle_set)
    rename_columns(final_set)

    base_set["measure"] = "0W"
    middle_set["measure"] = "6W"
    final_set["measure"] = "12W"

    df = base_set.append(middle_set).append(final_set)
    df = df.reset_index(drop=True)

    print("ready to launch the program.....")

    app = MyApp(stat, df)
    app.window.mainloop()
