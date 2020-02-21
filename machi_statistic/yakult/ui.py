import gc
import re
import json
import math
import tkinter
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

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

    def _onkey(self, event):
        text = str(self._entry.get()).strip()

        self._listbox.delete(0, tkinter.END)

        if text == "":
            self._filtered_items = self._items

            for item in self._items:
                self._listbox.insert(tkinter.END, str(item))

        else:
            if text == "~":
                if self._significant_cb is not None:
                    self._filtered_items = self._significant_cb()

                    for item in self._filtered_items:
                        self._listbox.insert(tkinter.END, str(item))

            else:
                filtered = []
                for item in self._items:
                    found = False
                    ts = text.split(",", -1)

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
                for item in filtered:
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
            command=lambda: self._plot_corr(chart_type="c"),
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
        # src = self._src.selected
        # tar = self._tar.selected

        # x = self._df[tar]
        # y = self._df[src]

        # s, i, _, _, _ = stats.linregress(x, y)
        # xl = np.linspace(x.min(), x.max())

        win = tkinter.Toplevel()
        win.wm_title("plot")

        fig, ax = plt.subplots(figsize=(7, 7))
        win.protocol("WM_DELETE_WINDOW", lambda: self._clear_plot(win, fig, ax))
        win.bind("<Escape>", lambda event: self._clear_plot(win, fig, ax))

        if chart_type == "g":
            self._plot_group(ax=ax)
        elif chart_type == "c":
            self._plot_corr(ax=ax)

        # ax.scatter(x, y, s=40, color="k")
        # ax.plot(xl, xl * s + i, color="k")

        # prop = fm.FontProperties(fname="fonts/Kosugi/Kosugi-Regular.ttf", size=12)

        # ax.set_xlabel(tar, fontproperties=prop)
        # ax.set_ylabel(src, fontproperties=prop)

        fig.tight_layout()

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

    def _plot_group(
        self,
        ax=None,
        point_size=6,
        linelength=0.7,
        linewidth=2,
        title="",
        title_len_limit=50,
        title_size=14,
    ):

        prop = fm.FontProperties(
            fname="fonts/Kosugi/Kosugi-Regular.ttf", size=title_size
        )

        col = self._src.selected

        vs = pd.DataFrame(
            {
                # "0W": base_set[col].astype(np.float),
                # "6W": middle_set[col].astype(np.float),
                # "12W": final_set[col].astype(np.float),
                "0W": self._df[self._df["measure"] == "0W"][col].astype(np.float),
                "6W": self._df[self._df["measure"] == "6W"][col].astype(np.float),
                "12W": self._df[self._df["measure"] == "12W"][col].astype(np.float),
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

    def _plot_corr(
        self, ax=None, pointsize=20, linewidth=2, labelsize=14, label_len_limit=50,
    ):
        prop = fm.FontProperties(
            fname="fonts/Kosugi/Kosugi-Regular.ttf", size=labelsize
        )

        coly = self._src.selected
        colx = self._tar.selected

        # xs = base_set[colx].append(middle_set[colx]).append(final_set[colx])
        # ys = base_set[coly].append(middle_set[coly]).append(final_set[coly])
        xs = self._df[colx]
        ys = self._df[coly]

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

    print("ready to launch the program.....")

    app = MyApp(stat, df)
    app.window.mainloop()
