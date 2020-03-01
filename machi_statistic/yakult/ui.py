import gc
import json
import math
import tkinter
from tkinter import ttk

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import utils

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
            if text.startswith("~"):
                if self._significant_cb is not None:
                    self._filtered_items = self._significant_cb()

                else:
                    self._filtered_items = self._items

            else:
                self._filtered_items = self._items

            if text == "~":
                pass
            else:
                filtered = []
                for item in self._filtered_items:
                    found = False
                    ts = text.replace("~", "", -1).split(",", -1)

                    for t in ts:
                        if t.strip() == "":
                            continue

                        if t.strip().lower() in str(item).lower():
                            found = True
                        else:
                            found = False
                            break

                    if found:
                        filtered.append(item)

                self._filtered_items = filtered

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
    def __init__(self, reports, df, title="Statistic"):

        self._reports = reports
        self._statistic = reports["delta_12w"]
        self._df = df

        self._window = tkinter.Tk()
        self._window.title(title)

        self._root = tkinter.Frame(self._window)
        self._root.pack(fill=tkinter.BOTH, expand=tkinter.YES)
        self._root.grid_columnconfigure(0, weight=1)
        self._root.grid_columnconfigure(1, weight=1)

        self._root.grid_rowconfigure(0, weight=1)
        self._root.grid_rowconfigure(1, weight=50)
        self._root.grid_rowconfigure(2, weight=1)
        self._root.grid_rowconfigure(3, weight=1)
        self._root.grid_rowconfigure(4, weight=1)
        self._root.grid_rowconfigure(5, weight=4)

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

        self._info.grid_columnconfigure(0, weight=1)
        self._info.grid_columnconfigure(1, weight=1)

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
        self._p_label.grid(column=0, row=2, sticky=tkinter.W)

        self._p_var.set("P:")

        self._p_threshold_label = tkinter.Label(self._info, text="P Threshold:")
        self._p_threshold_label.grid(column=1, row=2, sticky=tkinter.E)

        self._p_threshold = tkinter.StringVar()
        self._p_threshold_entry = tkinter.Entry(
            self._info, textvariable=self._p_threshold
        )
        self._p_threshold_entry.grid(column=2, row=2, sticky=tkinter.E)

        self._p_threshold_entry.bind(
            "<KeyRelease>", lambda event: self._threshold_update(event)
        )

        self._p_threshold.set("0.05")

        self._tau_var = tkinter.StringVar()
        self._tau_label = tkinter.Label(self._info, textvariable=self._tau_var)
        self._tau_label.grid(column=0, row=3, sticky=tkinter.W)

        self._tau_var.set("TAU:")

        self._tau_threshold_label = tkinter.Label(self._info, text="TAU Threshold:")
        self._tau_threshold_label.grid(column=1, row=3, sticky=tkinter.E)

        self._tau_threshold = tkinter.StringVar()
        self._tau_threshold_entry = tkinter.Entry(
            self._info, textvariable=self._tau_threshold
        )
        self._tau_threshold_entry.grid(column=2, row=3, sticky=tkinter.E)

        self._tau_threshold_entry.bind(
            "<KeyRelease>", lambda event: self._threshold_update(event)
        )

        self._tau_threshold.set("0.2")

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

        self._combo = ttk.Combobox(
            self._root, values=list(self._reports.keys()), state="readonly"
        )
        self._combo.set("delta_12w")
        self._combo.grid(
            column=0,
            row=5,
            columnspan=2,
            sticky=tkinter.W + tkinter.E + tkinter.S + tkinter.N,
        )

        self._combo.bind("<<ComboboxSelected>>", lambda e: self._combo_selection_cb())

    def _threshold_update(self, event):
        self._tar.init_search()

    def _significant_threshold(self):
        p_threshold = None
        tau_threshold = None

        if self._p_threshold.get() != "":
            try:
                p_threshold = float(self._p_threshold.get())
            except ValueError as e:
                print(e)
                pass

        if self._tau_threshold.get() != "":
            try:
                tau_threshold = float(self._tau_threshold.get())
            except ValueError as e:
                print(e)
                pass

        return p_threshold, tau_threshold

    def _combo_selection_cb(self):
        self._statistic = self._reports[self._combo.get()]
        self._selection_cb("src")

    def _significant_cb(self):
        src = self._src.selected

        p_threshold, tau_threshold = self._significant_threshold()

        if src is not None and src != "":
            items = []

            for k, v in self._statistic[src].items():

                ok = True
                if p_threshold is not None:
                    if math.isnan(float(v["p"])):
                        ok = False
                    if float(v["p"]) > p_threshold:
                        ok = False

                if tau_threshold is not None:
                    if math.isnan(float(v["tau"])):
                        ok = False
                    if abs(float(v["tau"])) < tau_threshold:
                        ok = False

                if ok:
                    items.append(k)

            return items

    def _significant_coloring_cb(self):
        src = self._src.selected
        p_threshold, tau_threshold = self._significant_threshold()

        if src is not None and src != "":
            indexes = []
            for i, v in enumerate(self._tar.filtered_items):

                ok = True
                if p_threshold is not None:
                    if math.isnan(float(self._statistic[src][v]["p"])):
                        ok = False
                    if float(self._statistic[src][v]["p"]) > p_threshold:
                        ok = False

                if tau_threshold is not None:
                    if math.isnan(float(self._statistic[src][v]["tau"])):
                        ok = False
                    if abs(float(self._statistic[src][v]["tau"])) < tau_threshold:
                        ok = False

                if ok:
                    indexes.append(i)

            self._tar.set_listbox_items_significant(indexes)

    def _selection_cb(self, tag):
        if tag == "src":
            self._tar.init_search()
            self._significant_coloring_cb()

        src = self._src.selected
        tar = self._tar.selected

        self._a_var.set(f"A: {src}")
        self._b_var.set(f"B: {tar}")

        if self._statistic is None:
            return

        sk = self._statistic.get(src, None)
        if sk is not None:
            vk = sk.get(tar, None)

            if vk is not None:
                p = vk.get("p", math.nan)
                tau = vk.get("tau", math.nan)

                self._p_var.set(f"P:      {float(p):.19f}")
                self._tau_var.set(f"TAU: {float(tau):.19f}")

    def _plot_chart(self, chart_type="g"):

        assert chart_type in ("g", "c")

        win = tkinter.Toplevel()
        win.wm_title("plot")

        fig, ax = plt.subplots(figsize=(7, 7))
        win.protocol("WM_DELETE_WINDOW", lambda: self._clear_plot(win, fig, ax))
        win.bind("<Escape>", lambda event: self._clear_plot(win, fig, ax))

        if chart_type == "g":
            if self._src.selected == "" or self._src.selected is None:
                return

            utils.plot_group(
                self._df, self._src.selected, ax=ax, title=self._src.selected
            )

        elif chart_type == "c":
            if self._src.selected == "" or self._src.selected is None:
                return

            if self._tar.selected == "" or self._tar.selected is None:
                return

            assert self._combo.get() in (
                None,
                "simple",
                "delta_6w",
                "delta_12w",
                "delta_6w_12w",
                "delta_0w_6w_12w",
            )

            delta = None
            if self._combo.get() == "simple":
                pass
            elif self._combo.get() == "delta_6w":
                delta = "mb"
            elif self._combo.get() == "delta_12w":
                delta = "fb"
            elif self._combo.get() == "delta_6w_12w":
                delta = "mbfb"
            elif self._combo.get() == "delta_0w_6w_12w":
                delta = "bbmbfb"
            else:
                raise ValueError("invalid combo value")

            utils.plot_correlation(
                self._df, self._tar.selected, self._src.selected, delta=delta, ax=ax
            )

        fig.tight_layout()

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


def read_data_csv(src="data.csv"):
    df = pd.read_csv("data.csv")

    df = df[:19]
    df = df.astype(np.float64)

    df = utils.clean_data(df)

    print("testing dataframe.....")

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

    base_set["measure"] = "0w"
    middle_set["measure"] = "6w"
    final_set["measure"] = "12w"

    df = base_set.append(middle_set).append(final_set)
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    reports = {}

    print("reading statistics.....")

    with open(f"statistics/statistic_simple.json", "r") as f:
        reports["simple"] = json.load(f)

    with open(f"statistics/statistic_delta_6W.json", "r") as f:
        reports["delta_6w"] = json.load(f)

    with open(f"statistics/statistic_delta_12W.json", "r") as f:
        reports["delta_12w"] = json.load(f)

    with open(f"statistics/statistic_delta_6W_12W.json", "r") as f:
        reports["delta_6w_12w"] = json.load(f)

    with open(f"statistics/statistic_delta_0W_6W_12W.json", "r") as f:
        reports["delta_0w_6w_12w"] = json.load(f)

    print("reading data.....")
    df = read_data_csv()

    print("ready to launch the program.....")

    app = MyApp(reports, df)
    app.window.mainloop()
