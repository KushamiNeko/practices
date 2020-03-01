import argparse
import json
import math
import re
import sys
from typing import cast

import numpy as np
import pandas as pd
import scipy.stats as stats

import utils


# def clean_data(df):
    # regex = re.compile(r"^.+[-_](\d{1}).*$", re.MULTILINE)

    # for i, col in enumerate(df.columns):
        # m = regex.match(col)
        # # na = df[col][df[col].isna()]
        # na = df[col][df[col].isnull()]
        # if len(na) > 0:
            # if m is not None:

                # pcol = (
                    # col[: m.start(1)] + str(int(m.group(1)) - 1) + col[m.start(1) + 1 :]
                # )

                # # df[col][df[col].isna()] = df[pcol][df[col].isna()]
                # df[col][df[col].isnull()] = df[pcol][df[col].isnull()]

            # else:
                # # raise Exception(f"\n{na}\n{m}\n")
                # raise Exception("\n{}\n{}\n".format(na, m))

    # return df


# def test_data_na(df):
    # for col in df.columns:
        # for i, d in enumerate(df[col]):
            # if math.isnan(d):
                # # raise Exception(f"data should not be nan, col: {col}, {i}")
                # raise Exception("data should not be nan, col: {}, {}".format(col, i))


def calculate_statistic(
    df, cols, progress_report=20,
):
    statistic = {}

    progress = 0.0
    works = float(len(cols) * len(cols))

    for src in cols:

        if statistic.get(src, None) is None:
            statistic[src] = {}

        for tar in cols:

            dsrc = df[src]
            dtar = df[tar]

            tau, p_value = stats.kendalltau(dsrc, dtar, nan_policy="omit")

            if tau == np.nan or p_value == np.nan:
                pass

            if math.isnan(tau) or math.isnan(p_value):
                pass

            progress += 1

            if progress % progress_report == 0:
                # print(f"{round((progress/works)*100.0,4)}%.....")
                print("{}%.....".format(round((progress / works) * 100.0, 4)))

            if statistic[src].get(tar, None) is None:
                statistic[src][tar] = {}

            statistic[src][tar] = {
                "p": str(p_value),
                "tau": str(tau),
            }

    return statistic


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", default="data.csv", metavar="", type=str, help="source directory"
    )

    parser.add_argument(
        "--output",
        default="statistic.json",
        metavar="",
        type=str,
        help="destination directory",
    )

    args = vars(parser.parse_args())

    if not args.get("input") or not args.get("output"):
        print("please specifiy input csv file and output json file")
        sys.exit(1)

    input_file = cast(str, args.get("input"))
    output_file = cast(str, args.get("output"))

    # print(f"input file: {input_file}")
    # print(f"output file: {output_file}")
    print("input file: {}".format(input_file))
    print("output file: {}".format(output_file))

    if not input_file.endswith(".csv") or not output_file.endswith(".json"):
        print("please specify valid input csv file and output json file")
        sys.exit(1)

    df = pd.read_csv(input_file)

    df = df[:19]
    df = df.astype(np.float64)

    df = utils.clean_data(df)

    # print(f"total columns: {len(df.columns)}")
    print("total columns: {}".format(len(df.columns)))

    print()

    print("testing dataframe.....")

    utils.test_data_na(df)

    print("finish testing dataframe.....")

    print()

    cols = df.columns[1:]

    print("calculating statistic.....")

    statistic = calculate_statistic(df, cols, progress_report=500,)

    print("finish calculating statistic.....")

    print()

    with open(output_file, "w") as output:
        json.dump(statistic, output, indent=2)
