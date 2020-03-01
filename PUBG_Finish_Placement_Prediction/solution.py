#######################################################
import gc
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from xgboost import XGBRegressor

matplotlib.use("gtk3agg")

#######################################################


def plot_scatter_correlation(colx, coly, sample_size=1000):

    choice = np.random.randint(0, len(train), size=sample_size)

    x = train.iloc[choice][colx]
    y = train.iloc[choice][coly]

    yrange = y.max() - y.min()
    yratio = 0.1

    f, ax = plt.subplots()

    s, i, _, _, _ = stats.linregress(x, y)

    ax.scatter(x=x, y=y)

    ax.plot(x, x * s + i, color="k")

    ax.set_xlabel(colx)
    ax.set_ylabel(coly)

    ax.set_ylim((y.min() - (yrange * yratio), y.max() + (yrange * yratio)))

    plt.show()

    return f


#######################################################


root = os.path.join(
    os.getenv("HOME"), "data_science", "PUBG_Finish_Placement_Prediction",
)

TRAIN_FILE = os.path.join(root, "train_V2.csv")
TEST_FILE = os.path.join(root, "test_V2.csv")

assert os.path.exists(TRAIN_FILE)
assert os.path.exists(TEST_FILE)

#######################################################

train = pd.read_csv(TRAIN_FILE)

#######################################################

encoder = LabelEncoder()
encoder.fit(train["matchType"])

#######################################################

# corr = train.corr(method="spearman")

#######################################################

# corr["winPlacePerc"].sort_values(ascending=False)

# winPlacePerc       1.000000
# walkDistance       0.866519
# boosts             0.681427
# weaponsAcquired    0.666185
# heals              0.565478
# longestKill        0.456501
# damageDealt        0.448591
# kills              0.432542
# rideDistance       0.432261
# killStreaks        0.393392
# assists            0.299501
# headshotKills      0.282433
# DBNOs              0.258729
# revives            0.253557
# swimDistance       0.234142
# vehicleDestroys    0.073959
# rankPoints         0.064634
# numGroups          0.050263
# maxPlace           0.045357
# winPoints          0.039435
# roadKills          0.038608
# teamKills          0.023208
# killPoints         0.016520
# matchDuration     -0.002240
# killPlace         -0.720767

#######################################################


def preprocessing(X, encoder=encoder):
    id_columns = [
        "Id",
        "groupId",
        "matchId",
    ]

    deprecated_columns = [
        "rankPoints",
        "winPoints",
        "killPoints",
    ]

    uncorrelated_columns = [
        # "teamKills",
        # "roadKills",
        "matchDuration",
        # "maxPlace",
        # "numGroups",
        # "vehicleDestroys",
    ]

    X = X.drop(X[X["winPlacePerc"].isna()].index)

    X = X.drop(deprecated_columns, axis=1)
    X = X.drop(uncorrelated_columns, axis=1)

    X["matchType"] = encoder.fit_transform(X["matchType"])

    return X


#######################################################

gc.collect()

X = train.copy()

#######################################################

match_aggs = {
    "teamKills": ["min", "max", "median", "mean", "std", "skew"],
    "roadKills": ["min", "max", "median", "mean", "std", "skew"],
    "vehicleDestroys": ["min", "max", "median", "mean", "std", "skew"],
    "numGroups": ["min", "max", "median", "mean", "std", "skew"],
    "maxPlace": ["min", "max", "median", "mean"],
    "walkDistance": ["min", "max", "median", "mean", "std", "skew"],
    "boosts": ["min", "max", "median", "mean", "std", "skew"],
    "weaponsAcquired": ["min", "max", "median", "mean", "std", "skew"],
    "heals": ["min", "max", "median", "mean", "std", "skew"],
    "longestKill": ["min", "max", "median", "mean", "std", "skew"],
    "damageDealt": ["min", "max", "median", "mean", "std", "skew"],
    "kills": ["min", "max", "median", "mean", "std", "skew"],
    "rideDistance": ["min", "max", "median", "mean", "std", "skew"],
    "killStreaks": ["min", "max", "median", "mean", "std", "skew"],
    "assists": ["min", "max", "median", "mean", "std", "skew"],
    "headshotKills": ["min", "max", "median", "mean", "std", "skew"],
    "DBNOs": ["min", "max", "median", "mean", "std", "skew"],
    "revives": ["min", "max", "median", "mean", "std", "skew"],
    "swimDistance": ["min", "max", "median", "mean", "std", "skew"],
    "killPlace": ["min", "max", "median", "mean", "std", "skew"],
}

group_aggs = {
    "teamKills": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "roadKills": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "vehicleDestroys": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "walkDistance": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "boosts": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "weaponsAcquired": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "heals": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "longestKill": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "damageDealt": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "kills": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "rideDistance": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "killStreaks": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "assists": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "headshotKills": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "DBNOs": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "revives": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "swimDistance": ["min", "max", "median", "mean", "std", "skew", "sum"],
    "killPlace": ["min", "max", "median", "mean", "std", "skew"],
}

match_stats = X.groupby("matchId").agg(match_aggs)
group_stats = X.groupby(["matchId", "groupId"]).agg(group_aggs)

#######################################################

match_columns = [f"match_{k}_{v}" for k in match_aggs.keys() for v in match_aggs[k]]
match_stats.columns = match_columns

group_columns = [f"group_{k}_{v}" for k in group_aggs.keys() for v in group_aggs[k]]
group_stats.columns = group_columns

#######################################################
#######################################################
#######################################################

# df_agg = X.groupby("matchId").agg(aggs)
# df_agg = X.groupby(["matchId", "groupId"]).agg(aggs)
# df_agg = X.groupby("groupId").agg(aggs)


# def agg(df):

# aggs = {
# "passband": ["min", "max", "median", "mean", "std", "skew"],
# "flux": ["min", "max", "mean", "median", "std", "skew"],
# "flux_err": ["min", "max", "mean", "median", "std", "skew"],
# "detected": ["sum"],
# }

# df_agg = df.groupby("object_id").agg(aggs)

# new_columns = [k + "_" + v for k in aggs.keys() for v in aggs[k]]
# df_agg.columns = new_columns

# return df_agg

# X = df_agg.reset_index().merge(right=dfm, on="object_id", how="outer")


#######################################################


def player_statistic(X):
    X["headshot_rate"] = X["headshotKills"] / (X["kills"] + 0.00001)
    X["kill_streak_rate"] = X["killStreaks"] / (X["kills"] + 0.00001)
    X["kills_assists"] = X["kills"] + X["assists"]
    X["heals_boosts"] = X["heals"] + X["boosts"]
    X["total_distance"] = X["walkDistance"] + X["rideDistance"] + X["swimDistance"]
    X["kills_assists_per_heal_boost"] = X["kills_assists"] / (X["heals_boosts"] + 1)
    X["damageDealt_per_heal_boost"] = X["damageDealt"] / (X["heals_boosts"] + 1)
    X["road_kills_per_rideDistance"] = X["roadKills"] / (X["rideDistance"] + 0.01)
    X["maxPlace_per_numGroups"] = X["maxPlace"] / X["numGroups"]
    X["assists_per_kill"] = X["assists"] / (X["kills"] + X["assists"] + 0.0001)
    X["killPlace"] = X["killPlace"] - 1
    return X


def group_statistic(X):
    group_cols = [
        "matchId",
        "groupId",
        "matchDuration",
        "matchType",
        "maxPlace",
        "numGroups",
        "maxPlace_per_numGroups",
        "winPlacePerc",
        "killPlace",
    ]
    if "winPlacePerc" not in X.columns:
        group_cols.remove("winPlacePerc")

    pl_data_grouped = X[group_cols].groupby(["matchId", "groupId"])
    gr_data = pl_data_grouped.first()
    gr_data.drop(columns="killPlace", inplace=True)

    gr_data["raw_groupSize"] = pl_data_grouped["numGroups"].count()
    gr_data["groupSize"] = gr_data["raw_groupSize"]
    gr_data["group_size_overflow"] = (gr_data["groupSize"] > 4).astype(np.int8)
    gr_data.loc[
        gr_data["groupSize"] > 4, ["groupSize"]
    ] = 2  # replace group sizes with median, since it's a bug, max group size is 4

    gr_data["meanGroupSize"] = gr_data.groupby("matchId")["groupSize"].transform(
        np.mean
    )
    gr_data["medianGroupSize"] = gr_data.groupby("matchId")["groupSize"].transform(
        np.median
    )
    # gr_data['maxGroupSize'] = gr_data.groupby('matchId')['groupSize'].transform(np.max)
    # gr_data['minGroupSize'] = gr_data.groupby('matchId')['groupSize'].transform(np.min)
    gr_data["maxKillPlace"] = (
        pl_data_grouped["killPlace"].max().groupby("matchId").transform(np.max)
    )

    gr_data["totalPlayers"] = gr_data.groupby("matchId")["groupSize"].transform(sum)
    # some matches have missing players, so I adjust the total number of players to account for that
    gr_data["totalPlayersAdjusted"] = (
        gr_data["maxPlace"].astype(float)
        * gr_data["totalPlayers"]
        / (gr_data["numGroups"] + 0.01)
    )
    # trim total number to 100 as it can't be higher than that
    gr_data["totalPlayersAdjusted"] = gr_data["totalPlayersAdjusted"].apply(
        lambda x: np.minimum(100.0, x)
    )
    # gr_data.drop(columns=['totalPlayers'], inplace=True)
    gr_data["num_opponents"] = gr_data["totalPlayersAdjusted"] - gr_data["groupSize"]

    X = X.merge(
        gr_data[
            [
                "num_opponents",
                "totalPlayersAdjusted",
                "groupSize",
                "raw_groupSize",
                "maxKillPlace",
            ]
        ],
        on=["matchId", "groupId"],
    )

    print("group size counts:")
    print(X["raw_groupSize"].value_counts())

    # normalizing some features
    X["revives_per_groupSize"] = X["revives"] / (X["groupSize"] - 1 + 0.001)
    X["kills_assists_norm_both"] = (
        X["kills_assists"].astype(np.float32) / X["num_opponents"] / X["matchDuration"]
    )

    X["killPlace_norm"] = X["killPlace"] / (X["maxKillPlace"] + 0.000001)

    # X['kills_assists_norm_opp_n'] = X['kills_assists'].astype(np.float32) / X['num_opponents']
    # X['kills_assists_norm_dur'] = X['kills_assists'].astype(np.float32) / X['matchDuration']
    X["damageDealt_norm_both"] = (
        X["damageDealt"].astype(np.float32) / X["num_opponents"] / X["matchDuration"]
    )
    # X['damageDealt_norm_opp_n'] = X['damageDealt'].astype(np.float32) / X['num_opponents']
    # X['damageDealt_norm_dur'] = X['damageDealt'].astype(np.float32) / X['matchDuration']
    X["DBNOs_norm"] = (
        X["DBNOs"].astype(np.float32) / X["num_opponents"] / X["matchDuration"]
    )
    X["heals_norm"] = X["heals"].astype(np.float32) / X["matchDuration"]
    X["boosts_norm"] = (
        X["boosts"].astype(np.float32) / X["matchDuration"]
    )  # - lowers correlation, don't do
    X["walkDistance_norm"] = X["walkDistance"].astype(np.float32) / X["matchDuration"]
    X["rideDistance_norm"] = X["rideDistance"].astype(np.float32) / X["matchDuration"]
    X["swimDistance_norm"] = X["swimDistance"].astype(np.float32) / X["matchDuration"]

    # gr_data.drop(columns=['groupSize'], inplace=True)

    gr_data = reduce_mem_usage(gr_data)
    gr_data.drop(columns=list(set(gr_data.columns) & all_useless_cols))
    return gr_data, X


def group_and_match_statistics(data, gr_data):

    group_stats_cols = [
        "assists",
        "boosts",
        "DBNOs",
        "killPoints",
        "longestKill",
        "rankPoints",
        "road_kills_per_rideDistance",
        "kills_assists_norm_both",
        "damageDealt_norm_both",
        "DBNOs_norm",
        "heals_boosts",
        "assists_per_kill",
        "killPlace_norm",
        "revives",
        "roadKills",
        "teamKills",
        "vehicleDestroys",
        "weaponsAcquired",
        "winPoints",
        "headshot_rate",
        "kill_streak_rate",
        "kills_assists",
        "heals_norm",
        "walkDistance_norm",
        "rideDistance_norm",
        "swimDistance_norm",
        "damageDealt_per_heal_boost",
        "kills_assists_per_heal_boost",
    ]
    # removed damageDealt
    match_stats_cols = [
        "assists",
        "boosts",
        "DBNOs",
        "killPoints",
        "longestKill",
        "rankPoints",
        "road_kills_per_rideDistance",
        "kills_assists_norm_both",
        "damageDealt_norm_both",
        "DBNOs_norm",
        "heals_boosts",
        "assists_per_kill",
        "revives",
        "roadKills",
        "teamKills",
        "vehicleDestroys",
        "weaponsAcquired",
        "winPoints",
        "headshot_rate",
        "kill_streak_rate",
        "kills_assists",
        "heals_norm",
        "walkDistance_norm",
        "rideDistance_norm",
        "swimDistance_norm",
        "damageDealt_per_heal_boost",
        "kills_assists_per_heal_boost",
    ]

    pl_data_grouped_by_group = data.groupby(["matchId", "groupId"])
    pl_data_grouped_by_match = data.groupby(["matchId"])

    # group_sizes = pl_data_grouped_by_group['groupSize'].count().values.reshape([-1])
    # fixed_group_sizes = pl_data_grouped_by_group['groupSize'].first().values.reshape([-1])
    # sum_multipliers = fixed_group_sizes.astype(np.float32) / group_sizes
    # print('min multiplier: {:.2f}, max multiplier: {:.2f}'.format(np.min(sum_multipliers), np.max(sum_multipliers)))
    # print('min group size: {:d}, max group size: {:d}'.format(np.min(group_sizes), np.max(group_sizes)))
    # print('min fixed group size: {:d}, max fixed group size: {:d}'.format(np.min(fixed_group_sizes), np.max(fixed_group_sizes)))
    # print(pd.Series(sum_multipliers).value_counts())

    group_funcs = {"min": np.min, "max": np.max, "sum": np.sum, "median": np.mean}
    match_funcs = {
        "min": np.min,
        "max": np.max,
        "sum": np.sum,
        "median": np.median,
        "std": np.std,
    }
    extra_group_stats = (
        pl_data_grouped_by_group[["matchId", "groupId"]].first().reset_index(drop=True)
    )
    extra_match_stats = (
        pl_data_grouped_by_match[["matchId"]].first().reset_index(drop=True)
    )

    for colname in group_stats_cols:
        for f_name, func in group_funcs.items():
            gr_col_name = f_name + "_group_" + colname
            if (gr_col_name not in all_useless_cols) or (
                (gr_col_name + "_rank") not in all_useless_cols
            ):
                if func is not sum:
                    extra_group_stats[gr_col_name] = (
                        pl_data_grouped_by_group[colname].agg(func).values
                    )
                else:
                    extra_group_stats[gr_col_name] = (
                        pl_data_grouped_by_group[colname].agg(func).values
                        * sum_multipliers
                    )

    for colname in match_stats_cols:
        for f_name, func in match_funcs.items():
            m_col_name = f_name + "_match_" + colname
            if m_col_name not in all_useless_cols:
                if func is np.std:
                    extra_match_stats[m_col_name] = (
                        pl_data_grouped_by_match[colname].agg(func).fillna(0).values
                    )
                elif func is np.min:
                    if m_col_name in min_match_useful_cols:
                        extra_match_stats[m_col_name] = (
                            pl_data_grouped_by_match[colname].agg(func).values
                        )
                else:
                    extra_match_stats[m_col_name] = (
                        pl_data_grouped_by_match[colname].agg(func).values
                    )

    extra_group_stats.set_index(["matchId", "groupId"], inplace=True)
    extra_match_stats.set_index(["matchId"], inplace=True)

    pl_data_grouped_by_group = None
    pl_data_grouped_by_match = None

    select_cols = []
    for col in extra_group_stats.columns:
        if ((col + "_rank") not in all_useless_cols) and (
            col not in ["matchId", "groupId"]
        ):
            select_cols.append(col)

    # adding rank information
    rank_data = extra_group_stats.groupby(["matchId"])
    rank_data = rank_data[select_cols].rank() - 1  # method='dense'
    gc.collect()
    max_rank_data = rank_data.groupby(["matchId"]).transform(np.max)
    rank_data = rank_data / (max_rank_data + 0.0001)
    max_rank_data = None
    gc.collect()
    print("rank data created")

    gr_col_to_drop = list(set(extra_group_stats.columns) & all_useless_cols)
    extra_group_stats.drop(columns=gr_col_to_drop, inplace=True)
    gc.collect()

    extra_group_stats = extra_group_stats.join(
        rank_data, on=["matchId", "groupId"], rsuffix="_rank"
    )
    extra_group_stats.reset_index(
        level=1, inplace=True
    )  # put groupId back into the columns

    rank_data = None
    gc.collect()
    print("rank data merged")
    extra_group_stats = reduce_mem_usage(extra_group_stats)
    extra_match_stats = reduce_mem_usage(extra_match_stats)

    merged_features = extra_group_stats.merge(extra_match_stats, on=["matchId"])
    extra_group_stats = None
    extra_match_stats = None
    gc.collect()
    print("extra match and group stats merged")
    merged_features = merged_features.merge(gr_data, on=["matchId", "groupId"])
    gr_data = None
    gc.collect()
    print("group data and stats merged")

    # one hot encoding of match type
    cats = merged_features["matchType"].unique()
    cats = set(cats) - set(all_useless_cols)
    encoded_data = np.empty(shape=(merged_features.shape[0], 0), dtype=np.int8)
    for category in cats:
        encoded_data = np.c_[
            encoded_data,
            (merged_features[["matchType"]] == category)
            .values.reshape(-1, 1)
            .astype(np.int8),
        ]
    encoded_data = pd.DataFrame(
        encoded_data, columns=cats, index=merged_features.index, dtype=np.int8
    )
    print("matchType data created")
    for col in encoded_data.columns:
        merged_features[col] = encoded_data[col]
    encoded_data = None
    gc.collect()
    print("match type data merged")
    cols_to_drop = ["matchType"]
    merged_features = merged_features.drop(columns=cols_to_drop)

    return merged_features


#######################################################


def reduce_mem_usage(df):
    # iterate through all the columns of a dataframe and modify the data type
    #   to reduce memory usage.

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


#######################################################
