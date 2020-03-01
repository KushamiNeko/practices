# %%

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

# %%


root = os.path.join(
    os.getenv("HOME"), "data_science", "PUBG_Finish_Placement_Prediction",
)

TRAIN_FILE = os.path.join(root, "train_V2.csv")
TEST_FILE = os.path.join(root, "test_V2.csv")

assert os.path.exists(TRAIN_FILE)
assert os.path.exists(TEST_FILE)

# %%

train = pd.read_csv(TRAIN_FILE)

# %%

# Id                  object
# groupId             object
# matchId             object
# assists              int64
# boosts               int64
# damageDealt        float64
# DBNOs                int64
# headshotKills        int64
# heals                int64
# killPlace            int64
# killPoints           int64
# kills                int64
# killStreaks          int64
# longestKill        float64
# matchDuration        int64
# matchType           object
# maxPlace             int64
# numGroups            int64
# rankPoints           int64
# revives              int64
# rideDistance       float64
# roadKills            int64
# swimDistance       float64
# teamKills            int64
# vehicleDestroys      int64
# walkDistance       float64
# weaponsAcquired      int64
# winPoints            int64
# winPlacePerc       float64
# dtype: object

# %%


def plot_scatter_correlation(colx, coly, sample_size=1000):

    choice = np.random.randint(0, len(train), size=sample_size)

    x = train.iloc[choice][colx]
    y = train.iloc[choice][coly]

    # h = train.iloc[choice]["matchType"]

    yrange = y.max() - y.min()
    yratio = 0.1

    f, ax = plt.subplots()

    s, i, _, _, _ = stats.linregress(x, y)

    # sns.scatterplot(x=x, y=y, ax=ax)
    ax.scatter(x=x, y=y)

    ax.plot(x, x * s + i, color="k")

    ax.set_xlabel(colx)
    ax.set_ylabel(coly)

    ax.set_ylim((y.min() - (yrange * yratio), y.max() + (yrange * yratio)))

    plt.show()

    return f


# %%

plt.close(plt.gcf())
gc.collect()

# %%

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

# %%

X = train.drop(deprecated_columns, axis=1)

# %%

# corr = X[~X["winPlacePerc"].isna()].corr()
# corr_sp = df[~df["winPlacePerc"].isna()].corr(method="spearman")


# %%

# winPlacePerc       1.000000
# walkDistance       0.810888
# boosts             0.634234
# weaponsAcquired    0.583806
# damageDealt        0.440507
# heals              0.427857
# kills              0.419916
# longestKill        0.410154
# killStreaks        0.377566
# rideDistance       0.342915
# assists            0.299441
# DBNOs              0.279970
# headshotKills      0.277722
# revives            0.240881
# swimDistance       0.149607
# vehicleDestroys    0.073436
# numGroups          0.039621
# maxPlace           0.037377
# roadKills          0.034544
# teamKills          0.015943
# matchDuration     -0.005171
# killPlace         -0.719069


drop_columns = [
    "teamKills",
    "matchDuration",
    "maxPlace",
    "numGroups",
    "vehicleDestroys",
]

# %%

encoder = LabelEncoder()
encoder.fit(train["matchType"])

# %%

X["matchType"] = encoder.transform(train["matchType"])
X = X.drop(drop_columns, axis=1)
X = X.drop(id_columns, axis=1)
X = X.drop(X[X["winPlacePerc"].isna()].index)

# %%

y = X["winPlacePerc"]
X = X.drop(["winPlacePerc"], axis=1)

# %%

# normalizer = Normalizer()
# normalizer.fit(X)

# %%

X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, test_size=0.1)

# X_train, X_valid, Y_train, Y_valid = train_test_split(
# normalizer.transform(X), y, test_size=0.2
# )

# %%


# model = LGBMRegressor(n_estimators=1000, num_leaves=50, n_jobs=8)

gc.collect()

model = LGBMRegressor(
    n_estimators=2048, learning_rate=0.025, num_leaves=1024, max_bin=300, n_jobs=8,
)


# %%


# model.fit(X_train, Y_train)

model.fit(
    X_train,
    Y_train,
    eval_metric="mae",
    early_stopping_rounds=7,
    eval_set=(X_valid, Y_valid),
    verbose=3,
)

gc.collect()

# pred = model.predict(X_valid)
# error = mean_absolute_error(Y_valid, pred)
# print("Error:", error)

# %%

# plot_scatter_correlation("walkDistance", "winPlacePerc")

# %%

test = pd.read_csv(TEST_FILE)

# %%

# test

# %%

# id_columns = [
# "Id",
# "groupId",
# "matchId",
# ]

TX = test.drop(deprecated_columns, axis=1)
TX["matchType"] = encoder.transform(test["matchType"])
TX = TX.drop(drop_columns, axis=1)
TX = TX.drop(["groupId", "matchId"], axis=1)

# %%

pred = model.predict(TX.drop(["Id"], axis=1))

# %%

TX["winPlacePerc"]= pred


# %%

TX[["Id", "winPlacePerc"]].set_index("Id").to_csv("submission.csv")


# %%


# %%
