############################################################################

import gc
import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

############################################################################

ROOT = os.getenv("HOME")

# CATEGORY = os.path.join(ROOT, "data_science/Predict_Future_Sales/item_categories.csv")
# SHOP = os.path.join(ROOT, "data_science/Predict_Future_Sales/shops.csv")

ITEM = os.path.join(ROOT, "data_science/Predict_Future_Sales/items.csv")
TRAIN = os.path.join(ROOT, "data_science/Predict_Future_Sales/sales_train.csv")

TEST = os.path.join(ROOT, "data_science/Predict_Future_Sales/test.csv")

JOIN_METHOD = "outer"

############################################################################

item_df = pd.read_csv(ITEM)

train_df = pd.read_csv(TRAIN)

test_df = pd.read_csv(TEST)

############################################################################


def preprocess(df, predicting=False, verify_sample=2000):
    df.loc[:, "item_category_id"] = item_df.loc[
        df["item_id"], "item_category_id"
    ].values

    choice = np.random.choice(len(df), verify_sample)
    choose = df.loc[choice]
    assert (
        choose["item_category_id"].values
        == item_df.loc[choose["item_id"], "item_category_id"].values
    ).all() == True

    return df


############################################################################

df = preprocess(train_df)

############################################################################

# df.columns

df.head(10)

df[["item_id", "item_category_id", "shop_id", "item_price"]]

############################################################################

# df.groupby(["item_id", "item_category_id"])

############################################################################

selector = ["date_block_num", "shop_id", "item_id"]

dfg = df.groupby(selector)

############################################################################

shop_item_cnt = dfg["item_cnt_day"].sum()
shop_item_cnt.name = "shop_item_cnt_month"

shop_item_cnt

############################################################################

# shop_item_cnt = shop_item_cnt.groupby(["shop_id", "item_id"]).mean()
# shop_item_cnt

############################################################################

df = pd.merge(df, shop_item_cnt, on=selector, how=JOIN_METHOD)
df.head(5)

############################################################################

df[["shop_item_cnt_month", "shop_id", "item_id"]]

############################################################################

selector = ["date_block_num", "shop_id", "item_category_id"]

dfg = df.groupby(selector)

############################################################################

shop_category_cnt = dfg["item_cnt_day"].sum()
shop_category_cnt.name = "shop_category_cnt_month"

shop_category_cnt

############################################################################

df = pd.merge(df, shop_category_cnt, on=selector, how=JOIN_METHOD)

############################################################################

df.columns

############################################################################

selector = ["date_block_num", "item_id"]

dfg = df.groupby(selector)

############################################################################

item_price_statistic = dfg.agg({"item_price": [np.max, np.min, np.mean]})

item_price_statistic.columns = ["item_price_max", "item_price_min", "item_price_mean"]

############################################################################

df = pd.merge(df, item_price_statistic, on=selector, how=JOIN_METHOD)

############################################################################

selector = ["date_block_num", "item_category_id"]

dfg = df.groupby(selector)

############################################################################

category_price_statistic = dfg.agg({"item_price": [np.max, np.min, np.mean]})

category_price_statistic.columns = [
    "category_price_max",
    "category_price_min",
    "category_price_mean",
]

############################################################################

df = pd.merge(df, category_price_statistic, on=selector, how=JOIN_METHOD)

############################################################################

selector = ["date_block_num", "shop_id"]

dfg = df.groupby(selector)

############################################################################

shop_price_statistic = dfg.agg({"item_price": [np.max, np.min, np.mean]})

shop_price_statistic.columns = [
    "shop_price_max",
    "shop_price_min",
    "shop_price_mean",
]

############################################################################

df = pd.merge(df, shop_price_statistic, on=selector, how=JOIN_METHOD)

############################################################################

# df = df.groupby(["shop_id", "item_id", "item_category_id"]).mean().reset_index()
# df = df.groupby(["shop_id", "item_id"]).mean().reset_index()

############################################################################

# df.isna().any()

############################################################################
############################################################################

shop_item_count = df.groupby("shop_id")["item_id"].count()
shop_item_count.name = "shop_item_count"

df = pd.merge(df, shop_item_count, on="shop_id", how=JOIN_METHOD)

############################################################################

category_item_count = df.groupby("item_category_id")["item_id"].count()
category_item_count.name = "category_item_count"

df = pd.merge(df, category_item_count, on="item_category_id", how=JOIN_METHOD)

############################################################################

# df.columns
# df[["shop_id", "item_id", "shop_item_cnt_month"]].head(10)
# df.groupby(["shop_id", "item_id"]).ngroup()

############################################################################


############################################################################
############################################################################
############################################################################

Y = df["shop_item_cnt_month"]
X = df.drop(["date", "date_block_num", "item_cnt_day", "shop_item_cnt_month"], axis=1)
# X = df.drop(["date_block_num", "item_cnt_day", "shop_item_cnt_month"], axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)

############################################################################


model = LGBMRegressor(
    n_estimators=8192, learning_rate=0.025, num_leaves=4096, max_bin=350, n_jobs=8,
    # n_estimators=16384, learning_rate=0.025, num_leaves=8192, max_bin=500, n_jobs=8,
)


############################################################################

model.fit(
    X_train,
    Y_train,
    eval_metric="mse",
    early_stopping_rounds=10,
    eval_set=(X_valid, Y_valid),
    verbose=3,
)

############################################################################

pred = model.predict(X_valid)
error = mean_squared_error(Y_valid, pred)
print("Error:", error)

############################################################################

gc.collect()

############################################################################

df.columns

############################################################################

test_df = preprocess(test_df)

############################################################################

test = pd.merge(
    test_df,
    # df.drop(["date", "date_block_num", "item_cnt_day", "shop_item_cnt_month"], axis=1),
    df.drop(["date_block_num", "item_cnt_day", "shop_item_cnt_month"], axis=1),
    on=["shop_id", "item_id", "item_category_id"],
)

############################################################################

pred = model.predict(test.drop("ID", axis=1))

############################################################################

test[["shop_id", "item_id", "item_category_id", "shop_price_mean"]]
# submission = pd.DataFrame({"ID": test["ID"], "item_cnt_month": pred,})

# submission = submission.set_index("ID")

# submission.to_csv("submission.csv")

############################################################################
############################################################################
############################################################################
############################################################################
