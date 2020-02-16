import config
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
# from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRFClassifier


def agg(df):
    # dfm.drop(labels=["distmod", "hostgal_specz"], axis=1, inplace=True)

    aggs = {
        "passband": ["min", "max", "median", "mean", "std", "skew"],
        "flux": ["min", "max", "mean", "median", "std", "skew"],
        "flux_err": ["min", "max", "mean", "median", "std", "skew"],
        "detected": ["sum"],
    }

    df_agg = df.groupby("object_id").agg(aggs)

    new_columns = [k + "_" + v for k in aggs.keys() for v in aggs[k]]
    df_agg.columns = new_columns

    return df_agg


def preprocess(df, dfm, training=True):
    dfm.drop(labels=["distmod", "hostgal_specz"], axis=1, inplace=True)

    # aggs = {
    # "passband": ["min", "max", "median", "mean", "std", "skew"],
    # "flux": ["min", "max", "mean", "median", "std", "skew"],
    # "flux_err": ["min", "max", "mean", "median", "std", "skew"],
    # "detected": ["sum"],
    # }

    # df_agg = df.groupby("object_id").agg(aggs)

    # new_columns = [k + "_" + v for k in aggs.keys() for v in aggs[k]]
    # df_agg.columns = new_columns

    df_agg = agg(df)

    X = df_agg.reset_index().merge(right=dfm, on="object_id", how="outer")

    if training:
        encoder = LabelEncoder()
        encoder.fit(X["target"])

        y = encoder.transform(X["target"])

        X.drop(labels=["object_id", "target"], axis=1, inplace=True)

        X = X.astype(np.float)

        return X, y, encoder

    else:
        return X


if __name__ == "__main__":

    df = pd.read_csv(config.TRAIN_CSV)
    dfm = pd.read_csv(config.TRAIN_METADATA_CSV)

    X, y, encoder = preprocess(df, dfm)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2,
    )

    # model = XGBRFClassifier(
    # n_estimators=5000,
    # max_depth=15,
    # subsample=0.8,
    # n_jobs=8,
    # learning_rate=0.5,
    # # objective="multi:softmax",
    # num_class=y.max() + 1,
    # )

    model = RandomForestClassifier(n_estimators=5000, criterion="gini", n_jobs=8,)

    model.fit(
        X,
        y,
        # X_train,
        # y_train,
        # early_stopping_rounds=10,
        # eval_set=[(X_valid, y_valid)],
        # verbose=False,
    )

    pred = model.predict(X_valid)

    score = accuracy_score(y_valid, pred)
    print("Accuracy Score:", score)

    # dft = pd.read_csv(config.TEST_CSV)
    # dftm = pd.read_csv(config.TEST_METADATA_CSV)

    # TX = preprocess(df, dfm, training=False)

    # pred = model.predict(TX)

    # parameters = {
    # "n_estimators": [3000, 5000, 7000, 9000, 12000],
    # "max_depth": [10, 15, 20, 25, 30],
    # "n_jobs": [8],
    # "random_state": [int(time.time())],
    # "learning_rate": [0.05, 0.025, 0.075],
    # }

    # clf = GridSearchCV(model, parameters, cv=2, scoring="accuracy", verbose=3)

    # clf.fit(
    # X, y, early_stopping_rounds=7, eval_set=[(X_valid, y_valid)], verbose=True,
    # )

    # print(clf.cv_results_)
