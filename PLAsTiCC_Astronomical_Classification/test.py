import pandas as pd
from scipy import stats

import os

ROOT = os.path.join("/run/media/neko/HDD/ARTIFICIAL_INTELLIGENCE_DATA_SCIENCE",
                    "KAGGLE_COMPETITIONS")
COMPETITION = "PLAsTiCC_Astronomical_Classification/all"

TRAIN_CSV = os.path.join(ROOT, COMPETITION, "training_set.csv")
TRAIN_METADATA_CSV = os.path.join(ROOT, COMPETITION,
                                  "training_set_metadata.csv")

TEST_CSV = os.path.join(ROOT, COMPETITION, "test_set.csv")
TEST_SAMPLE_CSV = os.path.join(ROOT, COMPETITION, "test_set_sample.csv")
TEST_METADATA_CSV = os.path.join(ROOT, COMPETITION, "test_set_metadata.csv")

assert (os.path.exists(TRAIN_CSV))
assert (os.path.exists(TRAIN_METADATA_CSV))
assert (os.path.exists(TEST_CSV))
assert (os.path.exists(TEST_SAMPLE_CSV))
assert (os.path.exists(TEST_METADATA_CSV))

f = pd.read_csv(TRAIN_CSV)
fm = pd.read_csv(TRAIN_METADATA_CSV)

aggs = {
    # "mjd": ["min", "max", "size"],
    "passband": ["min", "max", "mean", "median", "std", "sum", "skew"],
    "flux": ["min", "max", "mean", "median", "std", "skew"],
    "flux_err": ["min", "max", "mean", "median", "std", "skew"],
    "detected": ["mean", "median", "sum", "skew"],
    # "flux_ratio_sq": ["sum", "skew"],
    # "flux_by_flux_ratio_sq": ["sum", "skew"],
}

fagg = f.groupby("object_id").agg(aggs)
new_columns = [k + "_" + agg for k in aggs.keys() for agg in aggs[k]]
fagg.columns = new_columns

df_train = fagg.reset_index().merge(right=fm, on="object_id", how="outer")

# id_label = df_train[["object_id", "target"]]

print(df_train.head())
# print(fm.groupby("target"))
# print(df_train.iloc[0])
# print(df_train.iloc[0].drop("object_id").values)
# print(id_label.head())
