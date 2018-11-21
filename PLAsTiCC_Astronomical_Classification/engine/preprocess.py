import pandas as pd

import config

f = pd.read_csv(config.TRAIN_CSV)
fm = pd.read_csv(config.TRAIN_METADATA_CSV)

aggs = {
    # "mjd": ["min", "max", "size"],
    "passband": ["min", "max", "mean", "median", "std", "sum", "skew"],
    "flux": ["min", "max", "mean", "median", "std", "skew"],
    "flux_err": ["min", "max", "mean", "median", "std", "skew"],
    "detected": ["mean", "median", "sum", "skew"],
    # "flux_ratio_sq": ["sum", "skew"],
    # "flux_by_flux_ratio_sq": ["sum", "skew"],
}

df_agg = f.groupby("object_id").agg(aggs)
new_columns = [k + "_" + v for k in aggs.keys() for v in aggs[k]]
df_agg.columns = new_columns

df_train = df_agg.reset_index().merge(right=fm, on="object_id", how="outer")
