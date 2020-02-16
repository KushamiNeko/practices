import gc

import config
import numpy as np
import pandas as pd
import preprocess

columns = None


def readline_test_csv():
    global columns

    part = []
    object_id = None

    with open(config.TEST_CSV, "r") as f:
        line = f.readline()
        columns = line
        while line:
            line = f.readline()
            new_object_id = line.split(",", -1)[0]
            if object_id is None:
                object_id = new_object_id
                part.append(line)
            else:
                if new_object_id != object_id:
                    # print(new_object_id)
                    yield part

                    part = []
                    object_id = new_object_id

                else:
                    part.append(line)


def raw_df(batch_size=5, stop=None):
    global columns
    dft = None

    count = 0
    for part in readline_test_csv():
        count += 1

        raw = {}
        for i, k in enumerate(columns.split(",", -1)):
            raw[k.strip()] = [vs.split(",", -1)[i].strip() for vs in part]

        df_raw = pd.DataFrame(raw)
        df_raw = df_raw.astype(np.float)

        agg = preprocess.agg(df_raw)

        if dft is None:
            dft = agg.reset_index()
        else:
            dft = dft.append(agg.reset_index(), ignore_index=True)

        if stop is not None and count >= stop:
            break

        if count != 0 and count % batch_size == 0:
            yield dft
            dft = None
            gc.collect()

    if dft is not None:
        yield dft


if __name__ == "__main__":

    dftm = pd.read_csv(config.TEST_METADATA_CSV)

    dftm.drop(labels=["distmod", "hostgal_specz"], axis=1, inplace=True)

    dft = next(raw_df())

    dfti = dft.set_index("object_id")
    dftmi = dftm.set_index("object_id")

    TX = dft.merge(
        right=dftmi.loc[dfti.index].reset_index(), on="object_id", how="outer"
    )

    for i, dft in enumerate(raw_df()):
        # print(dft)

        if i > 1:
            break
