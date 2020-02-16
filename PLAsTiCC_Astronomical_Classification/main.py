import pickle

import config
import pandas as pd
import preprocess
import split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv(config.TRAIN_CSV)
    dfm = pd.read_csv(config.TRAIN_METADATA_CSV)

    X, y, encoder = preprocess.preprocess(df, dfm)

    # X_train, X_valid, y_train, y_valid = train_test_split(
    # X, y, train_size=0.8, test_size=0.2,
    # )

    model = RandomForestClassifier(n_estimators=5000, criterion="gini", n_jobs=8,)

    # model.fit(
    # X_train, y_train,
    # )

    # pred = model.predict(X_valid)

    # score = accuracy_score(y_valid, pred)
    # print("Accuracy Score:", score)

    model.fit(
        X, y,
    )

    # prediction

    dftm = pd.read_csv(config.TEST_METADATA_CSV)
    dftm.drop(labels=["distmod", "hostgal_specz"], axis=1, inplace=True)

    # processing

    preds = []

    for i, dft in enumerate(split.raw_df()):
        # dft = next(split.raw_df())

        dfti = dft.set_index("object_id")
        dftmi = dftm.set_index("object_id")

        TX = dft.merge(
            right=dftmi.loc[dfti.index].reset_index(), on="object_id", how="outer"
        )

        pred = model.predict(TX.drop(["object_id"], axis=1))

        preds.append(
            {"id": TX["object_id"], "prediction": encoder.inverse_transform(pred)}
        )

        print(encoder.inverse_transform(pred))

    print(preds)

    with open("predictions.pickle", "wb") as f:
        pickle.dump(preds, f)

    submit = []

    classes = list(encoder.classes_)
    classes.append(99)

    submit.append(
        # f"object_id,{','.join([f'class_{c}' for c in encoder.classes_])},class_99"
        f"object_id,{','.join([f'class_{c}' for c in classes])}"
    )

    for pred in preds:
        for i in range(len(pred["id"])):
            oid = int(pred["id"][i])
            p = pred["prediction"][i]

            submit.append(
                # f"{oid},{','.join(['1' if c == p else '0' for c in encoder.classes_ ])}"
                f"{oid},{','.join(['1' if c == p else '0' for c in classes ])}"
            )

    print(submit)

    with open("submit.pickle", "wb") as f:
        pickle.dump(submit, f)

    with open("submit.csv", "w") as f:
        f.write("\n".join(submit))
