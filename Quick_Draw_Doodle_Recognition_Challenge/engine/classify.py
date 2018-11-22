from mxnet import nd
import numpy as np
import pandas as pd

from typing import List, Dict

import os
import csv

import config
import preprocess
import helper
import net

###############################################################################


def test_generator():
    df = pd.read_csv(config.TEST_CSV_FILE, dtype="str")

    for k, s in df[["key_id", "drawing"]].values:
        img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
        img[:, :, 0] = helper.draw_strokes(s)

        yield k, img


###############################################################################

# def return_value(x):
# for k in x.keys():
# return x[k]

###############################################################################

# def return_keys(x):
# for k in x.keys():
# return k

###############################################################################

if __name__ == "__main__":

    classification: List[Dict[str, float]] = []

    dictionary = preprocess.LabelDictionary()

    files = os.listdir(config.TRAIN_CSV_FILES)

    conv_net = net.ConvNet(
        train=True,
        initialize=False,
        hibridize=True,
        parameters_file="check_points/third_attemp/145.params")

    generator = test_generator()
    # generator = helper.train_data_extract(dictionary)

    while True:
        try:
            key, x = next(generator)
            # x, y = next(generator)
        except StopIteration:
            print("end of the test file")
            exit(0)

        nd_data = nd.array([x])
        # nd_labels = nd.array([0])

        nd_data = nd.transpose(nd_data, (0, 3, 1, 2))

        nd_data = nd_data.as_in_context(config.CTX)
        # nd_labels = nd_labels.as_in_context(config.CTX)

        output = conv_net.net(nd_data)
        softmax = nd.softmax(output)
        prediction = nd.argmax(output, axis=1)

        id_rank = []

        for i, p in enumerate(output[0]):
            id_rank.append({i: p})
            id_rank.sort(key=lambda x: list(x.values())[0], reverse=True)

            if len(id_rank) > 3:
                del id_rank[len(id_rank) - 1]

        print(prediction)
        print(id_rank)
        print(dictionary.get_label_from_index(list(id_rank[0].keys())[0]))
        print(dictionary.get_label_from_index(list(id_rank[1].keys())[0]))
        print(dictionary.get_label_from_index(list(id_rank[2].keys())[0]))
        # print(output)
        # print(softmax)
        # break

        with open("classification.csv", "a", newline="") as csvfile:
            writter = csv.writer(
                csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            writter.writerow([
                key, " ".join([
                    dictionary.get_label_from_index(list(id_rank[0].keys())[0]),
                    dictionary.get_label_from_index(list(id_rank[1].keys())[0]),
                    dictionary.get_label_from_index(list(id_rank[2].keys())[0])
                ])
            ])

        # break
