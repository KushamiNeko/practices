import mxnet as mx
from mxnet import nd
import numpy as np
import pandas as pd

import typing

import os
import csv
# import sys
# import subprocess

import config
import preprocess
import helper
import net

###############################################################################


def test_generator():
    df = pd.read_csv(config.TEST_CSV_FILE, dtype="str")

    # for s in df["drawing"]:
    for k, s in df[["key_id", "drawing"]].values:
        img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
        img[:, :, 0] = helper.draw_strokes(s)

        yield k, img


###############################################################################


def return_value(x):
    for k in x.keys():
        return x[k]


###############################################################################


def return_keys(x):
    for k in x.keys():
        return k


###############################################################################

if __name__ == "__main__":

    classification: typing.List[typing.Dict[str, float]] = []

    dictionary = preprocess.LabelDictionary()

    files = os.listdir(config.TRAIN_CSV_FILES)

    conv_net = net.ConvNet(train=False, initialize=False, hibridize=True)

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
        nd_labels = nd.array([0])

        nd_data = nd.transpose(nd_data, (0, 3, 1, 2))

        nd_data = nd_data.as_in_context(config.CTX)
        nd_labels = nd_labels.as_in_context(config.CTX)

        for f in os.listdir(config.TRAIN_CSV_FILES):
            label = helper.get_label_from_filepath(f)
            params = os.path.join("check_points/separate_training",
                                  "{}_1.params".format(label))

            conv_net.load_parameters(params)
            # conv_net = net.ConvNet(
            # parameters_file=params,
            # train=False,
            # initialize=False,
            # hibridize=False)

            # with autograd.record():
            output = conv_net.net(nd_data)
            loss = conv_net.loss(output, nd_labels)

            # print(loss.asscalar())

            classification.append({label: loss.asscalar()})

            # if label == "calculator":
            # break

        classification.sort(key=return_value)

        # print(dictionary.get_label_from_index(y))
        # print(classification)

        with open("classification.csv", "a", newline="") as csvfile:
            writter = csv.writer(
                csvfile,
                delimiter=',',
                # quotechar=',',
                quoting=csv.QUOTE_MINIMAL)
            writter.writerow([
                key, " ".join([
                    return_keys(classification[0]),
                    return_keys(classification[1]),
                    return_keys(classification[2])
                ])
            ])

        # break

    # loss.backward()
    # conv_net.trainer.step(nd_data.shape[0])

    # print("Loss: {}".format(nd.mean(loss).asscalar()))
