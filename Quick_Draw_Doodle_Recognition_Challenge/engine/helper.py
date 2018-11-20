import numpy as np
import pandas as pd
import cv2

###############################################################################

import config
import preprocess
import json
import os

###############################################################################


def draw_strokes(strokes, size=config.SIZE, linewidth=4):
    img = np.ones((size, size), np.float32)

    for stroke in json.loads(strokes):
        x = stroke[0]
        y = stroke[1]

        for i in range(len(x) - 1):
            cv2.line(img, (x[i], y[i]), (x[i + 1], y[i + 1]), 0, linewidth)

    img = cv2.resize(img, (config.RESIZE, config.RESIZE))

    return img


###############################################################################


def train_data_extract(label_dictionary: preprocess.LabelDictionary):
    for idx, f in enumerate(os.listdir(config.TRAIN_CSV_FILES)):

        df = pd.read_csv(os.path.join(config.TRAIN_CSV_FILES, f), dtype="str")

        for s in df["drawing"]:
            # img = np.zeros((SIZE, SIZE, 1), np.float32)
            img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
            img[:, :, 0] = draw_strokes(s)

            index = label_dictionary.get_index_from_label(
                label_dictionary.get_label_from_filename(f))

            yield [img, index]


###############################################################################


def generator(csv_filepath: str, label_dictionary: preprocess.LabelDictionary):
    df = pd.read_csv(csv_filepath, dtype="str")

    for s in df["drawing"]:
        img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
        img[:, :, 0] = draw_strokes(s)

        index = label_dictionary.get_index_from_label(
            label_dictionary.get_label_from_filename(csv_filepath))

        yield [img, index]


###############################################################################
