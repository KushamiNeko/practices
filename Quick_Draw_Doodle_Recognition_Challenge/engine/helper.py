import numpy as np
import pandas as pd
import cv2
import random
import pickle

###############################################################################

import config
# import preprocess
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


def train_data_extract(label_dictionary):

    for idx, f in enumerate(os.listdir(config.TRAIN_CSV_FILES)):

        df = pd.read_csv(os.path.join(config.TRAIN_CSV_FILES, f), dtype="str")

        for s in df["drawing"]:
            # img = np.zeros((SIZE, SIZE, 1), np.float32)
            img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
            img[:, :, 0] = draw_strokes(s)

            index = label_dictionary.get_index_from_label(
                get_label_from_filepath(f))

            yield [img, index]


###############################################################################


def train_data_extract_limit(label_dictionary):

    random.seed()

    files = os.listdir(config.TRAIN_CSV_FILES)

    random.shuffle(files)

    for idx, f in enumerate(files):

        if os.path.exists("resource/pandas/{}.pickle".format(
                os.path.splitext(f)[0])):
            df_file = open(
                "resource/pandas/{}.pickle".format(os.path.splitext(f)[0]),
                "rb")
            df = pickle.load(df_file)
            df_file.close()
        else:
            df = pd.read_csv(
                os.path.join(config.TRAIN_CSV_FILES, f), dtype="str")

        if not os.path.exists("resource/pandas/{}.pickle".format(
                os.path.splitext(f)[0])):
            df_file = open(
                "resource/pandas/{}.pickle".format(os.path.splitext(f)[0]),
                'wb')
            pickle.dump(obj=df, file=df_file)
            df_file.close()

        i = random.randint(config.TRAIN_SAMPLES_LIMIT, len(df["drawing"]) - 1)

        for s in df["drawing"][i - config.TRAIN_SAMPLES_LIMIT:i]:
            # img = np.zeros((SIZE, SIZE, 1), np.float32)
            img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
            img[:, :, 0] = draw_strokes(s)

            index = label_dictionary.get_index_from_label(
                get_label_from_filepath(f))

            yield [img, index]


###############################################################################


def train_data_extract_random(label_dictionary):

    data = []

    files = os.listdir(config.TRAIN_CSV_FILES)

    random.shuffle(files)

    for idx, f in enumerate(files):
        # for idx, f in enumerate(os.listdir(config.TRAIN_CSV_FILES)):

        df = pd.read_csv(os.path.join(config.TRAIN_CSV_FILES, f), dtype="str")

        for s in df["drawing"][:config.TRAIN_SAMPLES_LIMIT]:
            # img = np.zeros((SIZE, SIZE, 1), np.float32)
            img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
            img[:, :, 0] = draw_strokes(s)

            index = label_dictionary.get_index_from_label(
                get_label_from_filepath(f))

            # yield [img, index]
            data.append((img, index))

    random.shuffle(data)

    for d in data:
        img, index = d
        yield [img, index]


###############################################################################


def get_label_from_filepath(filepath: str) -> str:
    label = os.path.splitext(os.path.basename(filepath))[0].replace(" ", "_")
    return label


###############################################################################


def generator(csv_filepath: str, label_dictionary):
    df = pd.read_csv(csv_filepath, dtype="str")

    for s in df["drawing"]:
        img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
        img[:, :, 0] = draw_strokes(s)

        index = label_dictionary.get_index_from_label(
            get_label_from_filepath(csv_filepath))

        yield [img, index]


###############################################################################


def random_generator(label_dictionary):
    random.seed()

    generators = []

    for f in os.listdir(config.TRAIN_CSV_FILES):
        g = generator(os.path.join(config.TRAIN_CSV_FILES, f), label_dictionary)
        generators.append(g)

    i = 0
    while True:
        try:
            if len(generators) == 0:
                break

            random.seed()
            i = random.randint(0, len(generators) - 1)

            img, index = next(generators[i])

            yield [img, index]
        except StopIteration:
            if len(generators) == 0:
                break

            del generators[i]

            continue


###############################################################################
