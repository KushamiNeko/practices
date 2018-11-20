import os

import pandas as pd
import numpy as np

import config

for idx, file in enumerate(os.listdir(config.TRAIN_CSV_FILES)):

    # if idx not in LABEL_DICT:
    # LABEL_DICT[idx] = os.path.splitext(file)[0]

    f = pd.read_csv(os.path.join(config.TRAIN_CSV_FILES, file), dtype="str")

    # print(f["word"])

    for s in f["word"]:
        if s != "broccoli":
            print(s)

    # for s in f["drawing"]:
    for s in f:
        print(s)

        break

    break
# img = np.zeros((SIZE, SIZE, 1), np.float32)
# img = np.zeros((RESIZE, RESIZE, 1), np.float32)
# img[:, :, 0] = draw_strokes(s)

# index = get_label_index(file)

# yield [img, index]

# def csv_extract(csv_file):
# f = pd.read_csv(csv_file, dtype="str")

# for s in f["drawing"]:
# img = np.zeros((RESIZE, RESIZE, 1), np.float32)
# img[:, :, 0] = draw_strokes(s)

# index = get_label_index(file)

# yield [img, index]

# def train_data_extract():
# for idx, file in enumerate(os.listdir(config.TRAIN_CSV_FILES)):

# # if idx not in LABEL_DICT:
# # LABEL_DICT[idx] = os.path.splitext(file)[0]

# f = pd.read_csv(os.path.join(config.TRAIN_CSV_FILES, file), dtype="str")

# for s in f["drawing"]:
# # img = np.zeros((SIZE, SIZE, 1), np.float32)
# img = np.zeros((RESIZE, RESIZE, 1), np.float32)
# img[:, :, 0] = draw_strokes(s)

# index = get_label_index(file)

# yield [img, index]
