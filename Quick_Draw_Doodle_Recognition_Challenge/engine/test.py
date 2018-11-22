import os

import pandas as pd
import numpy as np
import cv2

import config
import helper
import preprocess

dictionary = preprocess.LabelDictionary()

generator = helper.train_data_extract_limit(dictionary)

x, y = next(generator)
x, y = next(generator)

print(dictionary.get_label_from_index(y))
cv2.imshow('image', x)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for idx, file in enumerate(os.listdir(config.TRAIN_CSV_FILES)):

# # if idx not in LABEL_DICT:
# # LABEL_DICT[idx] = os.path.splitext(file)[0]

# f = pd.read_csv(os.path.join(config.TRAIN_CSV_FILES, file), dtype="str")

# # print(f["word"])

# for s in f["word"]:
# if s != "broccoli":
# print(s)

# # for s in f["drawing"]:
# for s in f:
# print(s)

# break

# break
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

# df = pd.read_csv(config.TEST_CSV_FILE, dtype="str")

# # print(df[["key_id", "drawing"]].head(1))

# i = 0
# for k, s in df[["key_id", "drawing"]].values:
# print(s)
# i += 1

# if i > 1:
# break

# i = 0
# for s in df["drawing"]:
# print(s)
# i += 1
# if i > 1:
# break
# print(s)
# img = np.zeros((config.RESIZE, config.RESIZE, 1), np.float32)
# img[:, :, 0] = helper.draw_strokes(s)
