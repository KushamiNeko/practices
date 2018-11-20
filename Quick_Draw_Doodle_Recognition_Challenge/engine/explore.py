# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import cv2

import mxnet as mx
from mxnet import autograd, gluon, nd

import os
import urllib
import json
import pickle

ROOT = "/run/media/neko/HDD/ARTIFICIAL_INTELLIGENCE_DATA_SCIENCE/KAGGLE_COMPETITIONS/"
COMPETIION = "Quick_Draw_Doodle_Recognition_Challenge"
TRAIN_CSV_FILES = os.path.join(ROOT, COMPETIION, "all/train_simplified")
TEST_CSV_FILE = os.path.join(ROOT, COMPETIION, "all/test_simplified.csv")

SIZE = 256

RESIZE = 74

LABEL_ID_DICT = {}
ID_LABEL_DICT = {}


def get_label_index(filename):
    label = os.path.splitext(os.path.basename(filename))[0].replace(" ", "_")
    return LABEL_ID_DICT[label]


def get_index_label(index):
    return ID_LABEL_DICT[index]


def get_label_dict():

    global LABEL_ID_DICT
    global ID_LABEL_DICT

    if not os.path.exists("resource/label_id.pickle") or not os.path.exists(
            "resource/id_label.pickle"):
        print("generate label dict")
        generate_label_dict()

    else:
        label_id = open("resource/label_id.pickle", "rb")
        LABEL_ID_DICT = pickle.load(label_id)
        label_id.close()

        id_label = open("resource/id_label.pickle", "rb")
        ID_LABEL_DICT = pickle.load(id_label)
        id_label.close()


def generate_label_dict():

    for index, f in enumerate(os.listdir(TRAIN_CSV_FILES)):
        label = os.path.splitext(f)[0].replace(" ", "_")

        if label not in LABEL_ID_DICT:
            LABEL_ID_DICT[label] = index

        if index not in ID_LABEL_DICT:
            ID_LABEL_DICT[index] = label

    label_id = open("resource/label_id.pickle", 'wb')
    pickle.dump(obj=LABEL_ID_DICT, file=label_id)
    label_id.close()

    id_label = open("resource/id_label.pickle", 'wb')
    pickle.dump(obj=ID_LABEL_DICT, file=id_label)
    id_label.close()


def print_label_dict():
    label_id = open("resource/label_id.pickle", "rb")
    t = pickle.load(label_id)
    label_id.close()

    print(t)
    print(len(t))

    id_label = open("resource/id_label.pickle", "rb")
    t = pickle.load(id_label)
    id_label.close()

    print(t)
    print(len(t))


# exit(0)


def draw_strokes(strokes, size=SIZE, linewidth=4):
    img = np.ones((size, size), np.float32)

    for stroke in json.loads(strokes):
        x = stroke[0]
        y = stroke[1]

        for i in range(len(x) - 1):
            cv2.line(img, (x[i], y[i]), (x[i + 1], y[i + 1]), 0, linewidth)

    img = cv2.resize(img, (RESIZE, RESIZE))

    return img


def train_data_extract():
    for idx, file in enumerate(os.listdir(TRAIN_CSV_FILES)):

        # if idx not in LABEL_DICT:
        # LABEL_DICT[idx] = os.path.splitext(file)[0]

        f = pd.read_csv(os.path.join(TRAIN_CSV_FILES, file), dtype="str")

        for s in f["drawing"]:
            # img = np.zeros((SIZE, SIZE, 1), np.float32)
            img = np.zeros((RESIZE, RESIZE, 1), np.float32)
            img[:, :, 0] = draw_strokes(s)

            index = get_label_index(file)

            yield [img, index]


get_label_dict()
# print(LABEL_ID_DICT)
generator = train_data_extract()

test = gluon.nn.Sequential()
test.add(
    gluon.nn.Conv2D(
        channels=96,
        kernel_size=11,
        # strides=(4, 4),
        strides=(1, 1),
        activation="relu",
    ))

test.collect_params().initialize(mx.init.Xavier(magnitude=2.24))

x, y = next(generator)

data = nd.array([x])
data = nd.transpose(data, (0, 3, 1, 2))

out = test(data)

# print(x)
# print(y)

print(data.shape)
print(out.shape)

exit(0)

# plt.imshow(x[:, :, 0], cmap=plt.cm.gray)
# plt.show()
# print("Label:  %d (%s)" % (y, LABEL_DICT[y]))

BATCH_SIZE = 64
CTX = mx.gpu()
EPOCHCS = 20

# alex_net = gluon.nn.Sequential()
alex_net = gluon.nn.HybridSequential()

with alex_net.name_scope():
    alex_net.add(
        gluon.nn.Conv2D(
            channels=96,
            kernel_size=11,
            # strides=(4, 4),
            strides=(1, 1),
            activation="relu",
        ))
    alex_net.add(gluon.nn.MaxPool2D(
        pool_size=3,
        strides=2,
    ))

    alex_net.add(
        gluon.nn.Conv2D(
            channels=192,
            kernel_size=5,
            activation="relu",
        ))
    alex_net.add(gluon.nn.MaxPool2D(
        pool_size=3,
        strides=2,
    ))

    alex_net.add(
        gluon.nn.Conv2D(
            channels=384,
            kernel_size=3,
            activation="relu",
        ))
    alex_net.add(
        gluon.nn.Conv2D(
            channels=384,
            kernel_size=3,
            activation="relu",
        ))

    alex_net.add(
        gluon.nn.Conv2D(
            channels=256,
            kernel_size=3,
            activation="relu",
        ))

    alex_net.add(gluon.nn.MaxPool2D(
        pool_size=3,
        strides=2,
    ))

    alex_net.add(gluon.nn.Flatten())

    alex_net.add(gluon.nn.Dense(
        4096,
        activation="relu",
    ))
    alex_net.add(gluon.nn.Dense(
        4096,
        activation="relu",
    ))
    alex_net.add(gluon.nn.Dense(340,))

alex_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=CTX)

data = data.as_in_context(CTX)
out = alex_net(data)

print(out.shape)

exit(0)

alex_net.hybridize()

trainer = gluon.Trainer(alex_net.collect_params(), "sgd", {
    "learning_rate": 0.001,
})

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# filename = os.path.join(dir_name, "testnet.params")
# net.save_parameters(filename)
# net2 = gluon.nn.Sequential()
# with net2.name_scope():
#     net2.add(gluon.nn.Dense(num_hidden, activation="relu"))
#     net2.add(gluon.nn.Dense(num_hidden, activation="relu"))
#     net2.add(gluon.nn.Dense(num_outputs))
# net2.load_parameters(filename, ctx=ctx)
# net2(nd.ones((1, 100), ctx=ctx))

for e in range(EPOCHCS):

    print(
        "========================================================================================="
    )

    print("epochs: %d" % (e))

    print(
        "========================================================================================="
    )

    print()

    generator = train_data_extract()

    looping = True
    while looping:
        imgs = []
        labels = []

        for _ in range(BATCH_SIZE):
            try:
                x, y = next(generator)
                imgs.append(x)
                labels.append(y)
            except StopIteration:
                looping = False
                break

        if len(imgs) == 0 or len(labels) == 0:
            break

        data = nd.array(imgs)
        labels = nd.array(labels)

        data = nd.transpose(data, (0, 3, 1, 2))

        data = data.as_in_context(CTX)
        labels = labels.as_in_context(CTX)

        with autograd.record():
            output = alex_net(data)
            loss = softmax_cross_entropy(output, labels)

        loss.backward()
        trainer.step(data.shape[0])

        print(nd.mean(loss).asscalar())
#     break

    alex_net.save_parameters("check_points/epochs_{}.params".format(e))

# def evaluate_accuracy(data_iterator, net):
#     acc = mx.metric.Accuracy()
#     for d, l in data_iterator:
#         data = d.as_in_context(CTX)
#         label = l.as_in_context(CTX)
#         output = net(data)
#         predictions = nd.argmax(output, axis=1)
#         acc.update(preds=predictions, labels=label)

#     return acc.get()[1]

# epochs = 20
# smoothing_constant = 0.01

# for e in range(epochs):
#     for i, (d, l) in enumerate(train_data):
#         data = d.as_in_context(CTX)
#         label = l.as_in_context(CTX)
#         with autograd.record():
#             output = alex_net(data)
#             loss = softmax_cross_entropy(output, label)

#         loss.backward()
#         trainer.step(data.shape[0])

#         curr_loss = nd.mean(loss).asscalar()
#         moving_loss = (curr_loss if ((i == 0) and (e == 0))
#                        else (1 - smoothing_constant) * moving_loss
#                        + (smoothing_constant) * curr_loss)

#     test_accuracy = evaluate_accuracy(test_data, alex_net)
#     train_accuracy = evaluate_accuracy(train_data, alex_net)
#     print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
#           (e, moving_loss, train_accuracy, test_accuracy))
