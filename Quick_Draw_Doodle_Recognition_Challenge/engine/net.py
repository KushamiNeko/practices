import mxnet as mx
from mxnet import gluon

import config

###############################################################################


class ConvNet():

    def __init__(self):

        self.net = None

        self.simple_conv_net()
        # self.alex_net()

        self.net.collect_params().initialize(
            mx.init.Xavier(magnitude=2.24), ctx=config.CTX)

        self.trainer = gluon.Trainer(self.net.collect_params(), "sgd", {
            "learning_rate": 0.001,
        })

        # self.softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        self.loss = gluon.loss.L2Loss()

        self.net.hybridize()

###############################################################################

    def alex_net(self):

        self.net = gluon.nn.HybridSequential()

        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(
                    channels=96,
                    kernel_size=11,
                    # strides=(4, 4),
                    strides=(1, 1),
                    activation="relu",
                ))
            self.net.add(
                gluon.nn.MaxPool2D(
                    # pool_size=3,
                    # strides=2,
                    pool_size=2,
                    strides=2,
                ))

            self.net.add(
                gluon.nn.Conv2D(
                    channels=192,
                    kernel_size=5,
                    activation="relu",
                ))
            self.net.add(
                gluon.nn.MaxPool2D(
                    # pool_size=3,
                    # strides=2,
                    pool_size=2,
                    strides=2,
                ))

            self.net.add(
                gluon.nn.Conv2D(
                    channels=384,
                    kernel_size=3,
                    activation="relu",
                ))
            self.net.add(
                gluon.nn.Conv2D(
                    channels=384,
                    kernel_size=3,
                    activation="relu",
                ))

            self.net.add(
                gluon.nn.Conv2D(
                    channels=256,
                    kernel_size=3,
                    activation="relu",
                ))

            self.net.add(
                gluon.nn.MaxPool2D(
                    # pool_size=3,
                    # strides=2,
                    pool_size=2,
                    strides=2,
                ))

            self.net.add(gluon.nn.Flatten())

            self.net.add(gluon.nn.Dense(
                4096,
                activation="relu",
            ))
            self.net.add(gluon.nn.Dense(
                4096,
                activation="relu",
            ))
            # self.net.add(gluon.nn.Dense(340))
            self.net.add(gluon.nn.Dense(1))

###############################################################################

    def simple_conv_net(self):
        self.net = gluon.nn.HybridSequential()

        channels = 196

        dense = 4096

        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(
                    channels=channels, kernel_size=5, activation="relu"))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            # self.net.add(
            # gluon.nn.Conv2D(channels=100, kernel_size=5, activation="relu"))

            self.net.add(
                gluon.nn.Conv2D(
                    channels=channels, kernel_size=5, activation="relu"))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            # self.net.add(
            # gluon.nn.Conv2D(channels=100, kernel_size=3, activation="relu"))

            self.net.add(
                gluon.nn.Conv2D(
                    channels=channels, kernel_size=3, activation="relu"))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            # self.net.add(gluon.nn.Conv2D(channels=100, kernel_size=5))
            # self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            # self.net.add(gluon.nn.Activation(activation="relu"))
            # self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            # self.net.add(gluon.nn.Conv2D(channels=100, kernel_size=5))
            # self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            # self.net.add(gluon.nn.Activation(activation="relu"))
            # self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            # self.net.add(gluon.nn.Conv2D(channels=50, kernel_size=3))
            # self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            # self.net.add(gluon.nn.Activation(activation="relu"))
            # self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Flatten())
            self.net.add(gluon.nn.Dense(dense, activation="relu"))
            self.net.add(gluon.nn.Dense(dense, activation="relu"))
            # self.net.add(gluon.nn.Dense(340))
            self.net.add(gluon.nn.Dense(1))

    # def evaluate_accuracy(csv_files):
    # acc = mx.metric.Accuracy()
    # for d, l in data_iterator:
    # data = d.as_in_context(CTX)
    # label = l.as_in_context(CTX)
    # output = net(data)
    # predictions = nd.argmax(output, axis=1)
    # acc.update(preds=predictions, labels=label)

    # return acc.get()[1]


###############################################################################
