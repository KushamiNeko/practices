import mxnet as mx
from mxnet import gluon

import config
import os

###############################################################################


class ConvNet():

    def __init__(self,
                 train: bool = True,
                 hibridize: bool = True,
                 initialize: bool = True,
                 parameters_file: str = ""):

        # self.simple_conv_net()
        self.alex_net()

        if initialize:
            self.net.collect_params().initialize(
                mx.init.Xavier(magnitude=2.24), ctx=config.CTX)
        else:
            if parameters_file:
                assert (os.path.exists(parameters_file))
                self.net.load_parameters(parameters_file, ctx=config.CTX)

        if train:
            self.trainer = gluon.Trainer(
                self.net.collect_params(),
                "adagrad",
                # "sgd",
                {
                    "learning_rate": 0.1,
                    # "learning_rate": 0.001,
                })

        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
        # self.loss = gluon.loss.L2Loss()

        if hibridize:
            self.net.hybridize()

###############################################################################

    def initialize(self):
        self.net.collect_params().initialize(
            mx.init.Xavier(magnitude=2.24), ctx=config.CTX)

###############################################################################

    def load_parameters(self, parameters_file: str):
        self.net.load_parameters(parameters_file, ctx=config.CTX)

###############################################################################

    def alex_net(self):

        self.net = gluon.nn.HybridSequential()

        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(
                    channels=96,
                    kernel_size=11,
                    strides=(4, 4),
                    activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
            self.net.add(
                gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2, 2)))
            self.net.add(
                gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
            self.net.add(
                gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
            self.net.add(
                gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
            self.net.add(gluon.nn.Flatten())
            self.net.add(gluon.nn.Dense(4096, activation="relu"))
            self.net.add(gluon.nn.Dense(4096, activation="relu"))
            self.net.add(gluon.nn.Dense(340))
#            self.net.add(
#                gluon.nn.Conv2D(
#                    channels=96,
#                    kernel_size=11,
#                    strides=(4, 4),
#                    activation="relu",
#                ))
#            self.net.add(gluon.nn.MaxPool2D(
#                pool_size=2,
#                strides=2,
#            ))
#
#            self.net.add(
#                gluon.nn.Conv2D(
#                    channels=192,
#                    kernel_size=5,
#                    activation="relu",
#                ))
#            self.net.add(gluon.nn.MaxPool2D(
#                pool_size=2,
#                strides=2,
#            ))
#
#            self.net.add(
#                gluon.nn.Conv2D(
#                    channels=384,
#                    kernel_size=3,
#                    activation="relu",
#                ))
#            self.net.add(
#                gluon.nn.Conv2D(
#                    channels=384,
#                    kernel_size=3,
#                    activation="relu",
#                ))
#
#            self.net.add(
#                gluon.nn.Conv2D(
#                    channels=256,
#                    kernel_size=3,
#                    activation="relu",
#                ))
#
#            self.net.add(gluon.nn.MaxPool2D(
#                pool_size=2,
#                strides=2,
#            ))
#
#            self.net.add(gluon.nn.Flatten())
#
#            self.net.add(gluon.nn.Dense(
#                4096,
#                activation="relu",
#            ))
#            self.net.add(gluon.nn.Dense(
#                4096,
#                activation="relu",
#            ))

# self.net.add(gluon.nn.Dense(340))
# self.net.add(gluon.nn.Dense(1))

###############################################################################

    def simple_conv_net(self):
        self.net = gluon.nn.HybridSequential()

        # channels = 144
        dense = 2048

        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(channels=100, kernel_size=5, activation="relu"))

            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(
                gluon.nn.Conv2D(channels=150, kernel_size=5, activation="relu"))

            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            # self.net.add(
            # gluon.nn.Conv2D(
            # channels=channels, kernel_size=3, activation="relu"))
            # self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Flatten())
            self.net.add(gluon.nn.Dense(dense, activation="relu"))
            self.net.add(gluon.nn.Dense(dense, activation="relu"))
            self.net.add(gluon.nn.Dense(340))
            # self.net.add(gluon.nn.Dense(1))


###############################################################################


class DenseNet():

    def __init__(self,
                 train: bool = True,
                 hibridize: bool = True,
                 initialize: bool = True,
                 parameters_file: str = ""):

        self.simple_dense_net()

        if initialize:
            self.net.collect_params().initialize(
                mx.init.Xavier(magnitude=2.24), ctx=config.CTX)
        else:
            if parameters_file:
                assert (os.path.exists(parameters_file))
                self.net.load_parameters(parameters_file, ctx=config.CTX)

        if train:
            self.trainer = gluon.Trainer(
                self.net.collect_params(),
                "adagrad",
                # "sgd",
                {
                    "learning_rate": 0.1,
                    # "learning_rate": 0.001,
                })

        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
        # self.loss = gluon.loss.L2Loss()

        if hibridize:
            self.net.hybridize()

###############################################################################

    def initialize(self):
        self.net.collect_params().initialize(
            mx.init.Xavier(magnitude=2.24), ctx=config.CTX)

###############################################################################

    def load_parameters(self, parameters_file: str):
        self.net.load_parameters(parameters_file, ctx=config.CTX)

###############################################################################

    def simple_dense_net(self):
        self.net = gluon.nn.HybridSequential()

        dense = 4096

        with self.net.name_scope():
            self.net.add(gluon.nn.Dense(dense))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))

            self.net.add(gluon.nn.Dense(dense))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))

            self.net.add(gluon.nn.Dense(dense))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))

            self.net.add(gluon.nn.Dense(dense))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))

            self.net.add(gluon.nn.Dense(dense))
            self.net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            self.net.add(gluon.nn.Activation(activation='relu'))

            # self.net.add(gluon.nn.Dense(dense, activation="relu"))
            # self.net.add(gluon.nn.Dense(dense, activation="relu"))
            # self.net.add(gluon.nn.Dense(dense, activation="relu"))
            # self.net.add(gluon.nn.Dense(dense, activation="relu"))
            # self.net.add(gluon.nn.Dense(dense, activation="relu"))

            self.net.add(gluon.nn.Flatten())
            self.net.add(gluon.nn.Dense(340))


###############################################################################
