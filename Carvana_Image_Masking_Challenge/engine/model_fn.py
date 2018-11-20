import tensorflow as tf

from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
# from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import fully_connected

from tensorflow.python.estimator.model_fn import ModeKeys

import loss_fn


def model_convnet(features, labels, mode, params):

  _KEEP_PROB = params["keep_prob"]
  _LEARNING_RATE = params["learning_rate"]
  _N_OUTPUTS = params["n_outputs"]

  _OUTPUT_IMAGE_HEIGHT = params["output_image_height"]
  _OUTPUT_IMAGE_WIDTH = params["output_image_width"]

  _SOURCE_IMAGE_HEIGHT = params["source_image_height"]
  _SOURCE_IMAGE_WIDTH = params["source_image_width"]

  # _BATCH_SIZE = params["batch_size"]

  _RINT_THRESHOLD = params["rint_threshold"]

  _FILTERS = [64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128]
  _KERNEL_SIZE = [7, 7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3]
  _STRIDES = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1]

  _N_HIDDEN = [4096, 4096]

  assert (len(_FILTERS) == len(_KERNEL_SIZE) == len(_STRIDES))

  is_training = False

  if mode == ModeKeys.TRAIN:
    is_training = True

  layer = features

  with tf.name_scope("conv_layers"):
    index = 0
    for filters, kernel_size, stride in zip(_FILTERS, _KERNEL_SIZE, _STRIDES):
      layer = conv2d(
        layer,
        filters,
        kernel_size,
        # stride,
        1,
        padding="SAME",
        # use elu activation function
        activation_fn=tf.nn.elu,
        # use batch normalization
        normalizer_fn=batch_norm,
        normalizer_params={
          "is_training": is_training,
          "decay": 0.99,
          "updates_collections": None,
          "fused": True,
        },
        # use he initializer
        weights_initializer=variance_scaling_initializer(),
        scope="conv-{}".format(index))

      if stride != 1:
        layer = max_pool2d(
          inputs=layer,
          kernel_size=3,
          stride=2,
          padding="VALID",
          scope="max_pool-{}".format(index))

      # layer = tf.nn.local_response_normalization(layer)

      index += 1

  # layer = max_pool2d(
  # inputs=layer, kernel_size=3, stride=2, padding="VALID", scope="max_pool")

  layer = tf.contrib.layers.flatten(layer, scope="flatten")

  with tf.name_scope("dense_layers"):
    index = 0
    for n_neurons in _N_HIDDEN:

      layer = fully_connected(
        layer,
        n_neurons,
        # use elu activation function
        activation_fn=tf.nn.elu,
        # use batch normalization
        normalizer_fn=batch_norm,
        normalizer_params={
          "is_training": is_training,
          "decay": 0.99,
          "updates_collections": None,
          "fused": True,
        },
        # use he initializer
        weights_initializer=variance_scaling_initializer(),
        scope="fc-{}".format(index),
      )

      layer = dropout(
        layer,
        _KEEP_PROB,
        is_training=is_training,
        scope="drop-{}".format(index),
      )

      index += 1

  # layer = dropout(
  # layer,
  # _KEEP_PROB,
  # is_training=is_training,
  # scope="dropout",
  # )

  logits = fully_connected(
    layer,
    _N_OUTPUTS,
    # use no activation function
    activation_fn=None,
    # use batch normalization
    normalizer_fn=batch_norm,
    normalizer_params={
      "is_training": True,
      "decay": 0.99,
      "updates_collections": None,
      "fused": True,
    },
    # use he initializer
    weights_initializer=variance_scaling_initializer(),
    scope="logits")

  predicts = tf.nn.sigmoid(logits, name="predicts")

  predicts_image = tf.reshape(
    predicts, [-1, _OUTPUT_IMAGE_HEIGHT, _OUTPUT_IMAGE_WIDTH, 1])

  if mode == ModeKeys.PREDICT:
    print("PREDICT")

    predicts_resize = tf.image.resize_images(
      predicts_image,
      [_SOURCE_IMAGE_HEIGHT, _SOURCE_IMAGE_WIDTH])

    # ones = tf.ones([_BATCH_SIZE, _N_OUTPUTS])
    # zeros = tf.zeros([_BATCH_SIZE, _N_OUTPUTS])

    predicts_resize_f = tf.contrib.layers.flatten(predicts_resize)

    shape = tf.shape(predicts_resize_f)

    ones = tf.ones(shape)
    zeros = tf.zeros(shape)

    condition = tf.greater_equal(predicts_resize_f, ones * _RINT_THRESHOLD)

    predicts_rint = tf.where(condition, ones, zeros)

    # predicts_rint = tf.rint(predicts, name="predicts_rint")

    predicts_rint_image = tf.reshape(
      predicts_rint, [-1, _SOURCE_IMAGE_HEIGHT, _SOURCE_IMAGE_WIDTH, 1])

    # predicts_rint = tf.image.resize_images(
    # predicts_rint,
    # [_SOURCE_IMAGE_HEIGHT, _SOURCE_IMAGE_WIDTH])

    features_summary = tf.summary.image("features", features)

    predicts_summary = tf.summary.image("predicts", predicts_resize)

    predicts_rint_summary = tf.summary.image(
      "predicts_rint", predicts_rint_image)

    return (
      predicts_resize, predicts_rint, features_summary, predicts_summary,
      predicts_rint_summary)

  # predicts_image = tf.reshape(predicts, tf.shape(labels))

  loss = tf.reduce_mean(loss_fn.mse_error(labels, predicts), name="loss")

  loss_summary = tf.summary.scalar("loss", loss)

  if mode == ModeKeys.EVAL:
    print("EVAL")

    return loss

  if mode == ModeKeys.TRAIN:
    print("TRAIN")

    features_summary = tf.summary.image("features", features)

    labels_summary = tf.summary.image("labels", labels)

    predict_summary = tf.summary.image("predict", predicts_image)

    # use AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=_LEARNING_RATE)

    # minimizing the loss in training op
    train_op = optimizer.minimize(loss)

    return (
      train_op, loss_summary, features_summary, labels_summary, predict_summary)

  raise ValueError("invalid mode")


# def model_dnn(features, labels, mode, params):

# _KEEP_PROB = params["keep_prob"]
# _LEARNING_RATE = params["learning_rate"]
# _N_OUTPUTS = params["n_outputs"]

# _DENSE_NEURONS = [4096, 4096, 4096, 2048, 2048]

# layer = features

# is_training = False

# if mode == ModeKeys.TRAIN:
# is_training = True

# layer = tf.contrib.layers.flatten(layer)

# index = 0
# for n_neurons in _DENSE_NEURONS:

# with tf.name_scope("dense_layers"):

# layer = fully_connected(
# layer,
# n_neurons,
# # use elu activation function
# activation_fn=tf.nn.elu,
# # use batch normalization
# normalizer_fn=batch_norm,
# normalizer_params={
# "is_training": is_training,
# "decay": 0.99,
# "updates_collections": None,
# "fused": True,
# },
# # use he initializer
# weights_initializer=variance_scaling_initializer(),
# scope="fc-{}".format(index),
# )

# layer = dropout(
# layer,
# _KEEP_PROB,
# is_training=is_training,
# scope="drop-{}".format(index),
# )

# index += 1

# logits = fully_connected(
# layer,
# _N_OUTPUTS,
# # use no activation function
# activation_fn=None,
# # use batch normalization
# normalizer_fn=batch_norm,
# normalizer_params={
# "is_training": True,
# "decay": 0.99,
# "updates_collections": None,
# "fused": True,
# },
# # use he initializer
# weights_initializer=variance_scaling_initializer(),
# scope="logits")

# predicts = tf.nn.sigmoid(logits, name="predicts")

# predicts = tf.reshape(predicts, tf.shape(labels))

# predicts_image = tf.reshape(predicts, tf.shape(labels))

# if mode == ModeKeys.PREDICT:
# print("PREDICT")

# return predicts

# loss = tf.reduce_mean(loss_fn.mse_error(labels, predicts), name="loss")

# loss_summary = tf.summary.scalar("loss", loss)

# if mode == ModeKeys.EVAL:
# print("EVAL")

# return loss

# if mode == ModeKeys.TRAIN:
# print("TRAIN")

# features_summary = tf.summary.image("features", features)

# labels_summary = tf.summary.image("labels", labels)

# predict_summary = tf.summary.image("predict", predicts_image)

# # use AdamOptimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=_LEARNING_RATE)

# # minimizing the loss in training op
# train_op = optimizer.minimize(loss)

# return (
# train_op, loss_summary, features_summary, labels_summary, predict_summary)

# raise ValueError("invalid mode")
