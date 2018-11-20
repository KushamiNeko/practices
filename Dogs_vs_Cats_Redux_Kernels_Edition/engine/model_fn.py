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

  _FILTERS = [64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128]
  _KERNEL_SIZE = [7, 7, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3]
  _STRIDES = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1]

  _N_HIDDEN = [4096, 4096]

  assert (len(_FILTERS) == len(_KERNEL_SIZE) == len(_STRIDES))

  is_training = False

  if mode == ModeKeys.TRAIN:
    is_training = True

  layer = features

  # x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 3])
  # y = tf.placeholder(tf.int32, [None])
  # is_training = tf.placeholder(tf.bool, shape=())

  # layer = x

  # layer = tf.map_fn(lambda img: random_flip_tf(img, vertical=False, horizontal=True), layer)

  # layer = tf.cond(
  # is_training,
  # lambda: tf.map_fn(lambda img: random_flip_tf(img, vertical=False, horizontal=True), layer),
  # lambda: layer,
  # )

  # layer = tf.image.convert_image_dtype(layer, tf.float32)

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

  # probability = tf.nn.softmax(logits)
  # predict = tf.argmax(probability, axis=1)

  # # calculate cross entropy and loss
  # xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  # labels=y, logits=logits)

  # loss = tf.reduce_mean(xentropy)

  # correct = tf.nn.in_top_k(logits, y, 1)
  # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

  # accuracy_summary = tf.summary.scalar("accuracy", accuracy)

  predicts = tf.nn.sigmoid(logits, name="predicts")

  predicts = tf.reshape(predicts, tf.shape(labels))

  predicts_image = tf.reshape(predicts, tf.shape(labels))

  if mode == ModeKeys.PREDICT:
    print("PREDICT")

    return predicts

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


def model_dnn(features, labels, mode, params):

  _KEEP_PROB = params["keep_prob"]
  _LEARNING_RATE = params["learning_rate"]
  _N_OUTPUTS = params["n_outputs"]

  _DENSE_NEURONS = [4096, 4096, 4096, 2048, 2048]

  layer = features

  is_training = False

  if mode == ModeKeys.TRAIN:
    is_training = True

  layer = tf.contrib.layers.flatten(layer)

  index = 0
  for n_neurons in _DENSE_NEURONS:

    with tf.name_scope("dense_layers"):

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

  predicts = tf.reshape(predicts, tf.shape(labels))

  predicts_image = tf.reshape(predicts, tf.shape(labels))

  if mode == ModeKeys.PREDICT:
    print("PREDICT")

    return predicts

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
