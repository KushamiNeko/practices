import tensorflow as tf


def mse_error(inputs, labels):
  inputs_flatten = tf.contrib.layers.flatten(inputs)
  labels_flatten = tf.contrib.layers.flatten(labels)

  difference = tf.subtract(labels_flatten, inputs_flatten)

  return tf.square(difference)
