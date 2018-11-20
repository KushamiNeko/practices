import datetime
import os
import random
import csv

import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import flatten
# from tensorflow.contrib.layers import max_pool2d
# from tensorflow.contrib.layers import avg_pool2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import fully_connected

from PIL import ImageEnhance
from PIL import Image

# path to dataset
DATA_ROOT = r"/run/media/onionhuang/HDD/ARTIFICIAL_INTELLIGENCE/KAGGLE_COMPETITIONS/Dogs_vs_Cats_Redux_Kernels_Edition"
assert (os.path.exists(DATA_ROOT))

TRAIN_DATA = os.path.join(DATA_ROOT, "train")
assert (os.path.exists(TRAIN_DATA))

TEST_DATA = os.path.join(DATA_ROOT, "test")
assert (os.path.exists(TEST_DATA))

BRIGHTNESS_FACTOR_MIN = 0.2
BRIGHTNESS_FACTOR_MAX = 1.8

CONTRAST_FACTOR_MIN = 0.2
CONTRAST_FACTOR_MAX = 1.8

SHARPENESS_FACTOR_MIN = 0
SHARPENESS_FACTOR_MAX = 2

SATURATION_FACTOR_MIN = 0
SATURATION_FACTOR_MAX = 2


def random_enhance_pil(
    image, brightness=True, contrast=True, saturation=True, sharpness=True):

  new_image = image

  if brightness:
    enhancer = ImageEnhance.Brightness(new_image)
    new_image = enhancer.enhance(
      random.uniform(BRIGHTNESS_FACTOR_MIN, BRIGHTNESS_FACTOR_MAX))

  if contrast:
    enhancer = ImageEnhance.Contrast(new_image)
    new_image = enhancer.enhance(
      random.uniform(CONTRAST_FACTOR_MIN, CONTRAST_FACTOR_MAX))

  if saturation:
    enhancer = ImageEnhance.Color(new_image)
    new_image = enhancer.enhance(
      random.uniform(SATURATION_FACTOR_MIN, SATURATION_FACTOR_MAX))

  if sharpness:
    enhancer = ImageEnhance.Sharpness(new_image)
    new_image = enhancer.enhance(
      random.uniform(SHARPENESS_FACTOR_MIN, SHARPENESS_FACTOR_MAX))

    return new_image


def resize_image_pil(
    image, size, random_enhance=True, pad=True, resample_filter=Image.LANCZOS):

  if random_enhance:
    image = random_enhance_pil(image)

  width, height = image.size

  if height > width:
    new_height = size
    ratio = new_height / height
    new_width = int(width * ratio)

  elif width > height:
    new_width = size
    ratio = new_width / width
    new_height = int(height * ratio)

  else:
    new_width = size
    new_height = size

  image_resize = image.resize((new_width, new_height), resample=resample_filter)

  if pad:
    image_pad = Image.new("RGB", (size, size), (0, 0, 0))
    ulc = ((size - new_width) // 2, (size - new_height) // 2)

    image_pad.paste(image_resize, ulc)

    return image_pad
  else:
    return image_resize


def get_label(file_name):
  label = file_name.split(".")[0]

  if label == "cat":
    return 0
  elif label == "dog":
    return 1

  # if label == "cat":
  # return [1.0, 0.0]
  # elif label == "dog":
  # return [0.0, 1.0]


def get_next_batch(iteration, batch_size, dataset):
  if batch_size * (iteration + 1) > len(dataset):
    return dataset[batch_size * iteration:]
  else:
    return dataset[batch_size * iteration:batch_size * (iteration + 1)]


def image_to_nparray_pil(image):
  return np.array(image.getdata()).reshape((SIZE, SIZE, 3))


def random_enhance_tf(image, brightness=True, contrast=True, saturation=True):
  new_image = image

  if brightness:
    seed = datetime.datetime.now().second
    new_image = tf.image.random_brightness(new_image, 5, seed=seed)

  if contrast:
    seed = datetime.datetime.now().second
    new_image = tf.image.random_contrast(new_image, 0.1, 1.5, seed=seed)

  if saturation:
    seed = datetime.datetime.now().second
    new_image = tf.image.random_saturation(new_image, 0., 1.5, seed=seed)

  return new_image


def random_flip_tf(image, vertical=True, horizontal=True):
  new_image = image

  if vertical:
    seed = datetime.datetime.now().second
    new_image = tf.image.random_flip_up_down(new_image, seed=seed)

  if horizontal:
    seed = datetime.datetime.now().second
    new_image = tf.image.random_flip_left_right(new_image, seed=seed)

  return new_image


def mse_error(inputs, labels):
  inputs_flatten = tf.contrib.layers.flatten(inputs)
  labels_flatten = tf.contrib.layers.flatten(labels)

  difference = tf.subtract(labels_flatten, inputs_flatten)

  return tf.square(difference)


# ImageNet Models Compatibility
# SIZE = 224
SIZE = 112

WIDTH = SIZE
HEIGHT = SIZE

# n_inputs = MNIST.train.images[0].size
n_outputs = 2

# parameters of convolution layers
# n_filters = [
# 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128,
# 128
# ]
# n_kernel_size = [7, 7, 7, 7, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# n_strides = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1]

# n_filters = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128]
# n_kernel_size = [7, 7, 7, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3]
# n_strides = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1]

n_filters = [128, 128, 128, 128, 128, 128, 128, 128, 128]
n_kernel_size = [7, 7, 7, 5, 5, 5, 3, 3, 3]
n_strides = [1, 1, 2, 1, 1, 2, 1, 1, 2]

assert (len(n_filters) == len(n_kernel_size) == len(n_strides))

n_hidden = [4096, 4096]

# parameters of fully connected layers
keep_prob = 0.6

# parameters of optimizer
learning_rate = 0.01

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 3])
y = tf.placeholder(tf.int32, [None])
# y = tf.placeholder(tf.float32, [None, 2])
is_training = tf.placeholder(tf.bool, shape=())

layer = x

layer = tf.cond(
  is_training,
  lambda: tf.map_fn(
    lambda img: random_flip_tf(img, vertical=False, horizontal=True), layer),
  lambda: layer,
)

image_summary = tf.summary.image("features", layer)

layer = tf.image.convert_image_dtype(layer, tf.float32)

# layer = tf.map_fn(lambda img: tf.image.per_image_standardization(img), layer)

with tf.name_scope("conv_layers"):
  index = 0
  for filters, kernel_size, stride in zip(n_filters, n_kernel_size, n_strides):
    layer = conv2d(
      layer,
      filters,
      kernel_size,
      stride,
      padding="SAME",
      activation_fn=tf.nn.elu,
      normalizer_fn=batch_norm,
      normalizer_params={
        "is_training": is_training,
        "decay": 0.99,
        "updates_collections": None,
        "fused": True,
      },
      weights_initializer=variance_scaling_initializer(),
      scope="conv_{}".format(index),
    )

    # if stride != 1:
    # layer = max_pool2d(
    # inputs=layer,
    # kernel_size=3,
    # stride=2,
    # padding="SAME",
    # scope="max_pool_{}".format(index))

    # layer = tf.nn.local_response_normalization(
    # layer, name="local_response_norm")

    index += 1

# layer = dropout(
# layer,
# keep_prob,
# is_training=is_training,
# scope="drop_to_avg_pool",
# )

# layer = avg_pool2d(
# inputs=layer, kernel_size=3, stride=1, padding="VALID", scope="avg_pool")

layer = flatten(layer, scope="flatten")

layer = dropout(
  layer,
  keep_prob,
  is_training=is_training,
  scope="drop_to_avg_pool",
)

with tf.name_scope("dense_layers"):
  index = 0
  for n_neurons in n_hidden:

    layer = fully_connected(
      layer,
      n_neurons,
      activation_fn=tf.nn.elu,
      normalizer_fn=batch_norm,
      normalizer_params={
        "is_training": is_training,
        "decay": 0.99,
        "updates_collections": None,
        "fused": True,
      },
      weights_initializer=variance_scaling_initializer(),
      scope="fc-{}".format(index),
    )

    layer = dropout(
      layer,
      keep_prob,
      is_training=is_training,
      scope="drop-{}".format(index),
    )

    index += 1

# layer = dropout(
# layer,
# keep_prob,
# is_training=is_training,
# scope="drop-{}".format(index),
# )

logits = fully_connected(
  layer,
  n_outputs,
  activation_fn=None,
  normalizer_fn=batch_norm,
  normalizer_params={
    "is_training": is_training,
    "decay": 0.99,
    "updates_collections": None,
    "fused": True,
  },
  weights_initializer=variance_scaling_initializer(),
  scope="output",
)

probability = tf.nn.softmax(logits)
predict = tf.argmax(probability, axis=1)

# mse = mse_error(y, probability)
# loss = tf.reduce_mean(mse)

# calculate cross entropy and loss
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  labels=y, logits=logits)

loss = tf.reduce_mean(xentropy)

loss_summary = tf.summary.scalar("loss", loss)

# use AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# minimizing the loss in training op
training_op = optimizer.minimize(loss)

# accuracy monitoring
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

accuracy_summary = tf.summary.scalar("accuracy", accuracy)

saver = tf.train.Saver()

now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
LOG_DIR = "logs/{}".format(now)

summary_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())

n_epochs = 200
batch_size = 10

# runing the graph
# with tf.Session() as sess:

# # we need to initialize all variables first
# tf.global_variables_initializer().run()
# tf.local_variables_initializer().run()

# # saver.restore(sess, "./my-convnet.ckpt")

# for epoch in range(n_epochs):

# DATASETS = np.random.permutation(
# [os.path.join(TRAIN_DATA, i) for i in os.listdir(TRAIN_DATA)])

# for iteration in range(len(DATASETS) // batch_size):

# # get mini bath of training data
# next_batch = get_next_batch(
# batch_size=batch_size, iteration=iteration, dataset=DATASETS)

# x_batch = []

# for file in next_batch:
# image = Image.open(file)
# image = resize_image_pil(image, SIZE)
# x_batch.append(image)

# x_batch = [image_to_nparray_pil(image) for image in x_batch]

# y_batch = np.array([get_label(os.path.basename(i)) for i in next_batch])

# _, image_log, loss_log, accuracy_log = sess.run(
# [
# training_op,
# image_summary,
# loss_summary,
# accuracy_summary,
# ],
# feed_dict={
# x: x_batch,
# y: y_batch,
# is_training: True,
# })

# step = epoch * (len(DATASETS) // batch_size) + iteration

# summary_writer.add_summary(loss_log, step)
# summary_writer.add_summary(accuracy_log, step)

# if step % 50 == 0:
# summary_writer.add_summary(image_log, step)

# print("STEP:", step)

# if iteration % 100 == 0:
# saver.save(sess, "./logs/checkpoint.ckpt")

# if iteration % 500 == 0:
# saver.save(sess, "./logs/checkpoint_{}.ckpt".format(step))

with tf.Session() as sess:

  # we need to initialize all variables first
  tf.global_variables_initializer().run()
  tf.local_variables_initializer().run()

  saver.restore(sess, "./logs/checkpoint_499500.ckpt")

  DATASETS = sorted(
    [os.path.join(TEST_DATA, i) for i in os.listdir(TEST_DATA)],
    key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

  print(len(DATASETS))

  ids = [os.path.splitext(os.path.basename(i))[0] for i in DATASETS]

  # next_batch = DATASETS

  results = {}

  for iteration in range(len(DATASETS) // batch_size):

    print(iteration)

    next_batch = get_next_batch(
      batch_size=batch_size, iteration=iteration, dataset=DATASETS)

    ids_batch = get_next_batch(
      batch_size=batch_size, iteration=iteration, dataset=ids)

    x_batch = []

    for file in next_batch:
      image = Image.open(file)
      image = resize_image_pil(image, SIZE, random_enhance=False)
      x_batch.append(image)

    x_batch = [image_to_nparray_pil(image) for image in x_batch]

    predict_val, probability_val = sess.run(
      [predict, probability], feed_dict={
        x: x_batch,
        is_training: False,
      })

    # print(len(ids))
    # print(len(predict_val))
    # print(len(probability_val))

    for index, image_id in enumerate(ids_batch):
      results[image_id] = probability_val[index][1]

  print(len(results))

with open("classification.csv", "w", newline="") as f:
  writer = csv.writer(
    f, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

  writer.writerow(["id", "label"])

  for key, value in results.items():
    writer.writerow([str(key), str(value)])

print("FINISH!!")
