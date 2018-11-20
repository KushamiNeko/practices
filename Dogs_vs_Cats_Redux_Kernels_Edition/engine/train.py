import os
import argparse

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys
# from tensorflow.contrib.layers import fully_connected
# from tensorflow.contrib.layers import dropout
# from tensorflow.contrib.layers import conv2d
# from tensorflow.contrib.layers import variance_scaling_initializer
# from tensorflow.contrib.layers import batch_norm

# import numpy as np

import inputs_fn
import image_preprocess_fn
import model_fn

# path to dataset
DATA_ROOT = "/run/media/onionhuang/HDD/ARTIFICIAL_INTELLIGENCE/KAGGLE_COMPETITIONS/Dogs_vs_Cats_Redux_Kernels_Edition"
assert (os.path.exists(DATA_ROOT))

TRAIN_DATA = os.path.join(DATA_ROOT, "train")
assert (os.path.exists(TRAIN_DATA))

TEST_DATA = os.path.join(DATA_ROOT, "test")
assert (os.path.exists(TEST_DATA))

# SOURCE_IMAGE_HEIGHT = 1280

# SOURCE_IMAGE_WIDTH = 1918

NEW_IMAGE_SIZE = 224
# NEW_IMAGE_SIZE = 112

KEEP_PROB = 0.6

LEARNING_RATE = 0.001

N_EPOCHS = 50

BATCH_SIZE = 20

# TARGET_WIDTH = NEW_IMAGE_SIZE

# TARGET_HEIGHT = int(
# float(SOURCE_IMAGE_HEIGHT) /
# (float(SOURCE_IMAGE_WIDTH) / float(NEW_IMAGE_SIZE)))

# N_OUTPUTS = TARGET_HEIGHT * TARGET_WIDTH

# def _labels_files(filename, labels_dir):
# separator = "/"
# basename = tf.string_split([filename], separator).values[-1]

# name = tf.string_split([basename], ".").values[0]

# new_name = tf.string_join([name, "mask.gif"], "_")

# lables_file = tf.string_join([labels_dir, new_name], separator)

# return lables_file


def _get_label(file_name):
  label = file_name.split(".")[0]

  if label == "cat":
    return 0
  elif label == "dog":
    return 1


def _read_inputs(images_dataset, labels_dataset):
  image = inputs_fn.tf_read_image_files(images_dataset)
  # image.set_shape([SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH, 3])

  # label = inputs_fn.tf_read_image_files(labels_dataset)
  # label = tf.gather(label, 0)
  # label = tf.image.rgb_to_grayscale(label)

  # label_max = tf.reduce_max(label)

  # label = label / label_max

  # label.set_shape([SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH, 1])

  image = image_preprocess_fn.random_adjustment(
    image,
    brightness=True,
    brightness_delta=0.5,
    contrast=True,
    contrast_lower=0.25,
    contrast_upper=1.75,
    hue=True,
    saturation=True)

  image = image_preprocess_fn.resize_pad(
    image,
    resize=True,
    resize_shape=(TARGET_HEIGHT, TARGET_WIDTH),
    pad=True,
    pad_shape=(NEW_IMAGE_SIZE, NEW_IMAGE_SIZE))

  # label = image_preprocess_fn.resize_pad(
  # label, resize=True, resize_shape=(TARGET_HEIGHT, TARGET_WIDTH), pad=False)

  image = image_preprocess_fn.standardize(image)

  image.set_shape([NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 3])

  # label.set_shape([TARGET_HEIGHT, TARGET_WIDTH, 1])

  return image


def train(
    features_dir, labels_dir, job_dir, pattern, checkpoint, env, max_steps,
    n_epochs, batch_size):

  with tf.name_scope("preprocess"):

    images_dataset = tf.train.match_filenames_once(
      [os.path.join(features_dir, pattern)] * n_epochs)

    labels_dataset = tf.map_fn(
      lambda x: _labels_files(x, labels_dir), images_dataset)

    with tf.device("/cpu:0"):
      image, label = _read_inputs(images_dataset, labels_dataset)

      features, labels = inputs_fn.tf_batch_inputs(
        image, label, batch_size, 8, batch_size * 2, shuffle=False)

  model_params = {
    "keep_prob": KEEP_PROB,
    "learning_rate": LEARNING_RATE,
    "n_outputs": N_OUTPUTS,
  }

  (train_op, loss_summary, features_summary, labels_summary,
   predicts_summary) = model_fn.model_convnet(
     features, labels, ModeKeys.TRAIN, model_params)

  saver = tf.train.Saver()

  summary_writer = tf.summary.FileWriter(job_dir, tf.get_default_graph())

  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.log_device_placement = True

  # runing the graph
  with tf.Session(config=config) as sess:

    # we need to initialize all variables first
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if checkpoint != "":
      print("RESTORE CHECKPOINT:", checkpoint)
      saver.restore(sess, checkpoint)

    step = 0

    try:
      while not coord.should_stop():
        _, loss_log, features_log, labels_log, predicts_log = sess.run(
          [
            train_op,
            loss_summary,
            features_summary,
            labels_summary,
            predicts_summary,
          ])

        if step % 2 == 0:
          summary_writer.add_summary(loss_log, step)

        if step % 10 == 0:
          summary_writer.add_summary(features_log, step)
          summary_writer.add_summary(labels_log, step)

          if step != 0:
            summary_writer.add_summary(predicts_log, step)

        if step % 100 == 0 and step != 0:
          saver.save(sess, os.path.join(job_dir, "checkpoint.ckpt"))

        if step % 500 == 0 and step != 0:
          saver.save(
            sess, os.path.join(job_dir, "checkpoint_{}.ckpt".format(step)))

        if env == "local":
          print("STEP:", step)

        if max_steps != "" and step > int(max_steps):
          break

        step += 1

    except tf.errors.OutOfRangeError:
      print("FINISH TRAINING")

    finally:
      coord.request_stop()

    coord.join(threads)

    saver.save(
      sess, os.path.join(job_dir, "checkpoint_{}.ckpt".format("final")))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Input Arguments
  parser.add_argument(
    "--features-dir",
    type=str,
    required=True,
    help="GCS or local paths to training data")

  parser.add_argument(
    "--labels-dir",
    type=str,
    required=True,
    help="GCS or local paths to training labels",
  )

  parser.add_argument(
    "--job-dir",
    type=str,
    required=True,
    help="GCS location to write checkpoints and export models")

  parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help="GCS location to read checkpoint and restore the models")

  parser.add_argument(
    "--env",
    choices=["local", "cloud"],
    default="local",
    help="Wether the model run on local or cloud environment")

  parser.add_argument(
    "--pattern", type=str, required=True, help="shots angle to train")

  parser.add_argument(
    "--max-steps", type=str, default="", help="max training steps")

  parser.add_argument(
    "--n-epochs", type=int, default=N_EPOCHS, help="training epochs")

  parser.add_argument(
    "--batch-size",
    type=int,
    default=BATCH_SIZE,
    help="Batch size for training steps")

  args = parser.parse_args()

  train(
    features_dir=args.features_dir,
    labels_dir=args.labels_dir,
    job_dir=args.job_dir,
    pattern=args.pattern,
    checkpoint=args.checkpoint,
    env=args.env,
    max_steps=args.max_steps,
    n_epochs=args.n_epochs,
    batch_size=args.batch_size)
