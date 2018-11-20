import argparse
import os
import pickle
import re

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.contrib.data import Dataset

import inputs_fn
import image_preprocess_fn
import model_fn
import encode

SOURCE_IMAGE_HEIGHT = 1280

SOURCE_IMAGE_WIDTH = 1918

NEW_IMAGE_SIZE = 112

KEEP_PROB = 0.7

LEARNING_RATE = 0.001

BATCH_SIZE = 20

RINT_THRESHOLD = 0.575

TARGET_WIDTH = NEW_IMAGE_SIZE

TARGET_HEIGHT = int(
  float(SOURCE_IMAGE_HEIGHT) /
  (float(SOURCE_IMAGE_WIDTH) / float(NEW_IMAGE_SIZE)))

N_OUTPUTS = TARGET_HEIGHT * TARGET_WIDTH


def _read_inputs(images_dataset):
  image = inputs_fn.tf_read_image_files(images_dataset)
  image.set_shape([SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH, 3])

  image = image_preprocess_fn.resize_pad(
    image,
    resize=True,
    resize_shape=(TARGET_HEIGHT, TARGET_WIDTH),
    pad=True,
    pad_shape=(NEW_IMAGE_SIZE, NEW_IMAGE_SIZE))

  image = image_preprocess_fn.standardize(image)

  image.set_shape([NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 3])

  return image


def _read_inputs_dataset(images):

  image_file = tf.read_file(images)

  image = tf.image.decode_image(image_file)
  image.set_shape([SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH, 3])

  image = image_preprocess_fn.resize_pad(
    image,
    resize=True,
    resize_shape=(TARGET_HEIGHT, TARGET_WIDTH),
    pad=True,
    pad_shape=(NEW_IMAGE_SIZE, NEW_IMAGE_SIZE))

  image = image_preprocess_fn.standardize(image)

  image.set_shape([NEW_IMAGE_SIZE, NEW_IMAGE_SIZE, 3])

  return image


def predict(
    features_dir, job_dir, pattern, checkpoint, env, max_steps, batch_size):

  images_dataset = []

  regex = re.compile(pattern)
  for x in os.listdir(features_dir):
    if regex.match(x):
      images_dataset.append(os.path.join(features_dir, x))

  with tf.name_scope("preprocess"):

    with tf.device("/cpu:0"):
      dataset = Dataset.from_tensor_slices(images_dataset)

      dataset = dataset.map(_read_inputs_dataset)

      batched_dataset = dataset.batch(BATCH_SIZE)

      iterator = batched_dataset.make_one_shot_iterator()
      features = iterator.get_next()

  model_params = {
    "keep_prob": KEEP_PROB,
    "learning_rate": LEARNING_RATE,
    "rint_threshold": RINT_THRESHOLD,
    "n_outputs": N_OUTPUTS,
    "output_image_height": TARGET_HEIGHT,
    "output_image_width": TARGET_WIDTH,
    "source_image_height": SOURCE_IMAGE_HEIGHT,
    "source_image_width": SOURCE_IMAGE_WIDTH,
  }

  (
    predicts_op, predicts_rint_op, features_summary, predicts_summary,
    predicts_rint_summary) = model_fn.model_convnet(
      features, None, ModeKeys.PREDICT, model_params)

  saver = tf.train.Saver()

  summary_writer = tf.summary.FileWriter(job_dir, tf.get_default_graph())

  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.log_device_placement = True

  # runing the graph
  with tf.Session(config=config) as sess:

    # # we need to initialize all variables first
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    if checkpoint != "":
      print("RESTORE CHECKPOINT:", checkpoint)
      saver.restore(sess, checkpoint)

    step = 0

    imgs = []
    rle_masks = []

    images = images_dataset

    try:
      while True:
        (
          predicts, predicts_rint, features_log, predicts_log,
          predicts_rint_log,) = sess.run(
            [
              predicts_op, predicts_rint_op, features_summary, predicts_summary,
              predicts_rint_summary
            ])

        for index in range(step * BATCH_SIZE, (step + 1) * BATCH_SIZE):

          img = os.path.basename(images[index])
          imgs.append(img)

          rle_mask = encode.rle_encode(predicts_rint[index % BATCH_SIZE])
          rle_masks.append(rle_mask)

          print(index)

        if env == "local":
          print("STEP:", step)

        if max_steps != "" and step > int(max_steps):
          break

        if step % 10 == 0:
          summary_writer.add_summary(features_log, step)
          summary_writer.add_summary(predicts_log, step)
          summary_writer.add_summary(predicts_rint_log, step)

        step += 1

    except (tf.errors.OutOfRangeError, IndexError):
      print("FINISH TRAINING")

    with open(os.path.join(job_dir, "temp_{}".format(pattern)), "wb") as f:
      pickle.dump({
        "imgs": imgs,
        "rle_masks": rle_masks,
      }, f)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Input Arguments
  parser.add_argument(
    "--features-dir",
    type=str,
    required=True,
    help="GCS or local paths to training data")

  # parser.add_argument(
  # "--features-dir-list",
  # type=str,
  # required=True,
  # help="GCS or local paths to training data list")

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
    "--batch-size",
    type=int,
    default=BATCH_SIZE,
    help="Batch size for training steps")

  args = parser.parse_args()

  predict(
    features_dir=args.features_dir,
    job_dir=args.job_dir,
    pattern=args.pattern,
    checkpoint=args.checkpoint,
    env=args.env,
    max_steps=args.max_steps,
    batch_size=args.batch_size)
