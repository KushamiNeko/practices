import tensorflow as tf

# default image size of 224 for being compatible with ImageNet
_DEFAULT_SIZE = 224


def resize_pad(
    inputs,
    resize=True,
    resize_shape=(_DEFAULT_SIZE, _DEFAULT_SIZE),
    pad=False,
    pad_shape=(_DEFAULT_SIZE, _DEFAULT_SIZE)):

  image = inputs

  if resize:
    image = tf.image.resize_images(inputs, resize_shape, align_corners=True)

  if pad:
    image = tf.image.resize_image_with_crop_or_pad(
      image, pad_shape[0], pad_shape[1])

  return image


def random_adjustment(
    inputs,
    brightness=True,
    brightness_delta=0.75,
    contrast=True,
    contrast_lower=0.25,
    contrast_upper=1.5,
    hue=False,
    hue_delta=0.5,
    saturation=True,
    saturation_lower=0.0,
    saturation_upper=1.25):

  image = inputs

  if brightness:
    image = tf.image.random_brightness(image, max_delta=brightness_delta)

  if contrast:
    image = tf.image.random_contrast(
      image, lower=contrast_lower, upper=contrast_upper)

  if hue:
    image = tf.image.random_hue(image, hue_delta)

  if saturation:
    image = tf.image.random_saturation(
      image, lower=saturation_lower, upper=saturation_upper)

    return image


def random_flip(inputs, vertical=False, horizontal=True):

  image = inputs

  if vertical:
    image = tf.image.random_flip_up_down(image)

  if horizontal:
    image = tf.image.random_flip_left_right(image)

  return image


def standardize(inputs):

  image = tf.image.per_image_standardization(inputs)

  return image


def fit01(inputs):

  image = inputs

  image = tf.image.convert_image_dtype(image, tf.float32)
  image_max = tf.reduce_max(tf.reshape(image, [-1]))
  image = image / image_max

  return image
