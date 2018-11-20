import random
from PIL import ImageEnhance
from PIL import Image

import tensorflow as tf
import numpy as np

# default image size of 224 for being compatible with ImageNet
_DEFAULT_SIZE = 224


def tf_resize_pad(
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


def tf_random_adjustment(
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


def tf_random_flip(inputs, vertical=False, horizontal=True):

  image = inputs

  if vertical:
    image = tf.image.random_flip_up_down(image)

  if horizontal:
    image = tf.image.random_flip_left_right(image)

  return image


def tf_standardize(inputs):

  image = tf.image.per_image_standardization(inputs)

  return image


def tf_fit01(inputs):

  image = inputs

  image = tf.image.convert_image_dtype(image, tf.float32)
  image_max = tf.reduce_max(tf.reshape(image, [-1]))
  image = image / image_max

  return image


def pil_random_enhance(
    image,
    brightness=True,
    brightness_factor_min=0.1,
    brightness_factor_max=2,
    contrast=True,
    contrast_factor_min=0.1,
    contrast_factor_max=2,
    saturation=True,
    saturation_factor_min=0,
    saturation_factor_max=2,
    sharpness=True,
    sharpness_factor_min=0,
    sharpness_factor_max=2):

  new_image = image

  if brightness:
    enhancer = ImageEnhance.Brightness(new_image)
    new_image = enhancer.enhance(
      random.uniform(brightness_factor_min, brightness_factor_max))

  if contrast:
    enhancer = ImageEnhance.Contrast(new_image)
    new_image = enhancer.enhance(
      random.uniform(contrast_factor_min, contrast_factor_max))

  if saturation:
    enhancer = ImageEnhance.Color(new_image)
    new_image = enhancer.enhance(
      random.uniform(saturation_factor_min, saturation_factor_max))

  if sharpness:
    enhancer = ImageEnhance.Sharpness(new_image)
    new_image = enhancer.enhance(
      random.uniform(sharpness_factor_min, sharpness_factor_max))

  return new_image


def pil_resize_pad_image(
    image, size, random_enhance=True, pad=True, resample_filter=Image.LANCZOS):

  if random_enhance:
    image = pil_random_enhance(image)

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


def pil_image_to_nparray(image, shape):
  return np.array(image.getdata()).reshape(shape)


def pil_image_to_float(image):
  return image.astype(float) / 255.0
