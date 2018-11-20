import numpy as np


def rle_encode(img):
  """
  img: numpy array, 1 - mask, 0 - background
  Returns run length as string formated
  """

  pixels = img.flatten()
  pixels[0] = 0
  pixels[-1] = 0
  runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
  runs[1::2] = runs[1::2] - runs[:-1:2]

  return " ".join(str(x) for x in runs)
