import tensorflow as tf


def tf_read_image_files(
    image_files, n_epochs=None, shuffle=False, seed=None, capacity=32):

  file_queue = tf.train.string_input_producer(
    image_files,
    num_epochs=n_epochs,
    shuffle=shuffle,
    seed=seed,
    capacity=capacity)

  image_reader = tf.WholeFileReader()

  _, image_file = image_reader.read(file_queue)

  image = tf.image.decode_image(image_file)

  return image


# def tf_read_image_files_with_patterns(
# files_pattern, n_epochs=None, shuffle=False, seed=None, capacity=32):

# file_queue = tf.train.string_input_producer(
# tf.train.match_filenames_once(files_pattern),
# num_epochs=n_epochs,
# shuffle=shuffle,
# seed=seed,
# capacity=capacity)

# image_reader = tf.WholeFileReader()

# _, image_file = image_reader.read(file_queue)

# image = tf.image.decode_image(image_file)

# return image

# def tf_batch_inputs(
# features, labels, batch_size, num_threads, min_after_dequeue,
# shuffle=False):

# tensors = []

# if features is not None:
# tensors.append(features)

# if labels is not None:
# tensors.append(labels)

# if features is None and labels is None:
# raise ValueError("features and labels can not be both empty")

# batch_data = None
# batch_labels = None

# if shuffle:
# batch_data, batch_labels = tf.train.shuffle_batch(
# tensors,
# batch_size=batch_size,
# num_threads=num_threads,
# capacity=min_after_dequeue + 3 * batch_size,
# min_after_dequeue=min_after_dequeue)

# return batch_data, batch_labels

# else:
# batch_data, batch_labels = tf.train.batch(
# tensors,
# batch_size=batch_size,
# num_threads=num_threads,
# capacity=min_after_dequeue + 3 * batch_size)

# return batch_data, batch_labels
