# Code significantly modified by ABL group, NYU from:

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'

If your data set involves bounding boxes, please look at build_imagenet_data.py.
"""

# python build_image_data.py --directory=/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/tmp_4_2Types/ --output_directory=/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/tmp_4_2Types/ --train_shards=20 --validation_shards=20 --num_threads=4 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import cv2
from scipy.ndimage import rotate
import scipy.interpolate as interp
from skimage.color import rgb2hed, hed2rgb
from fractions import gcd

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('directory', '/path_to_sorted_jpg_images/',
                           'Training data directory')
tf.app.flags.DEFINE_string('output_directory', '/path_to_output_directory/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1024,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('validation_shards', 128,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('test_shards', 200,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_integer('MaxNbImages', -1,
                            'Maximum number of images in each class - Will be taken randomly if >0, otherwise, all images are taken (may help in unbalanced datasets: undersample oneof the datasets) - if MaxNbImages>number of tiles, data augmentation will be done (rotation, mirroring, leading to possibility to increase dataset 8 fold)')

tf.app.flags.DEFINE_integer('rescale', 0,
			    'If you want the images to be rescaled to a certain dimension (299 for example), write the target size in rescale')

tf.app.flags.DEFINE_float('hed', 0,
                         'Color augmentation if hed > 0 (lim) and < 1')
 
tf.app.flags.DEFINE_float('hed_pc', 0.5,
                         'Proportion of tiles for which color augmentation should be applied')

tf.app.flags.DEFINE_integer('version', 1,
                            'replace with 1 for projects before December 2022; put 2 otherwise for new projects (prevent tenforflow from reading image as BGR instead of RGB')



# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
# tf.app.flags.DEFINE_string('labels_file', '', 'Labels file')


FLAGS = tf.app.flags.FLAGS
FLAGS_div = gcd(100,round(FLAGS.hed_pc*100))

def random_color(image):
  # inspired from https://www.biorxiv.org/content/10.1101/2022.05.17.492245v1.full
  nlim = FLAGS.hed
  nmult = [random.uniform(-nlim, nlim), random.uniform(-nlim, nlim), random.uniform(-nlim, nlim)]
  nadd = [random.uniform(-nlim, nlim), random.uniform(-nlim, nlim), random.uniform(-nlim, nlim)]
  ihc_hed = rgb2hed(image)
  # Augment the Haematoxylin channel.
  ihc_hed[:, :, 0] *= 1.0 + nmult[0]
  ihc_hed[:, :, 0] += nadd[0]
  # Augment the Eosin channel.
  ihc_hed[:, :, 1] *= 1.0 + nmult[1]
  ihc_hed[:, :, 1] += nadd[1]
  #  Augment the DAB channel.
  ihc_hed[:, :, 2] *= 1.0 + nmult[2]
  ihc_hed[:, :, 2] += nadd[2]
  # Convert back to RGB color coding.
  patch_rgb = hed2rgb(hed=ihc_hed)
  patch_transformed = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)
  patch_transformed *= 255.0
  patch_transformed = patch_transformed.astype(dtype=np.uint8)
  # cv2.imwrite(str(kk)+"bright.jpg",patch_transformed)
  return patch_transformed


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3

    if FLAGS.version == 1:
       return image
    elif FLAGS.version == 2:
       return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
       print("ERROR: version flag must be explicetely set to 1 or 2")
       return "error"
    #return image
    # return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #tf2 swaps channels colors....


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(filename, coder, flipRot, indx):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()
  #image_data = tf.gfile.FastGFile(filename, 'r').read()

  # Convert any PNG to JPEG's for consistency.
  #if _is_png(filename):
  #  print('Converting PNG to JPEG for %s' % filename)
  #  image_data = coder.png_to_jpeg(image_data)

  #print(filename)
  #print("flipRot:")
  #print(flipRot)
  # Rot_Mir = False
  # image = coder.decode_jpeg(image_data)
  image = coder.decode_jpeg(image_data)
  if flipRot > 0:
    #image = coder.decode_jpeg(image_data)
    rotate_img = rotate(image, 90 * (flipRot%4))
    if flipRot > 3:
      rotate_img = np.flipud(rotate_img)
    image = rotate_img
    #image_data = cv2.imencode('.jpg', rotate_img)[1].tostring()
  if FLAGS.rescale > 0:
    #image = coder.decode_jpeg(image_data)
    Factor = max(image.shape[0], image.shape[1]) / FLAGS.rescale;
    x = int(image.shape[1] / Factor);
    y = int(image.shape[0] / Factor);
    # res = np.resize(image, (int(y),int(x),3))
    print(image.shape, int(y),int(x))
    # image_data = cv2.imencode('.jpg', res)[1].tostring()    
    image = np.resize(image, (int(y),int(x),3))
    #image_data = cv2.imencode('.jpg', image)[1].tostring()
  #gcd(100,round(FLAGS.hed_pc*100))
  if (FLAGS.hed > 0)  & ( (indx%(100/FLAGS_div)) < (FLAGS.hed_pc*100/FLAGS_div)):
    #image = coder.decode_jpeg(image_data)
    image = random_color(image)
    #image_data = cv2.imencode('.jpg', image)[1].tostring()

  #image_data = cv2.imencode('.jpg', image)[1].tostring()
  # Decode the RGB JPEG.
  #image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  #print('height/width:' + height + ' '  +width)
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards, flipRot):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]
      flipRoti = flipRot[i]
#      image_buffer = tf.gfile.FastGFile(filename, 'r').read()
#      height, width = image_reader.read_image_dims(tf.Session(''), image_buffer)

      image_buffer, height, width = _process_image(filename, coder, flipRoti, i)
#      try:
#        image_buffer, height, width = _process_image(filename, coder)
#      except Exception as e:
#        print(e)
#        print('SKIPPED: Unexpected eror while decoding %s.' % filename)
#        continue

      example = _convert_to_example(filename, image_buffer, label,
                                    text, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0

  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards, flipRot):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards, flipRot)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished dealing with all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(name, data_dir):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the image data set resides in JPEG files located in
      the following directory structure.

        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg

      where 'dog' is the label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
#  unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
  unique_labels = []
  for item in os.listdir(data_dir):
  	if os.path.isdir(os.path.join(data_dir, item)):
  		unique_labels.append(os.path.join(item))
  unique_labels.sort()
  print("unique_labels:")
  print(unique_labels)
  labels = []
  filenames = []
  texts = []
  flipRot = []
  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
#    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    typeIm = name + '*.jpeg'
    jpeg_file_path = os.path.join(data_dir, text, typeIm)
    matching_files = tf.gfile.Glob(jpeg_file_path)
    if len(matching_files) < 1:
      typeIm = name + '*.jpg'
      jpeg_file_path = os.path.join(data_dir, text, typeIm)
      matching_files = tf.gfile.Glob(jpeg_file_path)
#    print(matching_files)
    if FLAGS.MaxNbImages > 0:
      tmp_label = [label_index] * len(matching_files)
      tmp_text = [text] * len(matching_files)
      tmp_filename = matching_files
      print("length filename:%d  " % len(tmp_filename))
      shuffled_index = list(range(len(tmp_filename)))
      random.seed(12345)
      random.shuffle(shuffled_index)
     
      tmp_label = [tmp_label[i] for i in shuffled_index]
      tmp_text = [tmp_text[i] for i in shuffled_index]
      tmp_filename = [tmp_filename[i] for i in shuffled_index]
      tmp_flipRot = [0 for i in shuffled_index]
      print("length filename:%d  " % len(tmp_filename))

      if FLAGS.MaxNbImages > len(tmp_label):
        # rotate or mirror images if more images needed
        more_label = []
        more_text= []
        more_filename = []
        more_flipRot = []
        for i in range(1,8): #coding rotation  (3 possibilities) and mirror w/o rotation (4 possibilities)
          more_label.extend(tmp_label)
          more_text.extend(tmp_text)
          more_filename.extend(tmp_filename)
          more_flipRot.extend([i] * len(matching_files))
        # more_shuffled_index = list(range(len(tmp_filename), len(tmp_filename)+len(more_label)))
        more_shuffled_index = list(range(len(more_label)))
        random.seed(12345)
        random.shuffle(more_shuffled_index)
        print("SIZE:")
        print(len(more_label))
        print(len(tmp_label))
        print(len(tmp_filename))
        print(len(more_shuffled_index))
        more_label = [more_label[i] for i in more_shuffled_index]
        more_text = [more_text[i] for i in more_shuffled_index]
        more_filename = [more_filename[i] for i in more_shuffled_index]
        more_flipRot = [more_flipRot[i] for i in more_shuffled_index]

        tmp_label.extend(more_label)
        tmp_text.extend(more_text)
        tmp_filename.extend(more_filename)
        tmp_flipRot.extend(more_flipRot)

      tmp_label = tmp_label[:min(FLAGS.MaxNbImages, len(tmp_filename))]
      tmp_text = tmp_text[:min(FLAGS.MaxNbImages, len(tmp_filename))]
      tmp_filename = tmp_filename[:min(FLAGS.MaxNbImages, len(tmp_filename))]
      tmp_flipRot = tmp_flipRot[:min(FLAGS.MaxNbImages, len(tmp_filename))]

      print("length filename:%d  " % len(tmp_filename))

      labels.extend(tmp_label)
      texts.extend(tmp_text)
      filenames.extend(tmp_filename)
      flipRot.extend(tmp_flipRot)
      print("length filename (no tmp):%d  " % len(filenames))

    else:
      labels.extend([label_index] * len(matching_files))
      texts.extend([text] * len(matching_files))
      filenames.extend(matching_files)
      flipRot.extend([0] * len(matching_files))

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]
  flipRot = [flipRot[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  #for nn in range(len(filenames)):
  #  print(filenames[nn], texts[nn], labels[nn])
  return filenames, texts, labels, flipRot


def _process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, texts, labels, flipRot = _find_image_files(name, directory)
  print('DONE***********************************************************')
  print(name, len(filenames), len(texts), len(labels), num_shards, len(flipRot))
  if len(filenames)>0:
    _process_image_files(name, filenames, texts, labels, num_shards, flipRot)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  #_process_dataset('valid', FLAGS.directory,
  #                 FLAGS.validation_shards)
  _process_dataset('train', FLAGS.directory,
                   FLAGS.train_shards)
  #_process_dataset('test', FLAGS.directory,
  #                 FLAGS.test_shards)


if __name__ == '__main__':
  tf.app.run()

