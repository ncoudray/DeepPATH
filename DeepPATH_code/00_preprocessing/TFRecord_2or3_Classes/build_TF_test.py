# Code significantly modified by ABL group, NYU from:
#
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
"""Converts image data to TFRecords file format for Lung cancer images. Can be 1 TFRecord per tile or 1 per slide
"""
# python  /ifs/home/coudrn01/NN/TensorFlowTest/00_TFRecord/build_TF_test.py --directory=/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/4_2Types/ --output_directory=/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/4_2Types_TFR/test_TFR --num_threads=1
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('directory', '/path_to_jpg_sorted_by_class/',
                           'Training data directory')
tf.app.flags.DEFINE_string('output_directory', '/path_to_output_directory/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 1024,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 128,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_integer('PatientID', -1,
                            'Aggregate TFRecord using first digit of basenane as set by PatientID.')

tf.app.flags.DEFINE_boolean('one_FT_per_Tile', False,
                            '1 TFrecord per tile if True, otherwise, 1 per slide.')

tf.app.flags.DEFINE_string('ImageSet_basename', 'test',
                            'test, train or valid')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
# tf.app.flags.DEFINE_string('labels_file', '', 'Labels file')


FLAGS = tf.app.flags.FLAGS


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
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename


def _process_image(filename, coder):
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

  print(filename)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  #print('height/width:' + height + ' '  +width)
  assert image.shape[2] == 3

  return image_data, height, width

def _get_slide_name(filename, name):
  """
  Extract root name of the file, remove "test_" and "indexes

  """
  if FLAGS.one_FT_per_Tile == True:
    return  os.path.basename(filename)
  else:
    if FLAGS.PatientID > 0:
      # print("basename is %s" % (os.path.basename(filename)[:FLAGS.PatientID+len(name)+1]) )
      return  os.path.basename(filename)[:FLAGS.PatientID+len(name)+1]
    else:
      return  '_'.join(os.path.basename(filename).split('_')[0:-2]) 



def _process_image_files_batch_test(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
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
  rootname = ''
  slide_counter = 0
  counter = 0
  #filenames.sort()
  for i in range(len(filenames)):
    filename = filenames[i]
    label = labels[i]
    text = texts[i]
    image_buffer, height, width = _process_image(filename, coder)

    next_file_root = _get_slide_name(filename, name)
    if rootname == next_file_root:
      # New tile of same slide
      tile_counter += 1
    else:
      # New slide
      rootname = next_file_root
      slide_counter += 1
      tile_counter = 0
      counter += 1
      output_filename = '%s_%s.TFRecord' % (rootname, label)
      output_file = os.path.join(FLAGS.output_directory, output_filename)
      writer = tf.python_io.TFRecordWriter(output_file)

    example = _convert_to_example(filename, image_buffer, label,
                                  text, height, width)
    writer.write(example.SerializeToString())

    if not counter % 1000:
      print('%s [thread %d]: Processed tile %d of slide %d.' %
            (datetime.now(), thread_index, tile_counter, slide_counter))
      sys.stdout.flush()

  writer.close()
  print('%s [thread %d]: Wrote %d images to %s' %
        (datetime.now(), thread_index, slide_counter, output_file))
  sys.stdout.flush()
  shard_counter = 0


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
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

  shard_ranges = np.linspace(ranges[thread_index][0],ranges[thread_index][1],num_shards_per_batch + 1).astype(int)
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

#      image_buffer = tf.gfile.FastGFile(filename, 'r').read()
#      height, width = image_reader.read_image_dims(tf.Session(''), image_buffer)

      image_buffer, height, width = _process_image(filename, coder)
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


def _process_image_files(name, filenames, texts, labels, num_shards):
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
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch_test, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
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
  labels = []
  filenames = []
  texts = []
  print(unique_labels)


  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
#    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    typeIm = name + '*.jpeg'
    jpeg_file_path = os.path.join(data_dir, text, typeIm)
    matching_files = tf.gfile.Glob(jpeg_file_path)
    #print(matching_files)
    if len(matching_files) < 1:
      typeIm = name + '*.jpg'
      jpeg_file_path = os.path.join(data_dir, text, typeIm)
      matching_files = tf.gfile.Glob(jpeg_file_path)
    matching_files.sort()
    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

  # do not shuffle for the moment
  filenames = filenames
  texts = texts
  labels = labels

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels


def _process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, texts, labels = _find_image_files(name, directory)
  print('DONE***********************************************************')
  _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  #_process_dataset('valid', FLAGS.directory, FLAGS.validation_shards)
  #_process_dataset('train', FLAGS.directory, FLAGS.train_shards)
  #_process_dataset('test', FLAGS.directory, FLAGS.train_shards)
  print(FLAGS.ImageSet_basename)
  _process_dataset(FLAGS.ImageSet_basename, FLAGS.directory,
                   FLAGS.train_shards)



if __name__ == '__main__':
  tf.app.run()

