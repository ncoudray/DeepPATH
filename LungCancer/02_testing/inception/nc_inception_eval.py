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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time
import pickle

import numpy as np
import tensorflow as tf

from . import image_processing
from . import inception_model as inception


FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
#                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_dir', '/ifs/home/coudrn01/NN/TensorFlowTest/6a_Inception_TensorFlow/models/inception/results/4_2Types/results/0_eval',
  """Directory where to write event logs.""")
#tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
#                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/ifs/home/coudrn01/NN/TensorFlowTest/6a_Inception_TensorFlow/models/inception/results/0_scratch',
  """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
#tf.app.flags.DEFINE_integer('num_examples', 50000,
#                            """Number of examples to run. Note that the eval """
#                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_integer('num_examples', 121094,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
#tf.app.flags.DEFINE_string('subset', 'validation',
#                           """Either 'validation' or 'train'.""")
tf.app.flags.DEFINE_string('subset', 'valid',
                           """Either 'valid' or 'train'.""")


def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, max_percent_op, all_filenames, filename_queue, net2048_op):
  """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
  """
  tf.initialize_all_variables()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Successfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      print("-num_examples: %d" % (FLAGS.num_examples))
      # num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size)) * 1
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      count_top_5 = 0.0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      current_score = []
      while step < num_iter and not coord.should_stop():
        #top_1, top_5 = sess.run([top_1_op, top_5_op])
        top_1, top_5, max_percent, out_filenames, _, net2048 = sess.run([top_1_op, top_5_op, max_percent_op, all_filenames, filename_queue, net2048_op])


        # for each file, save results in a ext file summarizing things
        print("out_filenames")
        print(out_filenames)
        print(top_1)
        print(len(top_1))
        print(len(out_filenames))
        print("max_percent")
        print(max_percent)
        print("net2048")
        print(net2048)
        net2048_path = os.path.join(FLAGS.eval_dir, 'net2048')
        if os.path.isdir(net2048_path):	
          pass
        else:
          os.makedirs(net2048_path)
        for kk in range(len(out_filenames)):
          imageName = os.path.splitext(out_filenames[kk].decode('UTF-8'))[0]
          imageName = imageName + '.net2048'
          print("net2048 length + values:")
          print(len(net2048[kk]))
          print(net2048[kk])
          with open(os.path.join(net2048_path, imageName), "w") as myfile:
           myfile.write(str(top_1[kk]) + "\t")
           myfile.write(str(max_percent[kk]) + "\t")
           class_1 = float(max_percent[kk][1])  
           class_2 = float(max_percent[kk][2])
           sum_class = class_1 + class_2
           class_1 = class_1 / sum_class
           class_2 = class_2 / sum_class
           if top_1[kk] == True:
             tmp = max(class_1, class_2)
             print("True found; score is %f" % (max(class_1, class_2)))
           else:
             tmp = min(class_1, class_2)
             print("False found; score is %f" % (min(class_1, class_2)))
           myfile.write(str(tmp) + "\t\n")

           for nn in range(len(net2048[kk])):
           #for nn in range(2048):
             myfile.write(str(net2048[kk][nn]))
             myfile.write("\n")


        with open(os.path.join(FLAGS.eval_dir, 'out_filename_Stats.txt'), "a") as myfile:
          for kk in range(len(out_filenames)):
              myfile.write(out_filenames[kk].decode('UTF-8') + "\t")
              myfile.write(str(top_1[kk]) + "\t")
              myfile.write(str(max_percent[kk]) + "\n")

              print(max_percent[kk][0])
              print(max_percent[kk][1])
              print(max_percent[kk][2])
              class_1 = float(max_percent[kk][1])
              class_2 = float(max_percent[kk][2])
              sum_class = class_1 + class_2
              class_1 = class_1 / sum_class
              class_2 = class_2 / sum_class
              if top_1[kk] == True:
                current_score.append(max(class_1, class_2))
                print("True found; score is %f" % (max(class_1, class_2)))
              else:
                current_score.append(min(class_1, class_2))
                print("False found; score is %f" % (min(class_1, class_2)))

        #print(top_1, max_percent)
        count_top_1 += np.sum(top_1)
        count_top_5 += np.sum(top_5)
        step += 1
        if step % 20 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      # Compute precision @ 1.
      precision_at_1 = count_top_1 / total_sample_count
      recall_at_5 = count_top_5 / total_sample_count
      print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
            (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
      summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
      precision_at_1 = -1

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return precision_at_1, np.mean(current_score)


def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels, all_filenames, filename_queue = image_processing.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _, end_points, net2048 = inception.inference(images, num_classes)

    # Calculate predictions.
    #max_percent =  tf.argmax(logits,1)
    #max_percent = tf.reduce_max(logits, reduction_indices=[1]) / tf.add_n(logits)
    max_percent = end_points['predictions']
    # max_percent = len(end_points)
    #for kk in range(len(labels)):
    #   #max_percent.append(end_points['predictions'][kk][labels[kk]])
    #   max_percent.append(labels[kk])
    top_1_op = tf.nn.in_top_k(logits, labels, 1)
    top_5_op = tf.nn.in_top_k(logits, labels, 5)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph_def=graph_def)

    while True:
      precision_at_1, current_score = _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, max_percent, all_filenames, filename_queue, net2048)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
    return precision_at_1, current_score
