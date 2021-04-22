# Code slightly modified by ABL group, NYU from:

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

from inception import image_processing
from inception import inception_model as inception


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run. """)
tf.app.flags.DEFINE_string('subset', 'valid',
                           """Either 'valid' or 'train' or 'test'.""")


def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, max_percent_op, all_filenames, filename_queue, net2048_op, endpoints_op, logits_op, labels_op):
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
      if "test" in FLAGS.TVmode:
        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size)*2.0)
      else:
        num_iter_1 = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size)*2.0)
        num_iter_2 = int(math.ceil(10000 / FLAGS.batch_size))
        num_iter = min(num_iter_1, num_iter_2)

      print("Nb of examples and iterations: %d, %d" %(FLAGS.num_examples,num_iter) )
      if num_iter == 0:
        num_iter = 1
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      # count_top_5 = 0.0
      total_sample_count = 0
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      current_score = []
      while step < num_iter and not coord.should_stop():
        if FLAGS.mode == '0_softmax':
          top_1, top_5, max_percent, out_filenames, _, net2048, endpoint, logits, labels = sess.run([top_1_op, top_5_op, max_percent_op, all_filenames, filename_queue, net2048_op, endpoints_op, logits_op, labels_op])
        elif FLAGS.mode == '1_sigmoid':
          max_percent, out_filenames, _, net2048, endpoint, logits, labels = sess.run([max_percent_op, all_filenames, filename_queue, net2048_op, endpoints_op, logits_op, labels_op])


        # for each file, save results in a ext file summarizing things
        print("iteration #%d" % step)
        print("out_filenames")
        print(out_filenames)
        print(len(out_filenames))
        print("max_percent")
        print(max_percent)

        # save overall stats
        #data_path = os.path.join(FLAGS.eval_dir, 'data')
        #if os.path.isdir(data_path):	
        #  pass
        #else:
        #  os.makedirs(data_path)
        np.set_printoptions(threshold=np.inf)
        if FLAGS.mode == '0_softmax':
          for kk in range(len(out_filenames)):
            imageName = os.path.splitext(out_filenames[kk].decode('UTF-8'))[0]
            imageName = imageName + '.dat'
            if True:
            #with open(os.path.join(data_path, imageName), "w") as myfile:
              #myfile.write(str(top_1[kk]) + "\t")
              #myfile.write(str(max_percent[kk]) + "\t")
              class_x = {}
              sum_class = 0
              max_Prob = 0
              # > background class '0' ignored
              for nC in range(1, len(max_percent[kk])):
                class_x[nC] = max_percent[kk][nC]
                sum_class = sum_class + max_percent[kk][nC]
                max_Prob = max(max_Prob, class_x[nC])
              for nC in range(1, len(class_x)):
                class_x[nC] = class_x[nC] / sum_class
              max_Prob = max_Prob / sum_class
              total_sample_count += 1
              '''
              class_1 = float(max_percent[kk][1])  
              class_2 = float(max_percent[kk][2])
              class_3 = float(max_percent[kk][3])
              sum_class = class_1 + class_2 + class_3
              class_1 = class_1 / sum_class
              class_2 = class_2 / sum_class
              class_3 = class_3 / sum_class
              '''
              if top_1[kk] == True:
                count_top_1 += 1
                tmp = max_Prob
                print("True found; score is %f" % (max_Prob))
              else:
                print(logits[kk])
                print(labels[kk])
                try:
                  tmp = class_x[labels[kk]]
                except:
                  tmp = -1
                '''
                if int(labels[kk]) == 1:
                  tmp = class_1
                elif int(labels[kk]) == 2:
                  tmp = class_2
                elif int(labels[kk]) == 3:
                  tmp = class_3
                else:
                  tmp = -1
                '''
                print("False found; score is %f" % (tmp))
              #myfile.write(str(tmp) + "\t\n")
              #myfile.write("labels: \t")
              #myfile.write(str(labels[kk]) + "\t")
              #myfile.write("\nlogits: \t")
              #for nlogit in logits[kk]:
              #  myfile.write(str(nlogit) + "\t")
            if "test" in FLAGS.TVmode:
              with open(os.path.join(FLAGS.eval_dir, 'out_filename_Stats.txt'), "a") as myfile:
                myfile.write(imageName + "\t")
                myfile.write(str(top_1[kk]) + "\t")
                myfile.write(" ".join(str(max_percent[kk]).splitlines()) + "\t")
                myfile.write(str(tmp) + "\t")
                myfile.write("labels: \t")
                myfile.write(str(labels[kk]) + "\n")
        elif FLAGS.mode == '1_sigmoid':
          top_1 = 0
          for inLog, inLab in zip(max_percent, labels):
            lab0 = True
            for inLog2, inLab2 in zip(inLog, inLab):
              #ignore label 0
              if lab0:
                lab0 = False
                continue
              total_sample_count += 1
              print(inLog2, inLab)
              print(round(inLog2), round(inLab2))
              if round(inLog2)==round(inLab2):
                top_1 += 1
          count_top_1 += np.sum(top_1)
          print("tmp", count_top_1 / total_sample_count)
          for kk in range(len(out_filenames)):
            imageName = os.path.splitext(out_filenames[kk].decode('UTF-8'))[0]
            imageName = imageName + '.dat'
            #with open(os.path.join(data_path, imageName), "w") as myfile:
              #myfile.write(str(labels[kk]) + "\t")
              #myfile.write(" ".join(str(max_percent[kk]).splitlines()) + "\t")
          with open(os.path.join(FLAGS.eval_dir, 'out_filename_Stats.txt'), "a") as myfile:
            for kk in range(len(out_filenames)):
                myfile.write(out_filenames[kk].decode('UTF-8') + "\t")
                myfile.write(str(labels[kk]) + "\t")
                myfile.write(" ".join(str(max_percent[kk]).splitlines()) + "\n")        



        if "test" in FLAGS.TVmode:
          # save end of each layer of the network
          save = False
          if save:
            for key in endpoint.keys():
              for kk in range(len(out_filenames)):
                endpoints_path = os.path.join(FLAGS.eval_dir, key)
                if os.path.isdir(endpoints_path):	
                  pass
                else:
                  os.makedirs(endpoints_path)
  
                imageName = os.path.splitext(out_filenames[kk].decode('UTF-8'))[0]
                output_tmp = open(os.path.join(endpoints_path ,imageName + '.pkl'), 'ab+')
                pickle.dump(endpoint[key][kk], output_tmp)
                output_tmp.close()

        # save last-but one layer in separate text files
          if False:
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
               for nn in range(len(net2048[kk])):
                 myfile.write(str(net2048[kk][nn]))
                 myfile.write("\n")

            #data_path = os.path.join(FLAGS.eval_dir, 'data')
            #if os.path.isdir(data_path):
            #  pass
            #else:
            #  os.makedirs(data_path)




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
      with open(os.path.join(FLAGS.eval_dir, 'precision_at_1.txt'), "a") as myfile:
        myfile.write(str(datetime.now()) + ":\tPrecision:" + str(precision_at_1) + "\n")

      # recall_at_5 = count_top_5 / total_sample_count
      '''
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
      summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)
      '''
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
      precision_at_1 = -1

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return precision_at_1, np.mean(current_score)


def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
  """Calculate the total loss on a single tower running the ImageNet model.

  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
  then each tower will operate on an batch of 16 images.

  Args:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
    num_classes: number of classes
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # When fine-tuning a model, we do not restore the logits but instead we
  # randomly initialize the logits. The number of classes in the output of the
  # logit is the number of classes in specified Dataset.
  restore_logits = not FLAGS.fine_tune

  # Build inference Graph.
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    logits = inception.inference(images, num_classes, for_training=True,
                                 restore_logits=restore_logits,
                                 scope=scope)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  split_batch_size = images.get_shape().as_list()[0]
  inception.loss(logits, labels, batch_size=split_batch_size)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on TensorBoard.
    loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(loss_name +' (raw)', l)
    tf.summary.scalar(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels, all_filenames, filename_queue = image_processing.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class
    num_classes = dataset.num_classes() + 1
    print("there are %d classes!" % dataset.num_classes())

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _, end_points, net2048, sel_end_points = inception.inference(images, num_classes)

    # Calculate predictions.
    #max_percent =  tf.argmax(logits,1)
    #max_percent = tf.reduce_max(logits, reduction_indices=[1]) / tf.add_n(logits)
    max_percent = end_points['predictions']
    # max_percent = len(end_points)
    #for kk in range(len(labels)):
    #   #max_percent.append(end_points['predictions'][kk][labels[kk]])
    #   max_percent.append(labels[kk])
    if FLAGS.mode == '0_softmax':
      top_1_op = tf.nn.in_top_k(logits, labels, 1)
      top_5_op = tf.nn.in_top_k(logits, labels, 5)
    elif FLAGS.mode == '1_sigmoid':
      top_1_op = None
      top_5_op = None
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = [] #tf.summary.FileWriter(FLAGS.eval_dir, graph_def=graph_def)

    while True:
      #precision_at_1, current_score = _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, max_percent, all_filenames, filename_queue, net2048, sel_end_points, logits, labels)
      precision_at_1, current_score = _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, max_percent, all_filenames, filename_queue, net2048, sel_end_points, logits, labels)
      print("%s: Precision: %.4f " % (datetime.now(), precision_at_1) )
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
    return precision_at_1, current_score
