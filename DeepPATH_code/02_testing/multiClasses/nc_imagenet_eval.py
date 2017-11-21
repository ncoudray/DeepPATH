"""A binary to evaluate Inception on the Lung data set.
Output generated:
** information for ROC curves with 2 aggregation methods (out_FPTPrate_PcTiles.txt, out_FPTPrate_ScoreTiles.txt)
** probability associated with each tile and info whether the max is a true positive or not (out_filename_Stats.txt)


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import csv

import tensorflow as tf

from inception import nc_inception_eval
from inception.nc_imagenet_data import ImagenetData
import numpy as np
#from inception import inception_eval
#from inception.imagenet_data import ImagenetData
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ImageSet_basename', 'test_',
                           """Either 'test_', 'valid_' or 'train_'.""")

tf.app.flags.DEFINE_integer('nbr_of_classes', 10,
                            """Number of possible classes.""")

tf.app.flags.DEFINE_string('labels_names', '/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/9_10mutations/label_names.txt',
                           'Names of the possible output labels ordered as desired')

def main(unused_argv=None):

  input_path = os.path.join(FLAGS.data_dir, FLAGS.ImageSet_basename + '*')
  print(input_path)
  #FLAGS.batch_size = 30
  data_files = tf.gfile.Glob(input_path)
  print(data_files)

  mydict={}
  count_slides = 0

  #with open(FLAGS.labels_names, "r") as f:
  #  for line in f:
  #    line = line.replace('\r','\n')
  #    line = line.split('\n')
  #    for eachline in line:
  #      if len(eachline)>0:
  #        unique_labels.append(eachline)
  if "test" in FLAGS.ImageSet_basename:
    for next_slide in data_files:
      print("New Slide ------------ %d" % (count_slides))
      labelindex = int(next_slide.split('_')[-1].split('.')[0])
      #if labelindex == 1:
      #  labelname = 'normal'
      #elif labelindex == 2:
      #  labelname = 'luad'
      #elif labelindex == 3:
      #  labelname = 'lusc'
      #else:
      #  labelname = 'error_label_name'
      # labelname = unique_labels[labelindex]

      # print("label %d: %s" % (labelindex, labelname))

      FLAGS.data_dir = next_slide
      dataset = ImagenetData(subset=FLAGS.subset)
      assert dataset.data_files()
      #if tf.gfile.Exists(FLAGS.eval_dir):
      #  tf.gfile.DeleteRecursively(FLAGS.eval_dir)
      #tf.gfile.MakeDirs(FLAGS.eval_dir)
      precision_at_1, current_score = nc_inception_eval.evaluate(dataset)
  elif "valid" in FLAGS.ImageSet_basename:
    #FLAGS.data_dir = FLAGS.data_dir + "/valid*"
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    nc_inception_eval.evaluate(dataset)



if __name__ == '__main__':
  tf.app.run()
