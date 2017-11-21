# Creation of Heat-map from tiles classified with inception v3.
#
# This code is a modification from image_classification.py:
# 
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

""" NYU modifications:
    Author: Nicolas Coudray
    Date created: March/2017
    Python Version: 3.5.3

	Use it with Lung images:
		For each slide in a given class, aggregate the predictions into a Heat-map
		
		If it is issued from a network with the last layer retrained:
			This program creates a graph from a saved GraphDef protocol buffer,
			and runs inference on ALL test images. It outputs human readable
			strings of the top prediction along with their probabilities.

		If it is aimed at a fully retrained inception v3 networK:
			read info from a text file "out_filename_Stats.txt" where each
			raw corresponds to a tile:
				tilename	Is_Tile_Correctly_Classified?	[bkg_inference	class1_inference   class2_inferece]
			example:
test_TCGA-77-8133-01A-01-TS1.1490c839-16d4-42b1-957b-01b962b14bfa_10_9.jpeg	True	[ 0.03734457  0.29187825  0.6707772 ]
test_TCGA-77-8133-01A-01-TS1.1490c839-16d4-42b1-957b-01b962b14bfa_16_8.jpeg	False	[ 0.03056222  0.57014841  0.39928937]
		


 # use for transfer-lerning network:
 # python 0f_HeatMap.py  --image_file '/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/4_2Types' --tiles_overlap 0 --output_dir '/ifs/home/coudrn01/NN/TensorFlowTest/6a_test/result_test_3' --tiles_stats '/ifs/home/coudrn01/NN/TensorFlowTest/6a_test/result_test_3/out_filename_Stats.txt' --resample_factor 2

# Additional options for fully trained NW and associated txt file

--tiles_stats '/ifs/home/coudrn01/NN/TensorFlowTest/6a_test/result_test_1/out_filename_Stats.txt' 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import pickle
import json
import csv
import numpy as np
from six.moves import urllib
import tensorflow as tf
from tensorflow.python.platform import gfile


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.misc
from scipy.misc import imsave
from scipy.misc import imread

FLAGS = None

# pylint: disable=line-too-long
# DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""
  # def __init__(self,               label_lookup_path=None,               uid_lookup_path=None):
  #    if not label_lookup_path:
  #      label_lookup_path = os.path.join(
  #          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
  #    if not uid_lookup_path:
  #      uid_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
  def __init__(self, uid_lookup_path=None):
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(FLAGS.model_dir, 'output_labels.txt')
    self.node_lookup = self.load(uid_lookup_path)

  def load(self, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    proto_as_ascii = tf.gfile.GFile(uid_lookup_path).readlines()
    node_id_to_name = {}
    IdxNb = 0
    for line in proto_as_ascii:
        node_id_to_name[IdxNb] = line.split('\n')[0]
        IdxNb += 1

    return node_id_to_name


  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, 'output_graph.pb'), 'rb') as f:
#  with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(image, mydict, cTileRootName, label_name):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  #create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

#    softmax_tensor = sess.graph.get_tensor_by_name('3_Lung_1634:0')
    softmax_tensor = sess.graph.get_tensor_by_name(FLAGS.tensor_name)



    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

# KeyError: "The name 'softmax:0' refers to a Tensor which does not exist. The operation, 'softmax', does not exist in the graph."


#    test_accuracy, predictions = sess.run([evaluation_step, prediction],      feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})



    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    Top = True
    current_score = 0
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        if human_string.startswith(label_name):
            current_score = score
#	else:
#		human_string = human_string + '_False'
        print('node_id %d: %s (score = %.5f)' % (node_id , human_string, score))
        if (human_string + '_SumScore') in mydict[cTileRootName].keys():
            mydict[cTileRootName][human_string + '_SumScore'] = mydict[cTileRootName][human_string + '_SumScore'] + score
            if Top:
                Top = False
                if human_string.startswith(label_name): 
                    # if best result is True positive:
                    print(mydict[cTileRootName][human_string + '_Selected'])
                    mydict[cTileRootName][human_string + '_Selected'] = mydict[cTileRootName][human_string + '_Selected'] + 1.0
        else:
            mydict[cTileRootName][human_string + '_SumScore'] = score
            if Top:
                Top = False
                if human_string.startswith(label_name): 
                    mydict[cTileRootName][human_string + '_Selected'] = 1.0
                else:
                    mydict[cTileRootName][human_string + '_Selected'] = 0.0
            else:
                mydict[cTileRootName][human_string + '_Selected'] = 0.0

    return mydict, current_score 

def get_inference_from_file(test_filename, mydict, cTileRootName, label_name, AllLabels):

	for nClass in AllLabels.keys():
		if nClass == label_name:
			TP_name = label_name
		else:
			TN_name = nClass


	basename = os.path.basename(test_filename)
	# remove extension to basename:
	basename = ('.').join(basename.split('.')[:-1])
	print("basename is :" + basename)
	current_score = -1
	with open(FLAGS.tiles_stats) as f:
		Found = False
		for line in f:
			if basename in line:
				Found = True
				line = line.replace('[','').replace(']','').split()
				print(line)
				is_TP = line[1]
				class_1 = float(line[3])
				class_2 = float(line[4])
				sum_class = class_1 + class_2
				class_1 = class_1 / sum_class
				class_2 = class_2 / sum_class
				if is_TP == 'True':
					current_score = max(class_1, class_2)
					print("True found; score is %f" % current_score)
				else:
					current_score = min(class_1, class_2)
					print("False found; score is %f" % current_score)

				if (TP_name + '_SumScore') in mydict[cTileRootName].keys():
					# key exists
					mydict[cTileRootName][TP_name + '_SumScore'] = mydict[cTileRootName][TP_name + '_SumScore'] + current_score
					mydict[cTileRootName][TN_name + '_SumScore'] = mydict[cTileRootName][TN_name + '_SumScore'] + (1.0 - current_score)
					if is_TP == 'True':
						mydict[cTileRootName][TP_name + '_Selected'] = mydict[cTileRootName][TP_name + '_Selected'] + 1.0

				else:
					# create key
					mydict[cTileRootName][TP_name + '_SumScore'] = current_score
					mydict[cTileRootName][TN_name + '_SumScore'] = (1 - current_score)
					if is_TP == 'True':
						mydict[cTileRootName][TP_name + '_Selected'] = 1.0
					else:
						mydict[cTileRootName][TP_name + '_Selected'] = 0.0

				break
		if Found ==False:
			print("image not found in text file... and that's weird...")


	return mydict, current_score 


def saveMap(HeatMap_divider_p, HeatMap_0_p, WholeSlide_0, cTileRootName, label_name, NewSlide):
	# save the previous heat maps if any
	HeatMap_divider = HeatMap_divider_p * 1.0 + 0.0
	HeatMap_0 = HeatMap_0_p
	HeatMap_divider[HeatMap_divider == 0] = 1.0
	HeatMap_0 = np.divide(HeatMap_0, HeatMap_divider)
	alpha = 0.33
	out = HeatMap_0 * 255 * (1.0 - alpha) + WholeSlide_0 * alpha
	heatmap_path = os.path.join(FLAGS.output_dir,'heatmaps')
	if os.path.isdir(heatmap_path):	
		pass
	else:
		os.makedirs(heatmap_path)
	
	label_name = re.sub(r"[^\w\s]", '_', label_name)
	label_name = re.sub(r"\s+", '_', label_name)
	filename = os.path.join(heatmap_path, cTileRootName + "_" + label_name + "_heatmap.jpg")
	#print(cTileRootName + "_heatmap.jpg")
#	try:
#		os.remove(filename)
#	except OSError:
#		pass

	# check if image was already processed in a previous screen
	
	if NewSlide:
		if os.path.isfile(filename):
			print(filename + " has already been processed in the past. skipped.")
			skip = True
			return skip
		else:
			print(filename + " has processed for the first times.")
	imsave(filename,out)

	#filename = os.path.join(heatmap_path, cTileRootName + "_" + label_name + "_heatmap_BW.jpg")
	#imsave(filename,HeatMap_0 * 255)

	#filename = os.path.join(heatmap_path, cTileRootName + "_" + label_name + "_heatmap_Div.jpg")
	#imsave(filename,HeatMap_divider * 255)

	skip = False
	return skip

def main(_):
	image_dir = FLAGS.image_file
	if os.path.isdir(FLAGS.output_dir):
		if len(os.listdir(FLAGS.output_dir)) > 0:
			print("WARNING: output folder is not empty")	
	else:
		sys.exit("output path not defined or does not exist")
		
  # For each sufolder (class)
  ## --> read all the test images in a loop
  ## --> for each slide:
  ##	Number of Tiles
  ##	Number of Tiles classified in each class
  ## 	Average score for each class (first sum then divide)



	# Read the name of the folder (= class names)
	sub_dirs = []
	if FLAGS.tiles_stats == '':
		create_graph()
	for item in os.listdir(image_dir):
    		if os.path.isdir(os.path.join(image_dir, item)):
        		sub_dirs.append(os.path.join(image_dir,item))

	print("sub_dirs:")
	print(sub_dirs)
	mydict={}
	SlideRootName = ''
	SlideNames = []
	AllLabels = {}
	skip = False
	# get all the label names
	for sub_dir in sub_dirs:
		dir_name = os.path.basename(sub_dir)
		label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
		AllLabels[label_name] = {}

	print("AllLabels:")
	print(AllLabels)
	# For each class
	for sub_dir in list(sub_dirs):
	#for sub_dir in list(reversed(sub_dirs)):
		dir_name = os.path.basename(sub_dir)
		label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
		extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
		file_list = []

		print("list the images in folder %s..." % (dir_name) )
		for extension in extensions:
      			#file_glob = os.path.join(image_dir, dir_name, 'test_*.' + extension)
      			file_glob = os.path.join(image_dir, dir_name, 'test_*' + FLAGS.tiles_filter + '*.' + extension)
      			file_list.extend(gfile.Glob(file_glob))
		if not file_list:
      			print('No images found')
      			continue


		## Aggregate the results and build heatmaps
		Start = True
		NewSlide = True
		# 1. For each slide, compute the number of good and bad classifications
		for test_filename in sorted(file_list):
			#cTileRootName = os.path.basename(test_filename).split('_files')[0]
			#cTileRootName = os.path.basename(test_filename)[0:50]	
			# remove slide number from image name:
			cTileRootName =  '_'.join(os.path.basename(test_filename).split('_')[0:-2]) 
			# extract coordinates of the tile
			ixTile = int(os.path.basename(test_filename).split('_')[-2])
			iyTile = int(os.path.basename(test_filename).split('_')[-1].split('.')[0])
			im2 = imread(test_filename)
			# check how bif the "re-combined" slide should be (invert col/row because of the swapaxes required)
			rTile = im2.shape[1]
			cTile = im2.shape[0]
						
			xTile =  (ixTile) * (rTile - FLAGS.tiles_overlap)
			yTile =  (iyTile) * (cTile - FLAGS.tiles_overlap)
			req_xLength = xTile + rTile
			req_yLength = yTile + cTile

			if FLAGS.resample_factor > 0:
				print("old / new r&cTile")
				print(rTile, cTile)
				rTile = round(rTile / FLAGS.resample_factor)
				cTile = round(cTile / FLAGS.resample_factor)
				print(rTile, cTile)
				im2s = scipy.misc.imresize(im2, (cTile, rTile))
				rTile = im2s.shape[1]
				cTile = im2s.shape[0]

				ixTile = round(ixTile / FLAGS.resample_factor)
				iyTile = round(iyTile / FLAGS.resample_factor)
				xTile = round(xTile / FLAGS.resample_factor)
				yTile = round(yTile / FLAGS.resample_factor)
				req_xLength = xTile + rTile
				req_yLength = yTile + cTile
			else:
				im2s = im2
			#print(ixTile, iyTile, rTile, cTile, req_xLength, req_yLength, xTile, yTile)
			if cTileRootName == SlideRootName:
				if skip:
					continue
				mydict[cTileRootName]['NbrTiles'] = mydict[cTileRootName]['NbrTiles'] + 1.0

				'''
				# append to the new re-combined slide		
				WholeSlide_temp = np.zeros([max(WholeSlide_0.shape[0], req_xLength), max(WholeSlide_0.shape[1],req_yLength), 3])
				WholeSlide_temp[0:WholeSlide_0.shape[0],0:WholeSlide_0.shape[1],:] = WholeSlide_0
				#print(WholeSlide_temp.shape)
				#print(xTile, req_xLength+1, yTile, req_yLength+1)
				# tmp = WholeSlide_temp[xTile:req_xLength+1, yTile:req_yLength+1,:]
				#print(tmp.shape)
				# WholeSlide_temp[xTile:req_xLength+1, yTile:req_yLength+1,:] = np.swapaxes(im2,0,1)
				# WholeSlide_temp[xTile:req_xLength, yTile:req_yLength,:] = np.swapaxes(im2,0,1)  or np.rot90(im2,1)
				WholeSlide_temp[xTile:req_xLength, yTile:req_yLength,:] = np.swapaxes(im2s,0,1)
				WholeSlide_0 = WholeSlide_temp
				del WholeSlide_temp

				HeatMap_temp = WholeSlide_0 * 0
				HeatMap_temp[0:HeatMap_0.shape[0], 0:HeatMap_0.shape[1],:] = HeatMap_0
				HeatMap_0 = HeatMap_temp
				del HeatMap_temp
			
				HeatMap_divider_tmp = WholeSlide_0 * 0
				HeatMap_divider_tmp[0:HeatMap_divider.shape[0], 0:HeatMap_divider.shape[1],:] = HeatMap_divider
				HeatMap_divider = HeatMap_divider_tmp
				del HeatMap_divider_tmp
				# save every time
				skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, SlideRootName, label_name, NewSlide)
				'''
				NewSlide = False
				#if skip:
				#	continue


			else:
				# Moved to a new slide
				print("Analyzing %s" % (cTileRootName) )
				if Start:
					Start = False
				#else:
				elif skip==False:
					# For previous the slide which is now finished, compute the averages 
					for key in mydict[SlideRootName].keys():
						if key.endswith('_SumScore'):
							mydict[SlideRootName][key] = mydict[SlideRootName][key] / mydict[SlideRootName]['NbrTiles']
						elif key.endswith('_Selected'):
							mydict[SlideRootName][key] = mydict[SlideRootName][key] / mydict[SlideRootName]['NbrTiles']

					# save the previous heat maps if any
					#HeatMap_divider[HeatMap_divider == 0] = 1
					#HeatMap_0 = np.divide(HeatMap_0, HeatMap_divider)
					#alpha = 0.5
					#out = HeatMap_0 * 255 * (1.0 - alpha) + WholeSlide_0 * alpha
					#heatmap_path = os.path.join(FLAGS.output_dir,'heatmaps')
					#os.makedirs(heatmap_path)
					#imsave(os.path.join(heatmap_path, [cTileRootName + '_heatmap.jpg']),out)
					skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, SlideRootName, label_name, NewSlide)

				NewSlide = True
				skip = False
				SlideRootName = cTileRootName
				mydict[SlideRootName] = {}
				mydict[SlideRootName]['NbrTiles'] = 1.0
				mydict[SlideRootName]['Read_Class'] = label_name

				
				# create a new re-combined slide	
				WholeSlide_0 = np.zeros([req_xLength, req_yLength, 3])
				#WholeSlide_0[xTile:req_xLength, yTile:req_yLength,:] = im2s
				# WholeSlide_0[xTile:req_xLength, yTile:req_yLength,:] = np.swapaxes(im2s,0,1)
				HeatMap_0 = np.zeros([req_xLength, req_yLength, 3])
				HeatMap_divider = np.zeros([req_xLength, req_yLength, 3])
				

			# Check score associated with that image:
			# print("FLAGS.tiles_stats")
			# print(FLAGS.tiles_stats)
			if FLAGS.tiles_stats == '':
				# transfer learning from retrained last layer of inception
				mydict, current_score = run_inference_on_image(test_filename, mydict, cTileRootName, label_name)
			else:
				# text file with stats from fully retrained network
				mydict, current_score = get_inference_from_file(test_filename, mydict, cTileRootName, label_name, AllLabels)
	

			# prepare heatmap
			print("current score")
			print(current_score)
			if current_score < 0:
				print("No probability found")
			else:
				if NewSlide == False:
					# append to the new re-combined slide		
					WholeSlide_temp = np.zeros([max(WholeSlide_0.shape[0], req_xLength), max(WholeSlide_0.shape[1],req_yLength), 3])
					WholeSlide_temp[0:WholeSlide_0.shape[0],0:WholeSlide_0.shape[1],:] = WholeSlide_0
					#print(WholeSlide_temp.shape)
					#print(xTile, req_xLength+1, yTile, req_yLength+1)
					# tmp = WholeSlide_temp[xTile:req_xLength+1, yTile:req_yLength+1,:]
					#print(tmp.shape)
					# WholeSlide_temp[xTile:req_xLength+1, yTile:req_yLength+1,:] = np.swapaxes(im2,0,1)
					# WholeSlide_temp[xTile:req_xLength, yTile:req_yLength,:] = np.swapaxes(im2,0,1)  or np.rot90(im2,1)
					WholeSlide_temp[xTile:req_xLength, yTile:req_yLength,:] = np.swapaxes(im2s,0,1)
					WholeSlide_0 = WholeSlide_temp
					del WholeSlide_temp

					HeatMap_temp = WholeSlide_0 * 0
					HeatMap_temp[0:HeatMap_0.shape[0], 0:HeatMap_0.shape[1],:] = HeatMap_0
					HeatMap_0 = HeatMap_temp
					del HeatMap_temp
			
					HeatMap_divider_tmp = WholeSlide_0 * 0
					HeatMap_divider_tmp[0:HeatMap_divider.shape[0], 0:HeatMap_divider.shape[1],:] = HeatMap_divider
					HeatMap_divider = HeatMap_divider_tmp
					del HeatMap_divider_tmp

					#skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, SlideRootName, label_name, NewSlide)

				else:
					WholeSlide_0[xTile:req_xLength, yTile:req_yLength,:] = np.swapaxes(im2s,0,1)


				#heattile = np.ones([512,512]) * current_score
				heattile = np.ones([req_xLength-xTile,req_yLength-yTile]) * current_score
				cmap = plt.get_cmap('BrBG')
				heattile = cmap(heattile)
				heattile = heattile[:,:,0:3]

				#HeatMap_0[xTile:req_xLength+1, yTile:req_yLength+1,:] = HeatMap_0[xTile:req_xLength+1, yTile:req_yLength+1,:] + heattile
				#HeatMap_divider[xTile:req_xLength+1, yTile:req_yLength+1,:] = HeatMap_divider[xTile:req_xLength+1, yTile:req_yLength+1,:] + 1
				HeatMap_0[xTile:req_xLength, yTile:req_yLength,:] = HeatMap_0[xTile:req_xLength, yTile:req_yLength,:] + heattile
				HeatMap_divider[xTile:req_xLength, yTile:req_yLength,:] = HeatMap_divider[xTile:req_xLength, yTile:req_yLength,:] + 1

				skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, SlideRootName, label_name, NewSlide)
				if skip:
					continue
		

		for key in mydict[SlideRootName].keys():
			if key.endswith('_SumScore'):
				mydict[SlideRootName][key] = mydict[SlideRootName][key] / mydict[SlideRootName]['NbrTiles']
			elif key.endswith('_Selected'):
				mydict[SlideRootName][key] = mydict[SlideRootName][key] / mydict[SlideRootName]['NbrTiles']

		skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, SlideRootName, label_name, NewSlide)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=2,
      help='Display this many predictions.'
  )
  parser.add_argument(
      '--tensor_name',
      type=str,
      default='3_Lung_1634:0',
      help='Name of the tensor.'
  )
  parser.add_argument(
      '--tiles_overlap',
      type=int,
      default=0,
      help='Overlap of the tiles in pixels.'
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      default='mustbedefined',
      help='Output directory.'
  )
  parser.add_argument(
      '--resample_factor',
      type=int,
      default=1,
      help='reduce the size of the output by this factor.'
  )
  parser.add_argument(
      '--tiles_stats',
      type=str,
      default='',
      help='text file where tile statistics are saved.'
  )
  parser.add_argument(
      '--tiles_filter',
      type=str,
      default='',
      help='process only images with this name.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


