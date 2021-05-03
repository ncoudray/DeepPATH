# Creation of Heat-map from tiles classified with inception v3.

""" 
The MIT License (MIT)

Copyright (c) 2017, Nicolas Coudray and Aristotelis Tsirigos (NYU)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import pickle
import json
import csv
import numpy as np
import glob
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import scipy.misc
#from scipy.misc import imsave
#from scipy.misc import imread
from imageio import imwrite as imsave
from imageio import imread
from PIL import Image

FLAGS = None






def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def dict_tiles_stats():
	stats_dict = {}
	with open(FLAGS.tiles_stats) as f:
		for line in f:
			line2 = line.replace('[','').replace(']','').split()
			if len(line2)>0:	
				#print(line2)
				#print(len(line2))
				tilename = '.'.join(line2[0].split('.')[:-1])
				cTileRootName =  '_'.join(os.path.basename(tilename).split('_')[0:-2])
				if cTileRootName not in stats_dict.keys():
					stats_dict[cTileRootName] = {}
					stats_dict[cTileRootName]['tiles'] = {}
					stats_dict[cTileRootName]['xMax'] = 0
					stats_dict[cTileRootName]['yMax'] = 0
				# stats_dict['.'.join(line2[0].split('.')[:-1])] = line
				# stats_dict[cTileRootName][tilename] = line  		
				ixTile = int(os.path.basename(tilename).split('_')[-2])
				iyTile = int(os.path.basename(tilename).split('_')[-1].split('.')[0])
				stats_dict[cTileRootName]['xMax'] = max(stats_dict[cTileRootName]['xMax'], ixTile)
				stats_dict[cTileRootName]['yMax'] = max(stats_dict[cTileRootName]['yMax'], iyTile)
				lineProb = line.split('[')[1]
				lineProb = lineProb.split(']')[0]
				lineProb = lineProb.split()
				stats_dict[cTileRootName]['tiles'][tilename] = [str(ixTile), str(iyTile), lineProb]
	return stats_dict



def get_inference_from_file(lineProb_st):
	lineProb = [float(x) for x in lineProb_st]
	if FLAGS.Cmap == 'CancerType':
		NumberOfClasses = len(lineProb)
		class_all = []
		sum_class = 0
		for nC in range(1,NumberOfClasses):
			class_all.append(float(lineProb[nC]))
			sum_class = sum_class + float(lineProb[nC])
		for nC in range(NumberOfClasses-1):
			class_all[nC] = class_all[nC] / sum_class
		current_score = max(class_all)
		oClass = class_all.index(max(class_all)) + 1
		if FLAGS.thresholds is not None:
			thresholds = FLAGS.thresholds
			thresholds = [float(x) for x in thresholds.split(',')]
			if len(thresholds) != len(class_all):
				print("Error: There must be one threshold per class:")
			probDiff = []
			for nC in range(len(class_all)):
				probDiff.append(class_all[nC] - thresholds[nC])
			oClass = probDiff.index(max(probDiff)) + 1
			current_score = class_all[oClass - 1]
			score_correction = thresholds[oClass-1]
		else:
			score_correction = 1.0 / len(class_all)
		if oClass == 1:
			if len(class_all) == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('red')])
				# cmap = plt.get_cmap('OrRd')
			else:
				cmap = plt.get_cmap('binary')
		elif oClass == 2:
			if len(class_all) == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('blue')])
				# cmap = plt.get_cmap('Blues')
			else:
				cmap = plt.get_cmap('OrRd')
		elif oClass == 3:
			cmap = plt.get_cmap('Blues')
		elif oClass == 4:
			cmap = plt.get_cmap('Oranges')
		elif oClass == 5:
			cmap = plt.get_cmap('Greens')
		else:
			cmap = plt.get_cmap('Purples')
	print(oClass, current_score, (current_score-score_correction)/(1.0-score_correction))
	return oClass, cmap, (current_score-score_correction)/(1.0-score_correction), [class_all[0], class_all[1]]

			


def saveMap(HeatMap_divider_p, HeatMap_0_p, WholeSlide_0, cTileRootName, NewSlide, dir_name, HeatMap_bin):
	# save the previous heat maps if any
	HeatMap_divider = HeatMap_divider_p * 1.0 + 0.0
	HeatMap_0 = HeatMap_0_p
	# Bkg = HeatMap_divider + 0.0
	HeatMap_divider[HeatMap_divider == 0] = 1.0
	HeatMap_0 = np.divide(HeatMap_0, HeatMap_divider)
	alpha = 0.33
	out = HeatMap_0 * 255 * (1.0 - alpha) + WholeSlide_0 * alpha
	out = out.transpose((1, 0, 2))
	heatmap_path = os.path.join(FLAGS.output_dir,'heatmaps')
	print(heatmap_path)
	if os.path.isdir(heatmap_path):	
		pass
	else:
		os.makedirs(heatmap_path)
	
	filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + dir_name + ".jpg")
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
			print(filename + " is processed for the first times.")

	out[out == [0,0,0]] = 255
	imsave(filename,out)

	if (NewSlide == False):
		HeatMap_bin = np.divide(HeatMap_bin, HeatMap_divider) 
		ImBin = HeatMap_bin * 0.
		# if FLAGS.thresholds is not None:
		# NonBkgTiles_c1 = HeatMap_0[HeatMap_divider[:,:,1]>0,0]
		# NonBkgTiles_c2 = HeatMap_0[HeatMap_divider[:,:,1]>0,1]
		# NonBkgTiles_c3 = HeatMap_0[HeatMap_divider[:,:,1]>0,2]
		if FLAGS.thresholds is not None:
			thresholds = FLAGS.thresholds
			thresholds = [float(x) for x in thresholds.split(',')]
			ImBin[:,:,0] = HeatMap_bin[:,:,0] >= thresholds[0]
			ImBin[:,:,2] = HeatMap_bin[:,:,2] >= thresholds[1]
		else:
			Tmax = np.max(HeatMap_bin,2)
			ImBin[:,:,0] = HeatMap_bin[:,:,0] == Tmax
			ImBin[:,:,2] = HeatMap_bin[:,:,2] == Tmax
		c1 = sum(ImBin[(HeatMap_divider_p[:,:,1] * 1.0 + 0.0)>0,0])
		c3 = sum(ImBin[(HeatMap_divider_p[:,:,1] * 1.0 + 0.0)>0,2]) 
	
		ImBin[HeatMap_divider_p==0] = 1
		ImBin = ImBin.transpose((1, 0, 2))
	
	
		print("*************")
		print(c1, c3)
		print(round(c1/(c1+c3),4))
		filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + dir_name + "_bin_" + str(int(c1)) + "_" + str(int(c3)) + "_r_" + str(round(c1/(c1+c3),4)) +  ".jpg")
		imsave(filename,ImBin * 255.)

	#filename = os.path.joineheatmap_path, cTileRootName + "_" + label_name + "_heatmap_BW.jpg")
	#imsave(filename,HeatMap_0 * 255)

	#filename = os.path.join(heatmap_path, cTileRootName + "_" + label_name + "_heatmap_Div.jpg")
	#imsave(filename,HeatMap_divider * 255)

	skip = False
	return skip

def main():
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

	# Read out_filename stats:
	stats_dict = dict_tiles_stats()

	# Read the name of the folder (= class names)
	sub_dirs = []
	for item in os.listdir(image_dir):
    		if os.path.isdir(os.path.join(image_dir, item)):
        		sub_dirs.append(os.path.join(image_dir,item))

	#sub_dirs = [image_dir]

	print("sub_dirs:")
	print(sub_dirs)
	SlideRootName = ''
	SlideNames = []
	skip = False
	# get all the label names

	#print(stats_dict)
	filtered_dict = {}
	for k in stats_dict.keys():
		#print(k)
		if FLAGS.slide_filter in k:
			filtered_dict[k] =stats_dict[k]
	#print(filtered_dict)

	## Aggregate the results and build heatmaps
	dir_name = 'unknown'
	# For each image in the out_filename_stats:
	for slide in sorted(filtered_dict.keys()):
		NewSlide = True
		t = time.time()
		ixTile = int(stats_dict[slide]['xMax'])
		iyTile = int(stats_dict[slide]['yMax'])
		req_xLength =  (ixTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap) + FLAGS.tiles_size
		req_yLength =  (iyTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap) + FLAGS.tiles_size
		# req_xLength =  (ixTile) * (FLAGS.tiles_size) + FLAGS.tiles_size
		# req_yLength =  (iyTile) * (FLAGS.tiles_size) + FLAGS.tiles_size
		if FLAGS.resample_factor > 0:
			req_xLength = int(req_xLength / FLAGS.resample_factor + 1)
			req_yLength = int(req_yLength / FLAGS.resample_factor + 1)
		WholeSlide_0 = np.zeros([req_xLength, req_yLength, 3])
		HeatMap_0 = np.zeros([req_xLength, req_yLength, 3])
		HeatMap_bin = np.zeros([req_xLength, req_yLength, 3])
		HeatMap_divider = np.zeros([req_xLength, req_yLength, 3])
		print("Checking slide " + slide)
		print(req_xLength, req_yLength)
		skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, slide, NewSlide, dir_name, HeatMap_bin)
		if skip:
			print("slide done --")
			continue
		for tile in stats_dict[slide]['tiles'].keys():		
			extensions = ['.jpeg', '.jpg']
			isError = True
			dir_name = 'unknown'
			for extension in extensions:
				for sub_dir in list(sub_dirs):
					try:
						test_filename = os.path.join(sub_dir, tile + extension)
						im2 = imread(test_filename)
						dir_name = os.path.basename(sub_dir)
						isError = False
					except:
						isError = True
					if isError == False:
						break
				if isError == False:
					break
			if isError == True:
				# print("image not found:" + tile)
				continue
			cTileRootName = slide
			ixTile = int(stats_dict[slide]['tiles'][tile][0])
			iyTile = int(stats_dict[slide]['tiles'][tile][1])
			rTile = im2.shape[1]
			cTile = im2.shape[0]
			xTile =  (ixTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap)
			yTile =  (iyTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap)
			# xTile =  (ixTile) * (FLAGS.tiles_size)
			# yTile =  (iyTile) * (FLAGS.tiles_size)
			req_xLength = xTile + rTile
			req_yLength = yTile + cTile					
			if FLAGS.resample_factor > 0:
				# print("old / new r&cTile")
				# print(rTile, cTile, xTile, yTile)
				rTile = int(rTile / FLAGS.resample_factor)
				cTile = int(cTile / FLAGS.resample_factor)
				if rTile<=0:
					im2s = im2
				elif cTile<=0:
					im2s = im2
				else:
					im2s = np.array(Image.fromarray(im2).resize((cTile, rTile)))
					rTile = im2s.shape[1]
					cTile = im2s.shape[0]
					xTile = int(xTile / FLAGS.resample_factor)
					yTile = int(yTile / FLAGS.resample_factor)
					req_xLength = xTile + rTile
					req_yLength = yTile + cTile
					print(rTile, cTile, xTile, yTile,req_xLength, req_yLength)
			else:
				im2s = im2
			# Check score associated with that image:
			oClass, cmap, current_score, class_prob = get_inference_from_file(stats_dict[slide]['tiles'][tile][2])
			if current_score < 0:
				print("No probability found")
			else:
				# print(np.swapaxes(im2s,0,1).shape)
				# print(xTile,req_xLength,yTile,req_yLength)
				# print(WholeSlide_0.shape)
				WholeSlide_0[xTile:req_xLength, yTile:req_yLength,:] = np.swapaxes(im2s,0,1)
				heattile = np.ones([req_xLength-xTile,req_yLength-yTile]) * current_score
				heattile = cmap(heattile)
				heattile = heattile[:,:,0:3]
				HeatMap_0[xTile:req_xLength, yTile:req_yLength,:] = HeatMap_0[xTile:req_xLength, yTile:req_yLength,:] + heattile
				HeatMap_divider[xTile:req_xLength, yTile:req_yLength,:] = HeatMap_divider[xTile:req_xLength, yTile:req_yLength,:] + 1
				HeatMap_bin[xTile:req_xLength, yTile:req_yLength,0] = HeatMap_bin[xTile:req_xLength, yTile:req_yLength,0] + np.ones([req_xLength-xTile,req_yLength-yTile]) * class_prob[0]
				HeatMap_bin[xTile:req_xLength, yTile:req_yLength,2] = HeatMap_bin[xTile:req_xLength, yTile:req_yLength,2] + np.ones([req_xLength-xTile,req_yLength-yTile]) * class_prob[1]	
			print("tile time: " + str((time.time() - t)/60))

		NewSlide = False
		skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, slide, NewSlide, dir_name, HeatMap_bin)
		print("slide time: " + str((time.time() - t)/60))	





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
      '--tiles_size',
      type=int,
      default=512,
      help='tile size in pixels.'
  )
  parser.add_argument(
      '--tiles_stats',
      type=str,
      default='',
      help='text file where tile statistics are saved.'
  )
  parser.add_argument(
      '--slide_filter',
      type=str,
      default='',
      help='process only images with this basename.'
  )
  parser.add_argument(
      '--filter_tile',
      type=str,
      default='/OurSoftware/TCGA-05-5425/tmp_class_30perTF/out_filename_Stats.txt',
      help='if map is a mutation, apply cmap of mutations only if tiles are LUAD.'
  )
  parser.add_argument(
      '--Cmap',
      type=str,
      default='CancerType',
      help='can be CancerType, of the name of a mutation (TP53, EGFR...)'
  )
  parser.add_argument(
      '--thresholds',
      type=str,
      default=None,
      help='thresholds to use for each label - string, for example: 0.285,0.288,0.628. If none, take the highest one.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.tiles_size = FLAGS.tiles_size + 2 * FLAGS.tiles_overlap
  FLAGS.tiles_overlap = 2 * FLAGS.tiles_overlap
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

