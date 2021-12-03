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
				stats_dict['.'.join(line2[0].split('.')[:-1])] = line
	return stats_dict



def get_inference_from_file(test_filename, cTileRootName, stats_dict):



	basename = os.path.basename(test_filename)
	# remove extension to basename:
	basename = ('.').join(basename.split('.')[:-1])
	print("basename is :" + basename)
	current_score = -1
	score_correction = 0
	oClass = -1
	cmap = plt.get_cmap('binary')
	Found = False
	Mutation = False
	if basename in stats_dict.keys():
		line = stats_dict[basename]
		lineProb = line.split('[')[1]
		lineProb = lineProb.split(']')[0]
		lineProb = lineProb.split()
		line = line.replace('[','').replace(']','').split()
		Found = True
		print(line)
		print(lineProb)
		if FLAGS.Cmap == 'CancerType':
			is_TP = line[1]
			NumberOfClasses = len(lineProb)
			class_all = []
			sum_class = 0
			for nC in range(1,NumberOfClasses):
				class_all.append(float(lineProb[nC]))
				sum_class = sum_class + float(lineProb[nC])
			for nC in range(NumberOfClasses-1):
				class_all[nC] = class_all[nC] / sum_class
			current_score = max(class_all)
			#class_1 = float(line[3])
			#class_2 = float(line[4])
			#class_3 = float(line[5])
			#sum_class = class_1 + class_2 + class_3
			#class_1 = class_1 / sum_class
			#class_2 = class_2 / sum_class
			#class_3 = class_3 / sum_class
			#current_score = max(class_1, class_2, class_3)
			oClass = class_all.index(max(class_all)) + 1
			#if current_score == class_1:
			#	oClass = 1
			#elif current_score == class_2:
			#	oClass = 2
			#else:
			#	oClass = 3
			if FLAGS.thresholds is not None:
				thresholds = FLAGS.thresholds
				thresholds = [float(x) for x in thresholds.split(',')]
				if len(thresholds) != len(class_all):
					print("Error: There must be one threshold per class:")
				probDiff = []
				for nC in range(len(class_all)):
					probDiff.append(class_all[nC] - thresholds[nC])
				print(probDiff)
				oClass = probDiff.index(max(probDiff)) + 1
				current_score = class_all[oClass - 1]
				#if class_1 > thresholds[0]:
				#	oClass = 1
				#elif class_2 > thresholds[1]:
				#	oClass = 2
				#	if class_3 > thresholds[2]:
				#		oClass = 4
				#if class_3 > thresholds[2]:
				#	oClass = 3
				score_correction = thresholds[oClass-1]
			else:
				score_correction = 1.0 / len(class_all)
			if oClass == 1:
				if len(class_all) == 2:
					cmap = plt.get_cmap('OrRd')
				else:
					cmap = plt.get_cmap('binary')
			elif oClass == 2:
				if len(class_all) == 2:
					cmap = plt.get_cmap('Blues')
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


		elif FLAGS.Cmap == 'EGFR':
			Mutation = True
			cmap = plt.get_cmap('Reds')
			oClass = 0
		elif FLAGS.Cmap == 'FAT1':
			Mutation = True
			cmap = plt.get_cmap('Oranges')
			oClass = 1
		elif FLAGS.Cmap == 'FAT4':
			Mutation = True
			c = mcolors.ColorConverter().to_rgb
			cmap = make_colormap([c('white'), c('yellow')])
			oClass = 2
		elif FLAGS.Cmap == 'KEAP1':
			Mutation = True
			c = mcolors.ColorConverter().to_rgb
			cmap = make_colormap([c('white'), c('green')])
			oClass = 3
		elif FLAGS.Cmap == 'KRAS':
			Mutation = True
			cmap = plt.get_cmap('Greens')
			oClass = 4
		elif FLAGS.Cmap == 'LRP1B':
			Mutation = True
			c = mcolors.ColorConverter().to_rgb
			cmap = make_colormap([c('white'), c('blue')])
			oClass = 5
		elif FLAGS.Cmap == 'NF1':
			Mutation = True
			cmap = plt.get_cmap('Blues')
			oClass = 6
		elif FLAGS.Cmap == 'SETBP1':
			Mutation = True
			cmap = plt.get_cmap('Purples')
			oClass = 7
		elif FLAGS.Cmap == 'STK11':
			Mutation = True
			c = mcolors.ColorConverter().to_rgb
			cmap = make_colormap([c('white'), c('magenta')])
			oClass = 8
		elif FLAGS.Cmap == 'TP53':
			Mutation = True
			cmap = plt.get_cmap('Greys')
			oClass = 9

		
		if Mutation:
			analyze = True
			if os.path.isfile(FLAGS.filter_tile):
				with open(FLAGS.filter_tile) as fstat2:
					for line2 in fstat2:
						print(line2)
						print(".".join(line[0].split(".")[:-1]))
						if ".".join(line[0].split(".")[:-1]) in line2:
							ref = line2.replace('[','').replace(']','').split()
							nMax = max([float(ref[3]), float(ref[4]), float(ref[5])])
							LUAD = float(ref[4])
							print("Found:")
							print(line2, nMax, LUAD)
							if LUAD != nMax:
								analyze = False
								current_score = -1
							#print(analyze)
							break
			if analyze == False:
				print("continue - Mutation not found")
			else:
				EGFR = float(line[13])
				FAT1 = float(line[14])
				FAT4 = float(line[15])
				KEAP1 = float(line[16])
				KRAS = float(line[17])
				LRP1B = float(line[18])
				NF1 = float(line[19])
				SETBP1 = float(line[20])
				STK11 = float(line[21])
				TP53 = float(line[22])
				Alldata = [EGFR, FAT1, FAT4, KEAP1, KRAS, LRP1B, NF1, SETBP1, STK11, TP53]
				current_score = Alldata[oClass]
		
	if Found ==False:
		print("image not found in text file... and that's weird...")

	print(oClass, current_score, (current_score-score_correction)/(1.0-score_correction))
	return oClass, cmap, (current_score-score_correction)/(1.0-score_correction)


def saveMap(HeatMap_divider_p, HeatMap_0_p, WholeSlide_0, cTileRootName, NewSlide, dir_name):
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
			print(filename + " has processed for the first times.")

	out[out == [0,0,0]] = 255
	imsave(filename,out)

	#filename = os.path.join(heatmap_path, cTileRootName + "_" + label_name + "_heatmap_BW.jpg")
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
	#		print("yes")
	#print(filtered_dict)

	## Aggregate the results and build heatmaps
	Start = True
	NewSlide = True
	old_dir_name = ''
	dir_name = 'unknown'
	# For each image in the out_filename_stats:
	for tile in sorted(filtered_dict.keys()):
		# remove slide number from image name:
		cTileRootName =  '_'.join(tile.split('_')[0:-2]) 
		extensions = ['.jpeg', '.jpg']
		isError = True
		old_dir_name =  dir_name 
		dir_name = 'unknown'
		#print(tile)
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
			print("image not found:")
			print(tile)
			# break
			continue

		# remove slide number from image name:
		cTileRootName =  '_'.join(os.path.basename(test_filename).split('_')[0:-2]) 
		print("cTileRootName")
		print(cTileRootName)
		print("test_filename")
		print(test_filename)
		print("dir_name")
		print(dir_name)


		# extract coordinates of the tile
		ixTile = int(os.path.basename(test_filename).split('_')[-2])
		iyTile = int(os.path.basename(test_filename).split('_')[-1].split('.')[0])
		# check how big the "re-combined" slide should be (invert col/row because of the swapaxes required)
		rTile = im2.shape[1]
		cTile = im2.shape[0]
					
		xTile =  (ixTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap)
		yTile =  (iyTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap)
		req_xLength = xTile + rTile
		req_yLength = yTile + cTile

		if FLAGS.resample_factor > 0:
			print("old / new r&cTile")
			print(rTile, cTile, xTile, yTile,req_xLength, req_yLength)
			rTile = int(rTile / FLAGS.resample_factor)
			cTile = int(cTile / FLAGS.resample_factor)
			#print(rTile, cTile)
			if rTile<=0:
				im2s = im2
			elif cTile<=0:
				im2s = im2
			else:
				# im2s = scipy.misc.imresize(im2, (cTile, rTile))
				im2s = np.array(Image.fromarray(im2).resize((cTile, rTile)))
				rTile = im2s.shape[1]
				cTile = im2s.shape[0]

				#ixTile = int(ixTile / FLAGS.resample_factor)
				#iyTile = int(iyTile / FLAGS.resample_factor)
				xTile = int(xTile / FLAGS.resample_factor)
				yTile = int(yTile / FLAGS.resample_factor)
				req_xLength = xTile + rTile
				req_yLength = yTile + cTile
				print(rTile, cTile, xTile, yTile,req_xLength, req_yLength)
		else:
			im2s = im2
		#print(ixTile, iyTile, rTile, cTile, req_xLength, req_yLength, xTile, yTile)
		if cTileRootName == SlideRootName:
			if skip:
				continue

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
				skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, SlideRootName, NewSlide, old_dir_name)

			NewSlide = True
			skip = False
			SlideRootName = cTileRootName
			
			# create a new re-combined slide	
			WholeSlide_0 = np.zeros([req_xLength, req_yLength, 3])
			HeatMap_0 = np.zeros([req_xLength, req_yLength, 3])
			HeatMap_divider = np.zeros([req_xLength, req_yLength, 3])
			

		# Check score associated with that image:
		# print("FLAGS.tiles_stats")
		# print(FLAGS.tiles_stats)
		# text file with stats from fully retrained network
		oClass, cmap, current_score = get_inference_from_file(test_filename, cTileRootName, stats_dict)


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

			heattile = cmap(heattile)
			heattile = heattile[:,:,0:3]

			HeatMap_0[xTile:req_xLength, yTile:req_yLength,:] = HeatMap_0[xTile:req_xLength, yTile:req_yLength,:] + heattile
			HeatMap_divider[xTile:req_xLength, yTile:req_yLength,:] = HeatMap_divider[xTile:req_xLength, yTile:req_yLength,:] + 1

			skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, SlideRootName, NewSlide, dir_name)
			if skip:
				continue
	

	skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, SlideRootName, NewSlide, dir_name)





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
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

