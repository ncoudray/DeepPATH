# Creation of Heat-map from tiles classified with inception v3.

""" NYU modifications:
    Author: Nicolas Coudray
    Date created: March/2017
    Python Version: 3.5.3
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
from scipy.misc import imsave
from scipy.misc import imread

FLAGS = None



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





def get_inference_from_file(TileName, stats_dict):

	# print("basename is :" + basename)
	current_score = -1
	score_correction = 0
	oClass = -1
	cmap = plt.get_cmap('binary')
	if TileName in stats_dict.keys():
		line = stats_dict[TileName]
		lineProb = line.split('[')[1]
		lineProb = lineProb.split(']')[0]
		lineProb = lineProb.split()
		line = line.replace('[','').replace(']','').split()
		print(line)
		print(lineProb)
	
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
		oClass = class_all.index(max(class_all)) + 1


		score_correction = 1.0 / len(class_all)
		
		'''analyze = True
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
		'''
	else:
		print("image not found in text file %s ... and that's weird..." % TileName)

	print(oClass, current_score, (current_score-score_correction)/(1.0-score_correction))
	#return oClass, (current_score-score_correction)/(1.0-score_correction)
	return class_all



def main():
	# Read out_filename stats:
	stats_dict = dict_tiles_stats()
  

	nClasses = [int(x) for x in FLAGS.Classes.split(',')]
	print("nClasses: " + str(nClasses))
	SlideRootName = ''
	SlideNames = []
	idx = {}
	iv1 = {}
	iv2 = {}
	iv3 = {}
	skip = False

	# print(stats_dict)
	# print(FLAGS.slide_filter)
	filtered_dict = {}
	for k in stats_dict.keys():
		#print(k)
		if FLAGS.slide_filter in k:
			filtered_dict[k] = stats_dict[k]
	# print("filtered:")
	# print(filtered_dict)

	#		print("yes")

	## Aggregate the results and build heatmaps
	Start = True
	NewSlide = True
	# For each image in the out_filename_stats:
	for tile in sorted(filtered_dict.keys()):
		# remove slide number from image name:
		cTileRootName =  '_'.join(tile.split('_')[0:-2]) 

		print("cTileRootName: %s" % cTileRootName)
		if cTileRootName == '':
			print("empty field")
			continue
		elif cTileRootName == SlideRootName:
			if skip:
				continue

			NewSlide = False
			#if skip:
			#	continue


		else:
			# Moved to a new slide
			print("Analyzing %s" % (cTileRootName) )
			idx[cTileRootName] = [(), ()]	
			iv1[cTileRootName] = []	
			iv2[cTileRootName] = []	
			iv3[cTileRootName] = []	

			#if skip:
				#	continue

			#elif skip==False:
				# For previous the slide which is now finished, compute the averages 

			NewSlide = True
			skip = False
		SlideRootName = cTileRootName
				#else:


		# extract coordinates of the tile
		ixTile = int(tile.split('_')[-2])
		iyTile = int(tile.split('_')[-1].split('.')[0])

		# check how big the "re-combined" slide should be (invert col/row because of the swapaxes required)
		#rTile = im2.shape[1]
		#cTile = im2.shape[0]
					
		#xTile =  (ixTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap)
		#yTile =  (iyTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap)
		#req_xLength = xTile + rTile
		#req_yLength = yTile + cTile

		class_all = get_inference_from_file(tile, stats_dict)

		# prepare heatmap
		print("current score: " + str(class_all))
		# print(idx, ixTile, iyTile)
		idx[cTileRootName][0] += (iyTile,)
		idx[cTileRootName][1] += (ixTile,)
		if len(nClasses) > 0:
			iv1[cTileRootName].append(class_all[nClasses[0]-1])
		if len(nClasses) >1:
			iv2[cTileRootName].append(class_all[nClasses[1]-1])
		if len(nClasses) >2:
			iv3[cTileRootName].append(class_all[nClasses[2]-1])

	for slide in idx.keys(): 
		req_yLength = max(FLAGS.tiles_size, max(idx[slide][0])+1 )
		req_xLength = max(FLAGS.tiles_size, max(idx[slide][1])+1 )
		X1 = np.zeros([req_yLength,req_xLength])
		print("idx, iv1:")
		print(idx[slide])
		print(len(idx[slide][0]), len(idx[slide][1]), len(iv1[slide]) )
		print(req_yLength, req_xLength )

		X1[idx[slide]] = iv1[slide]
		X2 = np.zeros([req_yLength,req_xLength])
		X3 = np.zeros([req_yLength,req_xLength])
		if len(nClasses) >1:
			X2[idx[slide]] = iv2[slide]
		if len(nClasses) >2:
			X3[idx[slide]] = iv3[slide]
		rgb = np.zeros((req_yLength, req_xLength, 3), dtype=np.uint8)
		rgb[..., 0] = X1 * 255
		rgb[..., 1] = X2 * 255
		rgb[..., 2] = X3 * 255
		filename = os.path.join(FLAGS.output_dir,"CMap_" + slide + ".jpg")
		imsave(filename, rgb)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  '''
  parser.add_argument(
      '--tiles_overlap',
      type=int,
      default=0,
      help='Overlap of the tiles in pixels.'
  )
  '''
  parser.add_argument(
      '--tiles_size',
      type=int,
      default=-1,
      help='tile size in pixels (resulting image padded to this size if smaller).'
  )
  
  parser.add_argument(
      '--output_dir',
      type=str,
      default='mustbedefined',
      help='Output directory.'
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
      default='',
      help='if map is a mutation, apply cmap of mutations only if tiles are LUAD (give here the out_filename_Stats.txt).'
  )
  parser.add_argument(
      '--Classes',
      type=str,
      default=None,
      help='Which classes to use  for each channel (up to 3; first class is 1, not 0) - string, for example: 2,1,4'
  )

  FLAGS, unparsed = parser.parse_known_args()
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

