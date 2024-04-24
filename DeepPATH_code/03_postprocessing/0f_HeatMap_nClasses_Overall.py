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
import cv2
import csv

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
from xml_writer import *

FLAGS = None
TYPE_CONTOUR =  0 
TYPE_BOX =      1
TYPE_ELLIPSE =  2 
TYPE_ARROW =    3 


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
	NotaClass = []
	#print("*** Enter get_inference_from_file")
	#print(lineProb)
	if FLAGS.combine is not '':
		#print("combine probabilities")
		#print(lineProb)
		classesIDstr = FLAGS.combine.split(',')
		classesID = [int(x) for x in classesIDstr]
		classesID = sorted(classesID, reverse = False)
		NotaClass = classesID
		for nCl in classesID[1:]:
			lineProb[classesID[0]] = lineProb[classesID[0]] + lineProb[nCl]
		classesID = sorted(classesID, reverse = True)
		for nCl in classesID[:-1]:
			# lineProb.pop(nCl)
			lineProb[nCl] = 0
		#print(lineProb)
	else:
		classesID = []
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
		# Apply correction if some classes are merged
		#print("class adjustment:")
		#print(oClass)
		#print(score_correction)
		#oClass = oClass + sum([oClass >= x for x in classesID[1:]])
		print(oClass)
		if FLAGS.project == '00_Adjacency':
			# compute adj
			if oClass == 1:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('red')])
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('orange')])
			elif oClass == 3:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('yellow')])
			elif oClass == 4:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('green')])
			elif oClass ==5:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('blue')])
			else:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('black')])
		elif FLAGS.project == '01_METbrain':
			if True:
				if oClass == 1:
					# cmap = plt.get_cmap('binary')
					c = mcolors.ColorConverter().to_rgb
					cmap = make_colormap([c('white'), c('black')])
				elif oClass == 2:
					c = mcolors.ColorConverter().to_rgb
					cmap = make_colormap([c('white'), c('#FFB000')])
				elif oClass == 3:
					c = mcolors.ColorConverter().to_rgb
					cmap = make_colormap([c('white'), c('#FE6100')])
					# cmap = plt.get_cmap('Blues')
				elif oClass == 4:
					c = mcolors.ColorConverter().to_rgb
					cmap = make_colormap([c('white'), c('cornflowerblue')])
					# cmap = plt.get_cmap('Oranges')
					# which is #6495ED
				elif oClass ==5:
					c = mcolors.ColorConverter().to_rgb
					cmap = make_colormap([c('white'), c('#785EF0')])
					# #DC267F
					# or Candy grape fizz #785EF0
				else:
					cmap = plt.get_cmap('Greens')
		elif FLAGS.project == '02_METliver':
			if oClass == 1:
				if len(class_all) == 2:
					c = mcolors.ColorConverter().to_rgb
					cmap = make_colormap([c('white'), c('#FFB000')])
				else:
					cmap = plt.get_cmap('binary')
			elif oClass == 2:
				if len(class_all) == 2:
					c = mcolors.ColorConverter().to_rgb
					cmap = make_colormap([c('white'), c('cornflowerblue')])
				else:
					c = mcolors.ColorConverter().to_rgb
					cmap = make_colormap([c('white'), c('cornflowerblue')])
			elif oClass == 3:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('#785EF0')])
			elif oClass == 4:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('#FFB000')])
			else:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('yellow')])
		elif FLAGS.project == '05_binary':
			if oClass == 1:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('yellow')])
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('darkviolet')])
		elif FLAGS.project == '06_TNBC_6folds':
			if oClass == 1:
				cmap = plt.get_cmap('binary')
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('red')])
			elif oClass == 3:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('orange')])
			elif oClass == 4:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('green')])
			elif oClass == 5:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('blue')])
			elif oClass == 6:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('yellow')])				
		elif FLAGS.project == '03_OSA':
			if oClass == 1:
				cmap = plt.get_cmap('binary')
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('#FFC107')])
				# green')])
			elif oClass == 3:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('#1E88E5')]) #blue')])
			elif oClass == 4:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('#D81B60')]) #red')])
				#cmap = plt.get_cmap('Oranges')
			else:
				cmap = plt.get_cmap('Purples')
		elif FLAGS.project == '04_HN':
			if oClass == 1:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('red')])
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('blue')])
			elif oClass == 3:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('black')])
			elif oClass == 4:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('green')])
			else:
				cmap = plt.get_cmap('Purples')
		elif  FLAGS.project == '07_Melanoma_Johannet':
			if oClass == 1:
				cmap = plt.get_cmap('binary')
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('blue')])
			elif oClass == 3:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('orange')])
		elif  FLAGS.project == '08_Melanoma_binary':
			if oClass == 1:
				c = mcolors.ColorConverter().to_rgb				
				cmap = make_colormap([c('white'), c('yellow')])
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('darkviolet')])
			else:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('darkviolet')])
		elif  FLAGS.project == '09_OSA_Surv':
			if oClass == 1:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('#98E6F2')])
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('#D866AB')])
			else:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('black')])
		elif FLAGS.project == '10_Melanoma_3Classes':
			if oClass == 1:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('red')])
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('yellow')])
			elif oClass == 3:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('darkviolet')])
			else:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('darkviolet')])
		elif FLAGS.project == '11_Melanoma_4Classes':
			if oClass == 1:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('red')])
			elif oClass == 2:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('yellow')])
			elif oClass == 3:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('#40E0D0')])
			elif oClass == 4:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('darkviolet')])
			else:
				c = mcolors.ColorConverter().to_rgb
				cmap = make_colormap([c('white'), c('darkviolet')])


				


	# print(lineProb)
	# print(oClass, current_score, (current_score-score_correction)/(1.0-score_correction),  [class_all[1], class_all[2], class_all[3]])
	class_allC = [class_all[k] for k in range(len(class_all))]
	#class_allC = [(class_all[k]-score_correction)/(1.0-score_correction) for k in range(len(class_all))]
	return oClass, cmap, (current_score-score_correction)/(1.0-score_correction), class_allC



def Get_Binary_stats2(bin_im):
	t, b_tmp= cv2.threshold(np.uint8(bin_im)*255,128,255,cv2.THRESH_BINARY)
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b_tmp , 4 , cv2.CV_16S)
	Each_Tumor_Area = []
	Each_Tumor_Mean_Dia = []
	nIt = 0
	ImBin1 = np.ascontiguousarray(bin_im)
	contoursC = []
	for i in range(0,labels.max()+1):
		if stats[i,0] > 0:
			mask = cv2.compare(labels,i,cv2.CMP_EQ)
			_,contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			Each_Tumor_Area.append(stats[i,4]* FLAGS.resample_factor * FLAGS.resample_factor)
			eachT = contours[0]
			if len(eachT) >= 5:
				ellipse = cv2.fitEllipse(eachT)
				MinAx = min(ellipse[1]) * FLAGS.resample_factor
				MaxAx = max(ellipse[1]) * FLAGS.resample_factor
				Each_Tumor_Mean_Dia.append( (MinAx + MaxAx) / 2 )
                        else:
				Each_Tumor_Mean_Dia.append(np.sqrt( Each_Tumor_Area[-1] / np.pi) )
			M = cv2.moments(eachT)
			if M["m00"] != 0:
				cx= int(M['m10']/M['m00'])
				cy= int(M['m01']/M['m00'])
			else:
				cx = 0
				cy = 0
			ImBin1 = cv2.putText(ImBin1, text = str(nIt), org=(cx, cy),  fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,211,25), thickness=4, lineType=cv2.LINE_AA)
			for eachT in contours:
				contoursC.append(np.array([nn[0] for nn in eachT]))

	Nb_Tumor = len(Each_Tumor_Area)
	fields = ['Nb_tumors', 'Nb_tumors_500px_Dia_or_more', 'Nb_tumors_1000px_Dia_or_more', 'Nb_tumors_2000px_Dia_or_more', 'Nb_tumors_3000px_Dia_or_more', 'Nb_tumors_4000px_Dia_or_more', 'Nb_tumors_5000px_Dia_or_more', 'List_of_tumor_diameter', 'List_of_tumor_areas']
	rows = [str(Nb_Tumor), str((np.asarray(Each_Tumor_Mean_Dia) > 500).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 1000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 2000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 3000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 4000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 5000).sum()), str(Each_Tumor_Mean_Dia), str(Each_Tumor_Area)]

	contoursC = np.array(contoursC)

	return fields, rows, sum(Each_Tumor_Area), contoursC
				
			

def Get_Binary_stats(bin_im):
	#print(bin_im)
	#imsave('bin_im.jpeg',bin_im)
	#print(np.array(np.uint8(bin_im)))
	#imsave('bin_imA8.jpeg',np.uint8(bin_im))
	# t, b_tmp= cv2.threshold(np.array(np.uint8(bin_im)),1,255,cv2.THRESH_BINARY_INV)
	t, b_tmp= cv2.threshold(np.uint8(bin_im)*255,128,255,cv2.THRESH_BINARY)
	#imsave('b_tmp.jpeg',b_tmp)
	a, contours,b = cv2.findContours(b_tmp,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	Each_Tumor_Area = []
	Each_Tumor_Mean_Dia = []
	nIt = 0
	ImBin1 = np.ascontiguousarray(bin_im)
	for eachT in contours:
		nIt += 1
		if len(eachT>2):
			Each_Tumor_Area.append(cv2.contourArea(eachT) * FLAGS.resample_factor * FLAGS.resample_factor)
		else:
			Each_Tumor_Area.append( len(eachT) * FLAGS.resample_factor * FLAGS.resample_factor)
		if len(eachT) >= 5:
			ellipse = cv2.fitEllipse(eachT)
			MinAx = min(ellipse[1]) * FLAGS.resample_factor
			MaxAx = max(ellipse[1]) * FLAGS.resample_factor
			Each_Tumor_Mean_Dia.append( (MinAx + MaxAx) / 2 )
		else:
			Each_Tumor_Mean_Dia.append(np.sqrt( Each_Tumor_Area[-1] / np.pi) )
		M= cv2.moments(eachT)
		if M["m00"] != 0:
			cx= int(M['m10']/M['m00'])
			cy= int(M['m01']/M['m00'])
		else:
			cx = 0
			cy = 0
		ImBin1 = cv2.putText(ImBin1, text = str(nIt), org=(cx, cy),  fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,211,25), thickness=4, lineType=cv2.LINE_AA)
	Nb_Tumor = len(Each_Tumor_Area)
	'''
	with open(filename, 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		fields = ['imageName', 'Percent_Tumor', 'Avg_Tumor_Prob', 'Nb_tumors', 'Nb_tumors_1000px_Dia_or_more', 'Nb_tumors_5000px_Dia_or_more', 'Tumor_areas', 'Tumor_avg_diam'] 
		csvwriter.writerow(fields)
		rows = [[cTileRootName, str(round(c1/(c1+c3)*100,2)), str(round(Avg_Prob_Class1*100, 2)), str(Nb_Tumor), str((np.asarray(Each_Tumor_Mean_Dia) > 1000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 5000).sum()), str(Each_Tumor_Area), str(Each_Tumor_Mean_Dia)]]
		csvwriter.writerows(rows)       
	'''


	#fields = ['Nb_tumors', 'Nb_tumors_500px_Dia_or_more', 'Nb_tumors_5000px_Dia_or_more', 'Tumor_areas', 'Tumor_avg_diam'] 
	#rows = [str(Nb_Tumor), str((np.asarray(Eiach_Tumor_Mean_Dia) > 500).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 5000).sum()), str(Each_Tumor_Area), str(Each_Tumor_Mean_Dia)]
	fields = ['Nb_tumors', 'Nb_tumors_500px_Dia_or_more', 'Nb_tumors_1000px_Dia_or_more', 'Nb_tumors_2000px_Dia_or_more', 'Nb_tumors_3000px_Dia_or_more', 'Nb_tumors_4000px_Dia_or_more', 'Nb_tumors_5000px_Dia_or_more', 'List_of_tumor_diameter', 'List_of_tumor_areas']
	rows = [str(Nb_Tumor), str((np.asarray(Each_Tumor_Mean_Dia) > 500).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 1000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 2000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 3000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 4000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 5000).sum()), str(Each_Tumor_Mean_Dia), str(Each_Tumor_Area)]

	# Convert contours to xml-compatible format
	contoursC = []
	for eachT in contours:
		contoursC.append(np.array([nn[0] for nn in eachT]))


	contoursC = np.array(contoursC)

	return fields, rows, sum(Each_Tumor_Area), contoursC

def saveMap(HeatMap_divider_p0, HeatMap_0_p, WholeSlide_0, cTileRootName, NewSlide, dir_name, HeatMap_bin_or):
	# HeatMap_0_p: heatmap coded
	# HeatMap_bin_or: sum of probabilities
	# WholeSlide_0: whole slide images

	# save the previous heat maps if any
	HeatMap_divider = HeatMap_divider_p0 * 1.0 + 0.0
	HeatMap_0 = HeatMap_0_p
	# Bkg = HeatMap_divider + 0.0
	HeatMap_divider[HeatMap_divider == 0] = 1.0
	HeatMap_0 = np.divide(HeatMap_0, HeatMap_divider[:,:,0:3])
	alpha = 0.33
	out = HeatMap_0 * 255 * (1.0 - alpha) + WholeSlide_0 * alpha
	# # #out = HeatMap_0 * 255 
	out = out.transpose((1, 0, 2))
	heatmap_path = os.path.join(FLAGS.output_dir,'heatmaps')
	# print(heatmap_path)
	if os.path.isdir(heatmap_path):	
		pass
	else:
		os.makedirs(heatmap_path)
	
	filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + ".jpg")
	# filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + dir_name + ".jpg")
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
		HeatMap_bin = np.divide(HeatMap_bin_or, HeatMap_divider) 
		ImBin = HeatMap_bin * 0.
		# if FLAGS.thresholds is not None:
		# NonBkgTiles_c1 = HeatMap_0[HeatMap_divider[:,:,1]>0,0]
		# NonBkgTiles_c2 = HeatMap_0[HeatMap_divider[:,:,1]>0,1]
		# NonBkgTiles_c3 = HeatMap_0[HeatMap_divider[:,:,1]>0,2]
		if FLAGS.thresholds is not None:
			thresholds = FLAGS.thresholds
			thresholds = [float(x) for x in thresholds.split(',')]
			ImBinT = ImBin
			for kT in range(len(thresholds)):
				ImBinT[:,:,kT] = (HeatMap_bin[:,:,kT] - thresholds[kT]) / (1 - thresholds[kT])
				# ImBinT[:,:,1] = (HeatMap_bin[:,:,1] - thresholds[1]) / (1 - thresholds[1])
				# ImBinT[:,:,2] = (HeatMap_bin[:,:,2] - thresholds[2]) / (1 - thresholds[2])
				# ImBinT[:,:,3] = (HeatMap_bin[:,:,3] - thresholds[3]) / (1 - thresholds[3])
				# ImBinT[:,:,4] = (HeatMap_bin[:,:,4] - thresholds[4]) / (1 - thresholds[4])
			Tmax = np.max(ImBinT,2)
			for kT in range(len(thresholds)):
				ImBin[:,:,kT] = ImBinT[:,:,kT] == Tmax
				# ImBin[:,:,1] = ImBinT[:,:,1] == Tmax
				# ImBin[:,:,2] = ImBinT[:,:,2] == Tmax
				# ImBin[:,:,3] = ImBinT[:,:,3] == Tmax
				# ImBin[:,:,4] = ImBinT[:,:,4] == Tmax

		else:
			Tmax = np.max(HeatMap_bin,2)
			ImBin[:,:,0] = HeatMap_bin[:,:,0] == Tmax
			ImBin[:,:,1] = HeatMap_bin[:,:,1] == Tmax
			ImBin[:,:,2] = HeatMap_bin[:,:,2] == Tmax
			ImBin[:,:,3] = HeatMap_bin[:,:,3] == Tmax
			ImBin[:,:,4] = HeatMap_bin[:,:,4] == Tmax
			ImBin[:,:,5] = HeatMap_bin[:,:,5] == Tmax

		if FLAGS.project == '00_Adjacency':
			class_rgb = {}
			class_rgb[0] = [1., 0., 0.]
			class_rgb[1] = [1., 0.84, 0.]
			class_rgb[2] = [1., 1., 0.]
			class_rgb[3] = [0, 1., 0.]
			class_rgb[4] = [0., 0., 1.]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '01_METbrain':
			class_rgb = {}
			class_rgb[0] = [0, 0, 0]
			class_rgb[1] = [255.0/255.0, 176.0/255.0, 0]
			class_rgb[2] = [254.0/255.0, 97.0/255.0, 0]
			class_rgb[3] = [100.0/255.0, 143.0/255.0, 1.0]
			# class_rgb[4] = [220.0/255.0, 38.0/255.0, 127.0/255.0]
			class_rgb[4] = [120.0/255.0, 94.0/255.0, 240.0/255.0]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '02_METliver':
			class_rgb = {}
			# class_rgb[0] = [1.0, 0, 0]
			# class_rgb[1] = [0, 0.0, 1.0]
			class_rgb[0] = [0.0, 0, 0]
			class_rgb[1] = [100.0/255.0, 143.0/255.0, 1.0]
			class_rgb[2] = [120.0/255.0, 94.0/255.0, 240.0/255.0]
			#class_rgb[2] = [220.0/255.0, 38.0/255.0, 127.0/255.0]
			class_rgb[3] = [255.0/255.0, 176.0/255.0, 0]
			class_rgb[4] = [1, 1, 1]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '03_OSA':
			class_rgb = {}
			class_rgb[0] = [0, 0, 0]
			#class_rgb[1] = [0, 1.0, 0]
			class_rgb[1] = [1.0, 193.0/255.0, 7.0/255.0]
			#class_rgb[2] = [0, 0.0, 1.0]
			class_rgb[2] = [30.0/255.0, 136.0/255.0, 229.0/255.0]
			#class_rgb[3] = [1.0, 0, 0.0]
			class_rgb[3] = [216.0/255.0, 27.0/255.0, 96.0/255.0]
			class_rgb[4] = [1, 1, 1]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '04_HN':
			class_rgb = {}
			class_rgb[0] = [1, 0, 0]
			class_rgb[1] = [0, 0.0, 1.0]
			class_rgb[2] = [1.0, 1.0, 1.0]
			class_rgb[3] = [0.0, 1.0, 0]
			class_rgb[4] = [186.0/255.0, 85.0/255.0, 211.0/255.0]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '05_binary':
			class_rgb = {}
			class_rgb[0] = [0, 1.0, 1.0]
			class_rgb[1] = [0.41, 0.0, 0.59]
			class_rgb[2] = [0, 0, 0]
			class_rgb[3] = [0, 0, 0]
			class_rgb[4] = [0, 0, 0]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '06_TNBC_6folds':
			class_rgb = {}
			class_rgb[0] = [0, 0, 0]
			class_rgb[1] = [1, 0, 0]
			class_rgb[2] = [1.0, 165.0/255.0, 0]
			class_rgb[3] = [0.0, 1.0, 0]
			class_rgb[4] = [0, 0.0, 1.0]
			class_rgb[5] = [1, 1, 0]
		elif FLAGS.project == '07_Melanoma_Johannet':
			class_rgb = {}
			class_rgb[0] = [0, 0, 0]
			class_rgb[1] = [0, 0, 1.0]
			class_rgb[2] = [255.0/255.0, 215.0/255.0, 0]
			class_rgb[3] = [0, 0, 0]
			class_rgb[4] = [0, 0, 0]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '08_Melanoma_binary':
			class_rgb = {}
			#class_rgb[0] = [0, 1.0, 1.0]
			#class_rgb[1] = [0.41, 0.0, 0.59]
			class_rgb[0] = [0.98, 1.0, 0.3125] #[0.41, 0.0, 0.59]
			class_rgb[1] = [0.41, 0.0, 0.59] #[0.98, 1.0, 0.3125]
			class_rgb[2] = [0, 0, 0]
			class_rgb[3] = [0, 0, 0]
			class_rgb[4] = [0, 0, 0]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '09_OSA_Surv':
			class_rgb = {}
			class_rgb[0] = [0.59,0.90,0.95]
			class_rgb[1] = [0.85,0.4,0.67]
			class_rgb[2] = [0, 0, 0]
			class_rgb[3] = [0, 0, 0]
			class_rgb[4] = [0, 0, 0]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '10_Melanoma_3Classes':
			class_rgb = {}
			class_rgb[0] = [1.0, .0, 0]
			class_rgb[1] = [0.98, 1.0, 0.3125]
			class_rgb[2] = [0.41, 0.0, 0.59]
			class_rgb[3] = [0, 0, 0]
			class_rgb[4] = [0, 0, 0]
			class_rgb[5] = [0, 0, 0]
		elif FLAGS.project == '11_Melanoma_4Classes':
			class_rgb = {}
			class_rgb[0] = [1.0, .0, 0]
			class_rgb[1] = [0.98, 1.0, 0.3125]
			class_rgb[2] = [64.0/255.0, 224.0/255.0, 208.0/255.0]
			class_rgb[3] = [0.41, 0.0, 0.59]
			class_rgb[4] = [0, 0, 0]
			class_rgb[5] = [0, 0, 0]


		#for kk in [0, 1, 2, 3]:
		#	tmp = HeatMap_bin[:,:,0:3] * 0 
		#	tmp[:,:,0]  = HeatMap_bin[:,:,kk]
		#	tmp[:,:,1]  = HeatMap_bin[:,:,kk]
		#	tmp[:,:,2]  = HeatMap_bin[:,:,kk]
		#	filename = os.path.join(heatmap_path,"classes_heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + str(kk) + ".jpg")
		#	imsave(filename,tmp * 255.)

		cl0 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,0])
		cl1 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,1])
		cl2 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,2])
		cl3 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,3]) 
		cl4 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,4])
		cl5 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,5])


		# HeatMap_divider_p = np.zeros([ImBin.shape[0],ImBin.shape[1], 3])
		# HeatMap_divider_p[:,:,0] = HeatMap_divider_p0[:,:,1]
		# HeatMap_divider_p[:,:,1] = HeatMap_divider_p0[:,:,2]
		# HeatMap_divider_p[:,:,2] = HeatMap_divider_p0[:,:,3]
		
		ImBinf = np.zeros([ImBin.shape[0],ImBin.shape[1], 3])
		for rgb in [0,1,2]:
			ImBinf[:,:,rgb] = ImBin[:,:,0] * class_rgb[0][rgb] + ImBin[:,:,1] * class_rgb[1][rgb] + ImBin[:,:,2] * class_rgb[2][rgb] + ImBin[:,:,3] * class_rgb[3][rgb] + ImBin[:,:,4] * class_rgb[4][rgb]  +  ImBin[:,:,5] * class_rgb[5][rgb] 

		# ImBinf[HeatMap_divider_p==0] = 1
		ImBinf[HeatMap_divider_p0[:,:,0:3]==0] = 1
		ImBinf = ImBinf.transpose((1, 0, 2))
		ImBinf = ImBinf * 255.
	
		print("*************")

		filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName +  ".csv")		
		# filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + dir_name +  ".csv")
		# Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p[:,:,1] * 1.0 + 0.0)>0,0])/np.sum(HeatMap_0[:,:,1]>0.0)
		# NbPixels_Class1 = int(c1) * FLAGS.resample_factor * FLAGS.resample_factor
		import csv
		with open(filename, 'w', newline='') as csvfile:
			csvwriter = csv.writer(csvfile)
			if FLAGS.project == '00_Adjacency':
				ClassMatrix = ImBin[:,:,0] + ImBin[:,:,1] * 2 + ImBin[:,:,2] * 3 + ImBin[:,:,3] * 4 +  ImBin[:,:,4] * 5
				AdjMatrix_UpDown = ClassMatrix[:-1,:] + ClassMatrix[1:,:] * 10
				AdjMatrix_RLeft = ClassMatrix[:,:-1] + ClassMatrix[:,1:] * 10
				# surfaces
				fields = ['imageName', 'class0_surface','class1_surface','class2_surface', 'class3_surface', 'class4_surface']
				fields_val = [ str(cl0), str(cl1), str(cl2), str(cl3), str(cl4) ]
				# Adjacency matrix
				for cl1 in range(1,5,1):
					val = cl1 + cl1 * 10
					fields.append(str(val))
					fields_val.append( sum(sum(AdjMatrix_UpDown == val)) + sum(sum(AdjMatrix_RLeft == val)) )
					for cl2 in range(cl1+1,5,1):
						val = cl1 + cl2 * 10
						valinv = cl2 + cl1 * 10
						fields.append(str(val))
						fields_val.append( sum(sum(AdjMatrix_UpDown == val)) + sum(sum(AdjMatrix_RLeft == val)) + sum(sum(AdjMatrix_UpDown == valinv)) + sum(sum(AdjMatrix_RLeft == valinv)))
				csvwriter.writerow(fields)
				rows = [cTileRootName]
				rows.extend([str(x) for x in fields_val])
				rows = [rows]
			elif FLAGS.project == '01_METbrain':
				Indx_Tumor = 1
				Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
				ImStat1 = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
				Indx_Tumor = 2
				Avg_Prob_Class2 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
				ImStat2 = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
				Avg_Prob_Class12 = Avg_Prob_Class1 + Avg_Prob_Class2
				fields2, rows2, TumorArea2, contours = Get_Binary_stats(ImStat1 + ImStat2)
				if (cl1+cl2+cl3) == 0:
					cl3p1 = 1
				else:
					cl3p1 = cl1 + cl2 + cl3


				if '2' and '3'  in FLAGS.combine.split(','):
					fields = ['imageName', 'Tumor_area','Non_tumor_area','Tumor_percentage', 'Tumor_avg_probability']
					fields.extend(fields2)
					csvwriter.writerow(fields)
					rows = [cTileRootName, str(round(cl1,0)+round(cl2,0)),str(round(cl3,1)),str(round(100*(cl1+cl2)/cl3p1,2)),str(round(Avg_Prob_Class12*100, 2))]
					rows.extend(rows2)
					rows = [rows]
				else:
					fields = ['imageName', 'Intraparaenchymal_area','Leptomeningeal_area','Non_tumor_area','Tumor_percentage', 'Tumor_avg_probability']
					fields.extend(fields2)
					csvwriter.writerow(fields)
					rows = [cTileRootName, str(round(cl1,0)),str(round(cl2,0)),str(round(cl3,1)),str(round(100*(cl1+cl2)/cl3p1,2)),str(round(Avg_Prob_Class12*100, 2))]
					rows.extend(rows2)
					rows = [rows]
			elif FLAGS.project == '02_METliver':
				Indx_Tumor = 3
				Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
				# fields2, rows2, TumorArea2, contours = Get_Binary_stats(HeatMap_bin_or[:,:,Indx_Tumor])
				ImStat = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
				fields2, rows2, TumorArea2, contours = Get_Binary_stats(ImStat)
				# ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor]
				fields = ['imageName', 'Tumor_area','Non_tumor_area','Tumor_percentage', 'Tumor_avg_probability']
				fields.extend(fields2)
				csvwriter.writerow(fields)
				if (cl3+cl1) == 0:
					cl3p1 = 1
				else:
					cl3p1 = cl3 + cl1
				rows = [cTileRootName, str(round(cl3,0)),str(round(cl1,0)), str(round(100*cl3/cl3p1,2)), str(round(Avg_Prob_Class1*100, 2))]
				rows.extend(rows2)
				rows = [rows]
			elif FLAGS.project == '03_OSA':
				fields = ['imageName', 'Necrotic tumor','Normal tissue','Viable Tumor']
				csvwriter.writerow(fields)
				rows = [[cTileRootName, str(round(cl1,1)),str(round(cl2,1)),str(round(cl3,1))]]
			elif FLAGS.project == '04_HN':
				fields = ['imageName', 'Invasive_scc','Normal epidermus','SCCIS']
				csvwriter.writerow(fields)
				rows = [[cTileRootName, str(round(cl0,1)),str(round(cl1,1)),str(round(cl3,1))]]
			elif FLAGS.project == '05_binary':
				fields = ['imageName','Normal tissue or class 1','Tumor or class2']
				csvwriter.writerow(fields)
				rows = [[cTileRootName, str(round(cl0,1)),str(round(cl1,1))]]
			elif FLAGS.project == '06_TNBC_6folds':
				fields = ['imageName','Art','DCIS','Inv','Nec','Other','Str']
				csvwriter.writerow(fields)
				rows = [[cTileRootName, str(round(cl0,1)),str(round(cl1,1)), str(round(cl2,1)), str(round(cl3,1)), str(round(cl4,1)), str(round(cl5,1))]]
			elif FLAGS.project == '07_Melanoma_Johannet':
				fields = ['imageName','Tumor','lymphocyte-rich','other']
				csvwriter.writerow(fields)
				rows = [[cTileRootName, str(round(cl2,1)),str(round(cl1,1)), str(round(cl0,1))]]
			elif FLAGS.project == '08_Melanoma_binary':
				Indx_Tumor = 1
				Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
				ImStat = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
				fields2, rows2, TumorArea2, contours = Get_Binary_stats(ImStat)
				fields = ['imageName','Tumor_area','Non_tumor_area','tumor_percentage','Tumor_avg_probability']
				fields.extend(fields2)
				csvwriter.writerow(fields)
				if (cl0+cl1) == 0:
					cl0p1 = 1
				else:
					cl0p1 = cl0 + cl1
				rows = [cTileRootName, str(round(cl1,0)),str(round(cl0,0)), str(round(100*cl1/cl0p1,2)), str(round(Avg_Prob_Class1*100, 2))]
				rows.extend(rows2)
				rows = [rows]
				contour_colors = [(250, 255, 80) for x in range(len(contours))]
				#from xml_writer import *
				writer = ImageScopeXmlWriter()
				if FLAGS.PxsMag == "Mag":
					#print(contours)
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.testPixelSizeMag/FLAGS.trainPixelSizeMag for nOb in contours]
					#print(contours2)
				else:
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.trainPixelSizeMag/FLAGS.testPixelSizeMag for nOb in contours]
				writer.add_contours(contours2, contour_colors)
				writer.write(os.path.join(heatmap_path,'_'.join(cTileRootName.split('_')[1:]) + '.xml'))
			elif FLAGS.project == '09_OSA_Surv':
				Indx = 0
				Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx])/np.sum(HeatMap_0[:,:,1]>0.0)
				ImStat = np.multiply(np.array(ImBin[:,:,Indx]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
				fields2, rows2, TumorArea2, contours = Get_Binary_stats(ImStat)
				writer = ImageScopeXmlWriter()
				contour_colors = [(152, 230, 242) for x in range(len(contours))]
				if FLAGS.PxsMag == "Mag":
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.testPixelSizeMag/FLAGS.trainPixelSizeMag for nOb in contours]
				else:
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.trainPixelSizeMag/FLAGS.testPixelSizeMag for nOb in contours]
				writer.add_contours(contours2, contour_colors)

				Indx = 1
				Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx])/np.sum(HeatMap_0[:,:,1]>0.0)
				ImStat = np.multiply(np.array(ImBin[:,:,Indx]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
				fields2, rows2, TumorArea2, contours = Get_Binary_stats(ImStat)
				#contour_colors = [(216, 102, 171) for x in range(len(contours))]
				contour_colors = [(255, 0, 0) for x in range(len(contours))]
				if FLAGS.PxsMag == "Mag":
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.testPixelSizeMag/FLAGS.trainPixelSizeMag for nOb in contours]
				else:
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.trainPixelSizeMag/FLAGS.testPixelSizeMag for nOb in contours]
				writer.add_contours(contours2, contour_colors)
				writer.write(os.path.join(heatmap_path,'_'.join(cTileRootName.split('_')[1:]) + '.xml'))
				rows = []
			elif FLAGS.project == '10_Melanoma_3Classes':
				Indx_Tumor = 2
				Avg_Prob_Class2 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
				Indx_Tumor = 0
				Avg_Prob_Class0 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)	
				Indx_Tumor = 2
				ImStat = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
				fields2, rows2, TumorArea2, contours = Get_Binary_stats2(ImStat)
				fields = ['imageName','Tumor_area','Necrosis_area','Other_area','tumor_percentage','Necrosis_percentage','Tumor_avg_probability','Necrosis_avg_probability']
				fields.extend(fields2)
				csvwriter.writerow(fields)
				if (cl0+cl1+cl2) == 0:
					cl0p1 = 1
				else:
					cl0p1 = cl0 + cl1 + cl2
				rows = [cTileRootName, str(round(cl2,0)),str(round(cl0,0)),str(round(cl1,0)), str(round(100*cl2/cl0p1,2)), str(round(100*cl0/cl0p1,2)), str(round(Avg_Prob_Class2*100, 2)), str(round(Avg_Prob_Class0*100, 2))]
				rows.extend(rows2)
				rows = [rows]
				#rows.extend(rows2)
				contour_colors = [(250, 255, 80) for x in range(len(contours))]
				writer = ImageScopeXmlWriter()
				if FLAGS.PxsMag == "Mag":
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.testPixelSizeMag/FLAGS.trainPixelSizeMag for nOb in contours]
				else:
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.trainPixelSizeMag/FLAGS.testPixelSizeMag for nOb in contours]
				writer.add_contours(contours2, contour_colors)
				writer.write(os.path.join(heatmap_path,'_'.join(cTileRootName.split('_')[1:]) + '.xml'))

			elif FLAGS.project == '11_Melanoma_4Classes':
				Indx_Tumor = 3
				Avg_Prob_Class3 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
				Indx_Tumor = 0
				Avg_Prob_Class0 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)	
				Indx_Tumor = 2
				Avg_Prob_Class2 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)	

				Indx_Tumor = 3
				ImStat = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
				fields2, rows2, TumorArea2, contours = Get_Binary_stats2(ImStat)
				fields = ['imageName','Tumor_area','Necrosis_area','RXeffects_area','Other_area','tumor_percentage','Necrosis_percentage','Tumor_avg_probability','Necrosis_avg_probability','RXeffects_avg_probability']
				fields.extend(fields2)
				csvwriter.writerow(fields)
				if (cl0+cl1+cl2+cl3) == 0:
					cl0p1 = 1
				else:
					cl0p1 = cl0 + cl1 + cl2+ cl3
				rows = [cTileRootName, str(round(cl3,0)),str(round(cl0,0)),str(round(cl2,0)),str(round(cl1,0)), str(round(100*cl3/cl0p1,2)), str(round(100*cl0/cl0p1,2)), str(round(Avg_Prob_Class3*100, 2)), str(round(Avg_Prob_Class0*100, 2)), str(round(Avg_Prob_Class2*100, 2))]
				rows.extend(rows2)
				rows = [rows]
				#rows.extend(rows2)
				contour_colors = [(250, 255, 80) for x in range(len(contours))]
				writer = ImageScopeXmlWriter()
				if FLAGS.PxsMag == "Mag":
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.testPixelSizeMag/FLAGS.trainPixelSizeMag for nOb in contours]
				else:
					contours2 = [(nOb * FLAGS.resample_factor - FLAGS.tiles_overlap/2) * FLAGS.trainPixelSizeMag/FLAGS.testPixelSizeMag for nOb in contours]
				writer.add_contours(contours2, contour_colors)
				writer.write(os.path.join(heatmap_path,'_'.join(cTileRootName.split('_')[1:]) + '.xml'))


			csvwriter.writerows(rows)       

		# filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + dir_name + "_bin_" + str(int(cgreen)) + "_" + str(int(cblue)) + "_" + str(int(cred)) + ".jpg")
		# filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_bin_" + str(int(cred)) + "_" + str(int(cgreen)) + "_" + str(int(cblue)) + ".jpg")
		filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_segmented.jpg")
		imsave(filename,ImBinf * 255.)


		# filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + dir_name + "_BW.jpg")
		# imsave(filename,b_tmp * 255.)

		# remove or blannk image
		filename_tmp = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + "unknown"  + ".jpg")
		print(filename_tmp)
		if os.path.exists(filename_tmp):
			os.remove(filename_tmp)
		filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_slide.jpg")
		WholeSlide_0[HeatMap_divider_p0[:,:,0:3]==0] = 255
		imsave(filename,np.swapaxes(WholeSlide_0,0,1))
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
	if os.path.isdir(image_dir):
		for item in os.listdir(image_dir):
    			if os.path.isdir(os.path.join(image_dir, item)):
        			sub_dirs.append(os.path.join(image_dir,item))
	else:
		# If there are different subsets and the path ends with "/*"
		for folder in glob.glob(image_dir):
			for item in os.listdir(folder):
				if os.path.isdir(os.path.join(folder, item)):
					sub_dirs.append(os.path.join(folder,item))

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
	# print(filtered_dict)

	## Aggregate the results and build heatmaps
	dir_name = 'unknown'
	# For each image in the out_filename_stats:
	for slide in sorted(filtered_dict.keys()):
		NewSlide = True
		t = time.time()
		ixTile = int(stats_dict[slide]['xMax'])
		iyTile = int(stats_dict[slide]['yMax'])
		if FLAGS.project == '00_Adjacency':
			FLAGS.tiles_size = 1
			FLAGS.tiles_overlap = 0
			FLAGS.resample_factor = 0
		req_xLength =  (ixTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap) + FLAGS.tiles_size
		req_yLength =  (iyTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap) + FLAGS.tiles_size
		# req_xLength =  (ixTile) * (FLAGS.tiles_size) + FLAGS.tiles_size
		# req_yLength =  (iyTile) * (FLAGS.tiles_size) + FLAGS.tiles_size
		if FLAGS.resample_factor > 0:
			req_xLength = int(req_xLength / FLAGS.resample_factor + 1)
			req_yLength = int(req_yLength / FLAGS.resample_factor + 1)
		WholeSlide_0 = np.zeros([req_xLength, req_yLength, 3])
		HeatMap_0 = np.zeros([req_xLength, req_yLength, 3])
		HeatMap_bin = np.zeros([req_xLength, req_yLength, 6])
		HeatMap_divider = np.zeros([req_xLength, req_yLength, 6])
		print("Checking slide " + slide)
		print(req_xLength, req_yLength)
		skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, slide, NewSlide, dir_name, HeatMap_bin)
		if skip:
			print("slide done --")
			continue
		cc = 0
		# print(stats_dict[slide]['tiles'].keys())
		for tile in stats_dict[slide]['tiles'].keys():
			# print(tile)
			cc+=1
			#if cc > 100:
			#	break		
			extensions = ['.jpeg', '.jpg']
			isError = True
			dir_name = 'unknownTMP'
			if FLAGS.project == '00_Adjacency':
				im2 = np.zeros([1, 1, 3])
				isError = False
				dir_name_old = dir_name
				dir_name = 'Adj'
			else:
				for extension in extensions:
					for sub_dir in list(sub_dirs):
						# print(extension, sub_dir)
						try:
							test_filename = os.path.join(sub_dir, tile + extension)
							im2 = imread(test_filename)
							dir_name_old = dir_name
							dir_name = os.path.basename(sub_dir)
							isError = False
						except:
							isError = True
						if isError == False:
							break
					if isError == False:
						break
			if isError == True:
				print("image not found:" + tile)
				continue
			# print("here 1")
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
			# print(rTile, cTile)					
			if rTile!= cTile:
                                continue
			if FLAGS.resample_factor > 0:
				# print("old / new r&cTile")
				# print(rTile, cTile, xTile, yTile)
				rTile = int(round(float(rTile) / FLAGS.resample_factor, 0))
				cTile = int(round(float(cTile) / FLAGS.resample_factor, 0))
				if rTile<=0:
					im2s = im2
				elif cTile<=0:
					im2s = im2
				else:
					im2s = np.array(Image.fromarray(im2).resize((cTile, rTile)))
					rTile = im2s.shape[1]
					cTile = im2s.shape[0]
					xTile = int(round(float(xTile) / FLAGS.resample_factor, 0))
					yTile = int(round(float(yTile) / FLAGS.resample_factor, 0))
					req_xLength = xTile + rTile
					req_yLength = yTile + cTile
					# print(rTile, cTile, xTile, yTile,req_xLength, req_yLength)
			else:
				im2s = im2
			# Check score associated with that image:
			# print("here 2")
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
				for kC in range(len(class_prob)):
					HeatMap_bin[xTile:req_xLength, yTile:req_yLength,kC] = HeatMap_bin[xTile:req_xLength, yTile:req_yLength,kC] + np.ones([req_xLength-xTile,req_yLength-yTile]) * class_prob[kC]
			if cc % 1000 == 0: 
				print("tile time (sec): " + str((time.time() - t) / cc))
			# print("here 3")
		NewSlide = False
		skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, slide, NewSlide, dir_name, HeatMap_bin)
		print("slide time (min): " + str((time.time() - t)/60))	





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
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
      type=float,
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
  parser.add_argument(
      '--project',
      type=str,
      default='01_METbrain',
      help='Project name (will define the number of classes and colors assigned). Can be: 00_Adjacency, 01_METbrain, 02_METliver, 03_OSA, 04_HN, 05_binary, 06_TNBC_6folds, 07_Melanoma_Johannet, 08_Melanoma_binary, 09_OSA_Surv, 10_Melanoma_3Classes', '11_Melanoma_4Classes'
  )
  parser.add_argument(
      '--combine',
      type=str,
      default='',
      help='combine classes (sum of the probabilities); comma separated string (2,3). Class ID starts at 1'
  )
  parser.add_argument(
      '--trainPixelSizeMag',
      type=float,
      default=20.0,
      help='Pixelsize of training set'
  )
  parser.add_argument(
      '--testPixelSizeMag',
      type=float,
      default=0.5,
      help='Pixelsize of test set - will be used to scale the xml file'
  )
  parser.add_argument(
      '--PxsMag',
      type=str,
      default='Mag',
      help='set to Mag or PixelSize depending on what output you want.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  FLAGS.tiles_size = FLAGS.tiles_size + 2 * FLAGS.tiles_overlap
  FLAGS.tiles_overlap = 2 * FLAGS.tiles_overlap
  if FLAGS.testPixelSizeMag == 0:
    FLAGS.testPixelSizeMag = FLAGS.trainPixelSizeMag
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

