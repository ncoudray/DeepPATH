""" NYU modifications:
    Author: Nicolas Coudray
    Date created: August/2017
    Python Version: 3.5.3

	Use it with Lung images stat file:
			Generate class probability histograms for each slide
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.misc
from scipy.misc import imsave
from scipy.misc import imread
from scipy.interpolate import spline
import numpy as np

def get_stats_from_file(tiles_stats):


	dict = {}
	with open(tiles_stats) as f:
		for line in f:
			line = line.replace('[','').replace(']','').split()
			basename = "_".join(".".join(line[0].split(".")[:-1]).split("_")[:-2])
			is_TP = line[1]
			class_1 = float(line[3])
			class_2 = float(line[4])
			class_3 = float(line[5])
			sum_class = class_1 + class_2 + class_3
			class_1 = class_1 / sum_class
			class_2 = class_2 / sum_class
			class_3 = class_3 / sum_class
			current_score = max(class_1, class_2, class_3)
			if current_score == class_1:
				oClass = 1
			elif current_score == class_2:
				oClass = 2
			else:
				oClass = 3
			#print(line)
			#print(basename + " " + is_TP)
			if basename in dict.keys():
				dict[basename].append([class_1, class_2 ,class_3])
			else:
				dict[basename] = [[class_1, class_2, class_3]]

	return dict 

def main(tiles_stats, output_dir):

	dict = get_stats_from_file(tiles_stats)
	for img in dict.keys():
		NbrOfClasses = np.array(dict[img]).shape[1]
		if NbrOfClasses == 3:
			plt.clf()
			ymax = 0
			lineL = {}
			color = ['k', 'r', 'b']
			labels = ['Normal', 'LUAD', 'LUSC']
			for nCl in range(NbrOfClasses):
				y,x = np.histogram( np.array(dict[img])[:,nCl], np.arange(0,1.1,0.02) )
				ymax = max(ymax, max(y/sum(y)))
				lineL[nCl] = plt.plot(x[:-1]*100,y/sum(y)*100, color[nCl], label=labels[nCl])
				meanV = np.mean(np.array(dict[img])[:,nCl])*100
				plt.plot([meanV, meanV],[0, 100], ':'+color[nCl])

			#plt.legend([lineL[0], lineL[1], lineL[2]], ['Normal', 'LUAD', 'LUSC'])
			lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
			plt.xlabel('Probability')
			plt.ylabel('Percentage of tiles')
			plt.title(img)
			plt.axis([0, 100, 0, ymax*100])
			plt.xticks(np.arange(0, 105, 5.0))
			plt.savefig(os.path.join(output_dir, img + "_histo.jpeg"), bbox_extra_artists=(lgd,), bbox_inches='tight')

		else:
			print("file %s has an unexpected number of classes" % img)






if __name__ == '__main__':



  parser = argparse.ArgumentParser()
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
  args = parser.parse_args()
  main(args.tiles_stats, args.output_dir)


