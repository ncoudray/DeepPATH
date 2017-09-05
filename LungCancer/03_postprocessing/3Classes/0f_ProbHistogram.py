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

def get_stats_from_file(tiles_stats, ctype, filter_file):


	dict = {}
	print(ctype)
	with open(tiles_stats) as f:
		for line in f:
			line = line.replace('[','').replace(']','').split()
			basename = "_".join(".".join(line[0].split(".")[:-1]).split("_")[:-2])
			if ctype =='Lung3Classes':
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
			elif  ctype == 'Mutations':
				analyze = True
				if os.path.isfile(filter_file):
					corr = 'corrected_'
					with open(filter_file) as fstat2:
						for line2 in fstat2:
							if ".".join(line[0].split(".")[:-1]) in line2:
								ref = line2.replace('[','').replace(']','').split()
								nMax = max([float(ref[3]), float(ref[4]), float(ref[5])])
								LUAD = float(ref[4])
								#print("Found:")
								#print(line2, nMax, LUAD)
								if LUAD != nMax:
									analyze = False
								#print(analyze)
								break
				if analyze == False:
					#print("continue")
					continue

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
				#print(line)
				#print([EGFR, FAT1, KRAS, SETBP1, STK11, TP53])
				if basename in dict.keys():
					# dict[basename].append([EGFR, FAT1, FAT4, KEAP1, KRAS, LRP1B, NF1, SETBP1, STK11, TP53])
					dict[basename].append([EGFR, FAT1, KRAS, SETBP1, STK11, TP53])
				else:
					# dict[basename] = [[EGFR, FAT1, FAT4, KEAP1, KRAS, LRP1B, NF1, SETBP1, STK11, TP53]]
					dict[basename] = [[EGFR, FAT1, KRAS, SETBP1, STK11, TP53]]

	return dict 

def main(tiles_stats, output_dir, ctype, filter_file):



	dict = get_stats_from_file(tiles_stats, ctype, filter_file)
	for img in dict.keys():
		NbrOfClasses = np.array(dict[img]).shape[1]
		plt.clf()
		ymax = 0
		lineL = {}
		if ctype =='Lung3Classes':
			color = ['k', 'r', 'b']
			labels = ['Normal', 'LUAD', 'LUSC']
		elif  ctype =='Mutations':
			color = ['r', 'y', 'g', 'b', 'm', 'k']
			labels = ['EGFR', 'FAT1', 'KRAS', 'SETBP1', 'STK11', 'TP53']

		else:
			print("type of classification not recognized !!!!")
			continue

		with open(os.path.join(output_dir, img +  "_" + ctype + "_histo.txt"), "a") as myfile:
			for nCl in range(NbrOfClasses):
				y,x = np.histogram( np.array(dict[img])[:,nCl], np.arange(0,1.1,0.01) )
				ymax = max(ymax, max(y/sum(y)))
				meanV = np.mean(np.array(dict[img])[:,nCl])*100
				lineL[nCl] = plt.plot(x[:-1]*100,y/sum(y)*100, color[nCl], label="%s (%.3f)" % (labels[nCl], meanV) )
				plt.plot([meanV, meanV],[0, 100], ':'+color[nCl])

				myfile.write(labels[nCl] + "\t")
				myfile.write(" ".join(str(y/sum(y)*100).splitlines()) + "\n")

			myfile.write("x-axis\t")
			myfile.write(" ".join(str(x[:-1]*100).splitlines()) + "\n")

		#plt.legend([lineL[0], lineL[1], lineL[2]], ['Normal', 'LUAD', 'LUSC'])
		lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
		plt.xlabel('Probability')
		plt.ylabel('Percentage of tiles')
		plt.title(img)
		plt.axis([0, 100, 0, ymax*100])
		plt.xticks(np.arange(0, 105, 5.0))
		plt.savefig(os.path.join(output_dir, img +  "_" + ctype + "_histo.jpeg"), bbox_extra_artists=(lgd,), bbox_inches='tight')


		








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
  parser.add_argument(
      '--ctype',
      type=str,
      default='Lung3Classes',
      help='Lung3Classes or Mutations'
  )  
  parser.add_argument(
      '--filter_file',
      type=str,
      default='',
      help='text file to filter LUAD Mutations.'
  )
  args = parser.parse_args()
  main(args.tiles_stats, args.output_dir, args.ctype, args.filter_file)

