"""
The MIT License (MIT)

Copyright (c) 2018, Nicolas Coudray and Aristotelis Tsirigos (NYU)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Objective: Create timelapse analysis (x vs y or y vs x) or datasets in out_filename_Stats.txt files (Embr. project)
"""
# module load  python3/intel/3.6.3 
import argparse
import os.path
import re
import sys


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.misc
from scipy.interpolate import spline
import numpy as np

# python ../../../03j_TimeLapse.py --tiles_stats out_filename_Stats.txt --output_dir /scratch/coudrn01/Embryons/validtest/00_All/test/test_10000k/plots --xy X --classIndx 1

def get_stats_from_file(tiles_stats):
	ndict = {}
	with open(tiles_stats) as f:
		for line in f:
			#line = line.replace('[','').replace(']','').split()
			basename = "_".join(".".join(line.split()[0].split(".")[:-1]).split("_")[:-2])
			X = float("_".join(".".join(line.split()[0].split(".")[:-1]).split("_")[-2:-1]))
			Y = float("_".join(".".join(line.split()[0].split(".")[:-1]).split("_")[-1:]))
			prob = line.split('[')[-1].split(']')[0].split()
			TrueLabel = line.split()[-1]
			#is_TP = line[1]
			class_all = []
			sum_class = 1 - float(prob[0])
			for nC in prob[1:]:
				class_all.append(float(nC)/sum_class)
			oClass = class_all.index(max(class_all))+1

			if basename in ndict.keys():
				ndict[basename]['TrueLabel'].append(TrueLabel)
				ndict[basename]['FoundLabel'].append(oClass)
				ndict[basename]['X'].append(X)
				ndict[basename]['Y'].append(Y)
				ndict[basename]['Prob_TrueLabel'].append(float(prob[oClass]))
				for nC in range(len(class_all)):
					ndict[basename]['Prob_'+str(nC+1)].append(class_all[nC])
			else:
				ndict[basename] = {}
				ndict[basename]['TrueLabel'] = [TrueLabel]
				ndict[basename]['FoundLabel'] = [oClass]
				ndict[basename]['X'] = [X]
				ndict[basename]['Y'] = [Y]
				ndict[basename]['Prob_TrueLabel'] = [float(class_all[oClass-1])]
				for nC in range(len(class_all)):
					ndict[basename]['Prob_'+str(nC+1)] = [class_all[nC]]

				


	return ndict, len(class_all)


def main(tiles_stats, output_dir, xy, classIndx):
	ndict, NbrOfClasses = get_stats_from_file(tiles_stats)
	# print(ndict)
	for basename in ndict.keys():
		# for each Y
		if xy=='Y':
			xAxis = ndict[basename]['Y']
			nlegend = np.unique(ndict[basename]['X'])
			ref = ndict[basename]['X']
		else:
			xAxis = ndict[basename]['X']
			nlegend = np.unique(ndict[basename]['Y'])
			ref = ndict[basename]['Y']
 			
		print(basename)
		color = ['k--', 'g--', 'y--', 'r', 'y', 'g', 'k']

		plt.clf()
		ymax = 0
		lineL = {}

		with open(os.path.join(output_dir, basename +  "_" + xy + "_plot_" + str(ndict[basename]['TrueLabel'][0]) + ".txt"), "a") as myfile:
			for ilegend, vlegend in enumerate(nlegend):
			#with open(os.path.join(output_dir, basename +  "_" + xy + "_" + str(vlegend) + "_plot.txt"), "a") as myfile:
				# find indexes for that legend
				indices = [i for i, x in enumerate(ref) if x == vlegend]
				# print(len(indices))
				# print(len(ref))
				x = [xAxis[i] for i in indices]
				if classIndx == 0:
					# prob of the expected class
					y = [ndict[basename]['Prob_TrueLabel'][i] for i in indices]
				else:
					# prob of a given class
					y = [ndict[basename]['Prob_'+str(classIndx)][i] for i in indices]

				#print(nlegend)
				# print(len(x))
				# print(len(y))
				x, xIndices = np.unique(x, return_index=True)
				y = [y[i] for i1, i in enumerate(xIndices)]


				ymax = max(ymax, max(y)/sum(y))

				lineL[ilegend] = plt.plot(x,y, color[ilegend], label="%s " % (str(vlegend)) )

				myfile.write(str(vlegend) + "\t")
				myfile.write(" ".join(str(y).splitlines()) + "\n")

			myfile.write("x-axis\t")
			myfile.write(" ".join(str(x).splitlines()) + "\n")

		#plt.legend([lineL[0], lineL[1], lineL[2]], ['Normal', 'LUAD', 'LUSC'])
		lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
		plt.xlabel('Frames or defocus')
		plt.ylabel(['Probability of class ' + str(classIndx)])
		plt.title(basename)
		plt.axis([0, max(x), 0, 1])
		plt.xticks(np.arange(0, max(x), 100))
		plt.savefig(os.path.join(output_dir, basename +  "_" + xy + "_" + str(vlegend) + "_plot_" + str(ndict[basename]['TrueLabel'][0]) + ".png"), bbox_extra_artists=(lgd,), bbox_inches='tight')




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
      help='text file where tile statistics are saved (out_filename_Stats.txt).'
  )
  parser.add_argument(
      '--xy',
      type=str,
      default='X',
      help='X to plot x vs probability for different y, Y to plot y vs probability for different x.'
  )
  parser.add_argument(
      '--classIndx',
      type=int,
      default='0',
      help='class which probability should be ploted (0 if the probability of the expected class).'
  )
  args = parser.parse_args()
  main(args.tiles_stats, args.output_dir, args.xy, args.classIndx)

