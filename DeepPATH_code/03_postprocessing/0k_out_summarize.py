 # Summarize supervised runs

""" 
The MIT License (MIT)

Copyright (c) 2021, Nicolas Coudray (NYU)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
          
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
        
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""



import os
import numpy as np
import argparse
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import glob
from itertools import cycle

def main(args):
	unique_labels = []
	with open(args.labels_names, "r") as f:
		for line in f:
			line = line.replace('\r','\n')
			line = line.split('\n')
			for eachline in line:
				if len(eachline)>0:
					unique_labels.append(eachline)
	if args.combine is not '':
		classesIDstr = args.combine.split(',')
		classesID = [int(x) for x in classesIDstr]
		classesID = sorted(classesID, reverse = True)
		NewID = ''
		for nCl in classesID[:-1]:
			NewID = NewID + '_' + unique_labels[nCl-1]
			unique_labels.pop(nCl-1)
		NewID = NewID + '_' + unique_labels[classesID[-1]-1]
		unique_labels[classesID[-1]-1] = NewID
		Labadd = 'comb' + ''.join(args.combine.split(','))
	else:
		Labadd = ''
	input_folders = args.input_folders
	nOut = args.out
	allIn = glob.glob(input_folders)
	allIn.sort()
	print(unique_labels)
	unique_labels.append('micro')
	unique_labels.append('macro')
	print(unique_labels)

	for idxL, curLabel in enumerate(unique_labels):
		AUC = {}
		AUCavg = {}
		PR = {}
		PRavg = {}
		PRref = {}
		print(curLabel)
		for kk in allIn:
			if curLabel=='micro':
				EachIterAUC = glob.glob(kk + "/test_*/" + nOut + "_roc_data_AvPb_micro_a*")
				EachIterPR = glob.glob(kk + "/test_*/" + nOut + "_PrecRec_data_AvPb_cmicro_A*")
				uPos = 4
				ustep = 6
				ustep2 = 8
				#print(EachIterAUC)
				#print(EachIterPR)
			elif curLabel=='macro':
				EachIterAUC = glob.glob(kk + "/test_*/" + nOut + "_roc_data_AvPb_macro_a*")
				EachIterPR = [] # glob.glob(kk + "/test_*/" + nOut + "_PrecRec_data_AvPb_cmacro_A*")
				uPos = 4
				ustep = 6
				ustep2 = 8
				print(EachIterAUC)
				print(EachIterPR)
			else:
				EachIterAUC = glob.glob(kk + "/test_*/" + nOut + "_roc_data_AvPb_c" + str(idxL+1) + "a*")
				EachIterPR = glob.glob(kk + "/test_*/" + nOut + "_PrecRec_data_AvPb_c" + str(idxL+1) + "A*")
				uPos = 6
				ustep = 5
				ustep2 = 5
			AUC[kk] = {}
			AUC[kk]['value'] = []
			AUC[kk]['iter'] = []
			PR[kk] = {}
			PR[kk]['value'] = []
			PR[kk]['iter'] = []
			PR[kk]['ref'] = []
			for nIter in EachIterAUC:
				print(nIter)
				print(nIter.split("_")[-uPos])
				AUC[kk]['value'].append(float(nIter.split("_")[-uPos]))
				iterV = int(nIter.split("_")[-uPos-ustep].split('k/out')[0])
				AUC[kk]['iter'].append(iterV)
				if iterV in AUCavg.keys():
					AUCavg[iterV].append(float(nIter.split("_")[-uPos]))
				else:
					AUCavg[iterV] = {}
					AUCavg[iterV] = [float(nIter.split("_")[-uPos])]
			nIndx = sorted(range(len(AUC[kk]['iter'])),key=AUC[kk]['iter'] .__getitem__)
			list1, list2 = (list(t) for t in zip(*sorted(zip(AUC[kk]['iter'], AUC[kk]['value']))))
			AUC[kk]['iter']  = list1
			AUC[kk]['value'] = list2
			for nIter in EachIterPR:
				print(nIter)
				print(nIter.split("_")[-uPos-2])
				PR[kk]['value'].append(float(nIter.split("_")[-uPos-2]))
				iterV = int(nIter.split("_")[-uPos-2-ustep2].split('k/out')[0])
				PR[kk]['iter'].append(iterV)
				PR[kk]['ref'].append(float(nIter.split("_")[-3]))
				if iterV in PRavg.keys():
					PRavg[iterV].append(float(nIter.split("_")[-uPos-2]))
				else:
					PRavg[iterV] = {}
					PRavg[iterV] = [float(nIter.split("_")[-uPos-2])]
			nIndx = sorted(range(len(PR[kk]['iter'])),key=PR[kk]['iter'] .__getitem__)
			if curLabel=='macro':
				list1 = []
				list2 = []
			else: 
				list1, list2 = (list(t) for t in zip(*sorted(zip(PR[kk]['iter'], PR[kk]['value']))))
			PR[kk]['iter']  = list1
			PR[kk]['value'] = list2
		Xauc = []
		Yauc = []
		XYstd = []
		for mm in AUCavg.keys():
			Xauc.append(mm)
			Yauc.append(np.average(AUCavg[mm]))
			XYstd.append(np.std(AUCavg[mm], ddof=1) / np.sqrt(np.size(AUCavg[mm])))
		Xauc, Yauc, XYstd = (list(t) for t in zip(*sorted(zip(Xauc, Yauc, XYstd))))
		Xpr = []
		Ypr = []
		XYprstd = []
		for mm in PRavg.keys():
			Xpr.append(mm)
			Ypr.append(np.average(PRavg[mm]))
			XYprstd.append(np.std(PRavg[mm], ddof=1) / np.sqrt(np.size(PRavg[mm])))
		if curLabel=='macro':
			Xpr = []
			Ypr = []
			XYprstd = []
		else:
			Xpr, Ypr, XYprstd = (list(t) for t in zip(*sorted(zip(Xpr, Ypr, XYprstd))))
		fig = plt.figure(figsize=(3, 6))
		FS = 8
		ax = plt.subplot(2, 1,1)
		ax.tick_params(labelsize=FS)
		colors = cycle(["black", "black", "gray", "gray","lightgray"])
		types = cycle(["solid","dashed","dotted","dashed","solid"])
		for i, color, type, img in zip(AUC.keys(), colors, types, allIn):
    	                    print(i, color, type)
    	                    plt.plot(
    	                       AUC[i]['iter'],
    	                       AUC[i]['value'],
    	                       color=color,
    	                       label=(img),
    	                       linewidth=0.5,
    	                       linestyle=type
    	                    )
		if args.OptIter is None:
			OptIndx = np.argmax(np.array(Yauc))
		else:
			#OptIndx = np.where(np.array(Xpr) == args.OptIter)[0][0]
			OptIndx = np.where(np.array(Xauc) == args.OptIter)[0][0]

		plt.plot(
    	                       Xauc,
    	                       Yauc,
    	                       color='red',
    	                       label=("average AUC (" + str(round(Yauc[OptIndx],4)) + " at " + str(Xauc[OptIndx]) + ")"),
    	                       linewidth=1.2,
    	                       linestyle='solid'
    	                    )
		plt.plot(
                                                [Xauc[0], Xauc[-1]],
                                                [0.5, 0.5],
                                                color='blue',
                                                label=('baseline=0.5'),
                                                linewidth=1,
                                                linestyle='dotted')
		plt.xlabel("Iterations", fontsize=FS)
		plt.ylabel("AUC", fontsize=FS)
		plt.fill_between(np.array(Xauc), np.array(Yauc) - np.array(XYstd), np.array(Yauc) + np.array(XYstd),
    	             color='lightcoral', alpha=0.2)
		plt.title(curLabel)
		handles, labels = ax.get_legend_handles_labels()
		plt.ylim([-0.05, 1.05])
		ax2 = plt.subplot(2, 1,2)
		plt.axis('off')
		ax2.legend(handles=handles, labels=labels,  loc="lower right", fontsize=6)
		plt.savefig(os.path.join(Labadd + "_" + args.out + "_" + input_folders.replace("*", "_") + 'avg_roc_data_' + curLabel + '.png'), dpi=1000, bbox_inches='tight')
		fig = plt.figure(figsize=(3, 6))
		ax = plt.subplot(2, 1,1)
		ax.tick_params(labelsize=FS)
		colors = cycle(["black", "black", "gray", "gray","lightgray"])
		types = cycle(["solid","dashed","dotted","dashed","solid"])
		for i, color, type, img in zip(PR.keys(), colors, types, allIn):
    	                    print(i, color, type)
    	                    plt.plot(
    	                       PR[i]['iter'],
    	                       PR[i]['value'],
    	                       color=color,
    	                       label=(img),
    	                       linewidth=0.5,
    	                       linestyle=type
    	                    )
		if args.OptIter is None:
			OptIndx = np.argmax(np.array(Ypr))
		else:
			OptIndx = np.where(np.array(Xpr) == args.OptIter)[0][0]
		plt.plot(
    	                       Xpr,
    	                       Ypr,
    	                       color='red',
    	                       label=("average PR (" + str(round(Ypr[OptIndx],4)) + " at " + str(Xpr[OptIndx]) + ")"),
    	                       linewidth=1.2,
    	                       linestyle='solid'
    	                    )
		plt.plot(
						[Xpr[0], Xpr[-1]],
    						[PR[kk]['ref'][0], PR[kk]['ref'][0]],
    						color='blue',
    						label=('baseline='+str(round(PR[kk]['ref'][0],4))),
    						linewidth=1,
    						linestyle='dotted')
		plt.xlabel("Iterations", fontsize=FS)
		plt.ylabel("PR", fontsize=FS)
		plt.fill_between(np.array(Xpr), np.array(Ypr) - np.array(XYprstd), np.array(Ypr) + np.array(XYprstd),
    	             color='lightcoral', alpha=0.2)
		plt.title(curLabel)
		handles, labels = ax.get_legend_handles_labels()
		plt.ylim([-0.05, 1.05])
		ax2 = plt.subplot(2, 1,2)
		plt.axis('off')
		ax2.legend(handles=handles, labels=labels,  loc="lower right", fontsize=6)
		plt.savefig(os.path.join(Labadd + "_" + args.out + "_" + input_folders.replace("*", "_") + 'avg_PR_data_' + curLabel + '.png'), dpi=1000, bbox_inches='tight')
		print(curLabel)
		print(Xauc, Yauc)
		print(Xpr, Ypr)




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_folders',
      type=str,
      default='valid_fold*',
      help="folder where outputs from previous steps were saved"
  )
  parser.add_argument(
      '--out',
      type=str,
      default='out1',
      help="out1, out2 or out3 to be summarized"
  )
  parser.add_argument(
      '--labels_names',
      type=str,
      default='',
      help='Names of the possible output labels ordered as desired example, /ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/9_10mutations/label_names.txt'
  )
  parser.add_argument(
      '--OptIter',
      type=int,
      default=None,
      help='Selected iteration (max by default)'
  )
  parser.add_argument(
      '--combine',
      type=str,
      default='',
      help='combine classes (sum of the probabilities); comma separated string (2,3). Class ID starts at 1 - to be used IF classes were already combined in the input directories (only the labels will be combined here)'
  )


  args = parser.parse_args()
  main(args)









