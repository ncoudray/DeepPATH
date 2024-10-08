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



def plot_and_save(args, ToPlot, ToAvg, whatisit, Label, Labadd, input_folders, allIn, kk):
		X_toPlot = []
		Y_toPlot = []
		std_toPlot = []
		for mm in ToAvg.keys():
			X_toPlot.append(mm)
			Y_toPlot.append(np.average(ToAvg[mm]))
			std_toPlot.append(np.std(ToAvg[mm], ddof=1) / np.sqrt(np.size(ToAvg[mm])))
		X_toPlot, Y_toPlot, std_toPlot = (list(t) for t in zip(*sorted(zip(X_toPlot, Y_toPlot, std_toPlot))))

		fig = plt.figure(figsize=(3, 6))
		FS = 8
		ax = plt.subplot(2, 1,1)
		ax.tick_params(labelsize=FS)
		colors = cycle(["black", "black", "gray", "gray","lightgray"])
		types = cycle(["solid","dashed","dotted","dashed","solid"])
		for i, color, type, img in zip(ToPlot.keys(), colors, types, allIn):
    	                    print(i, color, type)
    	                    plt.plot(
    	                       ToPlot[i][args.method][:args.MaxX],
    	                       ToPlot[i]['value'][:args.MaxX],
    	                       color=color,
    	                       label=(img),
    	                       linewidth=0.5,
    	                       linestyle=type
    	                    )
		if args.OptIter is None:
			OptIndx = np.argmax(np.array(Y_toPlot[:args.MaxX]))
		else:
			OptIndx = np.where(np.array(X_toPlot[:args.MaxX]) == args.OptIter)[0][0]

		plt.plot(
    	                       X_toPlot[:args.MaxX],
    	                       Y_toPlot[:args.MaxX],
    	                       color='red',
    	                       label=("average " + whatisit + " (" + str(round(Y_toPlot[OptIndx],4)) + " at " + str(X_toPlot[OptIndx]) + ")"),
    	                       linewidth=1.2,
    	                       linestyle='solid'
    	                    )
		plt.plot(
                                                [X_toPlot[0], X_toPlot[args.MaxX-1]],
                                                [ToPlot[kk]['ref'][0], ToPlot[kk]['ref'][0]],
                                                color='blue',
                                                label=('baseline=0.5'),
                                                linewidth=1,
                                                linestyle='dotted')
		if args.method == 'iter':
			plt.xlabel("Iterations", fontsize=FS)
		else:
			plt.xlabel("Epoch", fontsize=FS)
		plt.ylabel(whatisit, fontsize=FS)
		plt.fill_between(np.array(X_toPlot[:args.MaxX]), np.array(Y_toPlot[:args.MaxX]) - np.array(std_toPlot[:args.MaxX]), np.array(Y_toPlot[:args.MaxX]) + np.array(std_toPlot)[:args.MaxX],
    	             color='lightcoral', alpha=0.2)
		plt.title(Label)
		handles, labels = ax.get_legend_handles_labels()
		plt.ylim([-0.05, 1.05])
		ax2 = plt.subplot(2, 1,2)
		plt.axis('off')
		ax2.legend(handles=handles, labels=labels,  loc="lower right", fontsize=6)
		plt.savefig(os.path.join(Labadd + "_" + args.out + "_" + input_folders.replace("*", "_") + 'avg_' + whatisit + '_data_' +  Label + '.png'), dpi=1000, bbox_inches='tight')



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
	print(allIn)
	print(unique_labels)
	unique_labels.append('micro')
	unique_labels.append('macro')
	# print(unique_labels)

	for idxL, curLabel in enumerate(unique_labels):
		AUC = {}
		AUCavg = {}
		PR = {}
		PRavg = {}
		PRref = {}

		spec = {}
		accu = {}
		prec = {}
		rese = {}
		F1sc = {}

		specavg = {}
		accuavg = {}
		precavg = {}
		reseavg = {}
		F1scavg = {}

		print(curLabel)
		for kk in allIn:
			if curLabel=='micro':
				EachIterAUC = glob.glob(kk + "/test_*/" + nOut + "_roc_data_AvPb_micro_a*")
				EachIterPR = glob.glob(kk + "/test_*/" + nOut + "_PrecRec_data_AvPb_cmicro_A*")
				EachIterspecN = glob.glob(kk + "/test_*/" + nOut + "_ClassAvg_ConfusionMat_spec_*.png")
				uPos = 4
				ustep = 6
				ustep2 = 8
				ustep3 = 6
				print(EachIterAUC)
				print(EachIterPR)
			elif curLabel=='macro':
				EachIterAUC = glob.glob(kk + "/test_*/" + nOut + "_roc_data_AvPb_macro_a*")
				EachIterPR = [] # glob.glob(kk + "/test_*/" + nOut + "_PrecRec_data_AvPb_cmacro_A*")
				EachIterspecN = glob.glob(kk + "/test_*/" + nOut + "_ClassAvgNorm_ConfusionMat_Normalized_spec_*.png")
				uPos = 4
				ustep = 6
				ustep2 = 8
				ustep3 = 7
				print(EachIterAUC)
				print(EachIterPR)
			else:
				EachIterAUC = glob.glob(kk + "/test_*/" + nOut + "_roc_data_AvPb_c" + str(idxL+1) + "a*")
				EachIterPR = glob.glob(kk + "/test_*/" + nOut + "_PrecRec_data_AvPb_c" + str(idxL+1) + "A*")
				EachIterspecN = glob.glob(kk + "/test_*/" + nOut + "_Class" + str(idxL)  + "_ConfusionMat_spec_*.png")
				uPos = 6
				ustep = 5
				ustep2 = 5
				ustep3 = 4
				print(EachIterAUC)
				print(EachIterPR)
				print(EachIterspecN)
			AUC[kk] = {}
			AUC[kk]['value'] = []
			AUC[kk]['iter'] = []
			AUC[kk]['ref'] = []
			AUC[kk]['epoch'] = []
			PR[kk] = {}
			PR[kk]['value'] = []
			PR[kk]['iter'] = []
			PR[kk]['ref'] = []
			PR[kk]['epoch'] = []
			spec[kk] = {}
			spec[kk]['value'] = []
			spec[kk]['iter'] = []
			spec[kk]['ref'] = []
			spec[kk]['epoch'] = []
			accu[kk] = {}
			accu[kk]['value'] = []
			accu[kk]['iter'] = []
			accu[kk]['ref'] = []
			accu[kk]['epoch'] = []
			prec[kk] = {}
			prec[kk]['value'] = []
			prec[kk]['iter'] = []
			prec[kk]['ref'] = []
			prec[kk]['epoch'] = []
			rese[kk] = {}
			rese[kk]['value'] = []
			rese[kk]['iter'] = []
			rese[kk]['ref'] = []
			rese[kk]['epoch'] = []
			F1sc[kk] = {}
			F1sc[kk]['value'] = []
			F1sc[kk]['iter'] = []
			F1sc[kk]['ref'] = []
			F1sc[kk]['epoch'] = []

			print("AUC loop 1")
			for nIter in EachIterAUC:
				print(nIter)
				print(nIter.split("_")[-uPos])
				AUC[kk]['value'].append(float(nIter.split("_")[-uPos]))
				iterV = int(nIter.split("_")[-uPos-ustep].split('k/out')[0])
				AUC[kk]['iter'].append(iterV)
				AUC[kk]['ref'].append(float(0.5))

			nIndx = sorted(range(len(AUC[kk]['iter'])),key=AUC[kk]['iter'] .__getitem__)
			list1, list2 = (list(t) for t in zip(*sorted(zip(AUC[kk]['iter'], AUC[kk]['value']))))
			AUC[kk]['iter']  = list1
			AUC[kk]['value'] = list2
			AUC[kk]['epoch'] = np.array(list1) / list1[0]
			for IndxV in range(len(AUC[kk][args.method])):
				nIter = AUC[kk][args.method][IndxV]
				nIterV = AUC[kk]['value'][IndxV]
				if nIter in AUCavg.keys():
					AUCavg[nIter].append(nIterV)
				else:
					AUCavg[nIter] = {}
					AUCavg[nIter] = [float(nIterV)]

			print("PR loop 1")
			for nIter in EachIterPR:
				print(nIter)
				print(nIter.split("_")[-uPos-2])
				PR[kk]['value'].append(float(nIter.split("_")[-uPos-2]))
				iterV = int(nIter.split("_")[-uPos-2-ustep2].split('k/out')[0])
				PR[kk]['iter'].append(iterV)
				PR[kk]['ref'].append(float(nIter.split("_")[-3]))

			nIndx = sorted(range(len(PR[kk]['iter'])),key=PR[kk]['iter'] .__getitem__)
			if curLabel=='macro':
				list1 = []
				list2 = []
			else: 
				list1, list2 = (list(t) for t in zip(*sorted(zip(PR[kk]['iter'], PR[kk]['value']))))
			PR[kk]['iter']  = list1
			PR[kk]['value'] = list2
			if curLabel == 'macro':
				PR[kk]['epoch'] = []
			else:
				PR[kk]['epoch'] = np.array(list1) / list1[0]
				for IndxV in range(len(PR[kk]['value'])):
					nIter = PR[kk][args.method][IndxV]
					nIterV = PR[kk]['value'][IndxV]
					if nIter in PRavg.keys():
						PRavg[nIter].append(nIterV)
					else:
						PRavg[nIter] = {}
						PRavg[nIter] = [float(nIterV)]
			print("Matrix loop 1")
			print(EachIterspecN)
			for nIter in EachIterspecN:
				print(nIter)
				print(nIter.split("_")[-9])
				spec[kk]['value'].append(float(nIter.split("_")[-9]))
				accu[kk]['value'].append(float(nIter.split("_")[-7]))
				prec[kk]['value'].append(float(nIter.split("_")[-5]))
				rese[kk]['value'].append(float(nIter.split("_")[-3]))
				F1sc[kk]['value'].append(float(nIter.split("_")[-1].split('.png')[0]))

				print(nIter.split("_")[-uPos-3-ustep3].split('k/out'))
				iterV = int(nIter.split("_")[-uPos-3-ustep3].split('k/out')[0])
				spec[kk]['iter'].append(iterV)
				spec[kk]['ref'].append(float(-1))
				accu[kk]['iter'].append(iterV)
				accu[kk]['ref'].append(float(-1))
				prec[kk]['iter'].append(iterV)
				prec[kk]['ref'].append(float(-1))
				rese[kk]['iter'].append(iterV)
				rese[kk]['ref'].append(float(-1))
				F1sc[kk]['iter'].append(iterV)
				F1sc[kk]['ref'].append(float(-1))
				print(spec)
				#	AUCavg[iterV] = [float(nIter.split("_")[-uPos])]
			nIndx = sorted(range(len(spec[kk]['iter'])),key=spec[kk]['iter'] .__getitem__)
			list1, list2 = (list(t) for t in zip(*sorted(zip(spec[kk]['iter'], spec[kk]['value']))))
			spec[kk]['iter']  = list1
			spec[kk]['value'] = list2
			spec[kk]['epoch'] = np.array(list1) / list1[0]

			nIndx = sorted(range(len(accu[kk]['iter'])),key=accu[kk]['iter'] .__getitem__)
			list1, list2 = (list(t) for t in zip(*sorted(zip(accu[kk]['iter'], accu[kk]['value']))))
			accu[kk]['iter']  = list1
			accu[kk]['value'] = list2
			accu[kk]['epoch'] = np.array(list1) / list1[0]

			nIndx = sorted(range(len(prec[kk]['iter'])),key=prec[kk]['iter'] .__getitem__)
			list1, list2 = (list(t) for t in zip(*sorted(zip(prec[kk]['iter'], prec[kk]['value']))))
			prec[kk]['iter']  = list1
			prec[kk]['value'] = list2
			prec[kk]['epoch'] = np.array(list1) / list1[0]

			nIndx = sorted(range(len(rese[kk]['iter'])),key=rese[kk]['iter'] .__getitem__)
			list1, list2 = (list(t) for t in zip(*sorted(zip(rese[kk]['iter'], rese[kk]['value']))))
			rese[kk]['iter']  = list1
			rese[kk]['value'] = list2
			rese[kk]['epoch'] = np.array(list1) / list1[0]

			nIndx = sorted(range(len(F1sc[kk]['iter'])),key=F1sc[kk]['iter'] .__getitem__)
			list1, list2 = (list(t) for t in zip(*sorted(zip(F1sc[kk]['iter'], F1sc[kk]['value']))))
			F1sc[kk]['iter']  = list1
			F1sc[kk]['value'] = list2
			F1sc[kk]['epoch'] = np.array(list1) / list1[0]


			for IndxV in range(len(spec[kk][args.method])):
				nIter = spec[kk][args.method][IndxV]
				nIterV = spec[kk]['value'][IndxV]
				if nIter in specavg.keys():
					specavg[nIter].append(nIterV)
				else:
					specavg[nIter] = {}
					specavg[nIter] = [float(nIterV)]

				nIter = accu[kk][args.method][IndxV]
				nIterV = accu[kk]['value'][IndxV]
				if nIter in accuavg.keys():
					accuavg[nIter].append(nIterV)
				else:
					accuavg[nIter] = {}
					accuavg[nIter] = [float(nIterV)]

				nIter = prec[kk][args.method][IndxV]
				nIterV = prec[kk]['value'][IndxV]
				if nIter in precavg.keys():
					precavg[nIter].append(nIterV)
				else:
					precavg[nIter] = {}
					precavg[nIter] = [float(nIterV)]

				nIter = rese[kk][args.method][IndxV]
				nIterV = rese[kk]['value'][IndxV]
				if nIter in reseavg.keys():
					reseavg[nIter].append(nIterV)
				else:
					reseavg[nIter] = {}
					reseavg[nIter] = [float(nIterV)]

				nIter = F1sc[kk][args.method][IndxV]
				nIterV = F1sc[kk]['value'][IndxV]
				if nIter in F1scavg.keys():
					F1scavg[nIter].append(nIterV)
				else:
					F1scavg[nIter] = {}
					F1scavg[nIter] = [float(nIterV)]


		plot_and_save(args, AUC, AUCavg, 'AUC', curLabel, Labadd, input_folders, allIn, kk)
		if curLabel != 'macro':
			plot_and_save(args, PR, PRavg, 'PR', curLabel, Labadd, input_folders, allIn, kk)
			#continue
		plot_and_save(args, spec, specavg, 'specificity', curLabel, Labadd, input_folders, allIn, kk)
		plot_and_save(args, accu, accuavg, 'accuracy', curLabel, Labadd, input_folders, allIn, kk)
		plot_and_save(args, prec, precavg, 'precision', curLabel, Labadd, input_folders, allIn, kk)
		plot_and_save(args, rese, reseavg, 'recallsensitivity', curLabel, Labadd, input_folders, allIn, kk)
		plot_and_save(args, F1sc, F1scavg, 'F1score', curLabel, Labadd, input_folders, allIn, kk)


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
  parser.add_argument(
      '--method',
      type=str,
      default='iter',
      help='iter (if all folds have the same iterations) or epoch (if folds have different iterations, and if each folder corresponds to 1 epoch)'
  )
  parser.add_argument(
      '--MaxX',
      type=int,
      default=100,
      help='Truncate the x axis to the maximum number of checkpoints to visualize'
  )


  args = parser.parse_args()
  main(args)









