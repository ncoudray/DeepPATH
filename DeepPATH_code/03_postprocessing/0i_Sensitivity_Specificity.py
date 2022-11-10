# Extract statistics from out_filename_stat

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

def main(args): 
	# files_stats = "test_fold0_bal/test_130000k/out_filename_Stats.txt" 
	# nthreshold = [0.23, 0.77]
	files_stats = args.files_stats
	PatientID = args.PatientID
	nthreshold = [float(x) for x in args.threshold.split(',')]
	
	stats_dict = {}
	with open(files_stats) as f:
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
                                lineProb = [float(x) for x in lineProb]
                                if args.combine is not '':
                                	#print("***")
                                	#print(lineProb)
                                	classesIDstr = args.combine.split(',')
                                	classesID = [int(x) for x in classesIDstr]
                                	classesID = sorted(classesID, reverse = False)
                                	for nCl in classesID[1:]:
                                		lineProb[classesID[0]] = lineProb[classesID[0]] + lineProb[nCl]
                                	classesID = sorted(classesID, reverse = True)
                                	for nCl in classesID[:-1]:
                                		lineProb.pop(nCl)
                                	#print(lineProb)
                                stats_dict[cTileRootName]['tiles'][tilename] = [str(ixTile), str(iyTile), lineProb, lineProb.index(max(lineProb)), int(line2[-1])]

#balanced_accuracy_score(y_true = , y_pred= )


	y_true_perTil = {}
	y_pred_perTil = {}
	y_pred_perSli = {}
	y_true_perSli = {}
	y_pred_perPat = {}
	y_true_perPat = {}
	t_y_true_perTil = {}
	t_y_pred_perTil = {}
	t_y_pred_perPat = {}
	t_y_true_perPat = {}
	t_y_pred_perSli = {}
	t_y_true_perSli = {}
	for kk in range(len(lineProb)):
		y_true_perTil[kk] = []
		y_pred_perTil[kk] = []
		y_pred_perPat[kk] = []
		y_true_perPat[kk] = []
		y_pred_perSli[kk] = []
		y_true_perSli[kk] = []
		t_y_true_perTil[kk] = []
		t_y_pred_perTil[kk] = []
		t_y_pred_perPat[kk] = []
		t_y_true_perPat[kk] = []
		t_y_pred_perSli[kk] = []
		t_y_true_perSli[kk] = []

	#TPN_matrix_Til = [[0,0],[0,0]]
	#TPN_matrix_Sli = [[0,0],[0,0]]
	#TPN_matrix_Pat = [[0,0],[0,0]]
	#t_TPN_matrix_Til = [[0,0],[0,0]]
	#t_TPN_matrix_Sli = [[0,0],[0,0]]
	#t_TPN_matrix_Pat = [[0,0],[0,0]]

	TPN_matrix_Til  = [[0 for x in range(len(lineProb)-1)] for x in range(len(lineProb)-1)]
	TPN_matrix_Sli = [[0 for x in range(len(lineProb)-1)] for x in range(len(lineProb)-1)]
	TPN_matrix_Pat = [[0 for x in range(len(lineProb)-1)] for x in range(len(lineProb)-1)]
	t_TPN_matrix_Til = [[0 for x in range(len(lineProb)-1)] for x in range(len(lineProb)-1)]
	t_TPN_matrix_Sli = [[0 for x in range(len(lineProb)-1)] for x in range(len(lineProb)-1)]
	t_TPN_matrix_Pat = [[0 for x in range(len(lineProb)-1)] for x in range(len(lineProb)-1)]

	PerPatientData = {}		
	for nslide in stats_dict.keys():
		if nslide[:PatientID] not in PerPatientData.keys():
			PerPatientData[nslide[:PatientID]] = {}
			PerPatientData[nslide[:PatientID]]['NbTiles'] = 0
			PerPatientData[nslide[:PatientID]]['Sum_Labels'] = [0 for x in range(len(lineProb))]
		sum_Labels = [0 for x in range(len(lineProb))]
		for ntile in stats_dict[nslide]['tiles'].keys():
			print("***")
			true_label = stats_dict[nslide]['tiles'][ntile][-1] - 1
			print(true_label)
			if args.combine is not '':
				true_label = true_label - (np.asarray(classesID[:-1])<=true_label).sum()
			print(true_label)

			assigned_label = stats_dict[nslide]['tiles'][ntile][-3]
			for nkk in range(len(lineProb)):
				sum_Labels[nkk] = sum_Labels[nkk] + assigned_label[nkk]
				PerPatientData[nslide[:PatientID]]['Sum_Labels'][nkk]  = PerPatientData[nslide[:PatientID]]['Sum_Labels'][nkk] + assigned_label[nkk]
			#sum_Labels[0] = sum_Labels[0] + assigned_label[0]
			#sum_Labels[1] = sum_Labels[1] + assigned_label[1]
			#sum_Labels[2] = sum_Labels[2] + assigned_label[2]
			#PerPatientData[nslide[:PatientID]]['Sum_Labels'][0] = PerPatientData[nslide[:PatientID]]['Sum_Labels'][0] + assigned_label[0]
			#PerPatientData[nslide[:PatientID]]['Sum_Labels'][1] = PerPatientData[nslide[:PatientID]]['Sum_Labels'][1] + assigned_label[1]
			#PerPatientData[nslide[:PatientID]]['Sum_Labels'][2] = PerPatientData[nslide[:PatientID]]['Sum_Labels'][2] + assigned_label[2]
			if len(nthreshold) >= 2:
				# print(assigned_label, true_label)
				NewT = []
				for nC in range(len(nthreshold)):
					NewT.append( (assigned_label[nC + 1] - nthreshold[nC]) / (1 - nthreshold[nC]) )
				# print(assigned_label, NewT)
				t_assigned_label = NewT.index(max(NewT))
				#if assigned_label[true_label + 1] >= nthreshold[true_label]:
				#	t_assigned_label = true_label
				#else:
				#	t_assigned_label = 1 - true_label
			else:
				t_assigned_label =    stats_dict[nslide]['tiles'][ntile][-2]-1

			assigned_label = assigned_label.index(max(assigned_label)) - 1
			#print(true_label, assigned_label, t_assigned_label)
			#print(TPN_matrix_Til)
			TPN_matrix_Til[true_label][assigned_label] = TPN_matrix_Til[true_label][assigned_label] + 1
			y_true_perTil[0].append( true_label )
			y_pred_perTil[0].append( assigned_label )
			#print(true_label, t_assigned_label)
			# print(t_TPN_matrix_Til)
			t_TPN_matrix_Til[true_label][t_assigned_label] = t_TPN_matrix_Til[true_label][t_assigned_label] + 1
			t_y_true_perTil[0].append( true_label )
			t_y_pred_perTil[0].append( t_assigned_label )

		Av_Slide = [x / float(len(stats_dict[nslide]['tiles'].keys())) for x in sum_Labels]
		if len(nthreshold) >= 2:
			#print(true_label)
			#print(Av_Slide)
			#print(nthreshold)
			NewT = []
			for nC in range(len(nthreshold)):
				NewT.append( (Av_Slide[nC + 1] - nthreshold[nC]) / (1 - nthreshold[nC]) )
			t_assigned_label = NewT.index(max(NewT))
			#if Av_Slide[true_label + 1] >= nthreshold[true_label]:
			#	t_assigned_label = true_label
			#else:
			#	t_assigned_label = 1 - true_label
		else:
			t_assigned_label = Av_Slide.index(max(Av_Slide))-1
		assigned_label = Av_Slide.index(max(Av_Slide)) - 1
		TPN_matrix_Sli[true_label][assigned_label] = TPN_matrix_Sli[true_label][assigned_label] + 1
		y_true_perSli[0].append( true_label )
		y_pred_perSli[0].append( assigned_label )
		t_TPN_matrix_Sli[true_label][t_assigned_label] = t_TPN_matrix_Sli[true_label][t_assigned_label] + 1
		t_y_true_perSli[0].append( true_label )
		t_y_pred_perSli[0].append( t_assigned_label )
		PerPatientData[nslide[:PatientID]]['NbTiles'] = PerPatientData[nslide[:PatientID]]['NbTiles'] + float(len(stats_dict[nslide]['tiles'].keys()))
		PerPatientData[nslide[:PatientID]]['true_label'] = true_label

	for nPat in PerPatientData:
		true_label = PerPatientData[nPat]['true_label']
		Av_Slide = [x / PerPatientData[nPat]['NbTiles'] for x in PerPatientData[nPat]['Sum_Labels']]
		if len(nthreshold) >= 2:
			NewT = []
			for nC in range(len(nthreshold)):
				NewT.append( (Av_Slide[nC + 1] - nthreshold[nC]) / (1 - nthreshold[nC]) )
			t_assigned_label = NewT.index(max(NewT))
			#if Av_Slide[true_label + 1] >= nthreshold[true_label]:
			#	t_assigned_label = true_label
			#else:
			#	t_assigned_label = 1 - true_label
		else:
			t_assigned_label = Av_Slide.index(max(Av_Slide))-1
		assigned_label = Av_Slide.index(max(Av_Slide)) - 1
		#print(Av_Slide, assigned_label, t_assigned_label, true_label)
		TPN_matrix_Pat[true_label][assigned_label] = TPN_matrix_Pat[true_label][assigned_label]  + 1
		y_true_perPat[0].append( PerPatientData[nPat]['true_label']  )
		y_pred_perPat[0].append( assigned_label )
		t_TPN_matrix_Pat[true_label][t_assigned_label] = t_TPN_matrix_Pat[true_label][t_assigned_label]  + 1
		t_y_true_perPat[0].append( PerPatientData[nPat]['true_label']  )
		t_y_pred_perPat[0].append( t_assigned_label )


		

#balanced_accuracy_score(y_true = , y_pred= )

	print("************* PER TILE *************")
	compute_stats(TPN_matrix_Til, y_true_perTil, y_pred_perTil, lineProb, stats_dict, nthreshold, t_TPN_matrix_Til, t_y_true_perTil, t_y_pred_perTil, 'out1', args.labelFile, args.outputPath, args.combine)
	print("************* PER SLIDE *************")
	compute_stats(TPN_matrix_Sli, y_true_perSli, y_pred_perSli, lineProb, stats_dict, nthreshold, t_TPN_matrix_Sli, t_y_true_perSli, t_y_pred_perSli, 'out2', args.labelFile, args.outputPath, args.combine)
	print("************* PER PATIENT  *************")
	compute_stats(TPN_matrix_Pat, y_true_perPat, y_pred_perPat, lineProb, stats_dict, nthreshold, t_TPN_matrix_Pat, t_y_true_perPat, t_y_pred_perPat, 'out3', args.labelFile, args.outputPath, args.combine)

def plot_Confusion(TPN_matrix, labelFile, save_basename, Info, nClass,  combine):
	font = {'family' : 'Times New Roman',
	'weight' : 'medium',
	'size'   : 8}
	matplotlib.rc('font', **font)
	# fig, ax = plt.subplots(figsize=(3, 6))
	fig = plt.figure(figsize=(3, 4))
	gs = fig.add_gridspec(ncols = 1, nrows =2, width_ratios=[3], height_ratios=[3,1]) 
	# gs = gridspec.GridSpec(2, 1, width_ratios=[3, 1]) 
	# ax = plt.subplot(2, 1,1)
	ax = fig.add_subplot(gs[0, 0])
	ax.matshow(TPN_matrix, cmap=plt.cm.Blues)
	for i in range(len(TPN_matrix)):
		for j in range(len(TPN_matrix[0])):
			c = TPN_matrix[j,i]
			ax.text(i, j, str(c), va='center', ha='center')
	if os.path.exists(labelFile):
		text_file = open(labelFile)
		x = text_file.read().split('\n')
		if combine is not '':
			classesIDstr = combine.split(',')
			classesID = [int(x) for x in classesIDstr]
			classesID = sorted(classesID, reverse = True)
			NewID = ''
			for nCl in classesID[:-1]:
				NewID = NewID + '_' + x[nCl-1]
				x.pop(nCl-1)
			NewID = NewID + '_' + x[classesID[-1]-1]
			x[classesID[-1]-1] = NewID
		if nClass >= 0:
			x = [x[nClass], "all_other_classes"] 
		else:
			x = x[0:len(TPN_matrix)]
		default_x_ticks = range(len(TPN_matrix))
		plt.xticks(default_x_ticks, x)
		plt.yticks(default_x_ticks, x)
		ax.set_xticklabels(x,rotation=90)
	plt.xlabel("Assigned label")
	plt.ylabel("True label")
	# ax2 = plt.subplot(2, 1,2)
	ax2 = fig.add_subplot(gs[1, 0])
	plt.axis('off')
	ax2.text(0, 0, Info)
	plt.savefig(save_basename, dpi=1000, bbox_inches='tight')     



def compute_stats(TPN_matrix, y_true, y_pred, lineProb, stats_dict, nthreshold, t_TPN_matrix, t_y_true, t_y_pred, save_basename, labelFile, outputPath, combine):
	TPN_matrix_full = np.array(TPN_matrix)
	t_TPN_matrix_full = np.array(t_TPN_matrix)
	nspecificity_avg =  0
	naccuracy_avg = 0
	nprecision_avg = 0
	nRecall_sensitivity_avg = 0
	F1score_avg = 0
	FbalAcc_avg = 0

	nspecificity_avg_norm = 0
	naccuracy_avg_norm =  0
	nprecision_avg_norm = 0
	nRecall_sensitivity_avg_norm = 0
	F1score_avg_norm = 0
	FbalAcc_avg_norm = 0
	for nCl in range(np.size(TPN_matrix_full,1)):
		TPN_matrix = np.array([[0,0],[0,0]])
		TPN_matrix[0,0] = TPN_matrix_full[nCl,nCl]
		TPN_matrix[0,1] = sum(TPN_matrix_full[nCl,:]) - TPN_matrix[0,0]
		TPN_matrix[1,0] = sum(TPN_matrix_full[:,nCl]) - TPN_matrix[0,0] 
		TPN_matrix[1,1] = sum(sum(TPN_matrix_full)) -  sum(TPN_matrix[0,:]) - TPN_matrix[1,0]
		TPN_matrix
		nspecificity = TPN_matrix[1,1] / sum(TPN_matrix[1,:])
		naccuracy = (TPN_matrix[0,0]  + TPN_matrix[1,1] ) / sum(sum(TPN_matrix))
		nprecision  = (TPN_matrix[0,0]) / sum(TPN_matrix[:,0])
		# print(y_pred[0])
		nRecall_sensitivity = TPN_matrix[0,0] / sum(TPN_matrix[0,:])
		F1score = 2 * (nprecision * nRecall_sensitivity) / (nprecision + nRecall_sensitivity)
		FbalAcc = balanced_accuracy_score(y_true[0],y_pred[0])
	
		nspecificity_avg += nspecificity
		naccuracy_avg += naccuracy
		nprecision_avg += nprecision
		nRecall_sensitivity_avg += nRecall_sensitivity
		F1score_avg  += F1score
		FbalAcc_avg += FbalAcc

		print("*** Class " + str(nCl+1) + "***")
		print("**default threshold**")
		print("specificity: " + str(round(nspecificity,4)))
		print("accuracy: " + str(round(naccuracy,4)))
		print("precision: " + str(round(nprecision,4)))
		print("recall/sensitivity: " + str(round(nRecall_sensitivity,4)))
		print("F1score: " + str(round(F1score,4)))
		print("balanced accuracy: " + str(round(FbalAcc,4)))

		nInfo = "specificity: " + str(round(nspecificity,4)) + "\n" +\
			"accuracy: " + str(round(naccuracy,4))  + "\n" +\
			"precision: " + str(round(nprecision,4)) + "\n" +\
			"recall/sensitivity: " + str(round(nRecall_sensitivity,4)) + "\n" +\
			"F1score: " + str(round(F1score,4)) + "\n" +\
			"balanced accuracy: " + str(round(FbalAcc,4))


		plot_Confusion(TPN_matrix, labelFile, os.path.join(outputPath, save_basename + "_Class" + str(nCl) + "_ConfusionMat.png"), nInfo, nCl, combine)

		TPN_matrix = np.true_divide(TPN_matrix, TPN_matrix.sum(axis=1, keepdims=True))*100
		TPN_matrix = TPN_matrix.round(decimals=2)
		plot_Confusion(TPN_matrix, labelFile, os.path.join(outputPath, save_basename + "_Class" + str(nCl) + "_ConfusionMat_percent.png"), nInfo, nCl, combine)
	

		#TPN_matrix = t_TPN_matrix
		t_TPN_matrix = np.array([[0,0],[0,0]])
		t_TPN_matrix[0,0] = t_TPN_matrix_full[nCl,nCl]
		t_TPN_matrix[0,1] = sum(t_TPN_matrix_full[nCl,:]) - t_TPN_matrix[0,0]
		t_TPN_matrix[1,0] = sum(t_TPN_matrix_full[:,nCl]) - t_TPN_matrix[0,0]
		t_TPN_matrix[1,1] = sum(sum(t_TPN_matrix_full)) -  sum(t_TPN_matrix[0,:]) - t_TPN_matrix[1,0]
		TPN_matrix = t_TPN_matrix

		y_true = t_y_true
		y_pred = t_y_pred
		# print(y_pred[0])
		TPN_matrix = np.array(TPN_matrix)
		nspecificity = TPN_matrix[1,1] / sum(TPN_matrix[1,:])
		naccuracy = (TPN_matrix[0,0]  + TPN_matrix[1,1] ) / sum(sum(TPN_matrix))
		nprecision  = (TPN_matrix[0,0]) / sum(TPN_matrix[:,0])
		nRecall_sensitivity = TPN_matrix[0,0] / sum(TPN_matrix[0,:])
		F1score = 2 * (nprecision * nRecall_sensitivity) / (nprecision + nRecall_sensitivity)
		FbalAcc = balanced_accuracy_score(y_true[0],y_pred[0])

                
		nspecificity_avg_norm += nspecificity
		naccuracy_avg_norm += naccuracy
		nprecision_avg_norm += nprecision
		nRecall_sensitivity_avg_norm += nRecall_sensitivity
		F1score_avg_norm  += F1score
		FbalAcc_avg_norm += FbalAcc

		print("**chosen threshold**")
		print("specificity: " + str(round(nspecificity,4)))
		print("accuracy: " + str(round(naccuracy,4)))
		print("precision: " + str(round(nprecision,4)))
		print("recall/sensitivity: " + str(round(nRecall_sensitivity,4)))
		print("F1score: " + str(round(F1score,4)))
		print("balanced accuracy: " + str(round(FbalAcc,4)))

		nInfo = "specificity: " + str(round(nspecificity,4)) + "\n" +\
			"accuracy: " + str(round(naccuracy,4))  + "\n" +\
			"precision: " + str(round(nprecision,4)) + "\n" +\
			"recall/sensitivity: " + str(round(nRecall_sensitivity,4)) + "\n" +\
			"F1score: " + str(round(F1score,4)) + "\n" +\
			"balanced accuracy: " + str(round(FbalAcc,4))

	
		plot_Confusion(TPN_matrix, labelFile, os.path.join(outputPath, save_basename + "_Class" + str(nCl) + "_ConfusionMat_Normalized.png"), nInfo, nCl, combine)
	
		TPN_matrix = np.true_divide(TPN_matrix, TPN_matrix.sum(axis=1, keepdims=True))*100
		TPN_matrix = TPN_matrix.round(decimals=1)
		plot_Confusion(TPN_matrix, labelFile, os.path.join(outputPath, save_basename + "_Class" + str(nCl) + "_ConfusionMat_Normalized_percent.png"), nInfo, nCl, combine)

	nspecificity_avg = nspecificity_avg / np.size(TPN_matrix_full,1)
	nspecificity_avg_norm =  nspecificity_avg_norm / np.size(TPN_matrix_full,1)
	naccuracy_avg = naccuracy_avg /  np.size(TPN_matrix_full,1)
	naccuracy_avg_norm = naccuracy_avg_norm  /  np.size(TPN_matrix_full,1)
	nprecision_avg = nprecision_avg / np.size(TPN_matrix_full,1)
	nprecision_avg_norm = nprecision_avg_norm / np.size(TPN_matrix_full,1)
	nRecall_sensitivity_avg = nRecall_sensitivity_avg  / np.size(TPN_matrix_full,1)
	nRecall_sensitivity_avg_norm =  nRecall_sensitivity_avg_norm / np.size(TPN_matrix_full,1)
	F1score_avg   =  F1score_avg /   np.size(TPN_matrix_full,1)
	F1score_avg_norm  = F1score_avg_norm /  np.size(TPN_matrix_full,1)
	FbalAcc_avg = FbalAcc_avg  / np.size(TPN_matrix_full,1)
	FbalAcc_avg_norm = FbalAcc_avg_norm /  np.size(TPN_matrix_full,1)

	nInfo = "specificity: " + str(round(nspecificity_avg,4)) + "\n" +\
		"accuracy: " + str(round(naccuracy_avg,4))  + "\n" +\
		"precision: " + str(round(nprecision_avg,4)) + "\n" +\
		"recall/sensitivity: " + str(round(nRecall_sensitivity_avg,4)) + "\n" +\
		"F1score: " + str(round(F1score_avg,4)) + "\n" +\
		"balanced accuracy: " + str(round(FbalAcc_avg,4))

	plot_Confusion(TPN_matrix_full, labelFile, os.path.join(outputPath, save_basename + "_ClassAvg_ConfusionMat.png"), nInfo, -1, combine)
	TPN_matrix_full = np.true_divide(TPN_matrix_full, TPN_matrix_full.sum(axis=1, keepdims=True))*100
	TPN_matrix_full = TPN_matrix_full.round(decimals=1)
	plot_Confusion(TPN_matrix_full, labelFile, os.path.join(outputPath, save_basename + "_ClassAvg_ConfusionMat_percent.png"), nInfo, -1, combine)



	nInfo = "specificity: " + str(round(nspecificity_avg_norm,4)) + "\n" +\
		"accuracy: " + str(round(naccuracy_avg_norm,4))  + "\n" +\
		"precision: " + str(round(nprecision_avg_norm,4)) + "\n" +\
		"recall/sensitivity: " + str(round(nRecall_sensitivity_avg_norm,4)) + "\n" +\
		"F1score: " + str(round(F1score_avg_norm,4)) + "\n" +\
		"balanced accuracy: " + str(round(FbalAcc_avg_norm,4))
	
	plot_Confusion(t_TPN_matrix_full, labelFile, os.path.join(outputPath, save_basename + "_ClassAvgNorm_ConfusionMat_Normalized.png"), nInfo, -1, combine)
	t_TPN_matrix_full = np.true_divide(t_TPN_matrix_full, t_TPN_matrix_full.sum(axis=1, keepdims=True))*100
	t_TPN_matrix_full = t_TPN_matrix_full.round(decimals=1)
	plot_Confusion(t_TPN_matrix_full, labelFile, os.path.join(outputPath, save_basename + "_ClassAvgNorm_ConfusionMat_Normalized_percent.png"), nInfo, -1, combine)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--files_stats',
      type=str,
      default='out_filename_Stats.txt',
      help="out_filename_Stats.txt"
  )
  parser.add_argument(
      '--threshold',
      type=str,
      default='0',
      help="threshold to use for the classes in out_filename_Stats.txt"
  )
  parser.add_argument(
      '--PatientID',
      type=int,
      default=12,
      help="number of characters to use for patientID"
  )
  parser.add_argument(
      '--labelFile',
      type=str,
      default='',
      help="File with label names, 1 per line"
  )
  parser.add_argument(
      '--outputPath',
      type=str,
      default='',
      help="Path to save output files"
  )
  parser.add_argument(
      '--combine',
      type=str,
      default='',
      help='combine classes (sum of the probabilities); comma separated string (2,3). Class ID starts at 1'
  )


  args = parser.parse_args()
  main(args)





# files_stats = "test_fold0_bal/test_130000k/out_filename_Stats.txt"
# nthreshold = [0.23, 0.77]

