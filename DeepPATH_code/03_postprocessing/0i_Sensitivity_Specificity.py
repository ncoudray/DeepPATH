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


def main(args): 
	# files_stats = "test_fold0_bal/test_130000k/out_filename_Stats.txt" 
	# nthreshold = [0.23, 0.77]
	files_stats = args.files_stats
	PatientID = args.PatientID
	nthreshold = [float(x) for x in args.threshold.split(',')]
	# print(nthreshold)
	
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
                                stats_dict[cTileRootName]['tiles'][tilename] = [str(ixTile), str(iyTile), lineProb, int(line2[-1])]

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

	TPN_matrix_Til = [[0,0],[0,0]]
	TPN_matrix_Sli = [[0,0],[0,0]]
	TPN_matrix_Pat = [[0,0],[0,0]]
	t_TPN_matrix_Til = [[0,0],[0,0]]
	t_TPN_matrix_Sli = [[0,0],[0,0]]
	t_TPN_matrix_Pat = [[0,0],[0,0]]
	PerPatientData = {}		
	for nslide in stats_dict.keys():
		if nslide[:PatientID] not in PerPatientData.keys():
			PerPatientData[nslide[:PatientID]] = {}
			PerPatientData[nslide[:PatientID]]['NbTiles'] = 0
			PerPatientData[nslide[:PatientID]]['Sum_Labels'] = [0, 0, 0]
		sum_Labels = [0, 0, 0]
		for ntile in stats_dict[nslide]['tiles'].keys():
			true_label = stats_dict[nslide]['tiles'][ntile][-1] - 1 
			assigned_label = stats_dict[nslide]['tiles'][ntile][-2]
			sum_Labels[0] = sum_Labels[0] + assigned_label[0]
			sum_Labels[1] = sum_Labels[1] + assigned_label[1]
			sum_Labels[2] = sum_Labels[2] + assigned_label[2]
			PerPatientData[nslide[:PatientID]]['Sum_Labels'][0] = PerPatientData[nslide[:PatientID]]['Sum_Labels'][0] + assigned_label[0]
			PerPatientData[nslide[:PatientID]]['Sum_Labels'][1] = PerPatientData[nslide[:PatientID]]['Sum_Labels'][1] + assigned_label[1]
			PerPatientData[nslide[:PatientID]]['Sum_Labels'][2] = PerPatientData[nslide[:PatientID]]['Sum_Labels'][2] + assigned_label[2]
			if assigned_label[true_label + 1] >= nthreshold[true_label]:
				t_assigned_label = true_label
			else:
				t_assigned_label = 1 - true_label

			assigned_label = assigned_label.index(max(assigned_label)) - 1
			print(true_label, assigned_label)
			print(TPN_matrix_Til)
			TPN_matrix_Til[true_label][assigned_label] = TPN_matrix_Til[true_label][assigned_label] + 1
			y_true_perTil[0].append( true_label )
			y_pred_perTil[0].append( assigned_label )
			t_TPN_matrix_Til[true_label][t_assigned_label] = t_TPN_matrix_Til[true_label][t_assigned_label] + 1
			t_y_true_perTil[0].append( true_label )
			t_y_pred_perTil[0].append( t_assigned_label )

		Av_Slide = [x / float(len(stats_dict[nslide]['tiles'].keys())) for x in sum_Labels]
		if Av_Slide[true_label + 1] >= nthreshold[true_label]:
			t_assigned_label = true_label
		else:
			t_assigned_label = 1 - true_label
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
		if Av_Slide[true_label + 1] >= nthreshold[true_label]:
			t_assigned_label = true_label
		else:
			t_assigned_label = 1 - true_label
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
	compute_stats(TPN_matrix_Til, y_true_perTil, y_pred_perTil, lineProb, stats_dict, nthreshold, t_TPN_matrix_Til, t_y_true_perTil, t_y_pred_perTil)
	print("************* PER SLIDE *************")
	compute_stats(TPN_matrix_Sli, y_true_perSli, y_pred_perSli, lineProb, stats_dict, nthreshold, t_TPN_matrix_Sli, t_y_true_perSli, t_y_pred_perSli)
	print("************* PER PATIENT  *************")
	compute_stats(TPN_matrix_Pat, y_true_perPat, y_pred_perPat, lineProb, stats_dict, nthreshold, t_TPN_matrix_Pat, t_y_true_perPat, t_y_pred_perPat)

def compute_stats(TPN_matrix, y_true, y_pred, lineProb, stats_dict, nthreshold, t_TPN_matrix, t_y_true, t_y_pred):
	TPN_matrix = np.array(TPN_matrix)
	nspecificity = TPN_matrix[1,1] / sum(TPN_matrix[1,:])
	naccuracy = (TPN_matrix[0,0]  + TPN_matrix[1,1] ) / sum(sum(TPN_matrix))
	nprecision  = (TPN_matrix[0,0]) / sum(TPN_matrix[:,0])
	# print(y_pred[0])
	nRecall_sensitivity = TPN_matrix[0,0] / sum(TPN_matrix[0,:])
	F1score = 2 * (nprecision * nRecall_sensitivity) / (nprecision + nRecall_sensitivity)
	FbalAcc = balanced_accuracy_score(y_true[0],y_pred[0])

	print("**default threshold**")
	print("specificity: " + str(round(nspecificity,4)))
	print("accuracy: " + str(round(naccuracy,4)))
	print("precision: " + str(round(nprecision,4)))
	print("recall/sensitivity: " + str(round(nRecall_sensitivity,4)))
	print("F1score: " + str(round(F1score,4)))
	print("balanced accuracy: " + str(round(FbalAcc,4)))


	# y_true = {}
	# y_pred = {}
	# for kk in range(len(lineProb)):
	#	y_true[kk] = []
	#	y_pred[kk] = []

	# TPN_matrix = [[0,0],[0,0]]
	# for nslide in stats_dict.keys():
	#        for ntile in stats_dict[nslide]['tiles'].keys():
	#                true_label = stats_dict[nslide]['tiles'][ntile][-1] - 1 
	#                assigned_label = stats_dict[nslide]['tiles'][ntile][-2]
	#                if assigned_label[true_label + 1] >= nthreshold[true_label]:
	#                        assigned_label = true_label
	#                else:
	#                        assigned_label = 1 - true_label
	#                TPN_matrix[true_label][assigned_label] = TPN_matrix[true_label][assigned_label] + 1
	#                y_true[0].append( true_label )
	#                y_pred[0].append( assigned_label )
	

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

	print("**chosen threshold**")
	print("specificity: " + str(round(nspecificity,4)))
	print("accuracy: " + str(round(naccuracy,4)))
	print("precision: " + str(round(nprecision,4)))
	print("recall/sensitivity: " + str(round(nRecall_sensitivity,4)))
	print("F1score: " + str(round(F1score,4)))
	print("balanced accuracy: " + str(round(FbalAcc,4)))
	



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--files_stats',
      type=str,
      default='',
      help="out_filename_Stats.txt"
  )
  parser.add_argument(
      '--threshold',
      type=str,
      default='0.5,0.5',
      help="threshold to use for the classes in out_filename_Stats.txt"
  )
  parser.add_argument(
      '--PatientID',
      type=int,
      default=12,
      help="number of characters to use for patientID"
  )

  args = parser.parse_args()
  main(args)





# files_stats = "test_fold0_bal/test_130000k/out_filename_Stats.txt"
# nthreshold = [0.23, 0.77]

