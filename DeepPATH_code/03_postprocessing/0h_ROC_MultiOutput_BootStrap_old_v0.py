'''
	Author: Nicolas Coudray (NYU)
	Date created: 2017

	Compute ROC for multiple output-classes (several TP possible for a given input) using sklearn package
	(works on python 2.7 but not 3.5.3 on the phoenix clusters)
'''
import sys, getopt
import argparse
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp
from itertools import cycle
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import sem
import glob
FLAGS = None

def BootStrap(y_true, y_pred, isMacro,  n_classes = 1):
	#if True:
	#	return 0, 0
	# initialization by bootstraping
	n_bootstraps = 2000
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []
	print(y_true)
	print(y_pred)
	# y_true = y_true.reshape(1,-1)
	# y_pred = y_pred.reshape(1,-1)
	#print(y_true)
	#print(y_pred)
	#rng_seed = int(len(y_true) * 0.9)
	#print("rng_seed: %d" % rng_seed)
	rng = np.random.RandomState(rng_seed)
	if isMacro:
		for i in range(n_bootstraps):
			indices = rng.random_integers(0, len(y_pred[:,0]) - 1, len(y_pred[:,0]))
			if len(np.unique(y_true[indices,0])) < 2:
				# We need at least one positive and one negative sample for ROC AUC
				# to be defined: reject the sample
				#print("We need at least one positive and one negative sample for ROC AUC")
				continue
			else:
				y_true2 = dict()
				y_pred2 = dict()
				for n in range(n_classes):
					y_true2[n], y_pred2[n], _ = roc_curve(y_true[indices, n], y_pred[indices, n])
				# Then interpolate all ROC curves at this points
				all_f = np.unique(np.concatenate([y_true2[n]for n in range(n_classes)]))
				mean_tpr = np.zeros_like(all_f)
				#print(mean_tpr)
				#print(mean_tpr)
				#print(len(y_pred))
				#print(len(indices))
				#print(y_true[i])
				#print(indices)
				#print(y_true[i][indices])
				for n in range(n_classes):
					mean_tpr += interp(all_f, y_true2[n], y_pred2[n])
				mean_tpr /= n_classes
				score = auc(all_f, mean_tpr)
				bootstrapped_scores.append(score)
				#print("score: %f" % score)
	else:
		for i in range(n_bootstraps):
			# bootstrap by sampling with replacement on the prediction indices
			# indices = rng.random_integers(0, len(y_pred.reshape(1,-1) ) - 1, len(y_pred.reshape(1,-1) ))
			indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
			if len(np.unique(y_true[indices])) < 2:
				# We need at least one positive and one negative sample for ROC AUC
				# to be defined: reject the sample
				#print("We need at least one positive and one negative sample for ROC AUC")
				continue
			else:
				score = roc_auc_score(y_true[indices], y_pred[indices])
				bootstrapped_scores.append(score)
				#print("score: %f" % score)
	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.
	# Computing the lower and upper bound of the 90% confidence interval
	# You can change the bounds percentiles to 0.025 and 0.975 to get
	# a 95% confidence interval instead.
	#print(sorted_scores)
	#print(len(sorted_scores))
	#print(int(0.025 * len(sorted_scores)))
	#print(int(0.975 * len(sorted_scores)))
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
	print(confidence_lower)
	print(confidence_upper)
	print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
	return confidence_lower, confidence_upper

def main():
	jpg_dict = {}
	if FLAGS.PatientID > 0:
		# list sub-directories
		data_dir = FLAGS.jpgClass_path
		unique_labels = []
		# for item in os.listdir(data_dir):
		# 	if os.path.isdir(os.path.join(data_dir, item)):
 		# 		unique_labels.append(os.path.join(item))
		unique_labels = []
		with open(FLAGS.labels_names, "r") as f:
			for line in f:
				line = line.replace('\r','\n')
				line = line.split('\n')
				for eachline in line:
					if len(eachline)>0:
						unique_labels.append(eachline)
		unique_labels.sort()		
		for text in unique_labels:
			typeIm = 'train_*.jpeg'
			jpeg_file_path = os.path.join(data_dir, text, typeIm)
			#print(jpeg_file_path)
			#matching_files = tf.gfile.Glob(jpeg_file_path)
			matching_files = glob.glob(jpeg_file_path)
			#print(matching_files)
			if len(matching_files)>1:
				for img in matching_files:
					jpg_dict[os.path.basename(img)[6:6+FLAGS.PatientID]] = True
					#print(os.path.basename(img))
					#print(os.path.basename(img)[6:6+FLAGS.PatientID])

			typeIm = '*'
			jpeg_file_path = os.path.join(data_dir, text, typeIm)
			#print(jpeg_file_path)
			#matching_files = tf.gfile.Glob(jpeg_file_path)
			matching_files = glob.glob(jpeg_file_path)
			if len(matching_files)>1:
				for img in matching_files:
					jpg_dict[os.path.basename(img[0])[6:6+FLAGS.PatientID]] = True
		#print(jpg_dict)


	ref_file_data = {}
	if os.path.isfile(FLAGS.ref_file):
		corr = 'corrected_'
		with open(FLAGS.ref_file) as fstat2:
			for line in fstat2:
				if line.find('.dat') != -1:
					filename = line.split('.dat')[0]
				elif line.find('.jpeg') != -1:
					filename = line.split('.jpeg')[0]
				elif line.find('.net2048') != -1:
					filename = line.split('.net2048')[0]
				else:
					continue
				# remove set "train/valid/test"
				basenameXY = '_'.join(filename.split('_')[1:])

				ExpectedProb = line.split('[')[-1]
				ExpectedProb = ExpectedProb.split(']')[0]
				ref_file_data[basenameXY] = {}
				#ref_file_data[basenameXY]['value'] = float(ExpectedProb.split()[FLAGS.ref_label])
				ref_file_data[basenameXY]['value'] = (float(ExpectedProb.split()[FLAGS.ref_label])) / (1.0 - float(ExpectedProb.split()[0]))
				if 'True' in line:
					ref_file_data[basenameXY]['selected'] = True
				else:
					ref_file_data[basenameXY]['selected'] = False



	AllData = {}
	nstart = True
	y_score_Avg_PerTile = []
	y_score_PcS_PerTile = []
	y_ref_PerTile = []

	unique_labels = []
	with open(FLAGS.labels_names, "r") as f:
		for line in f:
			line = line.replace('\r','\n')
			line = line.split('\n')
			for eachline in line:
				if len(eachline)>0:
					unique_labels.append(eachline)
	TotNbTiles = 0
	if ', ' in FLAGS.file_stats:
		file1 = FLAGS.file_stats.split(', ')[0]
		file2 = FLAGS.file_stats.split(', ')[1]
		corr = ''
		with open(file1) as f:
			for line in f:
				#print(line)
				basename = line.split()[0]
				if FLAGS.PatientID > 0:
					#thisID = os.path.basename(basename[0])[5:17]
					thisID = os.path.basename(basename[0])[5:5+FLAGS.PatientID]
					if thisID in jpg_dict:
						continue
				tmp_out = line.split('[')[1]
				tmp_out = tmp_out.split(']')[0]
				AllData[basename] = {}				
				AllData[basename]['Labelvec'] = [float(x) for x in tmp_out.split(',')]

				tmp_out = line.split('Percent_Selected:')[1]
				PcSel = tmp_out.split('Average_Probability:')[0].split()
				AvgPrb = tmp_out.split('Average_Probability:')[1].split()
				tmp = 0
				AllData[basename]['Percent_Selected'] = {}
				for eachlabel in unique_labels:
					AllData[basename]['Percent_Selected'][eachlabel] = float(PcSel[tmp])
					tmp += 1
				tmp = 0
				AllData[basename]['Avg_Prob'] = {}
				AllData[basename]['All_Prob'] = {}
				for eachlabel in unique_labels:
					AllData[basename]['Avg_Prob'][eachlabel] = float(AvgPrb[tmp])
					AllData[basename]['All_Prob'][eachlabel] = [float(AvgPrb[tmp])]
					tmp += 1
				#print(line)
				#print(basename)
				#print(AllData[basename])

		with open(file2) as f:
			for line in f:
				#print(line)
				basename = line.split()[0]
				if FLAGS.PatientID > 0:
					thisID = os.path.basename(basename[0])[5:17]
					if thisID in jpg_dict:
						continue


				tmp_out = line.split('Percent_Selected:')[1]
				PcSel = tmp_out.split('Average_Probability:')[0].split()
				AvgPrb = tmp_out.split('Average_Probability:')[1].split()
				if basename in AllData.keys():
					tmp = 0
					for eachlabel in unique_labels:
						AllData[basename]['Percent_Selected'][eachlabel] = float(PcSel[tmp]) +AllData[basename]['Percent_Selected'][eachlabel]
						AllData[basename]['Percent_Selected'][eachlabel] = AllData[basename]['Percent_Selected'][eachlabel] / 2.0
						tmp += 1
					tmp = 0
					for eachlabel in unique_labels:
						AllData[basename]['Avg_Prob'][eachlabel] = float(AvgPrb[tmp]) + AllData[basename]['Avg_Prob'][eachlabel]
						AllData[basename]['Avg_Prob'][eachlabel] = AllData[basename]['Avg_Prob'][eachlabel] / 2.0
						AllData[basename]['All_Prob'][eachlabel].append(AllData[basename]['Avg_Prob'][eachlabel])
						tmp += 1					
				else:
					AllData[basename] = {}
					tmp_out = line.split('[')[1]
					tmp_out = tmp_out.split(']')[0]
					AllData[basename]['Labelvec'] = [float(x) for x in tmp_out.split(',')]
					tmp = 0
					AllData[basename]['Percent_Selected'] = {}
					for eachlabel in unique_labels:
						AllData[basename]['Percent_Selected'][eachlabel] = float(PcSel[tmp])
						tmp += 1
					tmp = 0
					AllData[basename]['Avg_Prob'] = {}
					for eachlabel in unique_labels:
						AllData[basename]['Avg_Prob'][eachlabel] = float(AvgPrb[tmp])
						AllData[basename]['All_Prob'][eachlabel] = [float(AvgPrb[tmp])]
						tmp += 1

				#print(line)
				#print(basename)
				#print(AllData[basename])


		y_score = []
		y_score_PcSelect = []
		y_ref = []
		n_classes = len(unique_labels)
		output = open(os.path.join(FLAGS.output_dir, 'out2_perSlideStats_avg.txt'),'w')
		for basename in AllData.keys():
			#print(basename)
			#print(AllData[basename])
			output.write("%s\ttrue_label: %s\t" % (basename, AllData[basename]['Labelvec']) )
			tmp_prob = []
			output.write("Percent_Selected: ")
			for eachlabel in unique_labels:
				tmp_prob.append(AllData[basename]['Percent_Selected'][eachlabel])
				output.write("%f\t" % (AllData[basename]['Percent_Selected'][eachlabel]) )
			y_score_PcSelect.append(tmp_prob)
			tmp_prob = []
			output.write("Average_Probability: ")
			for eachlabel in unique_labels: 
				tmp_prob.append(AllData[basename]['Avg_Prob'][eachlabel])
				output.write("%f\t" % (AllData[basename]['Avg_Prob'][eachlabel]) )
			output.write("\n")
			y_score.append(tmp_prob)
			y_ref.append(AllData[basename]['Labelvec'])
		output.close()
		#print("y_score")
		#print(y_score)
		#print("y_ref")
		#print(y_ref)
		#print("y_score_PcSelect")
		#print(y_score_PcSelect)

	else:
		XY = {}
		with open(FLAGS.file_stats) as f:
			for line in f:
				#print(line)
				if line.find('.dat') != -1:
					filename = line.split('.dat')[0]
				elif line.find('.jpeg') != -1:
					filename = line.split('.jpeg')[0]
				elif line.find('.net2048') != -1:
					filename = line.split('.net2048')[0]
				else:
					continue
				basename = '_'.join(filename.split('_')[:-2])
				#X = filename.split('_')[-2]
				#Y = filename.split('_')[-1]
				if filename in XY.keys():
					# tile already included
					continue
				else:
					XY[filename] = True

				if FLAGS.PatientID > 0:
					basename = basename[:(len(basename.split('_')[0]) + FLAGS.PatientID)+1]
				#	thisID = os.path.basename(basename)[5:5+FLAGS.PatientID]
				#	if thisID in jpg_dict:
				#		print("ID %s in jpg_dict" % thisID)
				#		continue
				#print("basename")
				#print(basename)
				#print("filename")
				#print(filename)
				# Check if tile should be considered for ROC (classified as LUAD) or not (Normal or LUSC)
				corr = ''
				analyze = True
				if os.path.isfile(FLAGS.ref_file):
					corr = 'corrected_'
					basenameXY = '_'.join(filename.split('_')[1:])
					if basenameXY in ref_file_data.keys():
						if FLAGS.ref_thresh == -1:
							# check if tile selected or not
							analyze = ref_file_data[basenameXY]['selected']
						elif ref_file_data[basenameXY]['value'] >= FLAGS.ref_thresh :
							analyze = True
							print("basenameXY %s identified with prob %f analyzed" % (basenameXY, ref_file_data[basenameXY]['value']) )
						else:
							analyze = False
							print("basenameXY %s identified with prob %f Not analyzed" % (basenameXY, ref_file_data[basenameXY]['value']) )
					else:
						analyze = False
						print("basenameXY %s not identified" % basenameXY)
					'''
					with open(FLAGS.ref_file) as fstat2:
						for line2 in fstat2:
							if filename in line2:
								#print("Found:")
								#print(line2)
								if "False" in line2:
									analyze = False
								#print(analyze)
								break
					'''

				if analyze == False:
					#print("continue")
					continue
				TotNbTiles += 1
				ExpectedProb = line.split('[')[1]
				ExpectedProb = ExpectedProb.split(']')[0]
				ExpectedProb = ExpectedProb.split()
				try: # mutations format
					IncProb = line.split('[')[2]
					IncProb = IncProb.split(']')[0]
					IncProb = IncProb.split()
				except:
					IncProb = []
					minProb_Indx = 1
					maxProb_Indx = 1
					minProb_Val = 2
					maxProb_Val = 0
					for kL in range(len(ExpectedProb)):
						if kL ==0:
							#IncProb.append(float(ExpectedProb[0]))
							IncProb.append(0)
						else:
							IncProb.append(float(ExpectedProb[kL]) / (1-float(ExpectedProb[0])))
							if IncProb[kL] < minProb_Val:
								minProb_Val = IncProb[kL]
								minProb_Indx = kL
							if IncProb[kL] >= maxProb_Val:
								maxProb_Val = IncProb[kL]
								maxProb_Indx = kL
					for kL in range(len(ExpectedProb)):
						ExpectedProb[kL] = 0
					try:
						True_Label = int(line.split('labels:')[1])
					except:
						# old filename format - assuming 2 classes only
						True_Label = line.split()[1]
						if True_Label == 'True':
							True_Label = maxProb_Indx
						else:
							True_Label = minProb_Indx
					#print("True label: %d " % True_Label)
					ExpectedProb[True_Label] = 1
				true_label = []
				true_label_name = ''
				for iprob in ExpectedProb:
					true_label.append(float(iprob))
					true_label_name = true_label_name + str(iprob)
				true_label.pop(0)
				OutProb = []
				for iprob in IncProb:
					OutProb.append(float(iprob))
				OutProb.pop(0)
				#print(true_label)
				#print(OutProb)
				tmp_prob_avg = []
				tmp_prob_pcs = []
				basename = basename + "_"+ true_label_name
				if basename in AllData:
					AllData[basename]['NbTiles'] += 1
					for eachlabel in range(len(OutProb)):
						AllData[basename]['Probs'][eachlabel] = AllData[basename]['Probs'][eachlabel] + OutProb[eachlabel]
						AllData[basename]['All_Prob'][eachlabel].append(OutProb[eachlabel])
						#if OutProb[eachlabel] >= 0.5:
						if FLAGS.MultiThresh > 0:
							PcS_thresh = FLAGS.MultiThresh
						else:
							PcS_thresh = max(OutProb)
						if OutProb[eachlabel] >= PcS_thresh:
							AllData[basename]['Nb_Selected'][eachlabel] = AllData[basename]['Nb_Selected'][eachlabel] + 1.0
				else:
					AllData[basename] = {}
					AllData[basename]['NbTiles'] = 1
					AllData[basename]['Labelvec'] = true_label
					AllData[basename]['Nb_Selected'] = {}
					AllData[basename]['Probs'] = {}
					AllData[basename]['All_Prob'] = {}
					for eachlabel in range(len(OutProb)):
						AllData[basename]['Nb_Selected'][eachlabel] = 0.0
						#AllData[basename]['LabelIndx_'+unique_labels(eachlabel)] = true_label(eachlabel)
						AllData[basename]['Probs'][eachlabel] = OutProb[eachlabel]
						AllData[basename]['All_Prob'][eachlabel] = [OutProb[eachlabel]]
						#if OutProb[eachlabel] >= 0.5:
						#print(eachlabel)
						#print(OutProb[eachlabel])
						#print(max(OutProb[eachlabel]))
						if FLAGS.MultiThresh > 0:
							PcS_thresh = FLAGS.MultiThresh
						else:
							PcS_thresh = max(OutProb)
						if OutProb[eachlabel] >= PcS_thresh:
							AllData[basename]['Nb_Selected'][eachlabel] = 1.0
					nstart = False
				for eachlabel in range(len(OutProb)):
					tmp_prob_avg.append(OutProb[eachlabel])
					if FLAGS.MultiThresh > 0:
						PcS_thresh = FLAGS.MultiThresh
					else:
						PcS_thresh = max(OutProb)
					if OutProb[eachlabel] >= PcS_thresh:
						tmp_prob_pcs.append(1.)
					else:
						tmp_prob_pcs.append(0.)
				y_score_Avg_PerTile.append(tmp_prob_avg)
				y_score_PcS_PerTile.append(tmp_prob_pcs)
				y_ref_PerTile.append(AllData[basename]['Labelvec'])



		#print("%d tiles used for the ROC curves" % TotNbTiles)
		output = open(os.path.join(FLAGS.output_dir, corr + 'out2_perSlideStats.txt'),'w')
		y_score = []
		y_score_PcSelect = []
		y_ref = []
		n_classes = len(unique_labels)
		#print(unique_labels)
		#print("DEBUG")
		#print(len(AllData.keys()))
		#print(len(AllData))
		for basename in AllData.keys():
			output.write("%s\ttrue_label: %s\t" % (basename, AllData[basename]['Labelvec']) )
			tmp_prob = []
			AllData[basename]['Percent_Selected'] = {}
			output.write("Percent_Selected: ")
			for eachlabel in range(len(unique_labels)):
				#print(eachlabel)
				#print(float(AllData[basename]['NbTiles']))
				#print(AllData[basename]['Nb_Selected'][eachlabel])
				AllData[basename]['Percent_Selected'][eachlabel] = AllData[basename]['Nb_Selected'][eachlabel] / float(AllData[basename]['NbTiles'])
				tmp_prob.append(AllData[basename]['Percent_Selected'][eachlabel])
				output.write("%f\t" % (AllData[basename]['Percent_Selected'][eachlabel]) )
			y_score_PcSelect.append(tmp_prob)
			AllData[basename]['Avg_Prob'] = {}
			tmp_prob = []
			output.write("Average_Probability: ")
			for eachlabel in range(len(AllData[basename]['Probs'])): 
				TopVal = -1
				if TopVal > 0:
					tmpDat = sorted(AllData[basename]['All_Prob'][eachlabel], reverse=True)
					AllData[basename]['Avg_Prob'][eachlabel] = sum(tmpDat[:min(10, len(tmpDat))]) / float(TopVal)
				else:
					AllData[basename]['Avg_Prob'][eachlabel] = AllData[basename]['Probs'][eachlabel] / float(AllData[basename]['NbTiles'])
				tmp_prob.append(AllData[basename]['Avg_Prob'][eachlabel])
				output.write("%f\t" % (AllData[basename]['Avg_Prob'][eachlabel]) )
			output.write("tiles#: %f\t" % float(AllData[basename]['NbTiles']))
			output.write("\n")
			y_score.append(tmp_prob)
			y_ref.append(AllData[basename]['Labelvec'])


		output.close()

	## Compute ROC per tile
	y_score_Avg_PerTile = np.array(y_score_Avg_PerTile)
	y_score_PcS_PerTile = np.array(y_score_PcS_PerTile)
	y_ref_PerTile = np.array(y_ref_PerTile)	
	#print(y_score_Avg_PerTile)
	#print(y_score_PcS_PerTile)
	#print(y_ref_PerTile)
	fpr = dict()
	tpr = dict()
	thresholds = dict()
	opt_thresh = dict()
	roc_auc = dict()
	fpr_PcSel = dict()
	tpr_PcSel = dict()
	roc_auc_PcSel = dict()
	#print("n_classes")
	#print(n_classes)
	confidence_low_avg = dict()
	confidence_up_avg = dict()
	confidence_low_PcS = dict()
	confidence_up_PcS = dict()
	
	print("DEBUG:")
	print(len(y_ref_PerTile))
	print(y_ref_PerTile)
	print(len(y_score_Avg_PerTile))
	print(y_score_Avg_PerTile)
	print(len(y_score_PcS_PerTile))
	print(y_score_PcS_PerTile)
	
	for i in range(n_classes):
		print(y_ref_PerTile[:, i], y_score_Avg_PerTile[:, i], y_score_PcS_PerTile[:, i])
		fpr[i], tpr[i], thresholds[i] = roc_curve(y_ref_PerTile[:, i], y_score_Avg_PerTile[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
		fpr_PcSel[i], tpr_PcSel[i], _ = roc_curve(y_ref_PerTile[:, i], y_score_PcS_PerTile[:, i])
		roc_auc_PcSel[i] = auc(fpr_PcSel[i], tpr_PcSel[i])
		euc_dist = []
		try:
			for jj in range(len(fpr[i])):
				euc_dist.append( euclidean_distances([[0, 1]], [[fpr[i][jj], tpr[i][jj]]]) )
			opt_thresh[i] = thresholds[i][euc_dist.index(min(euc_dist))]
		except:
			opt_thresh[i] = 0
		#print(y_ref_PerTile[:, i])
		#print(y_score_Avg_PerTile[:, i])
		confidence_low_avg[i], confidence_up_avg[i] = BootStrap(y_ref_PerTile[:, i], y_score_Avg_PerTile[:, i], False)
		confidence_low_PcS[i], confidence_up_PcS[i] = BootStrap(y_ref_PerTile[:, i], y_score_PcS_PerTile[:, i], False)


	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_ref_PerTile.ravel(), y_score_Avg_PerTile.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	confidence_low_avg["micro"], confidence_up_avg["micro"] = BootStrap(y_ref_PerTile.ravel(), y_score_Avg_PerTile.ravel(), False)

	fpr_PcSel["micro"], tpr_PcSel["micro"], thresholds["micro"] = roc_curve(y_ref_PerTile.ravel(), y_score_PcS_PerTile.ravel())
	roc_auc_PcSel["micro"] = auc(fpr_PcSel["micro"], tpr_PcSel["micro"])
	confidence_low_PcS["micro"], confidence_up_PcS["micro"] = BootStrap(y_ref_PerTile.ravel(), y_score_PcS_PerTile.ravel(), False)

	print('y_ref_PerTile.ravel(), y_score_PcS_PerTile.ravel()')
	print(y_ref_PerTile.ravel(), y_score_PcS_PerTile.ravel())
	euc_dist = []
	#for jj in range(len(fpr_PcSel["micro"])):
	#	print('[fpr_PcSel["micro"][jj], tpr_PcSel["micro"][jj]]')
	#	print([fpr_PcSel["micro"][jj], tpr_PcSel["micro"][jj]])
	#	euc_dist.append( euclidean_distances([[0, 1]], [[fpr_PcSel["micro"][jj], tpr_PcSel["micro"][jj]]]) )
	#opt_thresh["micro"] = thresholds["micro"][euc_dist.index(min(euc_dist))]


	## Compute macro-average ROC curve and ROC area
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	all_fpr_PcSel = np.unique(np.concatenate([fpr_PcSel[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	mean_tpr_PcSel= np.zeros_like(all_fpr_PcSel)
	for i in range(n_classes):
	    mean_tpr_PcSel += interp(all_fpr_PcSel, fpr_PcSel[i], tpr_PcSel[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
	confidence_low_avg["macro"], confidence_up_avg["macro"] = BootStrap(y_ref_PerTile, y_score_Avg_PerTile, True, n_classes)

	mean_tpr_PcSel /= n_classes
	fpr_PcSel["macro"] = all_fpr_PcSel
	tpr_PcSel["macro"] = mean_tpr_PcSel
	roc_auc_PcSel["macro"] = auc(fpr_PcSel["macro"], tpr_PcSel["macro"])
	confidence_low_PcS["macro"], confidence_up_PcS["macro"] = BootStrap(y_ref_PerTile, y_score_PcS_PerTile, True, n_classes)


	# save data
	print("******* FP / TP for average probability")
	print(fpr)
	print(tpr)
	for i in range(n_classes):
		output = open(os.path.join(FLAGS.output_dir, corr + 'out1_perTile_roc_data_AvPb_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc[i]) + '_CIs_' + str("%.4f" % confidence_low_avg[i]) + "_" + str("%.4f" % confidence_up_avg[i]) +  '_t' + str("%.6f" % opt_thresh[i]) + '.txt'),'w')

		for kk in range(len(tpr[i])):
			output.write("%f\t%f\n" % (fpr[i][kk], tpr[i][kk]) )
		output.close()

	output = open(os.path.join(FLAGS.output_dir, corr + 'out1_perTile_roc_data_AvPb_macro_auc_' + str("%.4f" % roc_auc["macro"]) + '_CIs_' + str("%.4f" % confidence_low_avg["macro"]) + "_" + str("%.4f" % confidence_up_avg["macro"]) + '.txt'),'w')
	for kk in range(len(tpr["macro"])):
		output.write("%f\t%f\n" % (fpr["macro"][kk], tpr["macro"][kk]) )
	output.close()

	#output = open(os.path.join(FLAGS.output_dir, corr + 'out1_perTile_roc_data_AvPb_micro_auc_' + str("%.4f" % roc_auc["micro"]) + '_CIs_' + str("%.4f" % confidence_low_avg["micro"]) + "_" + str("%.4f" % confidence_up_avg["micro"]) + '_t' + str("%.3f" % opt_thresh["micro"]) + '.txt'),'w')
	output = open(os.path.join(FLAGS.output_dir, corr + 'out1_perTile_roc_data_AvPb_micro_auc_' + str("%.4f" % roc_auc["micro"]) + '_CIs_' + str("%.4f" % confidence_low_avg["micro"]) + "_" + str("%.4f" % confidence_up_avg["micro"])  + '.txt'),'w')
	for kk in range(len(tpr["micro"])):
		output.write("%f\t%f\n" % (fpr["micro"][kk], tpr["micro"][kk]) )
	output.close()

	print("******* FP / TP for percent selected")
	print(fpr_PcSel)
	print(tpr_PcSel)
	for i in range(n_classes):
		output = open(os.path.join(FLAGS.output_dir, corr + 'out1_perTile_roc_data_PcSel_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc_PcSel[i]) + '_CIs_' + str("%.4f" % confidence_low_PcS[i]) + "_" + str("%.4f" % confidence_up_PcS[i]) + '.txt'),'w')
		for kk in range(len(tpr_PcSel[i])):
			output.write("%f\t%f\n" % (fpr_PcSel[i][kk], tpr_PcSel[i][kk]) )
		output.close()


	output = open(os.path.join(FLAGS.output_dir, corr+ 'out1_perTile_roc_data_PcSel_macro_auc_' + str("%.4f" % roc_auc_PcSel["macro"]) + '_CIs_' + str("%.4f" % confidence_low_PcS["macro"]) + "_" + str("%.4f" % confidence_up_PcS["macro"]) + '.txt'),'w')
	for kk in range(len(tpr_PcSel["macro"])):
		output.write("%f\t%f\n" % (fpr_PcSel["macro"][kk], tpr_PcSel["macro"][kk]) )
	output.close()

	output = open(os.path.join(FLAGS.output_dir, corr+ 'out1_perTile_roc_data_PcSel_micro_auc_' + str("%.4f" % roc_auc_PcSel["micro"]) + '_CIs_' + str("%.4f" % confidence_low_PcS["micro"]) + "_" + str("%.4f" % confidence_up_PcS["micro"]) + '.txt'),'w')
	for kk in range(len(tpr_PcSel["micro"])):
		output.write("%f\t%f\n" % (fpr_PcSel["micro"][kk], tpr_PcSel["micro"][kk]) )
	output.close()















	## Compute ROC per slide
	y_score = np.array(y_score)
	y_score_PcSelect = np.array(y_score_PcSelect)
	y_ref = np.array(y_ref)
	
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	thresholds = dict()
	opt_thresh = dict()
	roc_auc = dict()
	fpr_PcSel = dict()
	tpr_PcSel = dict()
	roc_auc_PcSel = dict()
	confidence_low_avg = dict()
	confidence_up_avg = dict()
	confidence_low_PcS = dict()
	confidence_up_PcS = dict()

	print("n_classes")
	print(n_classes)

	for i in range(n_classes):
		#print(y_ref[:, i], y_score[:, i])
		fpr[i], tpr[i], thresholds[i] = roc_curve(y_ref[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

		fpr_PcSel[i], tpr_PcSel[i], _ = roc_curve(y_ref[:, i], y_score_PcSelect[:, i])
		roc_auc_PcSel[i] = auc(fpr_PcSel[i], tpr_PcSel[i])
		euc_dist = []
		try:
			for jj in range(len(fpr[i])):
				euc_dist.append( euclidean_distances([[0, 1]], [[fpr[i][jj], tpr[i][jj]]]) )
			opt_thresh[i] = thresholds[i][euc_dist.index(min(euc_dist))]
		except:
			opt_thresh[i] = 0
		confidence_low_avg[i], confidence_up_avg[i] = BootStrap(y_ref[:, i], y_score[:, i], False)
		confidence_low_PcS[i], confidence_up_PcS[i] = BootStrap(y_ref[:, i], y_score_PcSelect[:, i], False)


	# Compute micro-average ROC curve and ROC area
	# print(len(y_ref.ravel()), len(y_score.ravel()))

	fpr["micro"], tpr["micro"], _ = roc_curve(y_ref.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	confidence_low_avg["micro"], confidence_up_avg["micro"] = BootStrap(y_ref.ravel(), y_score.ravel(), False)

	print(len(y_ref.ravel()), len(y_score_PcSelect.ravel()))

	fpr_PcSel["micro"], tpr_PcSel["micro"], thresholds["micro"] = roc_curve(y_ref.ravel(), y_score_PcSelect.ravel())
	roc_auc_PcSel["micro"] = auc(fpr_PcSel["micro"], tpr_PcSel["micro"])
	confidence_low_PcS["micro"], confidence_up_PcS["micro"] = BootStrap(y_ref.ravel(), y_score_PcSelect.ravel(), False)

	#euc_dist = []
	#for jj in range(len(fpr_PcSel["micro"])):
	#	euc_dist.append( euclidean_distances([[0, 1]], [[fpr_PcSel["micro"][jj], tpr_PcSel["micro"][jj]]]) )
	#print(min(euc_dist))
	#print(euc_dist.index(min(euc_dist)))
	#print(thresholds["micro"])
	#opt_thresh["micro"] = thresholds["micro"][euc_dist.index(min(euc_dist))]


	## Compute macro-average ROC curve and ROC area
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	all_fpr_PcSel = np.unique(np.concatenate([fpr_PcSel[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	mean_tpr_PcSel= np.zeros_like(all_fpr_PcSel)
	for i in range(n_classes):
	    mean_tpr_PcSel += interp(all_fpr_PcSel, fpr_PcSel[i], tpr_PcSel[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
	confidence_low_avg["macro"], confidence_up_avg["macro"] = BootStrap(y_ref, y_score, True, n_classes)

	mean_tpr_PcSel /= n_classes
	fpr_PcSel["macro"] = all_fpr_PcSel
	tpr_PcSel["macro"] = mean_tpr_PcSel
	roc_auc_PcSel["macro"] = auc(fpr_PcSel["macro"], tpr_PcSel["macro"])
	confidence_low_PcS["macro"], confidence_up_PcS["macro"] = BootStrap(y_ref, y_score_PcSelect, True, n_classes)



	# save data
	print("******* FP / TP for average probability")
	print(fpr)
	print(tpr)
	for i in range(n_classes):
		output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_AvPb_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc[i]) + '_CIs_' + str("%.4f" % confidence_low_avg[i]) + "_" + str("%.4f" % confidence_up_avg[i]) + '_t' + str("%.6f" % opt_thresh[i]) + '.txt'),'w')
		for kk in range(len(tpr[i])):
			output.write("%f\t%f\n" % (fpr[i][kk], tpr[i][kk]) )
		output.close()

	output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_AvPb_macro_auc_' + str("%.4f" % roc_auc["macro"]) + '_CIs_' + str("%.4f" % confidence_low_avg["macro"]) + "_" + str("%.4f" % confidence_up_avg["macro"]) + '.txt'),'w')
	for kk in range(len(tpr["macro"])):
		output.write("%f\t%f\n" % (fpr["macro"][kk], tpr["macro"][kk]) )
	output.close()

#	output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_AvPb_micro_auc_' + str("%.4f" % roc_auc["micro"]) + '_t' + str("%.3f" % opt_thresh["micro"]) + '_CIs_' + str("%.4f" % confidence_low_avg["micro"]) + "_" + str("%.4f" % confidence_up_avg["micro"])+ '.txt'),'w')
	output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_AvPb_micro_auc_' + str("%.4f" % roc_auc["micro"]) + '_CIs_' + str("%.4f" % confidence_low_avg["micro"]) + "_" + str("%.4f" % confidence_up_avg["micro"])+ '.txt'),'w')
	for kk in range(len(tpr["micro"])):
		output.write("%f\t%f\n" % (fpr["micro"][kk], tpr["micro"][kk]) )
	output.close()

	print("******* FP / TP for percent selected")
	print(fpr_PcSel)
	print(tpr_PcSel)
	for i in range(n_classes):
		output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_PcSel_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc_PcSel[i]) + '_CIs_' + str("%.4f" % confidence_low_PcS[i]) + "_" + str("%.4f" % confidence_up_PcS[i]) + '.txt'),'w')
		for kk in range(len(tpr_PcSel[i])):
			output.write("%f\t%f\n" % (fpr_PcSel[i][kk], tpr_PcSel[i][kk]) )
		output.close()


	output = open(os.path.join(FLAGS.output_dir, corr+ 'out2_roc_data_PcSel_macro_auc_' + str("%.4f" % roc_auc_PcSel["macro"]) + '_CIs_' + str("%.4f" % confidence_low_PcS["macro"]) + "_" + str("%.4f" % confidence_up_PcS["macro"])+ '.txt'),'w')
	for kk in range(len(tpr_PcSel["macro"])):
		output.write("%f\t%f\n" % (fpr_PcSel["macro"][kk], tpr_PcSel["macro"][kk]) )
	output.close()

	output = open(os.path.join(FLAGS.output_dir, corr+ 'out2_roc_data_PcSel_micro_auc_' + str("%.4f" % roc_auc_PcSel["micro"]) + '_CIs_' + str("%.4f" % confidence_low_PcS["micro"]) + "_" + str("%.4f" % confidence_up_PcS["micro"]) + '.txt'),'w')
	for kk in range(len(tpr_PcSel["micro"])):
		output.write("%f\t%f\n" % (fpr_PcSel["micro"][kk], tpr_PcSel["micro"][kk]) )
	output.close()




	# Plot all ROC curves
	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"],
		 label='micro-average ROC curve (area = {0:0.2f})'
		       ''.format(roc_auc["micro"]),
		 color='deeppink', linestyle=':', linewidth=4)

	plt.plot(fpr["macro"], tpr["macro"],
		 label='macro-average ROC curve (area = {0:0.2f})'
		       ''.format(roc_auc["macro"]),
		 color='navy', linestyle=':', linewidth=4)
	lw = 2
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=lw,
		     label='ROC curve of class {0} (area = {1:0.2f})' 
		     ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title("Some extension of Receiver operating characteristic to multi-class (Aggregation by averaging tiles'probabilities)")
	plt.legend(loc="lower right")
	plt.show()


	plt.figure()
	plt.plot(fpr_PcSel["micro"], tpr_PcSel["micro"],
		 label='micro-average ROC curve (area = {0:0.2f})'
		       ''.format(roc_auc_PcSel["micro"]),
		 color='deeppink', linestyle=':', linewidth=4)
	plt.plot(fpr_PcSel["macro"], tpr_PcSel["macro"],
		 label='macro-average ROC curve (area = {0:0.2f})'  
		       ''.format(roc_auc_PcSel["macro"]),
		 color='navy', linestyle=':', linewidth=4)
	lw = 2
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr_PcSel[i], tpr_PcSel[i], color=color, lw=lw,
		     label='ROC curve of class {0} (area = {1:0.2f})'  
		     ''.format(i, roc_auc_PcSel[i]))
	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class (Aggregation by percentage of TP tiles)')
	plt.legend(loc="lower right")
	plt.show()

if __name__ == '__main__':
  '''
  FLAGS = None
  try:
    opts, args = getopt.getopt(sys.argv,"hi:o:",["file_stats=","output_dir="])
  except getopt.GetoptError:
    print('python /ifs/home/coudrn01/NN/Lung/0h_ROC_sklearn.py -file_stats <> -output_dir <>')
    sys.exit(2)

  print("opts")
  print(opts)
  for opt, arg in opts:
    print(opt)
    if opt == '-h':
      print('0h_ROC_sklearn.py --file_stat=<inputfile> --output_dir=<outputfile>')
      sys.exit()
    elif opt in ("-file_stats", "--nfile_stat"):
      FLAGS.file_stats = arg
    elif opt in ("-output_dir", "--noutput_dir"):
      FLAGS.output_dir = arg

  main(FLAGS)
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--file_stats',
      type=str,
      default='',
      help='Statistics of each tiles.'
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      default='mustbedefined',
      help='Output directory.'
  )
  parser.add_argument(
      '--labels_names',
      type=str,
      default='',
      help='Names of the possible output labels ordered as desired example, /ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/9_10mutations/label_names.txt'
  )
  parser.add_argument(
      '--ref_file',
      type=str,
      default='',
      help='Stats files used as a reference (obtained from a previous classification Normal/LUAD/LUSC). If the tile is associated with "True" (proper classification), it is used for mutation analysis'
  )
  parser.add_argument(
      '--ref_label',
      type=int,
      default=1,
      help='Label ID in ref_file that needs to be checked.'
  )
  parser.add_argument(
      '--ref_thresh',
      type=float,
      default=0.5,
      help='threshold to apply to ref_label: if the probability is above it, the tile is included, otherwise it is excluded'
  )
  parser.add_argument(
      '--MultiThresh',
      type=float,
      default=-1,
      help='used for aggregation by percentage of tiles selected (that tile tiles above this threshold).'
  )
  parser.add_argument(
      '--PatientID',
      type=int,
      default=-1,
      help='Nb of characters for the patient ID.'
  )
  parser.add_argument(
      '--jpgClass_path',
      type=str,
      default='',
      help='path where the jpg were classified into train/valid/test (main folder)'
  )



  FLAGS, unparsed = parser.parse_known_args()
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

