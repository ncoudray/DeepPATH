'''
Compute ROC for multiple output-classes (several TP possible for a given input) using sklearn package
(works on python 2.7 but not 3.5.3 on the clusters)
'''
import sys, getopt
import argparse
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.metrics.pairwise import euclidean_distances

FLAGS = None


def main():
	AllData = {}
	nstart = True

	unique_labels = []
	with open(FLAGS.labels_names, "r") as f:
		for line in f:
			line = line.replace('\r','\n')
			line = line.split('\n')
			for eachline in line:
				if len(eachline)>0:
					unique_labels.append(eachline)
	TotNbTiles = 0
	with open(FLAGS.file_stats) as f:
		for line in f:
			print(line)
			if line.find('.dat') != -1:
				filename = line.split('.dat')[0]
			elif line.find('.jpeg') != -1:
				filename = line.split('.jpeg')[0]
			elif line.find('.net2048') != -1:
				filename = line.split('.net2048')[0]
			else:
				continue
			basename = '_'.join(filename.split('_')[:-2])
			print("basename")
			print(basename)
			print("filename")
			print(filename)

			# Check if tile should be considered for ROC (classified as LUAD) or not (Normal or LUSC)
			corr = ''
			analyze = True
			if os.path.isfile(FLAGS.ref_stats):
				corr = 'corrected_'
				with open(FLAGS.ref_stats) as fstat2:
					for line2 in fstat2:
						if filename in line2:
							print("Found:")
							print(line2)
							if "False" in line2:
								analyze = False
							print(analyze)
							break
			if analyze == False:
				print("continue")
				continue
			TotNbTiles += 1

			ExpectedProb = line.split('[')[1]
			ExpectedProb = ExpectedProb.split(']')[0]
			ExpectedProb = ExpectedProb.split()
			try: # old format file
				IncProb = line.split('[')[2]
				IncProb = IncProb.split(']')[0]
				IncProb = IncProb.split()
			except:
				tmp = []
				IncProb = []
				for kL in range(len(ExpectedProb)):
					if kL ==0:
						IncProb.append(float(ExpectedProb[0]))
						tmp.append(0)
					else:
						IncProb.append(float(ExpectedProb[kL]) / (1-float(ExpectedProb[0])))
						tmp.append(IncProb[kL])

				ExpectedProb = ExpectedProb * 0
				if 'True' in line:
					ExpectedProb[IncProb.index(max(tmp))] = 1
				else:
					ExpectedProb[IncProb.index(min(tmp))] = 1

				'''
				IncProb = [0, 0, 0]
				IncProb[0] = float(ExpectedProb[0])
				IncProb[1] = float(ExpectedProb[1]) / (1-float(ExpectedProb[0]))
				IncProb[2] = float(ExpectedProb[2]) / (1-float(ExpectedProb[0]))
				tmp = [IncProb[1], IncProb[2]]
				ExpectedProb = [0, 0, 0]
				if 'True' in line:
					ExpectedProb[IncProb.index(max(tmp))] = 1
				else:
					ExpectedProb[IncProb.index(min(tmp))] = 1
				'''


			true_label = []
			for iprob in ExpectedProb:
				true_label.append(float(iprob))
			true_label.pop(0)
			OutProb = []
			for iprob in IncProb:
				OutProb.append(float(iprob))
			OutProb.pop(0)
			print(true_label)
			print(OutProb)


			if basename in AllData:
				AllData[basename]['NbTiles'] += 1
				for eachlabel in range(len(OutProb)):
					AllData[basename]['Probs'][eachlabel] = AllData[basename]['Probs'][eachlabel] + OutProb[eachlabel]
					if OutProb[eachlabel] >= 0.5:
						AllData[basename]['Nb_Selected'][eachlabel] = AllData[basename]['Nb_Selected'][eachlabel] + 1.0
			else:
				AllData[basename] = {}
				AllData[basename]['NbTiles'] = 1
				AllData[basename]['Labelvec'] = true_label
				AllData[basename]['Nb_Selected'] = {}
				AllData[basename]['Probs'] = {}
				for eachlabel in range(len(OutProb)):
					AllData[basename]['Nb_Selected'][eachlabel] = 0.0
					#AllData[basename]['LabelIndx_'+unique_labels(eachlabel)] = true_label(eachlabel)
					AllData[basename]['Probs'][eachlabel] = OutProb[eachlabel]
					if OutProb[eachlabel] >= 0.5:
						AllData[basename]['Nb_Selected'][eachlabel] = 1.0

				nstart = False

	print("%d tiles used for the ROC curves" % TotNbTiles)
	output = open(os.path.join(FLAGS.output_dir, corr + 'out2_perSlideStats.txt'),'w')
	y_score = []
	y_score_PcSelect = []
	y_ref = []
	n_classes = len(unique_labels)
	print(unique_labels)
	print(AllData)
	for basename in AllData.keys():
		output.write("%s\ttrue_label: %s\t" % (basename, AllData[basename]['Labelvec']) )
		tmp_prob = []
		AllData[basename]['Percent_Selected'] = {}
		output.write("Percent_Selected: ")
		for eachlabel in range(len(unique_labels)):
			AllData[basename]['Percent_Selected'][eachlabel] = AllData[basename]['Nb_Selected'][eachlabel] / float(AllData[basename]['NbTiles'])
			tmp_prob.append(AllData[basename]['Percent_Selected'][eachlabel])
			output.write("%f\t" % (AllData[basename]['Percent_Selected'][eachlabel]) )
		y_score_PcSelect.append(tmp_prob)

		AllData[basename]['Avg_Prob'] = {}
		tmp_prob = []
		output.write("Average_Probability: ")
		for eachlabel in range(len(AllData[basename]['Probs'])): 
			AllData[basename]['Avg_Prob'][eachlabel] = AllData[basename]['Probs'][eachlabel] / float(AllData[basename]['NbTiles'])
			tmp_prob.append(AllData[basename]['Avg_Prob'][eachlabel])
			output.write("%f\t" % (AllData[basename]['Avg_Prob'][eachlabel]) )
		output.write("\n")
		y_score.append(tmp_prob)
		y_ref.append(AllData[basename]['Labelvec'])


	output.close()
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
	print("n_classes")
	print(n_classes)

	for i in range(n_classes):
		print(y_ref[:, i], y_score[:, i])
		fpr[i], tpr[i], thresholds[i] = roc_curve(y_ref[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

		fpr_PcSel[i], tpr_PcSel[i], _ = roc_curve(y_ref[:, i], y_score_PcSelect[:, i])
		roc_auc_PcSel[i] = auc(fpr_PcSel[i], tpr_PcSel[i])
		euc_dist = []
		for jj in range(len(fpr[i])):
			euc_dist.append( euclidean_distances([1, 0], [fpr[i][jj], tpr[i][jj]]) )
		opt_thresh[i] = thresholds[i][euc_dist.index(min(euc_dist))]

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_ref.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	fpr_PcSel["micro"], tpr_PcSel["micro"], thresholds["micro"] = roc_curve(y_ref.ravel(), y_score_PcSelect.ravel())
	roc_auc_PcSel["micro"] = auc(fpr_PcSel["micro"], tpr_PcSel["micro"])
	euc_dist = []
	for jj in range(len(fpr["micro"])):
		euc_dist.append( euclidean_distances([1, 0], [fpr["micro"][jj], tpr["micro"][jj]]) )
	opt_thresh["micro"] = thresholds["micro"][euc_dist.index(min(euc_dist))]


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

	mean_tpr_PcSel /= n_classes
	fpr_PcSel["macro"] = all_fpr_PcSel
	tpr_PcSel["macro"] = mean_tpr_PcSel
	roc_auc_PcSel["macro"] = auc(fpr_PcSel["macro"], tpr_PcSel["macro"])


	# save data
	print("******* FP / TP for average probabilitys")
	print(fpr)
	print(tpr)
	for i in range(n_classes):
		output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_AvPb_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc[i]) + '_t' + str(opt_thresh[i]) + '.txt'),'w')
		for kk in range(len(tpr[i])):
			output.write("%f\t%f\n" % (fpr[i][kk], tpr[i][kk]) )
		output.close()

	output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_AvPb_macro_auc_' + str("%.4f" % roc_auc["macro"]) + '.txt'),'w')
	for kk in range(len(tpr["macro"])):
		output.write("%f\t%f\n" % (fpr["macro"][kk], tpr["macro"][kk]) )
	output.close()

	output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_AvPb_micro_auc_' + str("%.4f" % roc_auc["micro"]) + '_t' + str(opt_thresh["micro"]) + '.txt'),'w')
	for kk in range(len(tpr["micro"])):
		output.write("%f\t%f\n" % (fpr["micro"][kk], tpr["micro"][kk]) )
	output.close()

	print("******* FP / TP for percent selected")
	print(fpr_PcSel)
	print(tpr_PcSel)
	for i in range(n_classes):
		output = open(os.path.join(FLAGS.output_dir, corr + 'out2_roc_data_PcSel_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc_PcSel[i]) + '.txt'),'w')
		for kk in range(len(tpr_PcSel[i])):
			output.write("%f\t%f\n" % (fpr_PcSel[i][kk], tpr_PcSel[i][kk]) )
		output.close()


	output = open(os.path.join(FLAGS.output_dir, corr+ 'out2_roc_data_PcSel_macro_auc_' + str("%.4f" % roc_auc_PcSel["macro"]) + '.txt'),'w')
	for kk in range(len(tpr_PcSel["macro"])):
		output.write("%f\t%f\n" % (fpr_PcSel["macro"][kk], tpr_PcSel["macro"][kk]) )
	output.close()

	output = open(os.path.join(FLAGS.output_dir, corr+ 'out2_roc_data_PcSel_micro_auc_' + str("%.4f" % roc_auc_PcSel["micro"]) + '.txt'),'w')
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
      default='/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/9_10mutations/label_names.txt',
      help='Names of the possible output labels ordered as desired'
  )
  parser.add_argument(
      '--ref_stats',
      type=str,
      default='',
      help='Stats files used as a reference (obtained from a previous classification Normal/LUAD/LUSC). If the tile is associated with "True" (proper classification), it is used for mutation analysis'
  )

  FLAGS, unparsed = parser.parse_known_args()
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

