'''
Compute ROC for multiple classes using sklearn package
(works on python 2.7 but not 3.5.3 pon the clusters)
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
	with open(FLAGS.file_stats) as f:
		for line in f:
			print(line)
			if line.find('.dat') != -1:
				filename = line.split('.dat')[0]
			elif line.find('.net2048') != -1:
				filename = line.split('.net2048')[0]
			else:
				continue
			basename = '_'.join(filename.split('_')[:-2])
			prob = line.split('[')[1]
			prob = prob.split(']')[0]
			prob = prob.split()
			nclass = []
			for iprob in prob:
				nclass.append(float(iprob))

			# remove background class
			nclass.pop(0)
			is_TP = line.split()[1]
			print(is_TP)
			if line.find('labels:') != -1:
				# if there is a "label" --> new file format with >2 classes. Otherwise, check /False/True.
				true_label = int(line.split('labels:')[-1])
			else:
				if is_TP == 'True':
					max_v = max(nclass)
					true_label = nclass.index(max_v)+1
				elif is_TP == 'False':
					min_v = min(nclass)
					true_label = nclass.index(min_v)+1
				else:
					true_label = -1

			print(true_label)

			if basename in AllData:
				AllData[basename]['NbTiles'] += 1
				if is_TP == 'True':
					AllData[basename]['Nb_Selected'] = AllData[basename]['Nb_Selected'] + 1.0
				
				for eachClass in range(len(nclass)):
					print("NEW")
					print(AllData[basename]['Probs'])
					print(nclass)
					print(AllData[basename]['Probs'][eachClass])
					print(nclass[eachClass])
					AllData[basename]['Probs'][eachClass] = AllData[basename]['Probs'][eachClass] + nclass[eachClass]
			else:
				AllData[basename] = {}
				AllData[basename]['NbTiles'] = 1
				AllData[basename]['Probs'] = nclass
				AllData[basename]['LabelIndx'] = true_label
				AllData[basename]['Labelvec'] = np.zeros(len(nclass))
				if is_TP == 'True':
					AllData[basename]['Nb_Selected'] = 1.0
				else:
					AllData[basename]['Nb_Selected'] = 0.0

				for kk in range(len(AllData[basename]['Labelvec'])):
					AllData[basename]['Labelvec'][kk] = 0
				AllData[basename]['Labelvec'][true_label-1] = 1
				nstart = False

	output = open(os.path.join(FLAGS.output_dir, 'out2_perSlideStats.txt'),'w')
	y_score = []
	y_ref = []
	n_classes = len(nclass)
	for basename in AllData.keys():
		output.write("%s\ttrue_label: %d\t" % (basename, AllData[basename]['LabelIndx']) )
		AllData[basename]['Avg_Prob'] = {}
		AllData[basename]['Percent_Selected'] = AllData[basename]['Nb_Selected'] / float(AllData[basename]['NbTiles'])
		tmp_prob = []
		for eachClass in range(len(AllData[basename]['Probs'])): 
			AllData[basename]['Avg_Prob'][eachClass] = AllData[basename]['Probs'][eachClass] / float(AllData[basename]['NbTiles'])
			tmp_prob.append(AllData[basename]['Avg_Prob'][eachClass])
			output.write("%f\t" % (AllData[basename]['Avg_Prob'][eachClass]) )
		output.write("\n")

		y_score.append(tmp_prob)
		y_ref.append(AllData[basename]['Labelvec'])
	output.close()
	y_score = np.array(y_score)
	y_ref = np.array(y_ref)
	

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	thresholds = dict()
	opt_thresh = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], thresholds[i] = roc_curve(y_ref[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
		euc_dist = []
		for jj in range(len(fpr[i])):
			euc_dist.append( euclidean_distances([1, 0], [fpr[i][jj], tpr[i][jj]]) )
		opt_thresh[i] = thresholds[i][euc_dist.index(min(euc_dist))]

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_ref.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	euc_dist = []
	for jj in range(len(fpr["micro"])):
		euc_dist.append( euclidean_distances([1, 0], [fpr["micro"][jj], tpr["micro"][jj]]) )
	opt_thresh["micro"] = thresholds["micro"][euc_dist.index(min(euc_dist))]

	## Compute macro-average ROC curve and ROC area
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



	# save data
	print("******* FP / TP ")
	print(fpr)
	print(tpr)
	for i in range(n_classes):
		output = open(os.path.join(FLAGS.output_dir, 'out2_roc_data_c' + str(i+1)+ 'auc_' + str("%.4f" % roc_auc[i]) + '_t' + str(opt_thresh[i]) + '.txt'),'w')
		for kk in range(len(tpr[i])):
			output.write("%f\t%f\n" % (fpr[i][kk], tpr[i][kk]) )
		output.close()

	output = open(os.path.join(FLAGS.output_dir, 'out2_roc_data_macro_auc_' + str("%.4f" % roc_auc["macro"]) + '.txt'),'w')
	for kk in range(len(tpr["macro"])):
		output.write("%f\t%f\n" % (fpr["macro"][kk], tpr["macro"][kk]) )
	output.close()

	output = open(os.path.join(FLAGS.output_dir, 'out2_roc_data_micro_auc_' + str("%.4f" % roc_auc["micro"])  + '_t' + str(opt_thresh["micro"]) + '.txt'),'w')
	for kk in range(len(tpr["micro"])):
		output.write("%f\t%f\n" % (fpr["micro"][kk], tpr["micro"][kk]) )
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
	plt.title('Some extension of Receiver operating characteristic to multi-class')
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
  FLAGS, unparsed = parser.parse_known_args()
  main()
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

