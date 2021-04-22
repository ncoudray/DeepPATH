"""A binary to evaluate Inception on the Lung data set.
Output generated:
** information for ROC curves with 2 aggregation methods (out_FPTPrate_PcTiles.txt, out_FPTPrate_ScoreTiles.txt)
** probability associated with each tile and info whether the max is a true positive or not (out_filename_Stats.txt)


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import csv
import sys
import tensorflow as tf

from inception import nc_inception_eval
from inception.nc_imagenet_data import ImagenetData
import numpy as np
#from inception import inception_eval
#from inception.imagenet_data import ImagenetData
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ImageSet_basename', 'test_',
                           """Either 'test_', 'valid' or 'train'.""")

tf.app.flags.DEFINE_string('TVmode', 'test',
                           """Either 'test' or 'valid' (test prep the output for AUC computation and expects 1 file per slide - valid only saves accuracy""")

tf.app.flags.DEFINE_string('mode', '0_softmax',
                            """0_softmax or 1_sigmoid.""")


def main(unused_argv=None):

  #input_path = os.path.join(FLAGS.data_dir, 'test_*')
  input_path = os.path.join(FLAGS.data_dir, FLAGS.ImageSet_basename + '*')
  print(input_path)
  #FLAGS.batch_size = 30
  data_files = tf.gfile.Glob(input_path)
  print(data_files)

  mydict={}
  count_slides = 0

  
  if "test" in FLAGS.TVmode:
    for next_slide in data_files:
      print("New Slide ------------ %d" % (count_slides))
      try:
        labelindex = int(next_slide.split('_')[-1].split('.')[0])
        labelname = 'label_' + str(labelindex)
      except:
        labelindex = 0
        labelname = 'label_0'
      print("label %d: %s" % (labelindex, labelname))

      FLAGS.data_dir = next_slide
      dataset = ImagenetData(subset=FLAGS.subset)
      assert dataset.data_files()
      #try:
      if True:
        precision_at_1, current_score = nc_inception_eval.evaluate(dataset)
        mydict[next_slide] = {}
        mydict[next_slide]['NbrTiles']  = FLAGS.num_examples
        mydict[next_slide][labelname+'_Selected'] = precision_at_1
        mydict[next_slide][labelname+'_Score'] = current_score
        mydict[next_slide]['Read_Class'] = labelname
        print(FLAGS.num_examples)
        count_slides += 1.0
      if False:
      #except Exception as e:
        print("%s FAILED to be processed properly" %next_slide)
        print("Unexpected error:", sys.exc_info()[0])
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    output = open(os.path.join(FLAGS.eval_dir, 'out_All_Stats.txt'), 'ab+')
    pickle.dump(mydict, output)
    output.close()
  elif "valid" in FLAGS.TVmode:
    #FLAGS.data_dir = FLAGS.data_dir + "/valid*"
    FLAGS.data_dir = os.path.join(FLAGS.data_dir, FLAGS.ImageSet_basename + '*')
    dataset = ImagenetData(subset=FLAGS.subset)
    print("Validation mode:")
    print(dataset.data_files())
    assert dataset.data_files()
    nc_inception_eval.evaluate(dataset)
  elif "train" in FLAGS.TVmode:
    FLAGS.data_dir = os.path.join(FLAGS.data_dir, FLAGS.ImageSet_basename + '*')
    dataset = ImagenetData(subset=FLAGS.subset)
    print(dataset.data_files())
    assert dataset.data_files()
    nc_inception_eval.evaluate(dataset)



  # # read data
  # output = open('out_All_Stats.txt', 'rb')
  # mydict = pickle.load(output) 

  """

  
  AllLabels = {}
  AllLabels['normal'] = {}
  AllLabels['luad'] = {}
  AllLabels['lusc'] = {}


  # Summarize the results
  for key, value in sorted(mydict.items()):
    # for each slide, check the label and do an array of values for each label
    current_label = mydict[key]['Read_Class']
    print("current label:" + current_label)
    if 'AllPercentSlidesTP' in AllLabels[current_label].keys():
      AllLabels[current_label]['AllPercentSlidesTP'].append(mydict[key][current_label +'_Selected'])
      AllLabels[current_label]['AllScoreSlidesTP'].append(mydict[key][current_label +'_Score'])
      AllLabels[current_label]['NbrSlides'] += 1.0
    else:
      AllLabels[current_label]['AllPercentSlidesTP'] = [ mydict[key][current_label +'_Selected'] ]
      AllLabels[current_label]['AllScoreSlidesTP'] = [ mydict[key][current_label +'_Score'] ] 
      AllLabels[current_label]['NbrSlides'] = 1.0

  for current_label in AllLabels.keys():
    print(current_label)
    AllLabels[current_label]['TP percent classified above 0.5'] = sum(1 if (x > 0.5) else 0 for x in (AllLabels[current_label]['AllPercentSlidesTP']) )
    AllLabels[current_label]['TP percent classified above 0.5'] = AllLabels[current_label]['TP percent classified above 0.5'] / AllLabels[current_label]['NbrSlides']
    print(current_label + " final scores:")
    print(" * %f %% slides have a majority of slides properly classified (true positive)" % (round( AllLabels[current_label]['TP percent classified above 0.5']*100 , 2)) )		
    AllLabels[current_label]['TP score classified above 0.5'] = sum(1 if (x > 0.5) else 0 for x in (AllLabels[current_label]['AllScoreSlidesTP']) )
    AllLabels[current_label]['TP score classified above 0.5'] = AllLabels[current_label]['TP score classified above 0.5'] / AllLabels[current_label]['NbrSlides']

  """



  """
  # Compute ROC curves for different thresholds

  if len(AllLabels) == 2:
    ROC = {}
    Label_names = list(AllLabels.keys())
    T_Label = Label_names[0]
    F_Label = Label_names[1]
    ROC[(T_Label + ' TPrate')] = {}
    ROC[(T_Label + ' FNrate')] = {}
    ROC[(F_Label + ' TNrate')] = {}
    ROC[(F_Label + ' FPrate')] = {}
    for threshold in np.linspace(0,1,101):
      # True positives and true negatives expressed in percentage
      # Class assigned
      rhist, rbins = np.histogram(AllLabels[T_Label]['AllPercentSlidesTP'], bins=np.linspace(0,1,101))
      cumulH = [sum(rhist)]
      for kk in range(len(rhist)):
        p = cumulH[kk]
        cumulH.append(float(p-rhist[kk]))
      ROC[(T_Label + ' TPrate')]['AllPercentSlides'] = cumulH / sum(rhist)
      ROC[(T_Label + ' FNrate')]['AllPercentSlides'] = 1 - ROC[(T_Label + ' TPrate')]['AllPercentSlides']
      rhist, rbins = np.histogram(AllLabels[F_Label]['AllPercentSlidesTP'], bins=np.linspace(0,1,101))
      cumulH = [sum(rhist)]
      for kk in range(len(rhist)):
        p = cumulH[kk]
        cumulH.append(float(p-rhist[kk]))
      cumulH = cumulH[::-1]
      ROC[(F_Label + ' TNrate')]['AllPercentSlides'] = cumulH / sum(rhist)
      ROC[(F_Label + ' FPrate')]['AllPercentSlides'] = 1 - ROC[(F_Label + ' TNrate')]['AllPercentSlides']


      rhist, rbins = np.histogram(AllLabels[T_Label]['AllScoreSlidesTP'], bins=np.linspace(0,1,101))
      cumulH = [sum(rhist)]
      for kk in range(len(rhist)):
        p = cumulH[kk]
        cumulH.append(float(p-rhist[kk]))
      ROC[(T_Label + ' TPrate')]['AllScoreSlides'] = cumulH / sum(rhist)
      ROC[(T_Label + ' FNrate')]['AllScoreSlides'] = 1 - ROC[(T_Label + ' TPrate')]['AllScoreSlides']
      rhist, rbins = np.histogram(AllLabels[F_Label]['AllScoreSlidesTP'], bins=np.linspace(0,1,101))
      cumulH = [sum(rhist)]
      for kk in range(len(rhist)):
        p = cumulH[kk]
        cumulH.append(float(p-rhist[kk]))
      cumulH = cumulH[::-1]
      ROC[(F_Label + ' TNrate')]['AllScoreSlides'] = cumulH / sum(rhist)
      ROC[(F_Label + ' FPrate')]['AllScoreSlides'] = 1 - ROC[(F_Label + ' TNrate')]['AllScoreSlides']




    ROC['AUC AllPercentSlides'] = 0.
    ROC['AUC AllScoreSlides'] = 0.

    #for threshold in np.linspace(0,1,100):
    for threshold in range(len(ROC[(F_Label + ' FPrate')]['AllPercentSlides'])-1):
      ROC['AUC AllPercentSlides'] += (- ROC[(F_Label + ' FPrate')]['AllPercentSlides'][threshold+1]  \
                                  + ROC[(F_Label + ' FPrate')]['AllPercentSlides'][threshold]) \
                                  * (ROC[(T_Label + ' TPrate')]['AllPercentSlides'][threshold+1]  \
                                  + ROC[(T_Label + ' TPrate')]['AllPercentSlides'][threshold])
      ROC['AUC AllScoreSlides'] += (- ROC[(F_Label + ' FPrate')]['AllScoreSlides'][threshold+1]  \
                                  + ROC[(F_Label + ' FPrate')]['AllScoreSlides'][threshold]) \
                                  * (ROC[(T_Label + ' TPrate')]['AllScoreSlides'][threshold+1]  \
                                  + ROC[(T_Label + ' TPrate')]['AllScoreSlides'][threshold])


    ROC['AUC AllPercentSlides'] = ROC['AUC AllPercentSlides'] /2
    ROC['AUC AllScoreSlides'] = ROC['AUC AllScoreSlides'] /2

    output = open(os.path.join(FLAGS.eval_dir, 'out_All_ROC.txt'), 'ab+')
    pickle.dump(ROC, output)
    output.close()

    #output = open(os.path.join(image_dir, 'out_FPTPrate_PcTiles.txt'),'w')
    output = open(os.path.join(FLAGS.eval_dir, 'out_FPTPrate_PcTiles.txt'),'w')
    for item in range(len(ROC[(F_Label + ' FPrate')]['AllPercentSlides'])):
      output.write("%f\t%f\n" % (ROC[(F_Label + ' FPrate')]['AllPercentSlides'][item], ROC[(T_Label + ' TPrate')]['AllPercentSlides'][item]) )

    output.close()

    output = open(os.path.join(FLAGS.eval_dir, 'out_FPTPrate_ScoreTiles.txt'),'w')
    for item in range(len(ROC[(F_Label + ' FPrate')]['AllScoreSlides'])):
      output.write("%f\t%f\n" % (ROC[(F_Label + ' FPrate')]['AllScoreSlides'][item], ROC[(T_Label + ' TPrate')]['AllScoreSlides'][item]) )

    output.close()


    # # read data
    # output = open('out_All_ROC.txt', 'rb')
    # ROC = pickle.load(output) 

    # Best value; option 1: when TP + (1-FP) is max
    Best_PercSlides = ROC[(T_Label + ' TPrate')]['AllPercentSlides'] + (1-ROC[(F_Label + ' FPrate')]['AllPercentSlides'])
    index_max_PercSlides = np.argmax(Best_PercSlides)
    # Best value; option 2: point closest to (1,0) of the ROC graph
    minIndx_PercSlides = np.argmin(pow((0.0-ROC[(F_Label + ' FPrate')]['AllPercentSlides']),2)+pow((1.0-ROC[(T_Label + ' TPrate')]['AllPercentSlides']),2))


    # Best value; option 1: when TP + (1-FP) is max
    Best_ScoreSlides = ROC[(T_Label + ' TPrate')]['AllScoreSlides'] + (1-ROC[(F_Label + ' FPrate')]['AllScoreSlides'])
    index_max_ScoreSlides = np.argmax(Best_PercSlides)
    # Best value; option 2: point closest to (1,0) of the ROC graph
    minIndx_ScoreSlides = np.argmin(pow((0.0-ROC[(F_Label + ' FPrate')]['AllScoreSlides']),2)+pow((1.0-ROC[(T_Label + ' TPrate')]['AllScoreSlides']),2))




    # plot ROC: FP (x) TP (y)
    gnucnt = open(os.path.join(FLAGS.eval_dir,'out_gnuplot.cnt'), 'w')
    gnucnt.write('set size 1,1\n')
    gnucnt.write('set terminal postscript portrait enhanced\n')
    gnucnt.write('set encoding iso_8859_1\n')
    gnucnt.write('set key autotitle columnhead\n')	
    gnucnt.write('unset key\n')
    gnucnt.write('set output "' + os.path.join(FLAGS.eval_dir, 'out_ROC.eps') + '"\n')
    gnucnt.write('set multiplot layout 2,1\n')
    gnucnt.write('set ylabel "TP(' + T_Label + ')"\n')
    gnucnt.write('set xlabel "FP(' + F_Label + ')"\n')
    gnucnt.write('set yrange [0:1]\n')
    gnucnt.write('set xrange [0:1]\n')
    gnucnt.write('set size square\n')
    gnucnt.write('set title "ROC based on percentage of correctly classified tiles\\n (AUC='+ str(round(ROC['AUC AllPercentSlides'],3)) +';\\n opt thresh='+str(round(np.linspace(0,1,101)[index_max_PercSlides],2))+'; TP='+ str(round(ROC[(T_Label + ' TPrate')]['AllPercentSlides'][index_max_PercSlides],3)) +'; FP='+ str(round(ROC[(F_Label + ' FPrate')]['AllPercentSlides'][index_max_PercSlides],3))+';\\n opt thresh2='+str(round(np.linspace(0,1,101)[minIndx_PercSlides],2))+'; TP='+ str(round(ROC[(T_Label + ' TPrate')]['AllPercentSlides'][minIndx_PercSlides],3)) +'; FP='+ str(round(ROC[(F_Label + ' FPrate')]['AllPercentSlides'][minIndx_PercSlides],3))+')"\n')
    gnucnt.write('plot "'+ os.path.join(FLAGS.eval_dir, 'out_FPTPrate_PcTiles.txt') +'" using 1:2 w l lc 1\n')

    gnucnt.write('set ylabel "TP(' + T_Label + ')"\n')
    gnucnt.write('set xlabel "FP(' + F_Label + ')"\n')
    gnucnt.write('set yrange [0:1]\n')
    gnucnt.write('set xrange [0:1]\n')
    gnucnt.write('set size square\n')
    gnucnt.write('set title "ROC based on average score of tiles\\n (AUC='+ str(round(ROC['AUC AllScoreSlides'],3)) +';\\n opt thresh='+str(round(np.linspace(0,1,101)[index_max_ScoreSlides],2))+'; TP='+ str(round(ROC[(T_Label + ' TPrate')]['AllScoreSlides'][index_max_ScoreSlides],3)) +'; FP='+ str(round(ROC[(F_Label + ' FPrate')]['AllScoreSlides'][index_max_ScoreSlides],3))+';\\n opt thresh2='+str(round(np.linspace(0,1,101)[minIndx_ScoreSlides],2))+'; TP='+ str(round(ROC[(T_Label + ' TPrate')]['AllScoreSlides'][minIndx_ScoreSlides],3)) +'; FP='+ str(round(ROC[(F_Label + ' FPrate')]['AllScoreSlides'][minIndx_ScoreSlides],3))+')"\n')
    gnucnt.write('plot "'+ os.path.join(FLAGS.eval_dir, 'out_FPTPrate_ScoreTiles.txt') +'" using 1:2 w l lc 1')
    gnucnt.close()
  """

  """
  dataset = ImagenetData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  nc_inception_eval.evaluate(dataset)
  """

if __name__ == '__main__':
  tf.app.run()
