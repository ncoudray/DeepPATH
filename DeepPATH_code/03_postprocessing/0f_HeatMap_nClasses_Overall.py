# Creation of Heat-map from tiles classified with inception v3.

""" 
The MIT License (MIT)

Copyright (c) 2017, Nicolas Coudray and Aristotelis Tsirigos (NYU)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import pickle
import json
import csv
import numpy as np
import glob
import time
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from imageio import imwrite as imsave
from imageio import imread
from PIL import Image

FLAGS = None

# ==========================================
# Centralized Configuration for Project Colors
# ==========================================
PROJECT_RGB_CLASSES = {
    '00_Adjacency': {
        0: [1., 0., 0.], 1: [1., 0.84, 0.], 2: [1., 1., 0.],
        3: [0, 1., 0.], 4: [0., 0., 1.], 5: [0, 0, 0]
    },
    '01_METbrain': {
        0: [0, 0, 0], 1: [255.0/255.0, 176.0/255.0, 0], 2: [254.0/255.0, 97.0/255.0, 0],
        3: [100.0/255.0, 143.0/255.0, 1.0], 4: [120.0/255.0, 94.0/255.0, 240.0/255.0], 5: [0, 0, 0]
    },
    '02_METliver': {
        0: [0.0, 0, 0], 1: [100.0/255.0, 143.0/255.0, 1.0], 2: [120.0/255.0, 94.0/255.0, 240.0/255.0],
        3: [255.0/255.0, 176.0/255.0, 0], 4: [1, 1, 1], 5: [0, 0, 0]
    },
    '03_OSA': {
        0: [0, 0, 0], 1: [0, 1.0, 0], 2: [0, 0.0, 1.0], 3: [1.0, 0, 0.0],
        4: [1, 1, 1], 5: [0, 0, 0]
    },
    '04_HN': {
        0: [1, 0, 0], 1: [0, 0.0, 1.0], 2: [1.0, 1.0, 1.0], 3: [0.0, 1.0, 0],
        4: [186.0/255.0, 85.0/255.0, 211.0/255.0], 5: [0, 0, 0]
    },
    '05_binary': {
        0: [0, 1.0, 1.0], 1: [0.41, 0.0, 0.59], 2: [0, 0, 0],
        3: [0, 0, 0], 4: [0, 0, 0], 5: [0, 0, 0]
    },
    '06_TNBC_6folds': {
        0: [0, 0, 0], 1: [1, 0, 0], 2: [1.0, 165.0/255.0, 0], 3: [0.0, 1.0, 0],
        4: [0, 0.0, 1.0], 5: [1, 1, 0]
    },
    '07_Melanoma_Johannet': {
        0: [0, 0, 0], 1: [0, 0, 1.0], 2: [255.0/255.0, 215.0/255.0, 0],
        3: [0, 0, 0], 4: [0, 0, 0], 5: [0, 0, 0]
    },
    '08_Melanoma_binary': {
        0: [0.41, 0.0, 0.59], 1: [0.98, 1.0, 0.3125], 2: [0, 0, 0],
        3: [0, 0, 0], 4: [0, 0, 0], 5: [0, 0, 0]
    }
}


def make_colormap(seq):
    """Return a LinearSegmentedColormap"""
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def get_project_colormap(project, oClass, num_classes):
    """Returns the appropriate colormap based on project and class."""
    c = mcolors.ColorConverter().to_rgb
    
    if project == '00_Adjacency':
        colors = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'blue'}
        return make_colormap([c('white'), c(colors.get(oClass, 'black'))])
        
    elif project == '01_METbrain':
        colors = {1: 'black', 2: '#FFB000', 3: '#FE6100', 4: 'cornflowerblue', 5: '#785EF0'}
        if oClass in colors:
            return make_colormap([c('white'), c(colors[oClass])])
        return plt.get_cmap('Greens')
        
    elif project == '02_METliver':
        if oClass == 1:
            return make_colormap([c('white'), c('#FFB000')]) if num_classes == 2 else plt.get_cmap('binary')
        elif oClass == 2:
            return make_colormap([c('white'), c('cornflowerblue')])
        elif oClass == 3:
            return make_colormap([c('white'), c('#785EF0')])
        elif oClass == 4:
            return make_colormap([c('white'), c('#FFB000')])
        else:
            return make_colormap([c('white'), c('yellow')])
            
    elif project == '03_OSA':
        if oClass == 1: return plt.get_cmap('binary')
        elif oClass == 2: return make_colormap([c('white'), c('green')])
        elif oClass == 3: return make_colormap([c('white'), c('blue')])
        elif oClass == 4: return plt.get_cmap('Oranges')
        else: return plt.get_cmap('Purples')
        
    elif project == '04_HN':
        colors = {1: 'red', 2: 'blue', 3: 'black', 4: 'green'}
        if oClass in colors:
            return make_colormap([c('white'), c(colors[oClass])])
        return plt.get_cmap('Purples')
        
    elif project == '05_binary':
        if oClass == 1: return make_colormap([c('white'), c('yellow')])
        elif oClass == 2: return make_colormap([c('white'), c('darkviolet')])
        
    elif project == '06_TNBC_6folds':
        if oClass == 1: return plt.get_cmap('binary')
        colors = {2: 'red', 3: 'orange', 4: 'green', 5: 'blue', 6: 'yellow'}
        return make_colormap([c('white'), c(colors.get(oClass, 'black'))])
        
    elif project == '07_Melanoma_Johannet':
        if oClass == 1: return plt.get_cmap('binary')
        elif oClass == 2: return make_colormap([c('white'), c('blue')])
        elif oClass == 3: return make_colormap([c('white'), c('orange')])
        
    elif project == '08_Melanoma_binary':
        if oClass == 1: return make_colormap([c('white'), c('yellow')])
        elif oClass == 2: return make_colormap([c('white'), c('darkviolet')])
        
    # Default fallback
    return make_colormap([c('white'), c('black')])


def dict_tiles_stats():
    stats_dict = {}
    with open(FLAGS.tiles_stats) as f:
        for line in f:
            line2 = line.replace('[','').replace(']','').split()
            if len(line2)>0:    
                tilename = '.'.join(line2[0].split('.')[:-1])
                cTileRootName =  '_'.join(os.path.basename(tilename).split('_')[0:-2])
                if cTileRootName not in stats_dict.keys():
                    stats_dict[cTileRootName] = {}
                    stats_dict[cTileRootName]['tiles'] = {}
                    stats_dict[cTileRootName]['xMax'] = 0
                    stats_dict[cTileRootName]['yMax'] = 0
                
                ixTile = int(os.path.basename(tilename).split('_')[-2])
                iyTile = int(os.path.basename(tilename).split('_')[-1].split('.')[0])
                stats_dict[cTileRootName]['xMax'] = max(stats_dict[cTileRootName]['xMax'], ixTile)
                stats_dict[cTileRootName]['yMax'] = max(stats_dict[cTileRootName]['yMax'], iyTile)
                lineProb = line.split('[')[1]
                lineProb = lineProb.split(']')[0]
                lineProb = lineProb.split()
                stats_dict[cTileRootName]['tiles'][tilename] = [str(ixTile), str(iyTile), lineProb]
    return stats_dict


def get_inference_from_file(lineProb_st):
    lineProb = [float(x) for x in lineProb_st]
    NotaClass = []
    
    if FLAGS.combine != '':
        classesIDstr = FLAGS.combine.split(',')
        classesID = [int(x) for x in classesIDstr]
        classesID = sorted(classesID, reverse = False)
        NotaClass = classesID
        for nCl in classesID[1:]:
            lineProb[classesID[0]] = lineProb[classesID[0]] + lineProb[nCl]
        classesID = sorted(classesID, reverse = True)
        for nCl in classesID[:-1]:
            lineProb[nCl] = 0
    else:
        classesID = []
        
    if FLAGS.Cmap == 'CancerType':
        NumberOfClasses = len(lineProb)
        class_all = []
        sum_class = 0
        for nC in range(1,NumberOfClasses):
            class_all.append(float(lineProb[nC]))
            sum_class = sum_class + float(lineProb[nC])
        for nC in range(NumberOfClasses-1):
            class_all[nC] = class_all[nC] / sum_class
            
        current_score = max(class_all)
        oClass = class_all.index(max(class_all)) + 1
        
        if FLAGS.thresholds is not None:
            thresholds = FLAGS.thresholds
            thresholds = [float(x) for x in thresholds.split(',')]
            if len(thresholds) != len(class_all):
                print("Error: There must be one threshold per class:")
            probDiff = []
            for nC in range(len(class_all)):
                probDiff.append(class_all[nC] - thresholds[nC])
            oClass = probDiff.index(max(probDiff)) + 1
            current_score = class_all[oClass - 1]
            score_correction = thresholds[oClass-1]
        else:
            score_correction = 1.0 / len(class_all)
            
        print("class adjustment:")
        print(oClass)
        print(score_correction)
        print(oClass)
        
        # Determine the map dynamically using the centralized config
        cmap = get_project_colormap(FLAGS.project, oClass, len(class_all))

    class_allC = [class_all[k] for k in range(len(class_all))]
    return oClass, cmap, (current_score-score_correction)/(1.0-score_correction), class_allC


def Get_Binary_stats(bin_im):
    t, b_tmp= cv2.threshold(np.uint8(bin_im)*255,128,255,cv2.THRESH_BINARY)
    a, contours,b = cv2.findContours(b_tmp,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    Each_Tumor_Area = []
    Each_Tumor_Mean_Dia = []
    nIt = 0
    ImBin1 = np.ascontiguousarray(bin_im)
    for eachT in contours:
        nIt += 1
        if len(eachT>2):
            Each_Tumor_Area.append(cv2.contourArea(eachT) * FLAGS.resample_factor * FLAGS.resample_factor)
        else:
            Each_Tumor_Area.append( len(eachT) * FLAGS.resample_factor * FLAGS.resample_factor)
        if len(eachT) >= 5:
            ellipse = cv2.fitEllipse(eachT)
            MinAx = min(ellipse[1]) * FLAGS.resample_factor
            MaxAx = max(ellipse[1]) * FLAGS.resample_factor
            Each_Tumor_Mean_Dia.append( (MinAx + MaxAx) / 2 )
        else:
            Each_Tumor_Mean_Dia.append(np.sqrt( Each_Tumor_Area[-1] / np.pi) )
        M= cv2.moments(eachT)
        if M["m00"] != 0:
            cx= int(M['m10']/M['m00'])
            cy= int(M['m01']/M['m00'])
        else:
            cx = 0
            cy = 0
        ImBin1 = cv2.putText(ImBin1, text = str(nIt), org=(cx, cy),  fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,211,25), thickness=4, lineType=cv2.LINE_AA)
    
    Nb_Tumor = len(Each_Tumor_Area)

    fields = ['Nb_tumors', 'Nb_tumors_500px_Dia_or_more', 'Nb_tumors_1000px_Dia_or_more', 'Nb_tumors_2000px_Dia_or_more', 'Nb_tumors_3000px_Dia_or_more', 'Nb_tumors_4000px_Dia_or_more', 'Nb_tumors_5000px_Dia_or_more', 'List_of_tumor_diameter', 'List_of_tumor_areas']
    rows = [str(Nb_Tumor), str((np.asarray(Each_Tumor_Mean_Dia) > 500).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 1000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 2000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 3000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 4000).sum()), str((np.asarray(Each_Tumor_Mean_Dia) > 5000).sum()), str(Each_Tumor_Mean_Dia), str(Each_Tumor_Area)]

    return fields, rows, sum(Each_Tumor_Area)


def saveMap(HeatMap_divider_p0, HeatMap_0_p, WholeSlide_0, cTileRootName, NewSlide, dir_name, HeatMap_bin_or):
    HeatMap_divider = HeatMap_divider_p0 * 1.0 + 0.0
    HeatMap_0 = HeatMap_0_p
    HeatMap_divider[HeatMap_divider == 0] = 1.0
    HeatMap_0 = np.divide(HeatMap_0, HeatMap_divider[:,:,0:3])
    alpha = 0.33
    out = HeatMap_0 * 255 * (1.0 - alpha) + WholeSlide_0 * alpha
    out = out.transpose((1, 0, 2))
    heatmap_path = os.path.join(FLAGS.output_dir,'heatmaps')
    
    if not os.path.isdir(heatmap_path):
        os.makedirs(heatmap_path)

    filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + ".jpg")

    if NewSlide:
        if os.path.isfile(filename):
            print(filename + " has already been processed in the past. skipped.")
            skip = True
            return skip
        else:
            print(filename + " is processed for the first times.")

    out[out == [0,0,0]] = 255
    imsave(filename,out)

    if (NewSlide == False):
        HeatMap_bin = np.divide(HeatMap_bin_or, HeatMap_divider) 
        ImBin = HeatMap_bin * 0.
        
        if FLAGS.thresholds is not None:
            thresholds = FLAGS.thresholds
            thresholds = [float(x) for x in thresholds.split(',')]
            ImBinT = ImBin
            for kT in range(len(thresholds)):
                ImBinT[:,:,kT] = (HeatMap_bin[:,:,kT] - thresholds[kT]) / (1 - thresholds[kT])
            Tmax = np.max(ImBinT,2)
            for kT in range(len(thresholds)):
                ImBin[:,:,kT] = ImBinT[:,:,kT] == Tmax
        else:
            Tmax = np.max(HeatMap_bin,2)
            ImBin[:,:,0] = HeatMap_bin[:,:,0] == Tmax
            ImBin[:,:,1] = HeatMap_bin[:,:,1] == Tmax
            ImBin[:,:,2] = HeatMap_bin[:,:,2] == Tmax
            ImBin[:,:,3] = HeatMap_bin[:,:,3] == Tmax
            ImBin[:,:,4] = HeatMap_bin[:,:,4] == Tmax
            ImBin[:,:,5] = HeatMap_bin[:,:,5] == Tmax

        # Fetch RGB classes dynamically from configuration dictionary
        class_rgb = PROJECT_RGB_CLASSES.get(FLAGS.project, {i: [0, 0, 0] for i in range(6)})

        cl0 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,0])
        cl1 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,1])
        cl2 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,2])
        cl3 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,3]) 
        cl4 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,4])
        cl5 = sum(ImBin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,5])

        ImBinf = np.zeros([ImBin.shape[0],ImBin.shape[1], 3])
        for rgb in [0,1,2]:
            ImBinf[:,:,rgb] = ImBin[:,:,0] * class_rgb[0][rgb] + ImBin[:,:,1] * class_rgb[1][rgb] + ImBin[:,:,2] * class_rgb[2][rgb] + ImBin[:,:,3] * class_rgb[3][rgb] + ImBin[:,:,4] * class_rgb[4][rgb]  +  ImBin[:,:,5] * class_rgb[5][rgb] 

        ImBinf[HeatMap_divider_p0[:,:,0:3]==0] = 1
        ImBinf = ImBinf.transpose((1, 0, 2))
        ImBinf = ImBinf * 255.

        print("*************")

        filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName +  ".csv")        
        
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if FLAGS.project == '00_Adjacency':
                ClassMatrix = ImBin[:,:,0] + ImBin[:,:,1] * 2 + ImBin[:,:,2] * 3 + ImBin[:,:,3] * 4 +  ImBin[:,:,4] * 5
                AdjMatrix_UpDown = ClassMatrix[:-1,:] + ClassMatrix[1:,:] * 10
                AdjMatrix_RLeft = ClassMatrix[:,:-1] + ClassMatrix[:,1:] * 10
                fields = ['imageName', 'class0_surface','class1_surface','class2_surface', 'class3_surface', 'class4_surface']
                fields_val = [ str(cl0), str(cl1), str(cl2), str(cl3), str(cl4) ]
                for cl1 in range(1,5,1):
                    val = cl1 + cl1 * 10
                    fields.append(str(val))
                    fields_val.append( sum(sum(AdjMatrix_UpDown == val)) + sum(sum(AdjMatrix_RLeft == val)) )
                    for cl2 in range(cl1+1,5,1):
                        val = cl1 + cl2 * 10
                        valinv = cl2 + cl1 * 10
                        fields.append(str(val))
                        fields_val.append( sum(sum(AdjMatrix_UpDown == val)) + sum(sum(AdjMatrix_RLeft == val)) + sum(sum(AdjMatrix_UpDown == valinv)) + sum(sum(AdjMatrix_RLeft == valinv)))
                csvwriter.writerow(fields)
                rows = [cTileRootName]
                rows.extend([str(x) for x in fields_val])
                rows = [rows]
            elif FLAGS.project == '01_METbrain':
                Indx_Tumor = 1
                Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
                ImStat1 = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
                Indx_Tumor = 2
                Avg_Prob_Class2 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
                ImStat2 = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
                Avg_Prob_Class12 = Avg_Prob_Class1 + Avg_Prob_Class2
                fields2, rows2, TumorArea2 = Get_Binary_stats(ImStat1 + ImStat2)
                if (cl1+cl2+cl3) == 0:
                    cl3p1 = 1
                else:
                    cl3p1 = cl1 + cl2 + cl3

                combine_list = FLAGS.combine.split(',')
                if '2' in combine_list and '3' in combine_list:
                    fields = ['imageName', 'Tumor_area','Non_tumor_area','Tumor_percentage', 'Tumor_avg_probability']
                    fields.extend(fields2)
                    csvwriter.writerow(fields)
                    rows = [cTileRootName, str(round(cl1,0)+round(cl2,0)),str(round(cl3,1)),str(round(100*(cl1+cl2)/cl3p1,2)),str(round(Avg_Prob_Class12*100, 2))]
                    rows.extend(rows2)
                    rows = [rows]
                else:
                    fields = ['imageName', 'Intraparaenchymal_area','Leptomeningeal_area','Non_tumor_area','Tumor_percentage', 'Tumor_avg_probability']
                    fields.extend(fields2)
                    csvwriter.writerow(fields)
                    rows = [cTileRootName, str(round(cl1,0)),str(round(cl2,0)),str(round(cl3,1)),str(round(100*(cl1+cl2)/cl3p1,2)),str(round(Avg_Prob_Class12*100, 2))]
                    rows.extend(rows2)
                    rows = [rows]
            elif FLAGS.project == '02_METliver':
                Indx_Tumor = 3
                Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
                ImStat = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
                fields2, rows2, TumorArea2 = Get_Binary_stats(ImStat)
                fields = ['imageName', 'Tumor_area','Non_tumor_area','Tumor_percentage', 'Tumor_avg_probability']
                fields.extend(fields2)
                csvwriter.writerow(fields)
                if (cl3+cl1) == 0:
                    cl3p1 = 1
                else:
                    cl3p1 = cl3 + cl1
                rows = [cTileRootName, str(round(cl3,0)),str(round(cl1,0)), str(round(100*cl3/cl3p1,2)), str(round(Avg_Prob_Class1*100, 2))]
                rows.extend(rows2)
                rows = [rows]
            elif FLAGS.project == '03_OSA':
                fields = ['imageName', 'Necrotic tumor','Normal tissue','Viable Tumor']
                csvwriter.writerow(fields)
                rows = [[cTileRootName, str(round(cl1,1)),str(round(cl2,1)),str(round(cl3,1))]]
            elif FLAGS.project == '04_HN':
                fields = ['imageName', 'Invasive_scc','Normal epidermus','SCCIS']
                csvwriter.writerow(fields)
                rows = [[cTileRootName, str(round(cl0,1)),str(round(cl1,1)),str(round(cl3,1))]]
            elif FLAGS.project == '05_binary':
                fields = ['imageName','Normal tissue or class 1','Tumor or class2']
                csvwriter.writerow(fields)
                rows = [[cTileRootName, str(round(cl0,1)),str(round(cl1,1))]]
            elif FLAGS.project == '06_TNBC_6folds':
                fields = ['imageName','Art','DCIS','Inv','Nec','Other','Str']
                csvwriter.writerow(fields)
                rows = [[cTileRootName, str(round(cl0,1)),str(round(cl1,1)), str(round(cl2,1)), str(round(cl3,1)), str(round(cl4,1)), str(round(cl5,1))]]
            elif FLAGS.project == '07_Melanoma_Johannet':
                fields = ['imageName','Tumor','lymphocyte-rich','other']
                csvwriter.writerow(fields)
                rows = [[cTileRootName, str(round(cl2,1)),str(round(cl1,1)), str(round(cl0,1))]]
            elif FLAGS.project == '08_Melanoma_binary':
                Indx_Tumor = 0
                Avg_Prob_Class1 = np.sum(HeatMap_bin[(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0,Indx_Tumor])/np.sum(HeatMap_0[:,:,1]>0.0)
                ImStat = np.multiply(np.array(ImBin[:,:,Indx_Tumor]), np.array(HeatMap_divider_p0[:,:,1] * 1.0 + 0.0)>0)
                fields2, rows2, TumorArea2 = Get_Binary_stats(ImStat)
                fields = ['imageName','Tumor_area','Non_tumor_area','tumor_percentage','Tumor_avg_probability']
                fields.extend(fields2)
                csvwriter.writerow(fields)
                if (cl0+cl1) == 0:
                    cl0p1 = 1
                else:
                    cl0p1 = cl0 + cl1
                rows = [cTileRootName, str(round(cl1,0)),str(round(cl0,0)), str(round(100*cl1/cl0p1,2)), str(round(Avg_Prob_Class1*100, 2))]
                rows.extend(rows2)
                rows = [rows]

            csvwriter.writerows(rows)        

        filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_segmented.jpg")
        imsave(filename,ImBinf * 255.)

        filename_tmp = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_" + "unknown"  + ".jpg")
        print(filename_tmp)
        if os.path.exists(filename_tmp):
            os.remove(filename_tmp)
        filename = os.path.join(heatmap_path,"heatmap_" + FLAGS.Cmap + "_" + cTileRootName + "_slide.jpg")
        WholeSlide_0[HeatMap_divider_p0[:,:,0:3]==0] = 255
        imsave(filename,np.swapaxes(WholeSlide_0,0,1))
    skip = False
    return skip


def main():
    image_dir = FLAGS.image_file
    if os.path.isdir(FLAGS.output_dir):
        if len(os.listdir(FLAGS.output_dir)) > 0:
            print("WARNING: output folder is not empty")    
    else:
        sys.exit("output path not defined or does not exist")

    stats_dict = dict_tiles_stats()

    sub_dirs = []
    if os.path.isdir(image_dir):
        for item in os.listdir(image_dir):
            if os.path.isdir(os.path.join(image_dir, item)):
                sub_dirs.append(os.path.join(image_dir,item))

    print("sub_dirs:")
    print(sub_dirs)
    SlideRootName = ''
    SlideNames = []
    skip = False

    filtered_dict = {}
    for k in stats_dict.keys():
        if FLAGS.slide_filter in k:
            filtered_dict[k] =stats_dict[k]

    dir_name = 'unknown'
    for slide in sorted(filtered_dict.keys()):
        NewSlide = True
        t = time.time()
        ixTile = int(stats_dict[slide]['xMax'])
        iyTile = int(stats_dict[slide]['yMax'])
        if FLAGS.project == '00_Adjacency':
            FLAGS.tiles_size = 1
            FLAGS.tiles_overlap = 0
            FLAGS.resample_factor = 0
        req_xLength =  (ixTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap) + FLAGS.tiles_size
        req_yLength =  (iyTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap) + FLAGS.tiles_size
        if FLAGS.resample_factor > 0:
            req_xLength = int(req_xLength / FLAGS.resample_factor + 1)
            req_yLength = int(req_yLength / FLAGS.resample_factor + 1)
        WholeSlide_0 = np.zeros([req_xLength, req_yLength, 3])
        HeatMap_0 = np.zeros([req_xLength, req_yLength, 3])
        HeatMap_bin = np.zeros([req_xLength, req_yLength, 6])
        HeatMap_divider = np.zeros([req_xLength, req_yLength, 6])
        print("Checking slide " + slide)
        print(req_xLength, req_yLength)
        skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, slide, NewSlide, dir_name, HeatMap_bin)
        if skip:
            print("slide done --")
            continue
        cc = 0
        for tile in stats_dict[slide]['tiles'].keys():
            cc+=1
            extensions = ['.jpeg', '.jpg']
            isError = True
            dir_name = 'unknownTMP'
            if FLAGS.project == '00_Adjacency':
                im2 = np.zeros([1, 1, 3])
                isError = False
                dir_name_old = dir_name
                dir_name = 'Adj'
            else:
                for extension in extensions:
                    for sub_dir in list(sub_dirs):
                        try:
                            test_filename = os.path.join(sub_dir, tile + extension)
                            im2 = imread(test_filename)
                            dir_name_old = dir_name
                            dir_name = os.path.basename(sub_dir)
                            isError = False
                        except:
                            isError = True
                        if isError == False:
                            break
                    if isError == False:
                        break
            if isError == True:
                print("image not found:" + tile)
                continue
            cTileRootName = slide
            ixTile = int(stats_dict[slide]['tiles'][tile][0])
            iyTile = int(stats_dict[slide]['tiles'][tile][1])
            rTile = im2.shape[1]
            cTile = im2.shape[0]
            xTile =  (ixTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap)
            yTile =  (iyTile) * (FLAGS.tiles_size - FLAGS.tiles_overlap)
            req_xLength = xTile + rTile
            req_yLength = yTile + cTile
            if rTile!= cTile:
                continue
            if FLAGS.resample_factor > 0:
                rTile = int(round(float(rTile) / FLAGS.resample_factor, 0))
                cTile = int(round(float(cTile) / FLAGS.resample_factor, 0))
                if rTile<=0:
                    im2s = im2
                elif cTile<=0:
                    im2s = im2
                else:
                    im2s = np.array(Image.fromarray(im2).resize((cTile, rTile)))
                    rTile = im2s.shape[1]
                    cTile = im2s.shape[0]
                    xTile = int(round(float(xTile) / FLAGS.resample_factor, 0))
                    yTile = int(round(float(yTile) / FLAGS.resample_factor, 0))
                    req_xLength = xTile + rTile
                    req_yLength = yTile + cTile
            else:
                im2s = im2

            oClass, cmap, current_score, class_prob = get_inference_from_file(stats_dict[slide]['tiles'][tile][2])
            if current_score < 0:
                print("No probability found")
            else:
                WholeSlide_0[xTile:req_xLength, yTile:req_yLength,:] = np.swapaxes(im2s,0,1)
                heattile = np.ones([req_xLength-xTile,req_yLength-yTile]) * current_score
                heattile = cmap(heattile)
                heattile = heattile[:,:,0:3]
                HeatMap_0[xTile:req_xLength, yTile:req_yLength,:] = HeatMap_0[xTile:req_xLength, yTile:req_yLength,:] + heattile
                HeatMap_divider[xTile:req_xLength, yTile:req_yLength,:] = HeatMap_divider[xTile:req_xLength, yTile:req_yLength,:] + 1
                for kC in range(len(class_prob)):
                    HeatMap_bin[xTile:req_xLength, yTile:req_yLength,kC] = HeatMap_bin[xTile:req_xLength, yTile:req_yLength,kC] + np.ones([req_xLength-xTile,req_yLength-yTile]) * class_prob[kC]
            if cc % 1000 == 0: 
                print("tile time (sec): " + str((time.time() - t) / cc))
                
        NewSlide = False
        skip = saveMap(HeatMap_divider, HeatMap_0, WholeSlide_0, slide, NewSlide, dir_name, HeatMap_bin)
        print("slide time (min): " + str((time.time() - t)/60)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--tiles_overlap',
        type=int,
        default=0,
        help='Overlap of the tiles in pixels.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='mustbedefined',
        help='Output directory.'
    )
    parser.add_argument(
        '--resample_factor',
        type=float,
        default=1,
        help='reduce the size of the output by this factor.'
    )
    parser.add_argument(
        '--tiles_size',
        type=int,
        default=512,
        help='tile size in pixels.'
    )
    parser.add_argument(
        '--tiles_stats',
        type=str,
        default='',
        help='text file where tile statistics are saved.'
    )
    parser.add_argument(
        '--slide_filter',
        type=str,
        default='',
        help='process only images with this basename.'
    )
    parser.add_argument(
        '--filter_tile',
        type=str,
        default='/OurSoftware/TCGA-05-5425/tmp_class_30perTF/out_filename_Stats.txt',
        help='if map is a mutation, apply cmap of mutations only if tiles are LUAD.'
    )
    parser.add_argument(
        '--Cmap',
        type=str,
        default='CancerType',
        help='can be CancerType, of the name of a mutation (TP53, EGFR...)'
    )
    parser.add_argument(
        '--thresholds',
        type=str,
        default=None,
        help='thresholds to use for each label - string, for example: 0.285,0.288,0.628. If none, take the highest one.'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='01_METbrain',
        help='Project name (will define the number of classes and colors assigned). Can be: 00_Adjacency, 01_METbrain, 02_METliver, 03_OSA, 04_HN, 05_binary, 06_TNBC_6folds, 07_Melanoma_Johannet, 08_Melanoma_binary.'
    )
    parser.add_argument(
        '--combine',
        type=str,
        default='',
        help='combine classes (sum of the probabilities); comma separated string (2,3). Class ID starts at 1'
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.tiles_size = FLAGS.tiles_size + 2 * FLAGS.tiles_overlap
    FLAGS.tiles_overlap = 2 * FLAGS.tiles_overlap
    main()
