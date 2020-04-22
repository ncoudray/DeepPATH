'''
The MIT License (MIT)

Copyright (c) 2020, Nicolas Coudray and Aristotelis Tsirigos

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''
import cv2 as cv
import spams
from PIL import Image
import os
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import csv

def stain_dict_Vahadane(img, thresh=0.8, vlambda=0.10):
        imgLab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        mask = (imgLab[:,:,0] / 255.0) < thresh
        if np.sum(mask==True) == 0 :
            mask = (imgLab[:,:,0] / 255.0) < (thresh+0.1)
            if np.sum(mask==True) == 0 :
                mask = (imgLab[:,:,0] / 255.0) < 1000
        mask = mask.reshape((-1,))
        # RGB to OD
        imgOD = img
        imgOD[(img == 0)] = 1
        imgOD = (-1) * np.log(imgOD / 255)
        imgOD = imgOD.reshape((-1, 3))
        # mask OD
        imgOD = imgOD[mask]
        WisHisHisv = spams.trainDL(imgOD.T, K=2, lambda1=vlambda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
        if WisHisHisv[0, 0] < WisHisHisv[1, 0]:
            WisHisHisv = WisHisHisv[[1, 0], :]
        # normalize rows
        WisHisHisv = WisHisHisv / np.linalg.norm(WisHisHisv, axis=1)[:, None]
        return WisHisHisv



if __name__ == '__main__':
	descr = """
	Apply Vahadane's normalization on list of images. Reference: 
	% @inproceedings{Vahadane2015ISBI,
	% 	Author = {Abhishek Vahadane and Tingying Peng and Shadi Albarqouni and Maximilian Baust and Katja Steiger and Anna Melissa Schlitter and Amit Sethi and Irene Esposito and Nassir Navab},
	% 	Booktitle = {IEEE International Symposium on Biomedical Imaging},
	% 	Title = {Structure-Preserved Color Normalization for Histological Images},
	% 	Year = {2015}}

	"""
	parser = ArgumentParser(description=descr)
	parser.add_argument("--Ref_Norm",
		help="Reference image for normalization with Vahadane method",
		dest='Ref_Norm')
	parser.add_argument("--ImgList",
                help="List of jpg images - First column is source path and name, second is destination path and name",
                dest='ImgList')
	args = parser.parse_args()

	tile = cv.imread(args.Ref_Norm)
	tile = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
	# standardize brightness
	p = np.percentile(tile, 90)
	tile = np.clip(tile * 255.0 / p, 0, 255).astype(np.uint8)
	# get stain dictionnary
	WisHisHisv = stain_dict_Vahadane(tile)
	print(WisHisHisv)
	

	fileIn = open(args.ImgList)
	reader = csv.reader(fileIn, delimiter='\t')
	for line in reader:
		TilePath = line[0]
		NewImageDir = line[1]
		print("TilePath:" + TilePath)
		print("NewImageDir:" + NewImageDir)

		tile = cv.imread(TilePath)
		tile = cv.cvtColor(np.asarray(tile), cv.COLOR_BGR2RGB)
		p = np.percentile(tile, 90)
		if p == 0:
			p = 1.0
		img2t = np.clip(tile * 255.0 / p, 0, 255).astype(np.uint8)
		WisHisHisv2 = stain_dict_Vahadane(img2t)
		# get concentration
		imgOD2 = img2t
		imgOD2[(img2t == 0)] = 1
		imgOD2 = (-1) * np.log(imgOD2 / 255.0)
		imgOD2 = imgOD2.reshape((-1, 3))
		start_values = spams.lasso(imgOD2.T, D=WisHisHisv2.T, mode=2, lambda1=0.01, pos=True).toarray().T
		img_end = (255 * np.exp(-1 * np.dot(start_values, WisHisHisv).reshape(tile.shape))).astype(np.uint8)
		imgout = Image.fromarray(img_end)
		imgout.save(NewImageDir, quality=100)


	fileIn.close()






