'''
    File name: nc
    Modified by: Nicolas Coudray
    Date created: March/2017
    Python Version: 2.7 (native on the cluster)

	Objective:
	Starting with tiles images, select images from a given magnification and order them according the the stage of the cancer, or type of cancer, etc...

	Usage:
		SourceFolder is the folder where all the svs images have been tiled . :
		It encloses:
		  * 1 subfolder per image, (the name of the subfolder being "imagename_files")
		  * each contains 14-17 subfolders which name is the magnification of the tiles and contains the tiles
		It should not enclose any other folder
		The output folder from which the script is run should be empty

'''
import json
from glob import glob
import os
import sys
import random
import numpy as np

if __name__ == '__main__':
	## Initialization
	# SourceFolder = '/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/512pxTiled'
	# JsonFile = '/ifs/home/coudrn01/NN/Lung/RawImages/metadata.cart.2017-03-02T00_36_30.276824.json'
	# Magnification = 20
	# MagDiffAllowed = 5
	# SortOption = 3
	# PercentValid = 15
	# PercentTest = 15

	if len(sys.argv) != 8:
		print('Usage: %prog <tiled images path> <JsonFilePath> <Magnification To copy> <Difference Allowed on Magnification> <Sorting option>  <percentage of images for validation> <Percentage of imaages for testing>')
		print("Example: python 0d_SortTiles_stage.py '/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/512pxTiled' '/ifs/home/coudrn01/NN/Lung/RawImages/metadata.cart.2017-03-02T00_36_30.276824.json' 20 5 3 15 15")
		print("     The images are expected to be in folders in this directory: '/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/512pxTiled'")
		print("     Each images should have its own folder with the svs image name followed by '_files'")
		print("     Each images should have subfolders with names corresponding to the magnification associated with the jpeg tiles saved inside it")
		print("     The sorting will be done using tiles corresponding to a magnification of 20 (+/- 5 if the 20 folder does not exist)")
		print("     15%% will be put for validation, 15%% for testing and the leftover for training")
		print("     linked images' names will start with 'train_', 'test_' or 'valid_' followed by the svs name and the tile ID")
		print("     Sorting options are: ")
		print("        1. sort according to cancer stage (i, ii, iii or iv) for each cancer separately (classification can be done separately for each cancer)")
		print("        2. sort according to cancer stage (i, ii, iii or iv) for each cancer  (classification can be done on everything at once)")
		print("        3. sort according to type of cancer (LUSC, LUAD, or Nomal Tissue)")
		print("        4. sort according to type of cancer (LUSC, LUAD)")
		print("        5. sort according to type of cancer / Normal Tissue (2 variables per type)")
		print("        6. sort according to cancer / Normal Tissue (2 variables)")
		print("        7. Random labels (3 variables for false positive control)")
		sys.exit()

	SourceFolder = str(sys.argv[1])
	JsonFile = str(sys.argv[2])
	Magnification = float(sys.argv[3])
	MagDiffAllowed = float(sys.argv[4])
	SortOption = int(sys.argv[5])
	PercentValid = float(sys.argv[6])
	PercentTest = float(sys.argv[7])




		# 1: stage only for each type of cancer separately
		# 2: stage and type of cancer in the same folder
		# 3 type of cancer only
	# if the required mag is not available, take the closest one available if it is with MagDiffAllowed. 
	## end Initialization



	SourceFolder = os.path.join(SourceFolder, "*_files")
	imgFolders = glob(SourceFolder)  
	jdata = json.loads(open (JsonFile).read())
	NbrTilesCateg = {}
	PercentTilesCateg = {}
	NbrImagesCateg = {}
	coMatrix = np.zeros((3,3))

	print("******************")
	print(imgFolders)
	for cFolderName in imgFolders:
		print("**************** starting %s" % cFolderName)
		SourceImgPath = os.path.join(SourceFolder, cFolderName)
		# imgRootName = 'TCGA-44-A479-01A-03-TSC.EE0DF4FB-5D66-40BF-A931-0363C7B5C559'
		imgRootName = os.path.basename(cFolderName)
		imgRootName = imgRootName.rstrip('_files')
		
	

		# for a given slide, check in the json file its parameters
		ID = -1
		for TotImg in range(len(jdata)):
			if jdata[TotImg]['file_name'].startswith(imgRootName):
				stage = jdata[TotImg]['cases'][0]['diagnoses'][0]['tumor_stage']
				cancer = jdata[TotImg]['cases'][0]['project']['project_id']
				ID = TotImg
				sample_type = jdata[TotImg]['cases'][0]['samples'][0]['sample_type']

		if ID==-1:
			print("File name not found in json file.")
			continue
			# Notes on these parameters:
				# cancer is TCGA-LUAD or TCGA-LUSC
				# Primary Tumor
				# Recurrent Tumor
				# Solid Tissue Normal	--> benign tissue



		# do not differentiate between a and b stages, just i, ii, iii and iv:
		stage = stage.replace(" ", "_")
		stage = stage.rstrip('a')
		stage = stage.rstrip('b')
		# Use a different class for the images representing normal adjacent tissues
		if SortOption == 1:
			# classificy according to cancer stage for each type of cancer separately
			if sample_type.find("Normal") > -1:
				IsHealthy = True
				stage = sample_type.replace(" ", "_")
			else:
				IsHealthy = False

			# create folders if needed
			if not os.path.exists(cancer):
				os.makedirs(cancer)
			SubDir = os.path.join(cancer, stage)
		elif SortOption == 2:
			# classificy according to cancer stage + type of cancer in the same folder
			SubDir = cancer + "_" +  stage
			if sample_type.find("Normal") > -1:
				IsHealthy = True
				SubDir = sample_type.replace(" ", "_")
			else:
				IsHealthy = False

		elif SortOption == 3:
			# Classify according to type of cancer or Normal tissue (3 classes)
			SubDir = cancer
			if sample_type.find("Normal") > -1:
				IsHealthy = True
				SubDir = sample_type.replace(" ", "_")
			else:
				IsHealthy = False
		elif SortOption == 4:
			# Classify according to type of cancer (2 classes); ignore normal tissue slides
			SubDir = cancer
			if sample_type.find("Normal") > -1:
				IsHealthy = True
				continue
			else:
				IsHealthy = False
		elif SortOption == 5:
			# Classify according to type of cancer vs  healthy (LUAD vs Normal or LUSC vs Normal)
			SubDir = cancer
			if sample_type.find("Normal") > -1:
				IsHealthy = True
				SubDir = os.path.join(cancer, sample_type.replace(" ", "_"))
			else:
				IsHealthy = False
				SubDir = os.path.join(cancer, cancer)
		elif SortOption == 6:
			# Classify according to type of cancer or Normal tissue (2 classes)
			SubDir = cancer
			if sample_type.find("Normal") > -1:
				IsHealthy = True
				SubDir = sample_type.replace(" ", "_")
			else:
				SubDir = "cancer"
				IsHealthy = False
		elif SortOption == 7:
			# Random classification
			# First, check the real label:
			SubDir_temp = cancer
			if sample_type.find("Normal") > -1:
				IsHealthy = True
				SubDir_temp = sample_type.replace(" ", "_")
			else:
				IsHealthy = False
			# Assign a random label
			AllOptions= ['TCGA-LUAD', 'TCGA-LUSC', 'Solid_Tissue_Normal']
			SubDir = AllOptions[random.randint(0,2)]
			coMatrix[AllOptions.index(SubDir_temp)][random.randint(0,2)] += 1
		else:
			sys.exit('Error: Option unknown')
					
		# create folders if needed
		if not os.path.exists(SubDir):
			os.makedirs(SubDir)
			
		if SubDir in NbrTilesCateg.keys():
			print SubDir + " already in dictionary"
		else:
			print SubDir + " not yet in dictionary"
			NbrTilesCateg[SubDir] = 0
			NbrTilesCateg[SubDir + "_train"] = 0
			NbrTilesCateg[SubDir + "_test"] = 0
			NbrTilesCateg[SubDir + "_valid"] = 0
			PercentTilesCateg[SubDir + "_train"] = 0
			PercentTilesCateg[SubDir + "_test"] = 0
			PercentTilesCateg[SubDir + "_valid"] = 0
			NbrImagesCateg[SubDir + "_train"] = 0
			NbrImagesCateg[SubDir + "_test"] = 0
			NbrImagesCateg[SubDir + "_valid"] = 0



		# Check in the reference directories if there is a set of tiles at the desired magnification
		AvailMagsDir = os.listdir(SourceImgPath)
		AvailMags = tuple(float(x) for x in AvailMagsDir)
		# check if the mag was known for that slide
		if max(AvailMags) < 0: 
			print("Magnification was not known for that file.")
			continue
		Mismatch = tuple(abs(x-Magnification) for x in AvailMags)
		if(min(Mismatch) <= MagDiffAllowed):
			 AvailMagsDir = AvailMagsDir[int(Mismatch.index(min(Mismatch)))]
		else: 	
			# No Tiles at the mag within the allowed range
			print("No Tiles found at the mag within the allowed range.")
			continue

		# Copy/symbolic link the images into the appropriate folder-type
		SourceImageDir = os.path.join(SourceImgPath, AvailMagsDir,"*")
		AllTiles = glob(SourceImageDir)
		NbTiles = 0
		for TilePath in AllTiles:
			TileName = os.path.basename(TilePath)
			# rename the images with the root name, and put them in train/test/valid 
			if NbrTilesCateg.get(SubDir) == 0:
				ttv = "train"
			elif PercentTilesCateg.get(SubDir + "_test") < PercentTest:
				ttv = "test"
			elif PercentTilesCateg.get(SubDir + "_valid") < PercentTest:
				ttv = "valid"
			else:
				ttv = "train"

			NewImageDir = os.path.join(SubDir, ("%s_%s_%s" %(ttv, imgRootName, TileName)))
			os.symlink(TilePath, NewImageDir)
			NbTiles += 1

		# update stats 
		NbrTilesCateg[SubDir] = NbrTilesCateg.get(SubDir) + NbTiles
		if ttv == "train":
			NbrTilesCateg[SubDir + "_train"] = NbrTilesCateg.get(SubDir + "_train") + NbTiles
			NbrImagesCateg[SubDir + "_train"] = NbrImagesCateg[SubDir + "_train"] + 1
		elif ttv == "test":
			NbrTilesCateg[SubDir + "_test"] = NbrTilesCateg.get(SubDir + "_test") + NbTiles
			NbrImagesCateg[SubDir + "_test"] = NbrImagesCateg[SubDir + "_test"] + 1
		elif ttv == "valid":
			NbrTilesCateg[SubDir + "_valid"] =  NbrTilesCateg.get(SubDir + "_valid") + NbTiles
			NbrImagesCateg[SubDir + "_valid"] = NbrImagesCateg[SubDir + "_valid"] + 1

		PercentTilesCateg[SubDir + "_train"] = float(NbrTilesCateg.get(SubDir + "_train")) / float(NbrTilesCateg.get(SubDir)) * 100.0
		PercentTilesCateg[SubDir + "_test"] = float(NbrTilesCateg.get(SubDir + "_test")) / float(NbrTilesCateg.get(SubDir)) * 100.0
		PercentTilesCateg[SubDir + "_valid"] = float(NbrTilesCateg.get(SubDir + "_valid")) / float(NbrTilesCateg.get(SubDir)) * 100.0


		print("Done. %d tiles linked to %s " % ( NbTiles, SubDir ) )
		print("Train / Test / Validation sets for %s = %f %%  / %f %% / %f %%" % (SubDir, PercentTilesCateg.get(SubDir + "_train"), PercentTilesCateg.get(SubDir + "_test"), PercentTilesCateg.get(SubDir + "_valid") ) )


	print("CoMatrix")
	print(coMatrix)
	for k, v in sorted(NbrTilesCateg.iteritems()):
		    print k, v
	for k, v in sorted(PercentTilesCateg.iteritems()):
		    print k, v
	for k, v in sorted(NbrImagesCateg.iteritems()):
		    print k, v

