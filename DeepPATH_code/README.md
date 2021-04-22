For details and citations about this work, please check:

Nicolas Coudray, Paolo Santiago Ocampo, Theodore Sakellaropoulos, Navneet Narula, Matija Snuderl, David Fenyö, Andre L. Moreira, Narges Razavian, Aristotelis Tsirigos. "Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning". Nature Medicine, 2018; DOI: 10.1038/s41591-018-0177-5

https://www.nature.com/articles/s41591-018-0177-5

This procedure is based on the inception v3 architecture from google. See [Inception v3](https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/inception/README.md) for information about it and the following paper:

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. "Rethinking the Inception Architecture for Computer Vision"
http://arxiv.org/abs/1512.00567

## Preliminary comments
### Overview

**Below is an overall description of our toolchain and is evolving as we move on with our projects.**

If you're more interested in the details of the process published in the paper above, see the description in the "example_TCGA_Lung" foder.**

### Datasets

Images can be obtained from the GDC data portal (https://portal.gdc.cancer.gov/). The easiest way is to:
* download the client from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
* Create and download a manifest and metadata json file from the gdc website for the whole slides images of interest
* Download images using the manifest and the API: gdc-client.exe download -m gdc_manifest.txt

### System requirement

This pipeline is currently developped on the [BigPurple cluster of NYU](https://med.nyu.edu/research/scientific-cores-shared-resources/high-performance-computing-core). See [this link](http://bigpurple-ws.nyumc.org/wiki/index.php/BigPurple_HPC_Cluster) for details (GPU nodes with Tesla V100 GPUs).

Major dependencies are:
- python 3.6.5 
- tensorflow-gpu 1.9.0
- numpy 1.14.3
- matplotlib 2.1.2
- sklearn
- scipy 1.1.0
- openslide-python 1.1.1
- Pillow 5.1.0

### Installation guide

Instructions Clone this repo to your local machine using:
```shell
git clone https://github.com/ncoudray/DeepPATH.git
```
Installation should take just a few seconds.

### Licence on our code
This license only concerns the code fully written by us. 

The MIT License (MIT)

Copyright (c) 2017, Nicolas Coudray, Theodoros Sakellaropoulos, and Aristotelis Tsirigos (NYU)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Other sources
For tiling, we modified the code from [this page](https://github.com/openslide/openslide-python/blob/master/examples/deepzoom/deepzoom_tile.py) from thge Carnegie Mellon University and  while "under the terms of version 2.1 of 

For the conversion from jpg to TFREcord, the training, testing and validation of inception v3, we modified the code from [this page](https://github.com/tensorflow/models/tree/master/research/inception) "Copyright 2016 Google Inc. All Rights Reserved", under the "Licensed under the Apache License, Version 2.0"


### Demo

see the example_TCGA_lung folder and instructions below for detailed list of options, usage and order of processing.

### Advice on folder organization


Preliminary comments:
* For the path, it is advised to always put the full path name and not the relative paths.
* For all the steps below, always submit the jobs via a script (if on BigPurple) and always check the output and error log files are fine. 
 

The overall process is:
* tile the svs images and convert into jpg
* sort the jpg images into train/valid/test at a given magnification and put them in appropriate classes
* convert each of the sets into TFRecord format
* run training and validation
* run testing
* run ROC curve & heatmap

There are several ways to run the pre-processing steps depending on the desired classes:
* if the classes come from the TCGA's metadata file: run the svs tiling step once using all the svs images inside 1 output folder, then run the sorting program (will generate 1 folder per class with the name of the folder being the name of the classes) and TFRecord conversion programs
* if you aim for a multi-output classification (mutations for example): run the svs tiling step once using all the svs images inside 1 output folder, then run the sorting program using option 10 (only 1 output folder generated with all the jpg. They will be sorted into train/valid/test but not assigned any label yet), then run the TFRecord conversion programs for multi-output classification (will assign the label to each tile)
* if a pathologist has selected/labelled regions with Aperio (contours of the ROIs saved in xml format, with label either in the Name or Value fields), then you run the svs tiling step on each class of xml file, run the sorting program using optin 10 on each of them (the output folder will have the same name as the one where each class has been tiled), and then run the TFRecord conversion programs

Advised folder organization (directories that may need to be created in plain, those generated by the programs in bold).
* For classes obtained from json files:

| directories                                     	| Comments           |
| ------------------------------------------------------|--------------------------|
| `images`                                  		| 				|
| `images/Raw/  `                           		| 				|
| `images/Raw/*svs`                         		| original svs images 		|
| `images/Raw/*json `                       		| json file from TCGA database 	|
| `images/<##>pxTiles_<##>Bkg`				| output folder for tiles. Replace ## tile size and background threshold used to run the tiling process` | 
| **`images/<##>pxTiles_<##>Bkg/<slide name>_files/20.0/<X index>_<Y index>.jpeg`**	| Each svs image will have a folder. Inside, there will be as many sub-folders as magnification available and the tiles jpeg images inside | 
| **`images/<##>pxTiles_<##>Bkg/<slide name>_files/10.0/<X index>_<Y index>.jpeg`**	|	|
| `images/01_Cancer_Tumor/				| output folder for sorted tiled	|
| **`images/01_Cancer_Tumor/Solid_Normal_Tissue/<t/v/t set>_<slide name>_<X index>_<Y index>.jpg`**	| folders with class names will be generated. Symbolic links to the tiled jpeg will be created and renamed.` 	|
| **`images/01_Cancer_Tumor/Tumor/<t/v/t set>_<slide name>_<X index>_<Y index>.jpg`**		|	|
| `images/TFRecord_TrainValid/`				| create folder for TFRecord |
| `images/TFRecord_Test/`					|				|
| **`images/TFRecord_TrainValid/train-#####-of-#####`**	| training tiles will be randomly assigned to different shards |
| **`images/TFRecord_TrainValid/valid-#####-of-#####`**	| validation tiles as well |
| **`images/TFRecord_Test/test_<slide name>_<label ID>.TFRecord`**	| For the test set, the tiles associated with a slides will be saved in the same TFRecord file. Check that the <label ID> are correct. |


* For classes obtained from Aperio's selected ROIs, you would have these additional folders:

| Additional directories                   	| Comments           |
| ------------------------------------------------------|--------------------------|
| `images/xml_<label 1>/*xml`		| 	|
| `images/xml_<label 2>/*xml`  		|	|
| `images/<##>pxTiles_<##>Bkg_<label 1>`	| You need as many output folders for the tiling process as input xml classes |
| `images/<##>pxTiles_<##>Bkg_<label 2>`	|	|
| `images/01_label12/`			| For the sorting, you should use the same output folder	|
| **`images/01_label12/<##>pxTiles_<##>Bkg_<label 1>`**			| The sorting program will need to be run for each class	|
| **`images/01_label12/<##>pxTiles_<##>Bkg_<label 2>`**			| using option 10. The name of the folder will be the same of the name of parent folder for each group of tiles	|


# 0 - Prepare the images.

Code in 00_preprocessing. 

All original images must start with the patient ID.

SVS images can be extremely large (+100,000 pixel wide). Optimal input size for inception is 299x299 pixels but the network has been designed to deal with variable image sizes. 

SVS images are first tiled, then sorted according to chosen labels. There will be one folder per label and all the jpg images in the corresponding folder. Also, tiles will be sorted into a train, test and validation set. All tiles generated from the same patient should be assigned to the same set. 

Finally, the jpg images are converted into TFRecords. For the train and validation set, there will be randomly assigned to 1024 and 128 shards respectively. For the test set, there will be 1 TFRecord per slide.




## 0.1 Tile the svs slide images

Example of script to submit this script on a SGE cluster (python 2.7 used):

```shell
#!/bin/tcsh
#$ -pe openmpi 32
#$ -A TensorFlow
#$ -N rqsub_tile
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q
#$ -l excl=true

python /path_to/0b_tileLoop_deepzoom4.py  -s 299 -e 0 -j 32 -B 25 -o <full_path_to_output_folder> "full_path_to_input_slides/*/*svs"  
```
Notes on the different version history:
* /path_to/0b_tileLoop_deepzoom4.py has been updated to deal with xml files having multiple layers, each having a different label. Tiles sharing the same label will be saved in similar sub-directories (name of the sub-directory will be the name of the layer, so it is better if the names are consistent throughout the different xml files, without space and only using alphanumeric characters). Unlike 0b_tileLoop_deepzoom3.py, the label is now expected to be registered in the 'Name' field of the xml's Attributes (and not in the 'Value' field).
* To see the list of images that failed to be tiled (usually because the file is corrupted), search for the work "Failed" in the output log file
* Also 0b_tileLoop_deepzoom4.py should now be working on dcm and jpg files. In this case, the mask can also be jpg instead xml files and "-x" would point to the directory where those masks are saved. Mask must have exactly the same basename as the original images and end in "mask.jpg". An additional "-t" parameter is required to save the temporary converted and renamed files (from dcm to jpg, assuming the folder name is of each dcm set represents the patient ID)
* 0b_tileLoop_deepzoom5.jpg is the next version and should also work with annotations coming either from Aperio or Qupath (after conversion to json). In the path option `-x`, if files extension are 'xml', Aperio format is assumed. If extension is 'json', QuPath format is assumed.



On a slurm cluster (Prince), you may want to try this header instead (and adjust option ```-j``` to ```28```):

```shell
#!/bin/bash
#SBATCH --job-name=rq_tile
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=50GB
#SBATCH --time=47:00:00
#SBATCH --output=rq_00tile_%A_%a.out
#SBATCH --error=rq_00tile_%A_%a.err

module load openslide-python/intel/1.1.1
module load pillow/python3.5/intel/4.2.1 
```

Example on bigpurple (should work with CPU nodes as well):
```shell
#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --job-name=Tile
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --output=rq_tile_%A_%a.out
#SBATCH --error=rq_tile_%A_%a.err
#SBATCH --mem=50GB

module load python/gpu/3.6.5
```


Mandatory parameters:
*  `-s` is tile_size: 299 (299x299 pixel tiles)
*  `-e` is overlap, 0 (no overlap between adjacent tiles). Important: the overlap is defined as "the number of extra pixels to add to each interior edge of a tile". Which means that the final tile size is `s + 2.e`. So to get a 299px tile with a 50% overlap, you need to set s to 149 and e to 75. Also, tile from the edges of the slide will be smaller (since up to two sides have no "interior" edge)
*  `-j` is number of threads: 32 (for a full GPU node on gpu0.q)
*  `-B` is Max Percentage of Background allowed: 25% (tiles removed if background percentage above this value)
*  `-o` is the path were the output images must be saved
*  The final mandatory parameter is the path to all svs images.
Optional parameters when regions have been selected with Aperio:
* `-x` is the path to the xml files. The rootname of the xml file must match exactly the one of the svs images. All the xml files sharing the same label should be in the same folder (named after this label, for example xml_<label>). If there are ROIs with different labels, they should be saved in separate folders and tiles independently in separate output folders (also named after the label, for example <###>pxTiles_<label>)
* `-F` in which field of the xml's Attributes tag are the labels saved, can be 'Name' (default), or 'Value'
* `-m` 1 or 0 if you want to tile the region inside the ROI, or outside (only tested with masks defined in xml files). If `-l=''`, then if will be everything outside all the ROIs, whatever their label. If `-l` is associated with a particular label, it will the inverse mask for that particular label. 
* `-R` minimum percentage of tile covered by ROI. If below the percentage, tile is not kept.
* `-l` To be used with xml file - Only do the tiling for the labels which name contains the characters in this option (string)
* `-S` Set it to true if you want to save ALL masks for ALL tiles (will be saved in same directory with <mask> suffix!!)
* `-M` set to -1 by default to tile the image at all magnifications. Set it to the value of the desired magnification to tile only at that magnification and save space
* `-N` normalize each tile according to the method described in E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, “Color transfer between images”. If normalization is needed, N list the mean and std for each channel in the Lab space. For example \'57,22,-8,20,10,5\' with the first 3 numbers being the targeted means, and then the targeted stds. To check what are the Lab values for a given jpg tile, you can use the `Get_Labstats_From_jpeg.py` script. 


Notes:
* This code can also be used to tile input jpg images: the full path to input images will end in <*jpg">, and you need to set the option `-x` to the `'.jpg'` string value and `-R` to the magnification at which the images were acquired (`20.0` for example)
* known bug: the library used fails to deal with images compressed as JPG 2000. These would lead to empty directories

Output:
* Each slide will have its own folder and inside, one sub-folder per magnification. Inside each magnification folder, tiles are named according to their position within the slide: ```<x>_<y>.jpeg```.
* If the extraction is made from masks defined in xml files, the tiles slides will be saved in folders named after the label of the layer (version 3 of the code only).

## 0.2a Sort the tiles into train/valid/test sets according to the classes defined


Then sort according to cancer type (script example for Phoenix):

```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rqsub_sort
#$ -cwd
#$ -S /bin/tcsh
#$ -q all.q

python /full_path_to/0d_SortTiles.py --SourceFolder=<tiled images path> --JsonFile=<JsonFilePath> --Magnification=<Magnification To copy>  --MagDiffAllowed=<Difference Allowed on Magnification> --SortingOption=<Sorting option> --PercentTest=15 --PercentValid=15 --PatientID=12 --nSplit 0
```

For Prince, the header of the script may be:
```shell
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=sort
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=rq_sort_%A_%a.out
#SBATCH --error=rq_sort_%A_%a.err
#SBATCH --mem=2G

module load numpy/intel/1.13.1
```

For BigPurple, the header of the script should be:
```shell
#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --job-name=Sort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=rq_sort_%A_%a.out
#SBATCH --error=rq_sort_%A_%a.err
#SBATCH --mem=20GB
#SBATCH --time=01:00:00

module load python/gpu/3.6.5
```

*  `--SourceFolder`: output of ``` 00_preprocessing/0b_tileLoop_deepzoom.py```, that is the main folder where the svs images were tiled
*  `--JsonFile`: file uploaded with the svs images and containing all the information regarding each slide (i.e, metadata.cart.2017-03-02T00_36_30.276824.json)
*  `--Magnification`: magnification at which the tiles should be considerted (example: 20)
*  `--MagDiffAllowed`: If the requested magnification does not exist for a given slide, take the nearest existing magnification but only if it is at +/- the amount allowed here(example: 5)
*  `--SortingOption` In the current directory, create one sub-folder per class, and fill each sub-folder with train_, test_ and valid_ test files. Images will be sorted into classes depending on the sorting option:
   - `1` sort according to cancer stage (i, ii, iii or iv) for each cancer separately (classification can be done separately for each cancer)
   - `2` sort according to cancer stage (i, ii, iii or iv) for each cancer  (classification can be done on everything at once)
   - `3` Sort according to type of cancer (LUSC, LUAD, or Nomal Tissue)
   - `4` Sort according to type of cancer (LUSC, LUAD)
   - `5` Sort according to type of cancer / Normal Tissue (2 variables per type)
   - `6` Sort according to cancer / Normal Tissue (2 variables)
   - `7` Random labels (3 labels. Can be used as a false positive control)
   - `8` Sort according to mutational load (High/Low). Must specify --TMB option.
   - `9` Sort according to BRAF mutations for metastatic only. Must specify --TMB option (BRAF mutant for each file).
   - `10` Do not sort. Just create symbolic links to all images in a single label folder and assign images to train/test/valid sets.
   - `11` Sample location (Normal, metastatic, etc...)
   - `12` temp
   - `13` temp
   - `14` Json is actually a text file. First column is ID, second is the labels
   - `15` Copy (not symlink) SVS slides (not jpeg tiles) to new directory if condition#1
   - `16` Copy (not symlink) SVS slides (not jpeg tiles) to new directory if condition#2
   - `17` Sort according to Normal (json file) vs other labels (from TMB text file)
   - `18` temp
   - `19` Slides are tiled in separate sub-folders. It will use  the sub-folders' names as labels

* `--TMB`: addional option 
   - for options 8: path to json file with mutational loads
   - for options 9: path to json file with mutant for metastatic
   - for option 17: text file, second column is label for non-normal tissues
*  `--PercentTest`: percentage of data (tiles/slides or patients depending on `Balance` option) for validation (example: 15); 
*  `--PercentValid` Percentage of data for testing (example: 15). All the other tiles will be used for training by default.
* `PatientID`: Number of digits used to code the patient ID (must be the first digits of the original image names)
* `nSplit`: interger n: Split into train/test in n different ways.  If split is > 0, then the data will be split in train/test only in "# split" non-overlapping ways (each way will have 100/(#split) % of test images). `PercentTest` and `PercentValid` will be ignored. If nSplit=0, then there will be one output split done according to `PercentValid` and `PercentTest`
* (optional) `outFilenameStats`: if an "out_filename_Stats.txt file" is given, check if the tile exists in it an only copy the tile if its value is "true".
* (optional) `expLabel`: Index of the label to sort on within the outFilenameStats file (if only True/False is needed, leave this option empty) - tiles will only be included if there labels is the one predicted (dominant) in the  outFilenameStats file. (should be a string, separated by commas if more than 1 label desired; label 0 is for inception background class; label 1 to n for the user's in alphabetical order)
* (optional) `threshold`: threshold above which the probability the class should be to be considered as true (if not specified, it would be considered as true if it has the max probability); (should be a string, separated by commas if more than 1 label desired)
* (optinal) `Balance`: balance the percentage of tiles in each datasets by: 0-tiles (default); 1-slides; 2-patients (must give PatientID)
* (optional) `outputtype`: Type of output: list source/destination in a file (```File```), do symlink (```Symlink```, default) or both (```Both```)

The output will be generated in the current directory where the program is launched (so start it from a new empty folder). Images will not be copied but a symbolic link will be created toward the <tiled images path>. The links will be renamed ```<type>_<slide_root_name>_<x>_<y>.jpeg``` with <type> being 'train_', 'test_' or 'valid_' followed by the svs name and the tile ID. 


## 0.2b Vahadane normalization
If Reinhard normalization hasn't been used, Vahadane's normalization can tried but needs to be applied after the sorting option  in section 0.2a was applied with `outputtype` set to ```File``` (some issues with the spams library when used in a multiprocessing environment)

To make things faster and submit batches of normalization, the img_list.txt file generated by the previous step can be split in multiple small files. For example:
```shell
split -l 200 img_list.txt splitted_img_list 
```

Then, multiple submissions can be done for each sub-list. Example for a slurm cluster:
```shell
nName='Vahadane_'
for f in  splitted_img_list*; do 
	echo $f
	sbatch --job-name=$nName$f --output=rq_$nName$f_%A.out --error=rq_$nName$f_%A.err sb_Vahadane.sh 'Ref_image.jpeg' $f
done
```

with  ```Ref_image.jpeg``` the reference image for normalization
and sb_Vahadane.sh:

```shell
#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --ntasks=1
#SBATCH --mem=10G

module load python/gpu/3.6.5

python 00_preprocessing/Normalize_Vahadane.py --Ref_Norm $1 --ImgList $2

```


Vahadane's method is described in  Vahadane, Abhishek, et al. "Structure-preserved color normalization for histological images." 2015 IEEE 12th International Symposium on Biomedical Imaging (ISBI). IEEE, 2015.



## 0.3a Convert the JPEG tiles into TFRecord format for 2 or 3 classes jobs

Notes:
* This code was adapted from [awslabs' deeplearning-benchmark code](https://github.com/awslabs/deeplearning-benchmark/blob/master/tensorflow/inception/inception/data/build_image_data.py)


Check code in subfolder 00_preprocessing/TFRecord_2or3_Classes/ if it aimed at classifying 2 or 3 different classes.

For the whole training set, the following code can be to convert JPEG to TFRecord:
```shell
#!/bin/tcsh
#$ -pe openmpi 4
#$ -A TensorFlow
#$ -N rqsub_TFR_trval
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q 

module load cuda/8.0
module load python/3.5.3

python build_image_data.py --directory='jpeg_label_directory' --output_directory='outputfolder' --train_shards=1024  --validation_shards=128 --num_threads=4
```

For Prince, the header of the script may be:

```shell
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=TFR_Vset
#SBATCH --cpus-per-task=1
#SBATCH --output=rq_TFR_%A_%a.out
#SBATCH --error=rq_TFR_%A_%a.err
#SBATCH --mem=20G

module load numpy/intel/1.13.1
module load cuda/8.0.44    
module load tensorflow/python2.7/1.0.1
module load bazel/gnu/0.4.3 

```

For BigPurple, the header can be (should work on CPU nodes too)
```shell
#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --job-name=TFR
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --output=rq_TFR_%A_%a.out
#SBATCH --error=rq_TFR_%A_%a.err
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1

module load python/gpu/3.6.5

```

The jpeg must not be directly inside 'jpeg_label_directory' but in subfolders with names corresponding to the labels (for example as `jpeg_label_directory/TCGA-LUAD/...jpeg` and `jpeg_label_directory/TCGA-LUSC/...jpeg`). The name of those tiles are : `<type>_name_x_y.jpeg` with type being "test", "train" or "valid", name the TCGA name of the slide, x and y the tile coordinates.

optinal parameter: `MaxNbImages`: (default: -1); Maximum number of images in each class - Will be taken randomly among images tiles if >0, otherwise, if -1, all images are taken (may help in unbalanced datasets: undersample oneof the datasets) - if MaxNbImages>number of tiles, data augmentation will be done (rotation, mirroring, leading to possibility to increase dataset up to 8 fold)


The same was done for the test and valid set with this slightly modified script:
```shell
python  build_TF_test.py --directory='jpeg_tile_directory'  --output_directory='output_dir_for_test' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test'

python  build_TF_test.py --directory='jpeg_tile_directory'  --output_directory='output_dir_for_valid' build_TF_test --one_FT_per_Tile=False --ImageSet_basename='valid'

```

Known bug: On many systems, it is better to always use `--num_threads=1`. Corrupted TFRecords can be generated when multi-threading is used.

The difference is that for the train set, the tiles are randomly assigned to the TFRecord files. For the test and validation set, it will created 1 TFRecord file per Slide (solution prefered) - though if `one_FT_per_Tile` is `True`, there will be 1 TFRecord file per Tile created. 

For the training, the option `MaxNbImages` can be used to threshold the maximum number of images in each class. Tiles will be randomly selected if the number of tiles available is higher (may be useful to downsample and balance datasets). If the `MaxNbImages` is at most 8 times larger than the number of tiles available, tiles will be augmented by randomly rotating and/or mirroring them. If it's much higher, tiles will be missing and the program won't be able to generate `MaxNbImages` tiles per class (it will not generate an output, just take all images available - you can double-check in the log files how many tiles are finally done). 



An optional parameter ```--ImageSet_basename='test'``` can be used to run it on 'test' (default), 'valid' or 'train' dataset

Also, by default, it creates 1 TFRecord for all files having the same basename (ignore the last two fields assumed to be the X,Y coordinates of the tile). If you want some other kind of aggregates (useful for dcm), you will use the  `PatientID` arguments to specify the number of characters in the filename that should be used as the basename. 

expected processing time for this step: a few seconds to a few minutes. Once done, check inside the resulting directory that the images have been properly linked.


## 0.3b Convert the JPEG tiles into TFRecord format for a multi-ouput prediction (example mutations)


Check subfolder 00_preprocessing/TFRecord_2or3_Classes/ if it aimed at multi-output classsification with 10 possibly concurent sclasses:

For the training and validation sets:
```shell
#!/bin/tcsh
#$ -pe openmpi 4
#$ -A TensorFlow
#$ -N rqsub_TFR_trval
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q 

python build_image_data_multiClass.py --directory='jpeg_main_directory' --output_directory='outputfolder' --train_shards=1024 --validation_shards=128 --num_threads=4  --labels_names=label_names.txt --labels=labels_files.txt  --PatientID=12
```
* ``` label_names.txt``` is a text file with the 10 possible labels, 1 per line
* ```labels_files.txt``` is a text file listing the mutations present for each patient. 1 patient per line, first column is patient ID (TCGA-38-4632 for example), second is mutation (TP53 for example). If a patient has several mutations, there would be as many lines as mutations for that patient. If a patient has no mutation, you can specify WT in the second column (and make sure WT in not present in the label_names file)
* ```--PatientID``` The file names are expected to start with the patient ID. This value represent the number of digits used for the PatientID
* ```jpeg_main_directory```: in this case the directory must be the unique subfolder where the jpg images are. They must all be within one single folder (not one folder per class). 

Check that the TFRecords are properly created and not empty. 


For the test set:
```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rqsub_sort
#$ -cwd
#$ -S /bin/tcsh
#$ -q all.q

python  build_TF_test_multiClass.py --directory='jpeg_tile_directory'  --output_directory='output_dir' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test' --labels_names=label_names.txt --labels=labels_files.txt  --PatientID=12
```


expected processing time for this step: a few seconds to a few minutes. Check the output log files and the resulting directory (check that the sizes of the created TFRecord files make sense)

#### Note on mutation file:
There are different ways of dealing with mutations. The sigmoid approach was used to identify several mutations than can occur at the same time. In this particular case, the label is associated to each tile during the conversion fro jpeg to TFRecord (otherwise, when using the softmax approach, jpg are classified in different folders and the folder name is used as the label). Thaty's why the label files need to be submitted in the parameters above. 

When working with the TCGA dataset from the GDC Data portal, the mutations can be found by looking for `Data Type == "Masked Somatic Mutations"`. The `Data Category` is "Simple Nucleotide Variation". Filtering based on that, 4 files per cancer type/project will be found (one for each mutation caller). We used mutect for our paper. A gzipped file can be downloaded and inside that there is a (gzipped also) maf file (a maf file is just a tab-separated file with specific columns).The fist column should be the Hugo Symbol and there should also be a column Tumor_Sample_Barcode with the patient/sample id. Silent mutations can also be filtered out if needed.


# 1 - Training
## 1.1 - Training from scratch
### 1.1.a Build the model

Code in the subfolders of 01_training/xClasses - can be used for any type of training.

Build the model from the proper directory, that means from ```cd 01_training/xClasses```:

```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rqs_build
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q
#$ -l excl=true

module load cuda/8.0
module load python/3.5.3
module load bazel/0.4.4


bazel build inception/imagenet_train
```

Note, on the Prince cluster, the header could be something like:
```shell
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=rq_train_%A_%a.out
#SBATCH --error=rq_train_%A_%a.err
#SBATCH --mem=20G
#SBATCH --time=168:00:00

module load numpy/intel/1.13.1
module load cuda/8.0.44    
module load tensorflow/python2.7/1.0.1
module load bazel/gnu/0.4.3 
```


On the bigpurple cluster (Note: you may have to adjust the partition and mem lines depending on your needs!! nodelist is optional but allows you to select which node exactly. Also, large batch sizes can be used [up to 320 tested]):

```shell
#!/bin/bash
#SBATCH --partition=gpu4_long
#SBATCH --job-name=Train
#SBATCH --ntasks=8
#SBATCH --output=rq_train_%A_%a.out
#SBATCH --error=rq_train_%A_%a.err
#SBATCH --mem=50G
#SBATCH --gres=gpu:4

module load python/gpu/3.6.5
module load bazel/0.15.2

```

### 1.1.b Train the model


Once the model is built, run it for all the training images (same header for the submission script):

```shell
bazel-bin/inception/imagenet_train --num_gpus=1 --batch_size=30 --train_dir='output_directory' --data_dir='TFRecord_images_directory' --ClassNumber=3 --mode='0_softmax' 
```

Notes on options and modifications in original inception code:
* The ```mode``` option has been added and must be set to either ```0_softmax``` (original inception - only one ouput label possible) or ```1_sigmoid``` (several output labels possible)
* Other options available:
- ```num_epochs_per_decay```: Epochs after which learning rate decays
- ```learning_rate_decay_factor```: factor of the rate decay
- ```NbrOfImages```: number of images in the training dataset (used for decay: NbrOfImages/batch_size gives the number of iterations for 1 epoch)
- ```max_steps```: number of batches to run
- ```save_step_for_chekcpoint```: frequency at which the checkpoints should be saved (default: 5,000)
* On bigpurple, you can use 4 or 8 GPUs (if available) to make it faster. You will need, in the header, to set ```gres=gpu:4``` or ```gres=gpu:8``` and in the parameters of imagenet_train, set ```num_gpus``` to 4 or 8.
* other notable modifications in the code:
- in ```image_processing.py```: ```image = tf.image.central_crop(image, central_fraction=0.875)``` commented out (winter 2019 version)
- in ```inception_distributed_train.py```: last 100 checkpoints saved instead of last 5



## 1.2 - Transfer learning

See [inception v3 github page](https://github.com/tensorflow/models/tree/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/inception#adjusting-memory-demands) for more details.


Bassically:

Build the model (the following two commands must be run from the proper directory, for example ```cd 01_training/xClasses```):

```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rqs_build
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q
#$ -l excl=true

module load cuda/8.0
module load python/3.5.3
module load bazel/0.4.4

bazel build inception/imagenet_train
```

download the checkpoints of the network trained by google on the 
```shell
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
```

This will create a directory called inception-v3 which contains the following files:
```shell
> ls inception-v3
README.txt
checkpoint
model.ckpt-157585
```
```

Run it for all the training images:
```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rqs_train
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q
#$ -l excl=true

module load cuda/8.0
module load python/3.5.3
module load bazel/0.4.4

bazel-bin/inception/imagenet_train --num_gpus=1 --batch_size=30 --train_dir='output_directory' --data_dir='TFRecord_images_directory' --pretrained_model_checkpoint_path="path_to/model.ckpt-157585" --fine_tune=True --initial_learning_rate=0.001  --ClassNumber=3 --mode='0_softmax'
```

Adjust the input parameters as required. For mode, this can also be '1_sigmoid'.




## 1.3 Validation
### 1.3.a. Validation's Accuracy (not used anymore - go 1.3.b)
Thi script should be run on the validation test set *at the same time* as the training but on a different node (memory issues occur otherwise).


Code is in 02_testing/xClasses/. 


run the job:

```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rqs_Valid
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q
#$ -l excl=true

module load cuda/8.0
module load python/3.5.3

python nc_imagenet_eval.py --checkpoint_dir='full_path_to/0_scratch/' --eval_dir='output_directory' --data_dir="full_path_to/TFRecord_valid/"  --batch_size 30 --ImageSet_basename='valid' --ClassNumber 2 --mode='0_softmax' --run_once --TVmode='valid'
```


Note, on the Prince cluster, the header could be something like:
```shell
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=valid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=150:00:00
#SBATCH --output=rq_valid_%A_%a.out
#SBATCH --error=rq_valid_%A_%a.err
#SBATCH --mem=10G

module load numpy/intel/1.13.1
module load cuda/8.0.44    
module load tensorflow/python2.7/1.0.1
```
On bigpurple, the head can be (adjust partition and mem as needed):

```shell
#!/bin/bash
#SBATCH --partition=gpu4_long
#SBATCH --job-name=Em0valid
#SBATCH --ntasks=4
#SBATCH --output=rq_valid_%A_%a.out
#SBATCH --error=rq_valid_%A_%a.err
#SBATCH --mem=100G
#SBATCH --gres=gpu:2

module load python/gpu/3.6.5
``` 

Replace ClassNumber with the number of classes used and mode by "1_sigmoid" if multi-output classification done (ex for mutations). 

You need to either:
* run it manually once in a while and keep track of the evolution of validation score.
* or run the script without the ```--run_once``` option (the program will run in an infinite loop and will need to be killed manually). To set how often the validation script needs to be run, you need to modify the code: in file ```02_testing/2Classes/inception/nc_inception_eval.py```, line 46, the default value of ```eval_interval_secs``` set to 5 minutes by default (for very long jobs, every 1 or 5 hours may be enough. This has to be changed before compilation with bazel).

Note: The current validation code only saves the validation accuracy, not the loss (saved in an output file named `precision_at_1.txt`). The code still needs to be changed for that. 

### 1.3.b. Validation's AUC
To get the validation (or training AUC), this script can be run at the same time as the training, or after on the saved checkpoints.
To achieve this, you would need to save the validation images in single TFRecord files (1 file per slide, and not random) as in section 0.2.b:

```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rqsub_TFR_test
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q 

module load cuda/8.0
module load python/3.5.3

python  build_TF_test.py --directory='jpeg_tile_directory'  --output_directory='output_dir' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='valid'

```

Note, the option `ImageSet_basename` is set to valid.

For Prince, the submission script header would be: 
```
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=TFRv
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=rq_TFR_%A_%a.out
#SBATCH --error=rq_TFR_%A_%a.err
#SBATCH --mem=30G
#SBATCH --time=48:00:00

module load numpy/intel/1.13.1
module load cuda/8.0.44    
module load tensorflow/python2.7/1.0.1

```



Once this is done, you can run the test/validation script (nc_imagenet_eval.py) BUT, using "valid" for the ```ImageSet_basename``` option and "test" for the ```TVmode``` option.
The following script shows an example on how to run it for all of the checkpoints on bigpurple:

```shell
#!/bin/bash
#SBATCH --partition=gpu4_long
#SBATCH --job-name=Valid
#SBATCH --ntasks=4
#SBATCH --output=rq_valid_%A_%a.out
#SBATCH --error=rq_valid_%A_%a.err
#SBATCH --mem=50G
#SBATCH --gres=gpu:2
### nodelist is optional - only if you want a specific node

export CHECKPOINT_PATH='/gpfs/scratch/coudrn01/NN_test/Embryoscopy/training/01_All/results_00'
export OUTPUT_DIR='/gpfs/scratch/coudrn01/NN_test/Embryoscopy/test/01_All/valid'
export DATA_DIR='/gpfs/scratch/coudrn01/NN_test/Embryoscopy/images/0_TFRecords_valid'
export LABEL_FILE='/gpfs/scratch/coudrn01/NN_test/Embryoscopy/test/01_All/labels.txt'

module load python/gpu/3.6.5


# create temporary directory for checkpoints
mkdir  -p $OUTPUT_DIR/tmp_checkpoints
export CUR_CHECKPOINT=$OUTPUT_DIR/tmp_checkpoints


# check if next checkpoint available
declare -i count=5000 
declare -i step=5000
declare -i NbClasses=2

while true; do
	echo $count
	if [ -f $CHECKPOINT_PATH/model.ckpt-$count.meta ]; then
		echo $CHECKPOINT_PATH/model.ckpt-$count.meta " exists"
		# check if there's already a computation for this checkpoinmt
		export TEST_OUTPUT=$OUTPUT_DIR/test_$count'k'
		if [ ! -d $TEST_OUTPUT ]; then
			mkdir -p $TEST_OUTPUT
			

			ln -s $CHECKPOINT_PATH/*-$count.* $CUR_CHECKPOINT/.
			touch $CUR_CHECKPOINT/checkpoint
			echo 'model_checkpoint_path: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' > $CUR_CHECKPOINT/checkpoint
			echo 'all_model_checkpoint_paths: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' >> $CUR_CHECKPOINT/checkpoint

			# Test
			python /gpfs/scratch/coudrn01/NN_test/code/DeepPATH/DeepPATH_code/02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir=$CUR_CHECKPOINT --eval_dir=$OUTPUT_DIR --data_dir=$DATA_DIR  --batch_size 80  --run_once --ImageSet_basename='valid_' --ClassNumber $NbClasses --mode='0_softmax'  --TVmode='test'
			# wait

			mv $OUTPUT_DIR/out* $TEST_OUTPUT/.

			# ROC
			export OUTFILENAME=$TEST_OUTPUT/out_filename_Stats.txt
			python /gpfs/scratch/coudrn01/NN_test/code/DeepPATH/DeepPATH_code/03_postprocessing/0h_ROC_MultiOutput_BootStrap.py --file_stats=$OUTFILENAME  --output_dir=$TEST_OUTPUT --labels_names=$LABEL_FILE

		else
			echo 'checkpoint '$TEST_OUTPUT' skipped'
		fi

	else
		echo $CHECKPOINT_PATH/model.ckpt-$count.meta " does not exist"
		break
	fi

	# next checkpoint
	count=`expr "$count" + "$step"`
done

# summarize all AUC per slide (average probability) for class 1: 
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c1*  | sed -e 's/k\/out2_roc_data_AvPb_c1/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_1.txt


# summarize all AUC per slide (average probability) for macro average: 
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_macro*  | sed -e 's/k\/out2_roc_data_AvPb_macro_/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_macro.txt


# summarize all AUC per slide (average probability) for micro average: 
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_micro*  | sed -e 's/k\/out2_roc_data_AvPb_micro_/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_micro.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c2*  | sed -e 's/k\/out2_roc_data_AvPb_c2/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_2.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c3*  | sed -e 's/k\/out2_roc_data_AvPb_c3/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_3.txt

```

and the same for Prince cluster:
```shell
#!/bin/tcsh
#SBATCH --gres=gpu:1
#SBATCH --job-name=loop32
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=rq_valid_%A_%a.out
#SBATCH --error=rq_valid_%A_%a.err
#SBATCH --mem=100G

#python -V
# module swap python/intel  python3/intel/3.5.3
# module load tensorflow/python3.5/1.0.1

module load numpy/intel/1.13.1
module load cuda/8.0.44    
module load tensorflow/python2.7/1.0.1
module load scikit-learn/intel/0.18.1

setenv CHECKPOINT_PATH /beegfs/coudrn01/OsmanLab/training/32c_POD/results_32c
setenv OUTPUT_DIR /beegfs/coudrn01/OsmanLab/valid_train/32c_POD/valid_all
setenv DATA_DIR /beegfs/coudrn01/OsmanLab/images/set_3more/32c_TFRecord_valid
setenv LABEL_FILE /beegfs/coudrn01/OsmanLab/valid_train/32c_POD/labels.txt

# create temporary directory for checkpoints
mkdir  -p $OUTPUT_DIR/tmp_checkpoints
setenv CUR_CHECKPOINT $OUTPUT_DIR/tmp_checkpoints


# check if next checkpoint available
@ count = 0 
@ step = 5000

@ DoWait = 1
while ($DoWait == 1)
	echo count
	if ( -f $CHECKPOINT_PATH/model.ckpt-$count.meta ) then
		echo $CHECKPOINT_PATH/model.ckpt-$count.meta " exists"
		# check if there's already a computation for this checkpoinmt
		setenv TEST_OUTPUT  $OUTPUT_DIR/test_$count'k'
		if (! -d $TEST_OUTPUT ) then
			mkdir -p $TEST_OUTPUT
			

			ln -s $CHECKPOINT_PATH/*-$count.* $CUR_CHECKPOINT/.
			touch $CUR_CHECKPOINT/checkpoint
			echo 'model_checkpoint_path: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' > $CUR_CHECKPOINT/checkpoint
			echo 'all_model_checkpoint_paths: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' >> $CUR_CHECKPOINT/checkpoint

			# Test
			python /beegfs/coudrn01/OsmanLab/code/DeepPATH/DeepPATH_code/02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir=$CUR_CHECKPOINT --eval_dir=$OUTPUT_DIR --data_dir=$DATA_DIR  --batch_size 30  --run_once --ImageSet_basename='valid_' --ClassNumber 2 --mode='0_softmax'  --TVmode='test'

			mv $OUTPUT_DIR/out* $TEST_OUTPUT/.

			# ROC
			setenv OUTFILENAME $TEST_OUTPUT/out_filename_Stats.txt
			python /beegfs/coudrn01/OsmanLab/code/DeepPATH/DeepPATH_code/03_postprocessing/0h_ROC_MultiOutput_BootStrap.py --file_stats=$OUTFILENAME  --output_dir=$TEST_OUTPUT --labels_names=$LABEL_FILE

		else
			echo 'checkpoint '$TEST_OUTPUT' skipped'
		endif

	else
		echo $CHECKPOINT_PATH/model.ckpt-$count.meta " does not exist"
		@ DoWait = 0
	endif

	# next checkpoint
	@ count = `expr "$count" + "$step"`
end

# summarize all AUC per slide (average probability) for class 1: 
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c1*  | sed -e 's/k\/out2_roc_data_AvPb_c1/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_1.txt


# summarize all AUC per slide (average probability) for macro average: 
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_macro*  | sed -e 's/k\/out2_roc_data_AvPb_macro_/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_macro.txt


# summarize all AUC per slide (average probability) for micro average: 
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_micro*  | sed -e 's/k\/out2_roc_data_AvPb_micro_/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_micro.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c2*  | sed -e 's/k\/out2_roc_data_AvPb_c1/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_2.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c3*  | sed -e 's/k\/out2_roc_data_AvPb_c1/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_3.txt

```

That script will run the validation script and the compute the ROC curve (code and namings explained in section 3). 



## 1.4 Comments on the code

This is inception v3 developped by google.  Full documentation on (re)-training can be found here: https://github.com/tensorflow/models/tree/master/inception


Main modifications when adjusting the code:
* in slim/inception_model.py: default ```num_classes``` in ```def inception_v3```
* in inception_train.py: default ```max_steps``` in  ```tf.app.flags.DEFINE_integer``` definition
* in imagenet_data.py: 
    * default number of classes in  ```def num_classes(self):```
    * size of the train and validation subsets in ```def num_examples_per_epoch(self)```
* Other changes for multi-output classification: 
    * - in slim/inception_model.py:
        * line 329 changed from ```end_points['predictions'] = tf.nn.softmax(logits, name='predictions')``` to ```end_points['predictions'] = tf.nn.sigmoid(logits, name='predictions')```
    * in slim/losses.py:
        * in ```def cross_entropy_loss``` (line 142 and next ones): ```tf.contrib.nn.deprecated_flipped_softmax_cross_entropy_with_logits``` replaced by  ```tf.contrib.nn.deprecated_flipped_sigmoid_cross_entropy_with_logits```
    * in ```image_processing.py``` (line 378):
        * ```'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1)``` changed to ```'image/class/label': tf.FixedLenFeature([FLAGS.nbr_of_classes+1], dtype=tf.int64, default_value=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])```
        * line 513: replaced ```return images, tf.reshape(label_index_batch, [batch_size])``` with ```return images, tf.reshape(label_index_batch, [batch_size, FLAGS.nbr_of_classes+1])```
    * in ```inception/inception_model.py``` ```sparse_labels = tf.reshape(labels, [batch_size, 1])  [....]  dense_labels = tf.sparse_to_dense(concated,
 [batch_size, num_classes], 1.0, 0.0)``` replaced by ```dense_labels = tf.reshape(labels, [batch_size, FLAGS.nbr_of_classes+1])```



--TVmode='test'



# 2 - Run the classification on the test images

Code in 02_testing/xClass:

Code is the same as the one used for the validation, but with different options: 


```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rq_Test
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q
#$ -l excl=true

module load cuda/8.0
module load python/3.5.3

python nc_imagenet_eval.py --checkpoint_dir='full_path_to/0_scratch/' --eval_dir='output_directory' --data_dir="full_path_to/TFRecord_perSlide_test/"  --batch_size 30 --ImageSet_basename='test_' --run_once --ClassNumber 2 --mode='0_softmax' --TVmode='test'
```

An optional parameter ```--ImageSet_basename='test'``` can be used to run it on 'test' (default), 'valid' or 'train' dataset

data_dir contains the images in TFRecord format, with 1 TFRecord file per slide.

In the eval_dir, it will generate the following files:
*  out_filename_Stats.txt: a text file with output information: <tilename> <True/False classification> [<output probabilities (with 1st one being the inception's background class>] <corrected output probability for the true label - adjusted to ignore the background class> labels: <true label number>. The order of the labels depends on the names of the folders where the images were saved before the conversion to TFRecord (alphabetical order).
*  node2048/: a subfolder where each file correspond to a tile such as the filenames are ```test_<svs name>_<tile ID x>_<tile ID y>.net2048``` and the first line of the file contains: ``` <True / False> \tab [<Background prob> <Prob class 1> <Prob class 2>]  <TP prob>```, and the next 2048 lines correspond to the output of the last-but-one layer


expected processing time for this step: on a gpu, about 1000 tiles per minute.


# 3 - Analyze the outcome

## Generate heat-maps per slides overlaid on original slide (all test slides in a given folder; code not optimized and slow):
code in 03_postprocessing/0f_HeatMap_nClasses.py:

on the Phoenix cluster, the header for the following commands could be:
```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rq_nHeatmap
#$ -cwd
#$ -S /bin/tcsh
#$ -q all.q
#$ -l mem_free=100G

module load python/3.5.3
```

code:
```shell
python 0f_HeatMap_nClasses.py  --image_file 'directory_to_jpeg_classes' --tiles_overlap 0 --output_dir 'result_folder' --tiles_stats 'out_filename_Stats.txt' --resample_factor 10 --slide_filter 'TCGA-05-5425' --filter_tile '' --Cmap 'CancerType' --tiles_size 512
```
* ```slide_filter```: process only images with this basename.
* ```filter_tile```: if map is a mutation, apply cmap of mutations only if tiles are LUAD (```out_filename_Stats.txt``` of Noemal/LUAD/LUSC classification)
* ```Cmap```: ```CancerType``` for Normal/LUAD/LUSC classification, or mutation name
* optiotnal: ```thresholds```: thresholds to use for each label - string, for example: 0.285,0.288,0.628. If none, take the highest one. 

colors are:
black for class 1, red for class 2, blue for class 3, orange for class 4, green for class 5, purple otherwise

## Generate heat-maps with no overlay (fast)

In the output CMap, each tile will be replaced by a single pixel which color is proportional to the probability associated to it in the out_filename_Stats.txt file.
Each class is associated to one of the RGB channels of the image, and the number of classes that can be display on a single heatmap is therefore limited to 3. Use the `Classes` option (string with digits separated by coma) to select which class should be associated to which channel. In the example below, class 3 is associated to channel R, class 1 to channel G and class 2 to chennel B. 


```shell
#!/bin/bash
#SBATCH --partition=fn_long
#SBATCH --job-name=CMap
#SBATCH --ntasks=1
#SBATCH --output=rq_heatmap_0_%A_%a.out
#SBATCH --error=rq_heatmap_0_%A_%a.err
#SBATCH --mem=50G

module load python/gpu/3.6.5


python 03_postprocessing/0g_HeatMap_MultiChannels.py --tiles_overlap=0 --tiles_size=512 --output_dir='CMap_output' --tiles_stats='test_125000k/out_filename_Stats.txt' --Classes='3,1,2' --slide_filter=''
```



## Code for ROC curves:


On the Phoenix  cluster, the header for the following commands could be something like:
```shell
#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rq_Analyze
#$ -cwd
#$ -S /bin/tcsh
#$ -q all.q

module load python/3.5.3
# Note: the scikit-learn seems to work properly only wiyh Native python, so unload 3.5.3 - heatmaps work with python/3.5.3
```

To also get confidence intervals (Bootstrap technique), use this code:

```shell
python 0h_ROC_MultiOutput_BootStrap.py  --file_stats /path_to/out_filename_Stats.txt  --output_dir /path_to/output_folder/ --labels_names /path_to/label_names.txt --ref_stats '' 
```


options:
* ```labels_names```: text file with the names of the labels, 1 per line
* ``` ref_file``` (only with multi-output) could be a out_filename_Stats.txt from a different run and used as a filter (will compute the ROC curve only with tiles labelled as "True" in that second out_filename_Stats.txt - could be usefull for example to select only tiles which are really LUAD within a slide). 
*  ``` ref_label``` number of the label in the ref_file to use for filter
* ```ref_thresh``` threshold to use for ref_label. Use "-1" to use True/False labels instead in the out_filenamestats file. Use "-2" to use label only if they have the max probability
* ```--MultiThresh 0.5```. There are two ways to aggregate the values per slide. One is computing the percentage of tiles "selected", that is, above a threshold. By default, for two classes, a tile is selected for a given class if the probability is above 0.5. That threshold can be changed with this option.


It will generate files starting with "out1" for non aggregated per tile ROC, and files starting with "out2" for per slide aggregated ROC curve. AUC will be show in the filename. 
File names of the outputs:
* start with out1 if the ROC are per tile
* start with out2 if the ROC are per slide
* then contain <AvPb> if the per slide aggregation was done by averaging probabilities
* or <PcSel> if the aggregation was done by computing the percentage of tile selected
* then, the names end with something like
........c1auc_0.6071_CIs_0.6023_0.6121_t0.367.txt
-> c1 (or c2 or c3...) means class 1
-> auc_0.6071. is the AUC for this class (if you have only 2 classes, the curves and AUC should be the same)
* the next two numbers are the CIs
* the last one with the "t" is the "optimal" threshold for this class (computed such as it's the nearest point on the ROC curve to the perfect (1,0) corner).




On the Prince cluster, the header could be something like:
```shell
#!/bin/bash
#SBATCH --job-name=ROC
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=rq_ROC_%A_%a.out
#SBATCH --error=rq_ROC_%A_%a.err
#SBATCH --mem=2G

module load numpy/intel/1.13.1
module load scikit-learn/intel/0.18.1

```

## Code in 03_postprocessing/multiClasses for  probability distributions (mutation analysis):


Generate probability distribution with means for each class for each slide:
```shell
python 0f_ProbHistogram.py --output_dir='result folder' --tiles_stats='out_filename_Statsout_filename_Stats.txt' --ctype='Lung3Classes'
```


