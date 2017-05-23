
Preliminary comment: On the NYUm HPC cluster, 3 modules are needed. The commands mentioned below must be run through qsub scripts and not on the head node! Module needed are:
module load cuda/8.0
module load python/3.5.3
module load bazel/0.4.4


# 0 - Prepare the images.

Code in 00_preprocessing.
See from https://github.com/tensorflow/models/blob/master/inception/README.md for specific format needed by inception


## 0.1 Tile the svs slide images

This step also required ```module load openjpeg/2.1.1```.
```shell
python 00_preprocessing/0b_tileLoop_deepzoom.py <svs images path> <tile_size> <overlap> <number of processes> <number of threads> <Max Percentage of Background>')
```
Example of parameters:
*  <svs images path> example: "/ifs/home/kerbosID/NN/Lung/RawImages/*/*svs"
*  <tile_size> 512 (512x512 pixel tiles)
*  <overlap> 0 (no overlap between adjacent tiles)
*  <number of processes> 4 
*  <number of threads> 10
*  <Max Percentage of Background> 30 (tiles removed if background percentage above this value)

The output will be generated in the current directory where the program is launched (so start it from a new empty folder).
Each slide will have its own folder and inside, one sub-folder per magnification. Inside each magnification folder, tiles are named according to their index within the slide: ```<x>_<y>.jpeg```.


## 0.2 Sort the tiles into train/valid/test sets according to the classes defined


# Then sort according to cancer type:
```shell
python 00_preprocessing/0d_SortTiles_stage.py <tiled images path> <JsonFilePath> <Magnification To copy> <Difference Allowed on Magnification> <Sorting option>  <percentage of images for validation> <Percentage of imaages for testing>
```
*  <tiled images path> output of ``` 00_preprocessing/0b_tileLoop_deepzoom.py```, that is the main folder where the svs images were tiled
*  <JsonFilePath> file uploaded with the svs images and containing all the information regarding each slide (i.e, metadata.cart.2017-03-02T00_36_30.276824.json)
*  <Magnification To copy> magnification at which the tiles should be considerted (example: 20)
*  <Difference Allowed on Magnification> If the requested magnification does not exist for a given slide, take the nearest existing magnification but only if it is at +/- the amount allowed here(example: 5)
*  <Sorting option> In the current directory, create one sub-folder per class, and fill each sub-folder with train_, test_ and valid_ test files. Images will be sorted into classes depending on the sorting option:
**  1. sort according to cancer stage (i, ii, iii or iv) for each cancer separately (classification can be done separately for each cancer)
**  2.sort according to cancer stage (i, ii, iii or iv) for each cancer  (classification can be done on everything at once)
**  3. sort according to type of cancer (LUSC, LUAD, or Nomal Tissue)
**  4. sort according to type of cancer (LUSC, LUAD)
**  5. sort according to type of cancer / Normal Tissue (2 variables per type)
**  6. sort according to cancer / Normal Tissue (2 variables)
*  <percentage of images for validation> (example: 15); 
*  <Percentage of images for testing> (example: 15). All the other tiles will be used for training by default.

The output will be generated in the current directory where the program is launched (so start it from a new empty folder). Images will not be copied but a symbolic link will be created toward the <tiled images path>. The links will be renamed ```<type>_<slide_root_name>_<x>_<y>.jpeg``` with <type> being 'train_', 'test_' or 'valid_' followed by the svs name and the tile ID. 


## 0.3 Convert the JPEG tiles into TFRecord format


For the whole training set, the following code was used to convert JPEG to TFRecord:
```shell
python build_image_data.py --directory='jpeg_main_directory' --output_directory='outputfolder' --train_shards=1024 --validation_shards=128 --num_threads=4

python  00_preprocessing/build_image_data.py --directory='jpeg_tile_directory' --output_directory=/ifs/home/coudrn01/NN/Lung/Test_All512pxTiled/6_Healthy_Cancer_bis_TFRecord/train/ --train_shards=1024 --validation_shards=128 --num_threads=4 

```

The jpeg must not be directly inside 'jpeg_tile_directory' must in subfolders with names corresponding to the labels
jpeg_tile_directory/TCGA-LUAD
jpeg_tile_directory/TCGA-LUSC

The name of the tiles are :
<type>_name_x_y.jpef
with type being "test", "train" or "valid", name the TCGA name of the slide, x and y the tile coordinates.


The same was done for the test set:
```shell
python  00_preprocessing/build_TF_test.py --directory='jpeg_tile_directory'  --output_directory='output_dir' --num_threads=1 --one_FT_per_Tile=False

```
IF "one_FT_per_Tile" is True, there will be 1 TFRecord file per Tile created. Otherwise, it will created 1 TFRecord file per Slide.

Note: This code was adapted from https://github.com/awslabs/deeplearning-benchmark/blob/master/tensorflow/inception/inception/data/build_image_data.py




# 1 - Re-training from scratch

Code in 01_training.


Build the model. Note that we need to make sure the TensorFlow is ready to use before this as this command will not build TensorFlow.
```shell
cd 01_training/inception
bazel build inception/imagenet_train
```

Run it for all the training images:
```shell
bazel-bin/inception/imagenet_train --num_gpus=1 --batch_size=30 --train_dir='output_directory' --data_dir='TFRecord_images_directory'
```

botteneck, graph, variables... are saved in the output_directory 

Evaluate as the training goes on: briefly, one can evaluate the model by running a test on the validation set (must be started on a different node from the one used for training):

To prepare the run:
```shell
bazel build inception/imagenet_eval
```
To actually run it:
```shell
bazel-bin/inception/imagenet_eval --checkpoint_dir='0_scratch/' --eval_dir='output_directory' --run_once --data_dir='validation_TFRecord_images'
```

The precision @ 1  measures how often the highest scoring prediction from the model matched the  label
Much like the training script, imagenet_eval.py also exports summaries that may be visualized in TensorBoard:

```shell
tensorboard --logdir='checkpoint_dir'
```




# 2 - Run the classification on the test images

Code in 02_testing.

Usage:
```shell
python 02_testing/nc_imagenet_eval.py --checkpoint_dir='0_scratch/' --eval_dir='output_directory' --run_once --data_dir='test_TFperSlide'
```

data_dir contains the images in TFRecord format, with 1 TFRecord file per slide.
In the eval_dir, it will generate files:
*  out_FPTPrate_PcTiles.txt and out_FPTPrate_ScoreTiles.txt: info for the ROC curve after aggregation of the results using the percentage of properly classified tiles or the average score aggregation technique
*  out_filename_Stats.txt: a text file with output information: <tilename> <True/False classification> [<output probilities>]
*  node2048/: a subfolder where each file correspond to a tile such as the filenames are ```test_<svs name>_<tile ID x>_<tile ID y>.net2048``` and the first line of the file contains: ``` <True / False> \tab [<Background prob> <Prob class 1> <Prob class 2>]  <TP prob>```, and the next 2048 lines correspond to the output of the last-but-one layer




# 3 - Analyze the outcome by creating heat-maps

Code in 03_postprocessing.


