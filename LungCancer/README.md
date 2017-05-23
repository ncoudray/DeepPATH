
Preliminary comment: On the NYUm HPC cluster, 3 modules are needed. The commands mentioned below must be run through qsub scripts and not on the head node! Module needed are:
module load cuda/8.0
module load python/3.5.3
module load bazel/0.4.4


# 1 - Prepare the images.

Code in 00_preprocessing.

See from https://github.com/tensorflow/models/blob/master/inception/README.md for specific format needed by inception


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




# 2 - Re-training from scratch

Code in 01_training.


Build the model. Note that we need to make sure the TensorFlow is ready to use before this as this command will not build TensorFlow.
```shell
cd 01_training/inception
bazel build inception/imagenet_train
```

# run it for all the training images:
```shell
bazel-bin/inception/imagenet_train --num_gpus=1 --batch_size=30 --train_dir='output_directory' --data_dir='TFRecord_images_directory'
```

botteneck, graph, variables... are saved in the output_directory 

# Evaluate as the training goes on
Briefly, one can evaluate the model by running a test on the validation set (must be started on a different node from the one used for training):

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




# 3 - Run the classification on the test images

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

