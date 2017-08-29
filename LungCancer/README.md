
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


Then sort according to cancer type:
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


Check subfolder 00_preprocessing/TFRecord_2or3_Classes/ if it aimed at classifying 2 or 3 different classes:

For the whole training set, the following code was used to convert JPEG to TFRecord:
```shell
python build_image_data.py --directory='jpeg_main_directory' --output_directory='outputfolder' --train_shards=1024 --num_threads=4
```

The jpeg must not be directly inside 'jpeg_tile_directory' must in subfolders with names corresponding to the labels
jpeg_tile_directory/TCGA-LUAD
jpeg_tile_directory/TCGA-LUSC

The name of the tiles are :
<type>_name_x_y.jpef
with type being "test", "train" or "valid", name the TCGA name of the slide, x and y the tile coordinates.


The same was done for the test set:
```shell
python  build_TF_test.py --directory='jpeg_tile_directory'  --output_directory='output_dir' --num_threads=1 --one_FT_per_Tile=False

```
If "one_FT_per_Tile" is True, there will be 1 TFRecord file per Tile created. Otherwise, it will created 1 TFRecord file per Slide.

An optional parameter ```--ImageSet_basename='test'``` can be used to run it on 'test' (default), 'valid' or 'train' dataset

Note: This code was adapted from https://github.com/awslabs/deeplearning-benchmark/blob/master/tensorflow/inception/inception/data/build_image_data.py



Check subfolder 00_preprocessing/TFRecord_2or3_Classes/ if it aimed at multi-output classsification with 10 possibly concurent sclasses:

For the training set:
```shell
python build_image_data_multiClass.py --directory='jpeg_main_directory' --output_directory='outputfolder' --train_shards=1024 --num_threads=4  --labels_names=label_names.txt --labels=labels_files.txt
```
* ``` label_names.txt``` is a text file with the 10 possible labels, 1 per line
* ```labels_files.txt``` is a text file listing the mutations present ifor each patient. 1 patient per line, first column is patient ID (TCGA-38-4632 for example), second is mutation (TP53 for example)

For the test and validation sets:
```shell
python  build_TF_test_multiClass.py --directory='jpeg_tile_directory'  --output_directory='output_dir' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test' --labels_names=label_names.txt --labels=labels_files.txt
```





# 1 - Re-training from scratch
## 1.1 Training

Code in one of the subfolders of 01_training (depending on the type of run: 2 classes, 3 classes, or 10 multiclass outputs with softmax layer replaced by sigmoid)


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

## 1.2 Validation

Should be run on the validation test set at the same time as the training but on a different node (memory issues occur otherwise). Same code as for testing (see section 2.), but without the ```--run_once``` option (the program will run in an infinite loop and will need to be killed manually). 

Note: The current validation code only saves the validation accuracy, not the loss. The code still needs to be changed for that. 


## 1.3 Comments on the code

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







# 2 - Run the classification on the test images

Code in one of the subfolders of 02_testing  (depending on the type of run: 2 classes, 3 classes, or 10 multiclass outputs with softmax layer replaced by sigmoid)


Usage:
```shell
python nc_imagenet_eval.py --checkpoint_dir='0_scratch/' --eval_dir='output_directory' --run_once --data_dir='test_TFperSlide' --batch_size 30 --ImageSet_basename='test_'

```
An optional parameter ```--ImageSet_basename='test'``` can be used to run it on 'test' (default), 'valid' or 'train' dataset

data_dir contains the images in TFRecord format, with 1 TFRecord file per slide.
In the eval_dir, it will generate files:
*  out_FPTPrate_PcTiles.txt and out_FPTPrate_ScoreTiles.txt: info for the ROC curve after aggregation of the results using the percentage of properly classified tiles or the average score aggregation technique
*  out_filename_Stats.txt: a text file with output information: <tilename> <True/False classification> [<output probilities>]
*  node2048/: a subfolder where each file correspond to a tile such as the filenames are ```test_<svs name>_<tile ID x>_<tile ID y>.net2048``` and the first line of the file contains: ``` <True / False> \tab [<Background prob> <Prob class 1> <Prob class 2>]  <TP prob>```, and the next 2048 lines correspond to the output of the last-but-one layer




# 3 - Analyze the outcome

## Code in 03_postprocessing/2Classes for 2 classes:
Generate heat-maps per slides (all test slides in a given folder; code not optimized and slow):
```shell
python 0f_HeatMap.py  --image_file 'directory_to_jpeg_classes' --tiles_overlap 0 --output_dir 'result_folder' --tiles_stats 'out_filename_Stats.txt' --resample_factor 4 --tiles_filter 'TCGA-05-5425'
```
*  ```image_file``` is the outcome folder of ```00_preprocessing/0d_SortTiles_stage.py``` (it has one sub-folder per class and jpeg tile images in each of them)
*  ```--tiles_stats out_filename_Stats.txt``` is one of the output files generated by ```02_testing/nc_imagenet_eval.py```. 
* the size of the heatmaps will be reduced by ```resample_factor```. For large slides, a high number is advised.
* ```--tiles_filter```: to be used if you want to process only some of the images


## Code in 03_postprocessing/3Classes for 3 classes:
ROC curves:
```shell
python 0h_ROC_sklearn.py  --file_stats out_filename_Stats3.txt  --output_dir 'output folder'
```
Generate heat-maps per slides (all test slides in a given folder; code not optimized and slow):
```shell
python 0f_0f_HeatMap_3classes.py  --image_file 'directory_to_jpeg_classes' --tiles_overlap 0 --output_dir 'result_folder' --tiles_stats 'out_filename_Stats.txt' --resample_factor 4 --tiles_filter 'TCGA-05-5425'
```
Generate probability distribution with means for each class for each slide:
```shell
python 0f_ProbHistogram.py --output_dir='result folder' --tiles_stats='out_filename_Statsout_filename_Stats.txt'
```

## Code in 03_postprocessing/multiClasses for 10-multi-output classification:
```shell
python  0h_ROC_MultiOutput.py  --file_stats 'MultiOuput/out_filename_Stats3.txt  --output_dir 'output folder' --labels_names label_names.txt --ref_stats 'LUAD/out_filename_Stats3.txt'
```
* ```--file_stats``` is the output generated by the multi-output classification
* ```--ref_stats``` (optional) is the output generated by the 2 or 3 classes classification and is used to filter and selected only LUAD tiles.



