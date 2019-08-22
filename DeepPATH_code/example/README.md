Example: processing of Lung Cancer images from the TCGA database:

* We originally downloaded the "Tissue Slides" dataset from the legacy website, "https://portal.gdc.cancer.gov/legacy-archive/search/f" via the gdc-client tool:
   - download the client from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
   - Create and download a manifest and metadata json file from the gdc website (examples attached here)
   - Download images using the manifest and the API:
   ```gdc-client.exe download -m gdc_manifest.txt```

 Some svs slides might be corrupted, in which case they could also be downloaded from the new website ("https://portal.gdc.cancer.gov/").

* Tile the images using the magnification (20x) and tile size of interest (512x512 px in this example):

```shell
python 00_preprocessing/0b_tileLoop_deepzoom4.py  -s 512 -e 0 -j 32 -B 50 -M 20 -o 512px_Tiled "downloaded_data/*/*svs"  
```



* Sort the dataset into a test, train and validation cohort for a 3-way classifier (LUAD/LUSC/Normal). You need to create a new directory and run this job from that directory

```shell
mkdir r1_sorted_3Cla
cd r1_sorted_3Cla
python ../00_preprocessing/0d_SortTiles.py --SourceFolder='../512px_Tiled/' --Magnification=20.0  --MagDiffAllowed=0 --SortingOption=3  --PatientID=12 --nSplit 0 --JsonFile='../downloaded_data/metadata.cart.2017-03-02T00_36_30.276824.json' --PercentTest=15 --PercentValid=15
```

Once the process is complete, it should display how many slides and images are assigned to each dataset and each class. In this particular test run, we obtained this for the number of tiles in each dataset (first number is total):

```shell
Solid_Tissue_Normal 249609
Solid_Tissue_Normal_test 33777
Solid_Tissue_Normal_train 176972
Solid_Tissue_Normal_valid 38860
TCGA-LUAD 531420
TCGA-LUAD_test 91377
TCGA-LUAD_train 368497
TCGA-LUAD_valid 71546
TCGA-LUSC 554999
TCGA-LUSC_test 85465
TCGA-LUSC_train 378424
TCGA-LUSC_valid 91110
```

meaning a total of 923,893 tiles for training. And for the number of tiles:
```shell
Solid_Tissue_Normal 591
Solid_Tissue_Normal_test 81
Solid_Tissue_Normal_train 416
Solid_Tissue_Normal_valid 94
TCGA-LUAD 823
TCGA-LUAD_test 126
TCGA-LUAD_train 572
TCGA-LUAD_valid 125
TCGA-LUSC 753
TCGA-LUSC_test 115
TCGA-LUSC_train 522
TCGA-LUSC_valid 116
```

Note: on the new website, it looks like the format and information in the metadata json files have changed (the sample type is now in the biospecimen file, and the architecture of the Json seems modified). The option "3" might only compatible with the old format. If you want to use the new formated Json files, you will need to modify the code or use option 14 and create your own label file from it. 

* Convert data into TFRecord files for each dataset

```shell
mkdir r1_TFRecord_test
mkdir r1_TFRecord_valid
mkdir r1_TFRecord_train

python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='r1_sorted_3Cla/'  --output_directory='r1_TFRecord_test' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test'

python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='r1_sorted_3Cla/'  --output_directory='r1_TFRecord_valid' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='valid'

python 00_preprocessing/TFRecord_2or3_Classes/build_image_data.py --directory='r1_sorted_3Cla/' --output_directory='r1_TFRecord_train' --train_shards=1024  --validation_shards=128 --num_threads=16
```

* Train the 3-way classifier

```shell
mkdir r1_results

bazel build inception/imagenet_train

bazel-bin/inception/imagenet_train --num_gpus=4 --batch_size=400 --train_dir='r1_results' --data_dir='r1_TFRecord_train' --ClassNumber=3 --mode='0_softmax' --NbrOfImages=923893 --save_step_for_chekcpoint=2300 --max_steps=230001
```


* As the first checkpoint appear, you can start running the validation set on it. Create a "labelref_r1.txt" text file with the list of possible classes (see attached example). To run it in on loop on all existing checkpoints, the following code can be adapted:


```shell
mkdir r1_valid
export CHECKPOINT_PATH='/fullpath_to/r1_results'
export OUTPUT_DIR='/fullpath_to/r1_valid'
export DATA_DIR='r1_TFRecord_valid'
export LABEL_FILE='labelref_r1.txt'

# check if next checkpoint available
declare -i count=2300 
declare -i step=2300
declare -i NbClasses=3

# create temporary directory for checkpoints
mkdir  -p $OUTPUT_DIR/tmp_checkpoints
export CUR_CHECKPOINT=$OUTPUT_DIR/tmp_checkpoints

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
			python 02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir=$CUR_CHECKPOINT --eval_dir=$OUTPUT_DIR --data_dir=$DATA_DIR  --batch_size 300  --run_once --ImageSet_basename='valid_' --ClassNumber $NbClasses --mode='0_softmax'  --TVmode='test'
			# wait

			mv $OUTPUT_DIR/out* $TEST_OUTPUT/.

			# ROC
			export OUTFILENAME=$TEST_OUTPUT/out_filename_Stats.txt
			python 03_postprocessing/0h_ROC_MultiOutput_BootStrap.py --file_stats=$OUTFILENAME  --output_dir=$TEST_OUTPUT --labels_names=$LABEL_FILE

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
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c1a*  | sed -e 's/k\/out2_roc_data_AvPb_c1a/ /' | sed -e 's/test_/ /' | sed -e 's
/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_1.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c2*  | sed -e 's/k\/out2_roc_data_AvPb_c2/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_2.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c3*  | sed -e 's/k\/out2_roc_data_AvPb_c3/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_3.txt

```

The same code can be adapted to run the checkpoints on the test set.

In the output directory, there will be 1 sub-folder per checkpoint with the data (per tile and per-slide AUCs and all raw data related to AUC compitation).
The valid_out2_AvPb_AUCs_1.txt summarizes the per slide AUC obtained by averaging the per-tile probabilities.


