# 1 - Retrieve the images.

Example: processing of Lung Cancer images from the TCGA database:

* We originally downloaded the "Tissue Slides" dataset from the legacy website, "https://portal.gdc.cancer.gov/legacy-archive/search/f" via the gdc-client tool:
   - download the client from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
   - Create and download a manifest and metadata json file from the gdc website (examples attached here)
   - Download images using the manifest and the API:
   ```gdc-client.exe download -m gdc_manifest.txt```

 Some svs slides might be corrupted, in which case they could also be downloaded from the new website ("https://portal.gdc.cancer.gov/").

# 2 - LUAD/LUSC/Normal classification
## 2.1 - Pre-processing - tiling

Tile the images using the magnification (20x) and tile size of interest (512x512 px in this example):

```shell
python 00_preprocessing/0b_tileLoop_deepzoom4.py  -s 512 -e 0 -j 32 -B 50 -M 20 -o 512px_Tiled "downloaded_data/*/*svs"  
```
It takes about 10sec to generaste 10k tiles when using 30 CPUs.

## 2.2 - Pre-processing - sorting

* Sort the dataset into a test, train and validation cohort for a 3-way classifier (LUAD/LUSC/Normal). You need to create a new directory and run this job from that directory

```shell
mkdir r1_sorted_3Cla
cd r1_sorted_3Cla
python ../00_preprocessing/0d_SortTiles.py --SourceFolder='../512px_Tiled/' --Magnification=20.0  --MagDiffAllowed=0 --SortingOption=3  --PatientID=12 --nSplit 0 --JsonFile='../downloaded_data/metadata.cart.2017-03-02T00_36_30.276824.json' --PercentTest=15 --PercentValid=15
```
Sorting takes about 10-15 minutes on 1 CPU for about 1.3 millions tiles.

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

## 2.2 - Pre-processing - Convert to TFRecord

Convert data into TFRecord files for each dataset

```shell
mkdir r1_TFRecord_test
mkdir r1_TFRecord_valid
mkdir r1_TFRecord_train

python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='r1_sorted_3Cla/'  --output_directory='r1_TFRecord_test' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test'

python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='r1_sorted_3Cla/'  --output_directory='r1_TFRecord_valid' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='valid'

python 00_preprocessing/TFRecord_2or3_Classes/build_image_data.py --directory='r1_sorted_3Cla/' --output_directory='r1_TFRecord_train' --train_shards=1024  --validation_shards=128 --num_threads=16
```
It takes about 90 sec to convert 10k images using 1 CPU (test and valid). Multi-treading implemented on training set lowers it to about 10 seconds for 10k images on 32 CPUs.

## 2.3 - Train the 3-way classifier

```shell
mkdir r1_results

bazel build inception/imagenet_train

bazel-bin/inception/imagenet_train --num_gpus=4 --batch_size=400 --train_dir='r1_results' --data_dir='r1_TFRecord_train' --ClassNumber=3 --mode='0_softmax' --NbrOfImages=923893 --save_step_for_chekcpoint=2300 --max_steps=230001
```

## 2.4 - Validation and test 

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


# 3 - Mutations
## 3.1 - Retrieve mutation information

The somatic mutation data can be retrieved from the GDC website, at https://portal.gdc.cancer.gov/, or  here: https://xenabrowser.net/datapages/, and the barecode can be used to match images to sample. In this particular example, we use mutect2 "masked somatic mutations" (ignoring low impact mutations). We label samples/slides as mutated with respect to every gene if it had a non-silent mutations. We used maftools to parse the Mutect2 variants from TCGA which by default uses Variant Classifications with High/Moderate variant consequences. These include: "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site", "Translation_Start_Site", "Nonsense_Mutation", "Nonstop_Mutation", "In_Frame_Del", "In_Frame_Ins", "Missense_Mutation". We then picked the top 10 "known cancer genes" (https://cancer.sanger.ac.uk/census) with respect to the number of (non-silent) mutation across our dataset, excluding genes like TNN which are known to be frequently mutated in general (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4267152/). We can then generate a label file where the first column is the slide ID, and the second the mutation name (if a slide has several mutations, then it will have several lines - example in file attached labels_r3.txt), and a reference file with the list of possible mutations (see labelref_r3.txt).


## 3.2 - slide segmentations
* To process mutations of LUAD images, there are different ways to do it. First, to extract probability of LUAD tiles on all LUAD tiles, we'll run them through the above classifier:

* Sort the tiles, assigning them all to "test"

```shell
mkdir r2_LUAD_segmentation
cd r2_LUAD_segmentation
python 00_preprocessing/0d_SortTiles.py --SourceFolder='../512px_Tiled/' --Magnification=20.0  --MagDiffAllowed=0 --SortingOption=3  --PatientID=12 --nSplit 0 --JsonFile='../downloaded_data/metadata.cart.2017-03-02T00_36_30.276824.json' --PercentTest=100 --PercentValid=0
```

Since Normal and LUSC do not interest us, delete their content (the content only - not the folder - the number of folders in that directory is used to identify the total number of possible classes):

```shell
rm -rf TCGA-lUSC/*
rm -rf Solid_Tissue_Normal/*
```

* convert to TFRecord: 
```shell
mkdir r2_TFRecord_test

python   00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='r2_LUAD_segmentation/'  --output_directory='r2_TFRecord_test' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test'
```

* Segment the LUAD tiles using the checkpoint giving the best validation/test AUC

```shell
export CHECKPOINT_PATH='r1_results'
export OUTPUT_DIR='r2_test'
export DATA_DIR='r2_TFRecord_test'
export LABEL_FILE='labelref_r1.txt'

# Best checkpoints
declare -i count=69000 
declare -i NbClasses=3

# create temporary directory for checkpoints
mkdir  -p $OUTPUT_DIR/tmp_checkpoints
export CUR_CHECKPOINT=$OUTPUT_DIR/tmp_checkpoints

export TEST_OUTPUT=$OUTPUT_DIR/test_$count'k'
mkdir -p $TEST_OUTPUT
			
ln -s $CHECKPOINT_PATH/*-$count.* $CUR_CHECKPOINT/.
touch $CUR_CHECKPOINT/checkpoint
echo 'model_checkpoint_path: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' > $CUR_CHECKPOINT/checkpoint
echo 'all_model_checkpoint_paths: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' >> $CUR_CHECKPOINT/checkpoint

# Test
python 02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir=$CUR_CHECKPOINT --eval_dir=$OUTPUT_DIR --data_dir=$DATA_DIR  --batch_size 300  --run_once --ImageSet_basename='test_' --ClassNumber $NbClasses --mode='0_softmax'  --TVmode='test'
			# wait

mv $OUTPUT_DIR/out* $TEST_OUTPUT/.
```


## 3.3 - multi-output classification

* sort the LUAD tiles identified as LUAD intro a train, valid a test set for mutation analysis, filtering with the "outFilenameStats" to only include LUAD tiles

```shell
mkdir r3_LUAD_sorted
cd r3_LUAD_sorted

python ../00_preprocessing/0d_SortTiles.py --SourceFolder='../512px_Tiled_NewPortal/'  --Magnification=20  --MagDiffAllowed=0 --SortingOption=10  --PatientID=-1 --PercentTest=15 --PercentValid=15 --nSplit 0 --outFilenameStats='../r2_test/test_69000k/out_filename_Stats.txt'

```

* Convert to TFRecord:
```shell
# valid
python 00_preprocessing/TFRecord_multi_Classes/build_TF_test_multiClass.py --directory='r3_LUAD_sorted/512px_Tiled_NewPortal/'  --output_directory='r3_TFRecord_valid' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='valid' --labels_names='labelref_r3.txt' --labels='labels_r3.txt' --PatientID=14

# test
python  00_preprocessing/TFRecord_multi_Classes/build_TF_test_multiClass.py --directory='r3_LUAD_sorted/512px_Tiled_NewPortal'  --output_directory='r3_TFRecord_test' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test' --labels_names='labelref_r3.txt' --labels='labels_r3.txt' --PatientID=14

# train
python 00_preprocessing/TFRecord_multi_Classes/build_image_data_multiClass.py --directory='r3_LUAD_sorted/512px_Tiled_NewPortal' --output_directory='r3_TFRecord_train' --train_shards=1024 --validation_shards=128 --num_threads=16  --labels_names='labelref_r3.txt' --labels='labels_r3.txt' --PatientID=14
```

* train the model with 10-class sigmoid classifier:
```shell
bazel-bin/inception/imagenet_train --num_gpus=4 --batch_size=400 --train_dir="r3_results_train" --data_dir="r3_TFRecord_train" --ClassNumber=10 --mode='1_sigmoid' --NbrOfImages=326613 --save_step_for_chekcpoint=815  --max_steps=81501
```

* once the checkpoints start being saved, we can start runing the valid and test sets:

```shell
export CHECKPOINT_PATH='full_ath_to/r3_results_train/'
export OUTPUT_DIR='full_path_to/r3_valid'
export DATA_DIR='r3_TFRecord_valid'
export LABEL_FILE='labelref_r3.txt'


# create temporary directory for checkpoints
mkdir  -p $OUTPUT_DIR/tmp_checkpoints
export CUR_CHECKPOINT=$OUTPUT_DIR/tmp_checkpoints


# check if next checkpoint available
declare -i count=815
declare -i step=815
declare -i NbClasses=10

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
                        python /gpfs/scratch/coudrn01/NN_test/code/DeepPATH/DeepPATH_code/02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir=$CUR_CHECKPOINT --eval_dir=$OUTPUT_DIR --data_dir=$DATA_DIR  --batch_size 200  --run_once --ImageSet_basename='valid_' --ClassNumber $NbClasses --mode='1_sigmoid'  --TVmode='test'
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
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c1a*  | sed -e 's/k\/out2_roc_data_AvPb_c1a/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_1.txt


# summarize all AUC per slide (average probability) for macro average: 
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_macro*  | sed -e 's/k\/out2_roc_data_AvPb_macro_/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_macro.txt

# summarize all AUC per slide (average probability) for micro average: 
ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_micro*  | sed -e 's/k\/out2_roc_data_AvPb_micro_/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_micro.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c2*  | sed -e 's/k\/out2_roc_data_AvPb_c2/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_2.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c3*  | sed -e 's/k\/out2_roc_data_AvPb_c3/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_3.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c4*  | sed -e 's/k\/out2_roc_data_AvPb_c4/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_4.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c5*  | sed -e 's/k\/out2_roc_data_AvPb_c5/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_5.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c6*  | sed -e 's/k\/out2_roc_data_AvPb_c6/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_6.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c7*  | sed -e 's/k\/out2_roc_data_AvPb_c7/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_7.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c8*  | sed -e 's/k\/out2_roc_data_AvPb_c8/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_8.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c9*  | sed -e 's/k\/out2_roc_data_AvPb_c9/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_9.txt

ls -tr $OUTPUT_DIR/test_*/out2_roc_data_AvPb_c10*  | sed -e 's/k\/out2_roc_data_AvPb_c10/ /' | sed -e 's/test_/ /' | sed -e 's/_/ /g' | sed -e 's/.txt//'   > $OUTPUT_DIR/valid_out2_AvPb_AUCs_10.txt

```

A similar code can be used for the test check by modifying the corresponding options and inputs.


