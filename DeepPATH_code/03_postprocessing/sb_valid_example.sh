#!/bin/bash
#SBATCH --partition=fn_short,gpu4_dev,gpu8_dev
#SBATCH --job-name=ts_ML101_valid
#SBATCH --ntasks=1
#SBATCH --output=rq_ts_ML101_valid__%A_%a.out
#SBATCH  --error=rq_ts_ML101_valid_%A_%a.err
#SBATCH --mem=10G

module load python/gpu/3.6.5

export CHECKPOINT_PATH='/path_to_checkpoints/train/ML101/resultsML101/'
export OUTPUT_DIR='/path_to_save_outputs/valid_test/ML101/valid'
export DATA_DIR='/path_to_TFRecord_files/preprocess/ML101_TFRecord_valid/'
export BASENAME='valid_'
export LABEL_FILE='labels.txt'
export ExpName='ML101_valid_'
export STAT_FILE_FILTER='FALSE'
declare -i RefLabel=0
# To run the validation only on a subset of tiles filtered according to a previous "out_filename_Stats.txt" file with a given class:
# export STAT_FILE_FILTER='/gpfs/scratch/coudrn01/NN_test/Melanoma_TMB/validtest/4020_both_xml_Normalized_10x/segment_P086/test_33000k/out_filename_Stats.txt'
# declare -i RefLabel=1

# check if next checkpoint available
declare -i count=2500 
declare -i step=2500
declare -i NbClasses=2
declare -i PatientID=10


while true; do
	echo $count
	if [ -f $CHECKPOINT_PATH/model.ckpt-$count.meta ]; then
		echo $CHECKPOINT_PATH/model.ckpt-$count.meta " exists"
		export TEST_OUTPUT=$OUTPUT_DIR/test_$count'k'
		if [ ! -d $TEST_OUTPUT ]; then
			mkdir -p $TEST_OUTPUT
			# create temporary directory for checkpoints
			mkdir  -p $TEST_OUTPUT/tmp_checkpoints
			export CUR_CHECKPOINT=$TEST_OUTPUT/tmp_checkpoints
		
	
			ln -s $CHECKPOINT_PATH/*-$count.* $CUR_CHECKPOINT/.
			touch $CUR_CHECKPOINT/checkpoint
			echo 'model_checkpoint_path: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' > $CUR_CHECKPOINT/checkpoint
			echo 'all_model_checkpoint_paths: "'$CUR_CHECKPOINT'/model.ckpt-'$count'"' >> $CUR_CHECKPOINT/checkpoint

			export OUTFILENAME=$TEST_OUTPUT/out_filename_Stats.txt

			sbatch --job-name=$ExpName$BASENAME$count  --output=rq_$ExpName$BASENAME$count_%A.out --error=rq_$ExpName$BASENAME$count_%A.err sb_TF_ROC_2.sh $TEST_OUTPUT $DATA_DIR $BASENAME $NbClasses $OUTFILENAME $LABEL_FILE $CUR_CHECKPOINT $PatientID $STAT_FILE_FILTER $RefLabel

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




