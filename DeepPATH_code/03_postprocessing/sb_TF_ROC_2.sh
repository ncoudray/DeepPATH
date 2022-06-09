#!/bin/bash
## #SBATCH --partition=gpu4_medium,gpu4_long,gpu8_long,gpu8_medium
#SBATCH --partition=gpu4_short,gpu8_short,gpu4_medium,gpu4_long,gpu8_long,gpu8_medium
#SBATCH --exclude=gpu-0007
#SBATCH --job-name=F001_valid
#SBATCH --ntasks=4
# #SBATCH --output=rq_F001_v_%A_%a.out
# #SBATCH --error=rq_F001_v_%A_%a.err
#SBATCH --mem=40G
#SBATCH --gres=gpu:2

module unload python/gpu/3.6.5
module load anaconda3/gpu/5.2.0 
if [[ $CONDA_SHLVL == 1 ]]; then conda deactivate; fi
conda activate /gpfs/data/coudraylab/NN/env/env_deepPath
unset PYTHONPATH

TEST_OUTPUT=$1
DATA_DIR=$2
BASENAME=$3
NbClasses=$4
OUTFILENAME=$5
LABEL_FILE=$6
CUR_CHECKPOINT=$7
PatientID=$8
STAT_FILE_FILTER=$9
RefLabel=${10}

echo $CUR_CHECKPOINT
echo $TEST_OUTPUT
echo $DATA_DIR
echo $BASENAME
echo $NbClasses

# Test
python /gpfs/data/coudraylab/NN/github/DeepPATH_code/02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir=$CUR_CHECKPOINT --eval_dir=$TEST_OUTPUT --data_dir=$DATA_DIR  --batch_size 300  --run_once --ImageSet_basename=$BASENAME --ClassNumber $NbClasses --mode='0_softmax'  --TVmode='test'


echo " ROC"
echo $OUTFILENAME
echo $TEST_OUTPUT
echo $LABEL_FILE
echo $PatientID
# ROC
export OUTFILENAME=$TEST_OUTPUT/out_filename_Stats.txt
export OUTFILENAMEUNIQ=$TEST_OUTPUT/out_filename_Stats_unique.txt

sort -u $OUTFILENAME > $OUTFILENAMEUNIQ



conda deactivate 
conda activate /gpfs/data/coudraylab/NN/env/env_deepPath_4_376

if [[ $RefLabel -eq "FALSE" ]]
then
	python /gpfs/data/coudraylab/NN/github/DeepPATH_code/03_postprocessing/0h_ROC_MultiOutput_BootStrap_2.py --file_stats=$OUTFILENAMEUNIQ  --output_dir=$TEST_OUTPUT --labels_names=$LABEL_FILE --PatientID=$PatientID --color="black,red,blue,blueviolet,yellow,darkolivegreen,green,cyan,royalblue,fuchsia"
# --color="black,darkred,red,orange,gold,yellow,darkolivegreen,green,cyan,royalblue,blueviolet,fuchsia"
else
        python /gpfs/data/coudraylab/NN/github/DeepPATH_code/03_postprocessing/0h_ROC_MultiOutput_BootStrap_2.py --file_stats=$OUTFILENAMEUNIQ  --output_dir=$TEST_OUTPUT --labels_names=$LABEL_FILE --PatientID=$PatientID  --ref_file=$STAT_FILE_FILTER --ref_label=$RefLabel --ref_thresh=-2 
fi

# THRESH=`ls $TEST_OUTPUT | grep out1_roc_data_AvP | sed -e 's/_/ /g' | sed -e 's/.txt/ /g' | awk '{print $NF}' | grep t | sed -e 's/t/ /g'`
THRESH=`ls $TEST_OUTPUT | grep out1_roc_data_AvP | sed -e 's/_/ /g' | sed -e 's/.txt/ /g' | awk '{print $(NF-1)}'  | grep J | sed -e 's/J/ /g'`
THRESH=`echo $THRESH | sed -e 's/ /,/g'`
echo $THRESH

python /gpfs/data/coudraylab/NN/github/DeepPATH_code/03_postprocessing/0i_Sensitivity_Specificity.py --threshold=$THRESH --labelFile $LABEL_FILE --PatientID $PatientID --files_stats $OUTFILENAMEUNIQ --outputPath=$TEST_OUTPUT




