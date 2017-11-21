#!/bin/tcsh
#$ -pe openmpi 1
#$ -A TensorFlow
#$ -N rqsub_3Class_train
#$ -cwd
#$ -S /bin/tcsh
#$ -q gpu0.q
# #$ -q gpu0.q@gpu001.cm.cluster 
# #$ -q gpu1.q@gpu102.cm.cluster

module load cuda/8.0
module load python/3.5.3
module load bazel/0.4.4
