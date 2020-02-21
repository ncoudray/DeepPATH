#!/bin/bash
#SBATCH --partition=cpu_dev,gpu4_dev,gpu8_dev
#SBATCH --job-name=Compile
#SBATCH --ntasks=1
#SBATCH --output=rq_compile_%A_%a.out
#SBATCH --error=rq_compile_%A_%a.err
#SBATCH --mem=50G
# #SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
module load python/gpu/3.6.5
module load bazel/0.15.2

bazel build inception/imagenet_train

