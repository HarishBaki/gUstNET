#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:3
#SBATCH --output=/home/harish/scratch/%j.out

# This needs to point to your conda env directory
CONDA_ENV=/home/harish/miniconda3/envs/gUstNET

# Load conda environment
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

python gpu_rank.py

