#!/bin/bash
#SBATCH --job-name=process_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --output=process_ecapa_output.log
#SBATCH --error=process_ecapa_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

rm -rf data

conda deactivate