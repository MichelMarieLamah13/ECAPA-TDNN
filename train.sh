#!/bin/bash
#SBATCH --job-name=train_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=5
#SBATCH --output=train_output.log
#SBATCH --error=train_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python trainECAPAModel.py --save_path exps/exp1

conda deactivate