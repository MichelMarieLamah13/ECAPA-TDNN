#!/bin/bash
#SBATCH --job-name=eval_ecapa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=5
#SBATCH --output=eval_output.log
#SBATCH --error=eval_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --eval --initial_model exps/pretrain.model

conda deactivate