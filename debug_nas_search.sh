#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 -m pdb trainNASSEARCHModel.py --config config_nas_search_vietnam_celeb.yml

conda deactivate