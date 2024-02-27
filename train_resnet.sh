#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# #SBATCH --constraint=GPURAM_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainRESNETModel.py --config config_resnet.yml

conda deactivate