#!/bin/bash
#SBATCH --job-name=resnet_multi_1cl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --constraint=GPURAM_Min_16GB
##SBATCH --nodelist=eris
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainRESNETModelMulti_1cl.py
#python3 -m pdb trainRESNETModelMulti_1cl.py

conda deactivate