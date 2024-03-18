#!/bin/bash
# resnet_multi_ncl (original), tmp_1(voxceleb)
#SBATCH --job-name=tmp_1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --constraint=GPURAM_Min_16GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainRESNETModelMulti_ncl.py
#python3 -m pdb trainRESNETModelMulti_ncl.py

conda deactivate