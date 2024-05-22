#!/bin/bash
#SBATCH --job-name=resnet_multi_1cl_7cp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainRESNETModelMulti_1cl.py --config config_resnet_multi_1cl_7cp.yml
#python3 -m pdb trainRESNETModelMulti_1cl.py --config config_resnet_multi_1cl_7cp.yml

conda deactivate