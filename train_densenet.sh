#!/bin/bash
# JobNames v1: densenet(voxceleb), densenet_cc (cn-celeb), densenet_vc (vietnam-celeb)
#SBATCH --job-name=densenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
##SBATCH --exclude=helios,apollon,eris
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 trainDENSENETModel.py --config config_densenet_vietnam_celeb.yml
python3 trainDENSENETModel.py --config config_densenet_cn_celeb.yml
# python3 trainDENSENETModel.py --config config_densenet.yml


#python3 -m pdb trainDENSENETModel.py

conda deactivate