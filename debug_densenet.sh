#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn


# python3 -m pdb trainDENSENETModel.py --config config_densenet_vietnam_celeb.yml
# python3 -m pdb trainDENSENETModel.py --config config_densenet_vietnam_celeb_2.yml

python3 -m pdb trainDENSENETModel_3.py --config config_densenet_vietnam_celeb_3.yml

conda deactivate