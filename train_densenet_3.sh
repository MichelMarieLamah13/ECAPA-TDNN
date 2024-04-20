#!/bin/bash
# JobNames v2: densenet_3(voxceleb), densenet_cc_3 (cn-celeb), densenet_vc_3 (vietnam-celeb)
#SBATCH --job-name=densenet_vc_3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --exclude=helios,apollon,eris
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainDENSENETModel_3.py --config config_densenet_vietnam_celeb_3.yml
# python3 trainDENSENETModel_3.py --config config_densenet_cn_celeb_3.yml
# python3 trainDENSENETModel_3.py --config config_densenet_3.yml


conda deactivate