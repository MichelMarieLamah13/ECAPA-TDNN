#!/bin/bash
# Using pooling not adaptative
# JobNames: densenet_cleaned_vox(voxceleb), densenet_cleaned_cc (cn-celeb), densenet_cleaned_vc (vietnam-celeb)
#SBATCH --job-name=densenet_cleaned_vc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
##SBATCH --exclude=helios,apollon,eris
##SBATCH --nodelist=eris
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainDENSENETModel_3.py --config config_densenet_vietnam_celeb_3_std_cleaned.yml
# python3 trainDENSENETModel_3.py --config config_densenet_cn_celeb_3_std_cleaned.yml
# python3 trainDENSENETModel_3.py --config config_densenet_3_std_cleaned.yml


conda deactivate