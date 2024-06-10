#!/bin/bash
# Job names: resnet_cleaned_vox(voxceleb) resnet_cleaned_vc(vietnam-celeb), resnet_cleaned_cc(cn-celeb)
#SBATCH --job-name=resnet_cleaned_vox
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
##SBATCH --exclude=eris,apollon,helios
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainRESNETModel.py --config config_resnet_cleaned.yml
# python3 trainRESNETModel.py --config config_resnet_cn_celeb_cleaned.yml
# python3 trainRESNETModel.py --config config_resnet_vietnam_celeb_cleaned.yml


conda deactivate