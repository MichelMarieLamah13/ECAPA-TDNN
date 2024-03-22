#!/bin/bash
# Job names: resnet(voxceleb) resnet_vc(vietnam-celeb), resnet_cc(cn-celeb)
#SBATCH --job-name=resnet_vc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_16GB
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 trainRESNETModel.py
# python3 trainRESNETModel.py --config config_resnet_cn_celeb.yml
python3 trainRESNETModel.py --config config_resnet_vietnam_celeb.yml
# python3 -m pdb trainRESNETModel.py

conda deactivate