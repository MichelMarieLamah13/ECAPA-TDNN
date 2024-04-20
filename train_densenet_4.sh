#!/bin/bash
# JobNames std: densenet_4_std(voxceleb), densenet_cc_4_std(cn-celeb), densenet_vc_4_std(vietnam-celeb)
#SBATCH --job-name=densenet_vc_4_std
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=eris
##SBATCH --exclude=helios,apollon
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainDENSENETModel_4.py --config config_densenet_vietnam_celeb_4_std.yml
# python3 trainDENSENETModel_4.py --config config_densenet_cn_celeb_4_std.yml
# python3 trainDENSENETModel_4.py --config config_densenet_4_std.yml


conda deactivate