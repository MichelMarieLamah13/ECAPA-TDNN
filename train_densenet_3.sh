#!/bin/bash
# Using pooling not adaptative
# JobNames: densenet_3_std(voxceleb), densenet_cc_3_std (cn-celeb), densenet_vc_3_std (vietnam-celeb)
# JobNames: densenet_3_std_2(voxceleb), densenet_cc_3_std_2 (cn-celeb), densenet_vc_3_std_2 (vietnam-celeb)
# JobNames: densenet_3_std_3(voxceleb), densenet_cc_3_std_3 (cn-celeb), densenet_vc_3_std_3 (vietnam-celeb)
#SBATCH --job-name=densenet_cc_3_std
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPURAM_Min_16GB
#SBATCH --time=7-00:00:00
##SBATCH --exclude=helios,apollon,eris
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j_output.log
#SBATCH --error=%x_%j_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

# python3 trainDENSENETModel_3.py --config config_densenet_vietnam_celeb_3_std.yml
python3 trainDENSENETModel_3.py --config config_densenet_cn_celeb_3_std.yml
# python3 trainDENSENETModel_3.py --config config_densenet_3_std.yml

# python3 trainDENSENETModel_3.py --config config_densenet_vietnam_celeb_3_std_2.yml

# python3 trainDENSENETModel_3.py --config config_densenet_vietnam_celeb_3_std_3.yml


conda deactivate