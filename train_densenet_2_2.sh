#!/bin/bash
# JobNames v5: densenet_2_2(voxceleb), densenet_cc_2_2 (cn-celeb), densenet_vc_2_2 (vietnam-celeb)
#SBATCH --job-name=densenet_vc_2_2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint=GPURAM_Min_12GB
#SBATCH --time=7-00:00:00
##SBATCH --nodelist=eris
##SBATCH --exclude=helios,apollon,eris
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainDENSENETModel.py --config config_densenet_vietnam_celeb_2_2.yml

#python3 -m pdb trainDENSENETModel.py

conda deactivate