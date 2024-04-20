#!/bin/bash
# resnet_multi_ncl (original), tmp_1(voxceleb),
# tmp_2(voxceleb + cn_celeb), tmp_2_1 Moyenne(voxceleb + cn_celeb)
# tmp_3(voxceleb + vietnam_celeb), tmp_3_1 Moyenne(voxceleb + vietnam_celeb)
# resnet_multi_ncl_2 Moyenne(voxceleb + vietnam_celeb + cn_celeb)
#SBATCH --job-name=resnet_multi_ncl_2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=32GB
#SBATCH --constraint=GPURAM_Min_16GB
#SBATCH --exclude=helios,apollon,eris
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_output.log
#SBATCH --error=%x_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainRESNETModelMulti_ncl.py
#python3 -m pdb trainRESNETModelMulti_ncl.py

conda deactivate