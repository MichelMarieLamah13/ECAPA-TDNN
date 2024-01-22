#!/bin/bash
#SBATCH --job-name=bw2d
#SBATCH --partition=gpu
# #SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
# #SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
#SBATCH --gres=gpu:1
#SBATCH --exclude=apollon,eris
#SBATCH --time=7-00:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=10
#SBATCH --output=bw2d_output.log
#SBATCH --error=bw2d_error.log


source /etc/profile.d/conda.sh
conda activate ecapa_tdnn

python3 trainECAPAModel.py --save_path exps/exp3 --feat_type wav2vec2 --feat_dim 768 --n_cpu 10 --batch_size 128 --is_2d

conda deactivate